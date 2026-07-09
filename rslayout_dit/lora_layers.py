"""
LoRA layers for RSLayout-DiT
Based on the RSLayout-DiT implementation with causal attention masking
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class LoRALinearLayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer
    Implements: output = input + scale * up(down(input))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        network_alpha: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cond_width: int = 512,
        cond_height: int = 512,
        number: int = 0,
        n_loras: int = 1,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        # Network alpha for scaling
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        # Initialize weights
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

        # Condition parameters
        self.cond_height = cond_height
        self.cond_width = cond_width
        self.number = number  # Which LoRA this is (for multi-LoRA)
        self.n_loras = n_loras

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation with masking for condition tokens

        Args:
            hidden_states: [B, N, D] where N = img_tokens + cond_tokens

        Returns:
            LoRA delta: [B, N, D]
        """
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        # Calculate sizes
        batch_size = hidden_states.shape[0]
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        block_size = hidden_states.shape[1] - cond_size * self.n_loras

        # Create mask: only apply LoRA to this condition's tokens
        shape = (batch_size, hidden_states.shape[1], self.in_features)
        mask = torch.ones(shape, device=hidden_states.device, dtype=dtype)
        mask[:, :block_size + self.number * cond_size, :] = 0
        mask[:, block_size + (self.number + 1) * cond_size:, :] = 0
        hidden_states_masked = mask * hidden_states

        # Apply LoRA
        down_hidden_states = self.down(hidden_states_masked.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        # Scale by network alpha
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class MultiSingleStreamBlockLoraProcessor(nn.Module):
    """
    LoRA processor for FLUX single stream blocks
    Applies LoRA to Q, K, V projections with causal attention masking
    """

    def __init__(
        self,
        dim: int,
        ranks: list,
        lora_weights: list,
        network_alphas: list,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cond_width: int = 512,
        cond_height: int = 512,
        n_loras: int = 1,
        use_causal_mask: bool = True,
    ):
        super().__init__()
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.use_causal_mask = use_causal_mask

        # Create LoRA layers for Q, K, V
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            )
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            )
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            )
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond: bool = False,
    ) -> torch.FloatTensor:
        """
        Apply LoRA-modified attention with causal masking

        Args:
            attn: Attention module
            hidden_states: [B, N, D] concatenated [img_tokens, cond_tokens]
            use_cond: Whether to return separate condition outputs

        Returns:
            hidden_states or (hidden_states, cond_hidden_states)
        """
        batch_size = hidden_states.shape[0]

        # Compute Q, K, V with LoRA
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Add LoRA deltas
        for i in range(self.n_loras):
            query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
            key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
            value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        mask = None
        if self.use_causal_mask:
            # Build causal attention mask
            cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
            block_size = hidden_states.shape[1] - cond_size * self.n_loras
            seq_len = query.shape[2]

            # Mask: image tokens can see everything, condition tokens only see themselves
            mask = torch.ones((seq_len, seq_len), device=hidden_states.device)
            mask[:block_size, :] = 0  # Image tokens see all
            for i in range(self.n_loras):
                start = i * cond_size + block_size
                end = (i + 1) * cond_size + block_size
                mask[start:end, start:end] = 0  # Each condition sees itself
            mask = mask * -1e20
            mask = mask.to(query.dtype)

        # Scaled dot-product attention with mask
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
        )

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Keep the full sequence for current diffusers FLUX blocks. The block
        # owns splitting text/image tokens after attention.
        return hidden_states


class MultiDoubleStreamBlockLoraProcessor(nn.Module):
    """
    LoRA processor for FLUX double stream blocks
    Applies LoRA to Q, K, V, and projection layers
    """

    def __init__(
        self,
        dim: int,
        ranks: list,
        lora_weights: list,
        network_alphas: list,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cond_width: int = 512,
        cond_height: int = 512,
        n_loras: int = 1,
    ):
        super().__init__()
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height

        # Create LoRA layers for Q, K, V, and projection
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            )
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            )
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            )
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            )
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Apply LoRA-modified double stream attention

        Returns:
            (encoder_hidden_states, hidden_states, cond_hidden_states)
        """
        batch_size = hidden_states.shape[0]
        inner_dim = 3072
        head_dim = inner_dim // attn.heads

        # Context projections (text encoder hidden states)
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # Image projections with LoRA
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Add LoRA deltas
        for i in range(self.n_loras):
            query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
            key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
            value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Current diffusers FLUX applies RoPE to the joint text+image sequence.
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Attention over the joint text+image sequence
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        text_seq_len = encoder_hidden_states.shape[1]
        context_attn_output, hidden_states = hidden_states.split_with_sizes(
            [text_seq_len, hidden_states.shape[1] - text_seq_len], dim=1
        )

        # Apply output projection with LoRA
        hidden_states = attn.to_out[0](hidden_states)
        for i in range(self.n_loras):
            hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)

        hidden_states = attn.to_out[1](hidden_states)
        context_attn_output = attn.to_add_out(context_attn_output)

        return hidden_states, context_attn_output
