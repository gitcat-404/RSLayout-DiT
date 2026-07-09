# RSLayout-DiT

Official code release for **RSLayout-DiT: Parameter-Efficient Layout-Controlled Remote Sensing Image Generation with Diffusion Transformers**.

RSLayout-DiT trains a lightweight layout-control adapter for a FLUX-style remote-sensing diffusion transformer. Layout annotations are rendered as category-colored condition images, encoded into latent layout tokens, and injected into the transformer with condition-to-image causal attention. This release contains the training, inference, ablation, and evaluation code. Datasets, checkpoints, generated images, and logs are not included.

## News

- Code for RSLayout-DiT training and inference is provided.
- FLUX-ControlNet and text-only baseline scripts are included for ablation.
- CLIPScore, FID preparation, YOLOScore, and fine-grained evaluation helpers are included.

## Installation

Create an environment with CUDA-enabled PyTorch, then install the dependencies:

```bash
git clone https://github.com/gitcat-404/RSLayout-DiT.git
cd RSLayout-DiT

conda create -n rslayout-dit python=3.10 -y
conda activate rslayout-dit

pip install -r requirements.txt
```

The code was developed with `torch 2.5.1`, `torchvision 0.20.1`, `diffusers 0.36.0`, `transformers 4.57.6`, and `accelerate 1.10.1`.

You can check the basic imports with:

```bash
bash test_import.sh
```

## Model and Data

Download the remote-sensing FLUX backbone from Hugging Face:

- RS-FLUX: https://huggingface.co/BxuanZ/FLUX-RS

Download or prepare the datasets:

- DIOR-RSVG: https://github.com/zhanyang-nwpu/RSVG-pytorch
- DOTA: https://captain-whu.github.io/DOTA/

The original CC-Diff repository also provides processed remote-sensing resources:

- CC-Diff repository: https://github.com/AZZMM/CC-Diff
- DIOR-RSVG resource link: https://pan.baidu.com/s/1nBad0IK8BSM_pUBwdm5Z7g?pwd=ccdp
- DOTA resource link: https://pan.baidu.com/s/1P5vshdqUHFPOxH4JNveMbQ?pwd=ccdp

After preparation, each split should contain a `metadata.jsonl` file and the corresponding images. Images can be placed directly in the split directory or under an `images/` subdirectory.

```text
path_to_data/DIOR/
├── train/
│   ├── metadata.jsonl
│   └── *.jpg
├── val/
│   ├── metadata.jsonl
│   └── *.jpg
└── test/
    ├── metadata.jsonl
    └── *.jpg
```

Each metadata row should contain:

```json
{
  "file_name": "example.jpg",
  "caption": ["scene description", "airplane", "ship"],
  "bndboxes": [[x1, y1, x2, y2]],
  "obboxes": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
}
```

For HBB-only data, `obboxes` can be omitted or filled with horizontal-box polygons. For DOTA-style OBB data, use the four annotated vertices.

## Training

Set the backbone and dataset paths:

```bash
export MODEL_PATH=/path/to/FLUX-RS
export DATA_ROOT=path_to_data/DIOR
```

Single-GPU training:

```bash
bash scripts/train_rslayout_dit.sh
```

Four-GPU training:

```bash
export GPUS=0,1,2,3
export NUM_PROCESSES=4
export OUTPUT_DIR=checkpoint-rslayout-dit

bash scripts/train_rslayout_dit_multigpu.sh
```

Object-aware refinement can be launched from a trained LoRA checkpoint:

```bash
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 \
  train_rslayout_dit.py \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --resume_lora_path /path/to/base/lora.safetensors \
  --train_data_dir "$DATA_ROOT/train" \
  --validation_data_dir "$DATA_ROOT/val" \
  --max_train_steps 98550 \
  --learning_rate 2e-5 \
  --weighting_scheme logit_normal \
  --timestep_min 0.02 \
  --timestep_max 0.98 \
  --object_loss_weight 1.0 \
  --small_object_loss_weight 1.0 \
  --class_balance_loss_weight 0.5 \
  --boundary_loss_weight 0.5 \
  --output_dir checkpoint-rslayout-dit-objaware
```

## Inference

Generate images from the validation or test metadata:

```bash
export MODEL_PATH=/path/to/FLUX-RS
export DATA_ROOT=path_to_data/DIOR
export OUTPUT_DIR=outputs-rslayout-dit
export NUM_SAMPLES=0

bash scripts/infer_rslayout_dit.sh /path/to/lora.safetensors
```

You can also run inference directly:

```bash
python infer_rslayout_dit.py \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --lora_path /path/to/lora.safetensors \
  --data_dir "$DATA_ROOT/test" \
  --num_samples 0 \
  --output_dir outputs-rslayout-dit \
  --resolution 512 \
  --num_inference_steps 28 \
  --guidance_scale 3.5 \
  --lora_weight 1.0 \
  --render_style colored_polygons \
  --draw_arrows \
  --torch_dtype bf16
```

For custom layout-controlled generation, pass a JSON file with `--layout_json`.

## Evaluation

First arrange real and generated images into the evaluation folder:

```bash
python scripts/prepare_rslayout_eval.py \
  --metadata "$DATA_ROOT/test/metadata.jsonl" \
  --real-image-dir "$DATA_ROOT/JPEGImages" \
  --generated-dir outputs-rslayout-dit \
  --output-dir eval_work/rslayout_dit_test
```

Compute CLIPScore:

```bash
python eval/eval_clip_score.py \
  --folder eval_work/rslayout_dit_test/generated \
  --ann eval_work/rslayout_dit_test/metadata_clip.jsonl
```

Compute FID:

```bash
fidelity \
  --input1 eval_work/rslayout_dit_test/real \
  --input2 eval_work/rslayout_dit_test/generated \
  -b 16 -g 0 -f --samples-resize-and-crop 512
```

YOLOScore requires an external detector checkpoint:

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("/path/to/yolo-detector.pt")
results = model.val(
    data="eval_work/rslayout_dit_test/yolo/config.yaml",
    imgsz=800,
    batch=16,
    project="eval_work/rslayout_dit_test/yolo_runs",
    name="yoloscore",
    exist_ok=True,
)
print(results.results_dict)
PY
```

Fine-grained small-object, rare-class, and localization analyses are available in:

```bash
python tools/finegrained_yoloscore.py --help
python tools/sizewise_yoloscore.py --help
```

## Ablations

Text-only baseline:

```bash
python infer_text_only_baseline.py \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --data_dir "$DATA_ROOT/test" \
  --num_samples 0 \
  --output_dir outputs-text-only-baseline
```

FLUX-ControlNet ablation:

```bash
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 \
  train_controlnet_ablation.py \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --train_data_dir "$DATA_ROOT/train" \
  --validation_data_dir "$DATA_ROOT/val" \
  --output_dir checkpoint-rslayout-controlnet
```

The no-causal-mask ablation can be trained by adding:

```bash
--disable_causal_mask
```

to `train_rslayout_dit.py`.

## Acknowledgement

This project builds on the open-source remote-sensing generation ecosystem, including RS-FLUX, DIOR-RSVG, DOTA, and CC-Diff.
