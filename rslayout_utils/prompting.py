import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from .layout_condition import LayoutObject


CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "vehicle": "vehicle",
    "baseballfield": "baseball field",
    "groundtrackfield": "ground track field",
    "windmill": "windmill",
    "bridge": "bridge",
    "overpass": "overpass",
    "ship": "ship",
    "airplane": "airplane",
    "tenniscourt": "tennis court",
    "airport": "airport",
    "Expressway-Service-area": "expressway service area",
    "basketballcourt": "basketball court",
    "stadium": "stadium",
    "storagetank": "storage tank",
    "chimney": "chimney",
    "dam": "dam",
    "Expressway-toll-station": "expressway toll station",
    "golffield": "golf field",
    "trainstation": "train station",
    "harbor": "harbor",
    "plane": "airplane",
    "storage-tank": "storage tank",
    "baseball-diamond": "baseball diamond",
    "tennis-court": "tennis court",
    "basketball-court": "basketball court",
    "ground-track-field": "ground track field",
    "large-vehicle": "large vehicle",
    "small-vehicle": "small vehicle",
    "helicopter": "helicopter",
    "roundabout": "roundabout",
    "soccer-ball-field": "soccer ball field",
    "swimming-pool": "swimming pool",
    "solar-panel": "solar panel",
    "parking-lot": "parking lot",
    "railway-track": "railway track",
    "container-yard": "container yard",
    "pier": "pier",
    "helipad": "helipad",
    "oil-refinery": "oil refinery",
    "greenhouse": "greenhouse",
}

SCENE_DISPLAY_NAMES: Dict[str, str] = {
    "airport": "an airport",
    "sports field": "a sports field",
    "freeway": "a freeway",
    "water area": "a water area",
    "golffield": "a golf field",
    "golf field": "a golf field",
    "harbor": "a harbor",
}

NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def number_word(value: int) -> str:
    return NUMBER_WORDS.get(int(value), str(value))


def pluralize(name: str, count: int) -> str:
    if count == 1:
        return name
    irregular = {
        "chimney": "chimneys",
    }
    if name in irregular:
        return irregular[name]
    if name.endswith("y"):
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def display_class(label: str) -> str:
    return CLASS_DISPLAY_NAMES.get(label, label.replace("-", " ").replace("_", " ").lower())


def extract_scene(caption: Sequence[str]) -> str:
    first = str(caption[0]).strip()
    match = re.search(r"aerial image of ([^.]+)", first, flags=re.IGNORECASE)
    if match:
        scene = match.group(1).strip().lower()
        return SCENE_DISPLAY_NAMES.get(scene, f"a {scene}")
    return "a remote sensing scene"


def object_summary(objects: Sequence[LayoutObject]) -> str:
    counts = Counter(display_class(obj.label) for obj in objects if obj.valid)
    if not counts:
        return "the specified scene"

    parts = []
    for name, count in counts.items():
        parts.append(f"{number_word(count)} {pluralize(name, count)}")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _position_from_bbox(bbox: Sequence[float]) -> str:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5

    if 0.33 <= cx <= 0.67 and 0.33 <= cy <= 0.67:
        return "near the center"

    vertical = "upper" if cy < 0.33 else "lower" if cy > 0.67 else ""
    horizontal = "left" if cx < 0.33 else "right" if cx > 0.67 else ""
    if vertical and horizontal:
        return f"near the {vertical}-{horizontal} corner"
    if vertical:
        return f"near the {vertical} edge"
    if horizontal:
        return f"near the {horizontal} edge"
    return "within the image"


def _layout_sentences(objects: Sequence[LayoutObject]) -> List[str]:
    grouped = defaultdict(list)
    for obj in objects:
        if obj.valid:
            grouped[obj.label].append(obj)

    sentences: List[str] = []
    for label, items in grouped.items():
        name = display_class(label)
        for obj in items:
            position = _position_from_bbox(obj.bbox)
            sentences.append(f"One {name} is {position}.")
    return sentences


def build_satellite_prompts(caption: Sequence[str], objects: Sequence[LayoutObject]) -> Tuple[str, str]:
    scene = extract_scene(caption)
    summary = object_summary(objects)
    clip_prompt = f"A satellite image of {scene}, showing {summary}."
    t5_prompt = (
        f"A satellite image of {scene}, showing {summary} with the specified spatial layout and orientations. "
        + " ".join(_layout_sentences(objects))
    ).strip()
    return clip_prompt, t5_prompt


def legacy_clip_prompt(prompt: str, objects: Sequence[LayoutObject], max_words: int = 60) -> str:
    labels = [obj.label for obj in objects if obj.label]
    if labels:
        compact = f"A remote sensing image with {', '.join(labels)}."
    else:
        compact = prompt.split(".")[0].strip() + "."
    words = compact.split()
    if len(words) <= max_words:
        return compact
    return " ".join(words[:max_words]).rstrip(" ,.") + "."
