"""
Hazard Grounding — phrase-based detection helpers
==================================================
Standalone functions: no class state, easy to unit-test in isolation.

Debug tips:
    from grounding import build_scene_phrases, calculate_iou
    phrases = build_scene_phrases("smoke rising from a barrel", vocab, always_check)
    print(phrases)
"""

from PIL import Image
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from florence_engine import FlorenceEngine


# ── Geometry ──────────────────────────────────────────────────────────────────

def calculate_iou(box1, box2) -> float:
    """Intersection over Union for two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


# ── Phrase selection ──────────────────────────────────────────────────────────

def build_scene_phrases(caption: str, hazmat_vocab: dict, always_check: list) -> list:
    """
    Score all vocabulary phrases against caption keywords.
    Returns a sorted list of (score, phrase, keywords) tuples.
    ALWAYS_CHECK phrases are force-included at score 0.

    Args:
        caption:       Scene description from Florence captioning task.
        hazmat_vocab:  {phrase: [keywords]} dict (from HAZMAT_VOCABULARY / prompts.py).
        always_check:  High-priority phrases grounded regardless of caption content.
    """
    caption_lower = caption.lower()
    scored = []
    seen_phrases = set()

    for phrase, keywords in hazmat_vocab.items():
        hits = sum(1 for kw in keywords if kw in caption_lower)
        if hits > 0:
            scored.append((hits, phrase, keywords))
            seen_phrases.add(phrase)

    for phrase in always_check:
        if phrase not in seen_phrases:
            scored.append((0, phrase, hazmat_vocab.get(phrase, [])))

    scored.sort(reverse=True)
    return scored


# ── Grounding ─────────────────────────────────────────────────────────────────

def detect_hazards_by_grounding(
    florence_engine: "FlorenceEngine",
    image: Image.Image,
    caption: str,
    display_labels: dict,
    hazmat_vocab: dict,
    always_check: list,
    min_area_fraction: float = 0.005,
    max_per_phrase: int = 3,
    iou_threshold: float = 0.75,
) -> dict:
    """
    Dense multi-instance hazard grounding using Florence phrase grounding.

    Pipeline per phrase:
      1. Call florence_engine.ground_phrase(image, phrase)
      2. Filter boxes smaller than min_area_fraction of image area
      3. Keep top max_per_phrase largest boxes
      4. Type-aware IoU dedup (same phrase type at same location → drop duplicate)

    Args:
        florence_engine:   FlorenceEngine instance (handles inference).
        image:             PIL Image already preprocessed to ≤1024px.
        caption:           Scene caption for phrase scoring.
        display_labels:    phrase → "⚠ label" mapping (from hazard_config.yaml).
        hazmat_vocab:      phrase → [keywords] vocab (from prompts.py).
        always_check:      Force-included phrases (from hazard_config.yaml).
        min_area_fraction: Drop boxes smaller than this fraction of image area.
        max_per_phrase:    Max bbox instances kept per phrase type.
        iou_threshold:     Overlap above this (same phrase type) → duplicate, drop.
    """
    img_area      = image.width * image.height
    min_bbox_area = min_area_fraction * img_area

    phrases = build_scene_phrases(caption, hazmat_vocab, always_check)
    print(f"  🎯 Grounding {len(phrases)} phrases (full scene vocabulary)...")

    all_bboxes      = []
    all_labels      = []
    claimed_regions = []   # [{"bbox": [...], "phrase_type": str}]

    for _, phrase, _ in phrases:
        try:
            grounding = florence_engine.ground_phrase(image, phrase)
            bboxes    = list(grounding.get("bboxes", []))
            if not bboxes:
                continue

            # Filter tiny boxes
            bboxes = [b for b in bboxes if (b[2] - b[0]) * (b[3] - b[1]) >= min_bbox_area]
            if not bboxes:
                continue

            # Largest-first, keep top N
            bboxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            bboxes = bboxes[:max_per_phrase]

            display_label = display_labels.get(phrase, f"⚠ {phrase}")

            for bbox in bboxes:
                # Only suppress if SAME phrase type overlaps heavily (type-aware dedup)
                already_claimed = any(
                    calculate_iou(bbox, r["bbox"]) > iou_threshold
                    and r["phrase_type"] == phrase
                    for r in claimed_regions
                )
                if not already_claimed:
                    all_bboxes.append(bbox)
                    all_labels.append(display_label)
                    claimed_regions.append({"bbox": bbox, "phrase_type": phrase})

        except Exception as e:
            print(f"  ⚠ Grounding failed for '{phrase}': {e}")
            continue

    print(f"  ✓ Found {len(all_bboxes)} hazard region(s) across {len(phrases)} phrase(s)")
    return {"bboxes": all_bboxes, "labels": all_labels}


# ── OD cross-reference ────────────────────────────────────────────────────────

def od_results_to_hazards(objects_result: dict, od_hazard_map: dict) -> dict:
    """
    Promote OD-detected objects matching od_hazard_map into hazard detections.
    Reuses existing OD bboxes — zero extra inference calls.

    Args:
        objects_result: {"bboxes": [...], "labels": [...]} from detect_objects().
        od_hazard_map:  {keyword: "⚠ display label"} (from hazard_config.yaml).
    """
    extra_bboxes = []
    extra_labels = []

    for label, bbox in zip(
        objects_result.get("labels", []),
        objects_result.get("bboxes", [])
    ):
        label_lower = label.lower()
        for kw, display in od_hazard_map.items():
            if kw in label_lower:
                extra_bboxes.append(bbox)
                extra_labels.append(display)
                break

    return {"bboxes": extra_bboxes, "labels": extra_labels}
