"""
Hazard Grounding — phrase-based detection helpers
==================================================
Standalone functions: no class state, easy to unit-test in isolation.

Debug tips:
    from utils.grounding import build_scene_phrases, calculate_iou
    phrases = build_scene_phrases("smoke rising from a barrel", vocab, always_check)
    print(phrases)
"""

import re

from PIL import Image
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from engines.florence_engine import FlorenceEngine


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

_PLACARD_PHRASES = {"hazmat sign", "hazmat placard", "warning sign", "safety sign"}


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
    hazmat_classes: Optional[dict] = None,
    hazmat_placard_keywords: Optional[dict] = None,
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

    phrases     = build_scene_phrases(caption, hazmat_vocab, always_check)
    phrase_list = [phrase for _, phrase, _ in phrases]
    print(f"  🎯 Grounding {len(phrase_list)} phrases in one batch call...")

    # Single Florence inference for all phrases at once
    combined_caption = ". ".join(phrase_list)
    try:
        grounding  = florence_engine.ground_phrase(image, combined_caption)
    except Exception as e:
        print(f"  ⚠ Batch grounding failed: {e}")
        return {"bboxes": [], "labels": []}

    raw_bboxes = grounding.get("bboxes", [])
    raw_labels = grounding.get("labels", [])

    all_bboxes      = []
    all_labels      = []
    claimed_regions = []
    phrase_counts   = {}

    for bbox, label in zip(raw_bboxes, raw_labels):
        phrase = label.lower().strip().rstrip(".")

        # Filter tiny boxes
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < min_bbox_area:
            continue

        # Per-phrase instance cap
        if phrase_counts.get(phrase, 0) >= max_per_phrase:
            continue

        # Type-aware IoU dedup
        already_claimed = any(
            calculate_iou(bbox, r["bbox"]) > iou_threshold
            and r["phrase_type"] == phrase
            for r in claimed_regions
        )
        if already_claimed:
            continue

        # Try OCR-based placard identification
        is_placard = phrase in _PLACARD_PHRASES
        if is_placard and hazmat_classes and hazmat_placard_keywords:
            placard_id = identify_hazmat_placard(
                florence_engine, image, bbox,
                hazmat_classes, hazmat_placard_keywords,
            )
            display_label = f"⚠ hazmat: {placard_id}" if placard_id else display_labels.get(phrase, f"⚠ {phrase}")
        else:
            display_label = display_labels.get(phrase, f"⚠ {phrase}")

        all_bboxes.append(bbox)
        all_labels.append(display_label)
        claimed_regions.append({"bbox": bbox, "phrase_type": phrase})
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    print(f"  ✓ Found {len(all_bboxes)} hazard region(s) from {len(phrase_list)} phrase(s)")
    return {"bboxes": all_bboxes, "labels": all_labels}


# ── Hazmat placard OCR ────────────────────────────────────────────────────────

def identify_hazmat_placard(
    florence_engine: "FlorenceEngine",
    image: Image.Image,
    bbox,
    hazmat_classes: dict,
    hazmat_placard_keywords: dict,
) -> Optional[str]:
    """
    Crop the bbox from image, run Florence OCR, and parse for:
      1. UN numbers       (4-digit, e.g. "1203")
      2. Hazard class     (e.g. "3", "6.1")
      3. Keyword labels   (e.g. "FLAMMABLE", "CORROSIVE")

    Returns a human-readable label like:
      "Flammable Liquid (Class 3, UN 1203)"
    or None if no placard text is recognised.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # Guard against degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image.crop((x1, y1, x2, y2))

    try:
        ocr_text = florence_engine.read_text(crop)
    except Exception as e:
        print(f"  ⚠ Placard OCR failed: {e}")
        return None

    if not ocr_text:
        return None

    text_upper = ocr_text.upper()
    text_lower = ocr_text.lower()
    print(f"  🔤 Placard OCR: {ocr_text!r}")

    # 1 — UN number (4 consecutive digits, optionally prefixed "UN")
    un_match = re.search(r'\bUN\s*(\d{4})\b|\b(\d{4})\b', text_upper)
    un_number = None
    if un_match:
        un_number = un_match.group(1) or un_match.group(2)

    # 2 — Hazard class number (e.g. "3", "6.1", "2.3")
    class_match = re.search(r'\b(\d(?:\.\d{1,2})?)\b', ocr_text)
    class_name = None
    if class_match:
        candidate = class_match.group(1)
        # Only use if it maps to a known class (avoid matching stray digits)
        class_name = hazmat_classes.get(candidate)

    # 3 — Keyword labels (scan all registered keywords)
    keyword_name = None
    for kw, label in hazmat_placard_keywords.items():
        if kw in text_lower:
            keyword_name = label
            break  # first match wins (keywords are ordered by priority in YAML)

    # Build result string from whatever was found
    parts = []
    if keyword_name:
        parts.append(keyword_name)
    elif class_name:
        parts.append(class_name)

    if class_name and class_name not in parts:
        parts.append(f"Class {class_match.group(1)}")
    elif class_name:
        parts.append(f"Class {class_match.group(1)}")

    if un_number:
        parts.append(f"UN {un_number}")

    if not parts:
        return None

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    return " · ".join(deduped)


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
