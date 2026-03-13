"""
Online pipeline post-processing helpers.
=========================================
Handles everything that happens *after* the VLM API call returns:

  _norm_to_pixel()   — convert normalised 0-999 bbox coords to pixel coords
  extract_bboxes()   — parse objects + hazards out of the VLM JSON response
  postprocess()      — deduplicate hazard types, enforce severity consistency,
                       fill clarifying question if absent

Kept separate from online.py so each step can be unit-tested independently:

    from utils.online_processing import extract_bboxes, postprocess

    fake_response = {
        "objects_detected": [{"label": "barrel", "bbox": [100, 200, 300, 400]}],
        "hazards": [{"type": "chemical", "severity": "high", "bbox": [100, 200, 300, 400]}],
        "overall_severity": "medium",
        "confidence": 0.8,
    }
    objects_detail, hazards_detail, labels = extract_bboxes(fake_response, 640, 480)
    assessment = postprocess(fake_response, labels)
"""

from utils.assessment import (
    normalize_hazard_type,
    _clarifying_question_for,
    generate_explanation,
)

# ── Severity constants ────────────────────────────────────────────────────────

SEVERITY_MESSAGES = {
    "critical": "CRITICAL - Immediate evacuation required.",
    "high":     "HIGH RISK - Exercise extreme caution.",
    "medium":   "MODERATE RISK - Proceed with safety measures.",
    "low":      "LOW RISK - Continue assessment.",
}

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

VALID_HAZARD_TYPES = {
    "fire", "smoke", "chemical", "structural", "electrical", "biological", "spill"
}

# Consistent annotation colours (shared with offline pipeline)
HAZARD_COLORS = {
    "fire": "#FF3300", "flame": "#FF6600", "smoke": "#9933FF",
    "explosion": "#FF0000", "chemical": "#00FF00", "debris": "#FF9900",
    "injured": "#FF00FF", "damage": "#0099FF", "spill": "#00FFFF",
    "wire": "#FFFF00", "person": "#FF66FF", "barrel": "#FF8800",
    "tank": "#FF8800", "pipe": "#AAAAAA", "vehicle": "#88AAFF",
}


# ── Bbox helpers ─────────────────────────────────────────────────────────────

def _norm_to_pixel(bbox: list, img_w: int, img_h: int) -> dict:
    """Convert normalised [x1,y1,x2,y2] (0-999 scale) to pixel coordinates."""
    x1 = max(0, min(img_w, int(bbox[0] / 999 * img_w)))
    y1 = max(0, min(img_h, int(bbox[1] / 999 * img_h)))
    x2 = max(0, min(img_w, int(bbox[2] / 999 * img_w)))
    y2 = max(0, min(img_h, int(bbox[3] / 999 * img_h)))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def extract_bboxes(llm_result: dict, img_w: int, img_h: int):
    """
    Parse objects_detected and hazards from the VLM JSON response and convert
    their bbox fields to pixel-coordinate detail dicts expected by visualize().

    Returns:
        (objects_detail, hazards_detail, object_labels)
    """
    objects_detail = []
    object_labels  = []

    for obj in llm_result.get("objects_detected", []):
        label = obj.get("label", "object")
        bbox  = obj.get("bbox")
        object_labels.append(label)
        if bbox and len(bbox) == 4:
            objects_detail.append({
                "label":        label,
                "bounding_box": _norm_to_pixel(bbox, img_w, img_h),
            })

    hazards_detail = []
    for h in llm_result.get("hazards", []):
        label = f"⚠ {h.get('type', 'hazard')}"
        bbox  = h.get("bbox")
        if bbox and len(bbox) == 4:
            hazards_detail.append({
                "label":        label,
                "bounding_box": _norm_to_pixel(bbox, img_w, img_h),
            })

    return objects_detail, hazards_detail, object_labels


def postprocess(llm_result: dict, object_labels: list) -> dict:
    """
    Apply the same post-processing guards used in the offline pipeline:
      - Deduplicate hazard types
      - Enforce severity consistency (overall >= worst individual)
      - Generate clarifying question if LLM returned null
      - Generate explanation if decision_support is empty

    Returns a flat assessment dict ready to merge into the final result.
    """
    # Deduplicate hazard types
    seen_types      = set()
    deduped_details = []
    for h in llm_result.get("hazards", []):
        canonical = normalize_hazard_type(h.get("type", ""), VALID_HAZARD_TYPES)
        if canonical not in seen_types:
            seen_types.add(canonical)
            h["type"] = canonical
            deduped_details.append(h)

    # Enforce severity consistency
    declared_overall = llm_result.get("overall_severity", "low").lower()
    worst_individual = max(
        (h.get("severity", "low").lower() for h in deduped_details),
        key=lambda s: SEVERITY_RANK.get(s, 0),
        default="low",
    )
    enforced_severity = (
        worst_individual
        if SEVERITY_RANK.get(worst_individual, 0) > SEVERITY_RANK.get(declared_overall, 0)
        else declared_overall
    )

    # Clarifying question fallback
    clarifying = llm_result.get("clarifying_question")
    if not clarifying and seen_types:
        clarifying = _clarifying_question_for(list(seen_types))

    assessment = {
        "detected_hazard_types": list(seen_types),
        "hazards_detail":        deduped_details,
        "severity":              enforced_severity,
        "confidence":            llm_result.get("confidence", 0.7),
        "decision_support":      llm_result.get("decision_support", ""),
        "recommendations":       llm_result.get("recommendations", []),
        "clarifying_question":   clarifying,
    }

    explanation = assessment["decision_support"]
    if not explanation:
        explanation = generate_explanation(object_labels, assessment, SEVERITY_MESSAGES)
    assessment["explanation"] = explanation

    return assessment
