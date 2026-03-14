"""
Hazard Assessment — LLM-based and keyword-based reasoning
==========================================================
Standalone functions: pass engines and config dicts as arguments.

Debug tips:
    # Test JSON parsing without running the full LLM
    from utils.assessment import normalize_hazard_type, hedge_caption
    print(normalize_hazard_type("fire|smoke", valid_types))
    print(hedge_caption("oil spill near the tank", substance_hedges))

    # Test keyword fallback without any model
    from utils.assessment import assess_hazards_keywords
    result = assess_hazards_keywords(["barrel", "fire"], "smoke rising", hazard_keywords)
"""

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engines.llm_engine import LLMEngine


# ── Caption hedging ───────────────────────────────────────────────────────────

def hedge_caption(text: str, substance_hedges: list) -> str:
    """
    Replace overconfident substance identifications with hedged language.

    Args:
        text:             Caption or description from Florence.
        substance_hedges: List of (regex_pattern, replacement) pairs
                          loaded from hazard_config.yaml.
    """
    for pattern, replacement in substance_hedges:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Hazard type normalisation ─────────────────────────────────────────────────

def normalize_hazard_type(htype: str, valid_hazard_types: set) -> str:
    """
    Normalise a free-form LLM hazard type string to a canonical value.

    Handles pipe-delimited types ("fire|smoke"), fuzzy matches ("blaze" → "fire"),
    and falls back to "chemical" for anything unrecognised.

    Args:
        htype:              Raw type string from LLM JSON.
        valid_hazard_types: Set of accepted canonical names (from hazard_config.yaml).
    """
    htype = htype.lower().strip()

    if "|" in htype:
        for token in htype.split("|"):
            token = token.strip()
            if token in valid_hazard_types:
                return token
        htype = htype.split("|")[0].strip()

    if "spill"      in htype or "leak"     in htype: return "spill"
    if "fire"       in htype or "flame"    in htype or "burn"   in htype: return "fire"
    if "smoke"      in htype or "fume"     in htype or "haze"   in htype: return "smoke"
    if "chemical"   in htype or "toxic"    in htype or "hazmat" in htype: return "chemical"
    if "structural" in htype or "collapse" in htype or "debris" in htype: return "structural"
    if "electrical" in htype or "wire"     in htype or "spark"  in htype: return "electrical"
    if "biological" in htype or "person"   in htype or "body"   in htype: return "biological"

    return htype if htype in valid_hazard_types else "chemical"


# ── LLM assessment ────────────────────────────────────────────────────────────

def assess_hazards_with_llm(
    llm_engine: "LLMEngine",
    system_prompt: str,
    objects: list,
    caption: str,
    dense_regions: list,
    hazards_grounded: list,
    substance_hedges: list,
    valid_hazard_types: set,
) -> dict:
    """
    Use LLMEngine to produce a structured hazard assessment dict.

    Steps:
      1. Hedge caption and regions to remove overconfident substance names.
      2. Build user prompt with all scene evidence.
      3. Call llm_engine.query() and parse the JSON response.
      4. Deduplicate hazard types (LLM sometimes lists the same type twice).
      5. Enforce severity consistency (overall >= worst individual).

    Raises on failure — caller should catch and fall back to assess_hazards_keywords.

    Args:
        llm_engine:         LLMEngine instance.
        system_prompt:      HAZARD_SYSTEM_PROMPT from prompts.py.
        objects:            List of object label strings from OD.
        caption:            Scene description from Florence captioning.
        dense_regions:      List of {"description", "bbox"} from dense region task.
        hazards_grounded:   List of {"label", "bbox"} from phrase grounding.
        substance_hedges:   (pattern, replacement) pairs from hazard_config.yaml.
        valid_hazard_types: Set of canonical hazard type names.
    """
    from engines.llm_engine import LLMEngine  # local import to avoid circular dependency

    hedged_caption  = hedge_caption(caption, substance_hedges)
    hedged_regions  = [
        {**r, "description": hedge_caption(r["description"], substance_hedges)}
        for r in dense_regions
    ]
    hedged_grounded = [
        {**g, "label": hedge_caption(g["label"], substance_hedges)}
        for g in hazards_grounded
    ]

    user_prompt = f"""Analyze this hazardous scene evidence:

DETECTED OBJECTS: {json.dumps(objects)}
SCENE DESCRIPTION: {hedged_caption}
REGION DESCRIPTIONS: {json.dumps(hedged_regions)}
GROUNDED HAZARDS: {json.dumps(hedged_grounded)}

Note: Substance names in the scene description are unconfirmed visual estimates.
Apply the SUBSTANCE UNCERTAINTY RULE - use hedged language for any unconfirmed substance.

Provide your JSON hazard assessment based ONLY on this evidence."""

    response   = llm_engine.query(system_prompt, user_prompt)
    llm_result = LLMEngine.parse_json(response)

    # Deduplicate hazard types
    seen_types      = set()
    deduped_details = []
    for h in llm_result.get("hazards", []):
        canonical = normalize_hazard_type(h.get("type", ""), valid_hazard_types)
        if canonical not in seen_types:
            seen_types.add(canonical)
            h["type"] = canonical
            deduped_details.append(h)

    # Enforce severity consistency: overall must equal highest individual severity
    severity_rank    = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    declared_overall = llm_result.get("overall_severity", "low").lower()
    worst_individual = max(
        (h.get("severity", "low").lower() for h in deduped_details),
        key=lambda s: severity_rank.get(s, 0),
        default="low",
    )
    enforced_severity = (
        worst_individual
        if severity_rank.get(worst_individual, 0) > severity_rank.get(declared_overall, 0)
        else declared_overall
    )
    if enforced_severity != declared_overall:
        print(f"  Severity corrected: '{declared_overall}' -> '{enforced_severity}'")

    clarifying = llm_result.get("clarifying_question")

    return {
        "detected_hazard_types": list(seen_types),
        "hazards_detail":        deduped_details,
        "severity":              enforced_severity,
        "confidence":            llm_result.get("confidence", 0.7),
        "scene_description":     llm_result.get("scene_description", ""),
        "summary":               llm_result.get("summary", ""),
        "decision_support":      llm_result.get("decision_support", ""),
        "recommendations":       llm_result.get("recommendations", []),
        "clarifying_question":   clarifying,
        "source":                "llm",
    }


# ── Keyword fallback ──────────────────────────────────────────────────────────

def _clarifying_question_for(detected_hazards: list, objects: list = None) -> "str | None":
    """
    Return the single most operationally useful clarifying question for the
    detected hazard set. Priority order mirrors response urgency.
    Uses detected objects to generate more scene-specific questions where possible.
    Returns None only when no hazards were detected.
    """
    if not detected_hazards:
        return None

    hazard_set  = set(detected_hazards)
    obj_text    = " ".join(objects or []).lower()

    if "biological" in hazard_set:
        return "Are personnel still inside the affected area, and have they been accounted for?"

    if "fire" in hazard_set:
        if "vehicle" in obj_text or "forklift" in obj_text or "truck" in obj_text:
            return "Is the vehicle fire self-contained, or has it spread to surrounding structures or fuel storage?"
        return "Is the fire contained to one area, or has it spread — and has the zone been evacuated?"

    if "chemical" in hazard_set or "spill" in hazard_set:
        if "pipe" in obj_text or "pipeline" in obj_text or "valve" in obj_text:
            return "Has the supply valve or isolation point for the leaking pipe been shut off?"
        if "tank" in obj_text or "vessel" in obj_text or "silo" in obj_text:
            return "What substance is stored in the tank, and what is the current pressure or fill level?"
        if "barrel" in obj_text or "drum" in obj_text or "canister" in obj_text:
            return "What does the barrel or container label identify as the stored substance?"
        # Generic chemical/spill — ask about the source rather than a specific container
        return "Has the source of the spill or release been identified, and is it still actively leaking?"

    if "structural" in hazard_set:
        return "Is the structure currently occupied, and are there signs of progressive collapse?"

    if "electrical" in hazard_set:
        return "Has the power supply to this zone been isolated at the breaker?"

    if "smoke" in hazard_set:
        return "Is the smoke originating from a fire or a chemical release? This determines whether fire crews or HAZMAT teams respond."

    primary = detected_hazards[0]
    return f"Can the operator confirm whether the {primary} hazard is active or contained?"


def assess_hazards_keywords(
    objects: list,
    caption: str,
    hazard_keywords: dict,
) -> dict:
    """
    Keyword-based hazard assessment — no model required.
    Used when use_llm=False or when LLM raises an exception.

    Args:
        objects:         List of object label strings from OD.
        caption:         Scene description from Florence captioning.
        hazard_keywords: {hazard_type: [keywords]} from hazard_config.yaml.
    """
    combined_text    = " ".join(objects).lower() + " " + caption.lower()
    detected_hazards = []
    hazard_details   = {}

    for hazard_type, keywords in hazard_keywords.items():
        matching = [kw for kw in keywords if kw in combined_text]
        if matching:
            detected_hazards.append(hazard_type)
            hazard_details[hazard_type] = matching

    severity = "low"
    if len(detected_hazards) >= 3 or "fire" in detected_hazards:
        severity = "critical" if "fire" in detected_hazards else "high"
    elif len(detected_hazards) >= 2:
        severity = "high"
    elif any(h in detected_hazards for h in ["chemical", "biological"]):
        severity = "medium"

    return {
        "detected_hazard_types": detected_hazards,
        "hazard_details":        hazard_details,
        "severity":              severity,
        "confidence":            0.5,
        "summary":               "",
        "recommendations":       [],
        "clarifying_question":   _clarifying_question_for(detected_hazards, objects),
        "source":                "keyword_fallback",
    }


# ── Fallback explanation ──────────────────────────────────────────────────────

def generate_explanation(
    objects: list,
    hazard_assessment: dict,
    severity_messages: dict,
) -> str:
    """
    Build an operator-facing RISK ASSESSMENT — not a scene description.

    Focuses on WHY detected hazards are dangerous and what to do, NOT on
    re-describing what was seen. The LLM's hazard descriptions (hdesc) are
    intentionally NOT echoed — they describe appearance; this function
    reasons about consequence and required action.

    Args:
        objects:           Detected object labels.
        hazard_assessment: Output dict from assess_hazards_* functions.
        severity_messages: {severity: message} from hazard_config.yaml.
    """
    severity = hazard_assessment.get("severity", "low")
    details  = hazard_assessment.get("hazards_detail", [])
    hazards  = hazard_assessment.get("detected_hazard_types", [])
    parts    = []

    if not hazards:
        parts.append("No active hazards were identified in this scene.")
        parts.append(severity_messages.get(severity, "Continue monitoring."))
        return " ".join(parts)

    # ── Risk consequence per hazard type ──────────────────────────────────────
    # Each entry explains WHY the hazard is dangerous, not what it looks like.
    _risk_reason = {
        "fire":       "poses an immediate life-safety threat — ignition of surrounding materials, burns, and smoke inhalation are primary concerns.",
        "smoke":      "indicates an active fire or chemical release — the source must be confirmed before personnel approach.",
        "chemical":   "means appropriate PPE cannot be selected until the substance is identified — treat as hazardous until confirmed otherwise.",
        "spill":      "creates slip, contamination, and potential ignition risk — the substance must be identified before containment can begin.",
        "electrical": "poses electrocution risk — the power supply to this zone must be isolated at the breaker before any approach.",
        "structural": "presents collapse risk to anyone in or near the area — load-bearing elements must be assessed before re-entry.",
        "biological": "personnel are inside the hazard zone — their condition and evacuation status must be confirmed immediately.",
    }

    # Sort most-severe first so the lead sentence addresses the highest risk
    _sev_rank    = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_details = sorted(
        details, key=lambda h: _sev_rank.get(h.get("severity", "low"), 3)
    )

    for h in sorted_details:
        htype   = h.get("type", "unknown")
        hsev    = h.get("severity", "unknown").upper()
        hloc    = h.get("location", "")
        reason  = _risk_reason.get(htype, f"a {htype} hazard has been detected and requires assessment.")
        loc_str = f" at {hloc}" if hloc else ""
        parts.append(f"{hsev} {htype.upper()} hazard{loc_str}: {reason}".capitalize())

    # If only keyword fallback ran (no detail records), use hazard type names
    if not sorted_details and hazards:
        hazard_list = " and ".join(hazards) if len(hazards) <= 3 else ", ".join(hazards)
        parts.append(f"Potential {hazard_list} hazard(s) detected — scene requires immediate assessment.")

    # ── How specific detected objects compound the risk ───────────────────────
    _compounding = {
        "barrel": "storage barrels near the hazard zone raise contamination risk if contents are unknown or containers are damaged",
        "drum":   "drums in proximity to the hazard raise contamination concern if unsealed",
        "tank":   "pressurised tanks near an active hazard create explosion or uncontrolled release risk",
        "person": "workers detected inside the hazard perimeter — evacuation status must be confirmed",
        "worker": "workers detected inside the hazard perimeter — evacuation status must be confirmed",
        "wire":   "exposed wiring adjacent to a spill or fire significantly elevates electrical risk",
    }

    compound_notes = []
    seen_notes     = set()
    for obj in objects:
        for kw, note in _compounding.items():
            if kw in obj.lower() and note not in seen_notes:
                compound_notes.append(note)
                seen_notes.add(note)

    if compound_notes:
        parts.append("Compounding factors: " + "; ".join(compound_notes) + ".")

    # ── Required action ───────────────────────────────────────────────────────
    parts.append(severity_messages.get(severity, "Proceed with caution."))

    return " ".join(parts)
