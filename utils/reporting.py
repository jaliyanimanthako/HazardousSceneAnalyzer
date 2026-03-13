"""
Reporting — visualization and text output for hazard analysis results
=====================================================================
Standalone functions: pass output dicts and colour config as arguments.

Debug tips:
    # Re-draw annotations without re-running the pipeline
    from reporting import visualize, print_report
    import yaml, json
    cfg    = yaml.safe_load(open("hazard_config.yaml"))
    result = json.load(open("result.json"))
    print_report(result)
    visualize("scene.jpg", result, cfg["hazard_colors"], save_path="debug_out.jpg")
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Union, Optional


def visualize(
    image_input: Union[str, Image.Image],
    output: dict,
    hazard_colors: dict,
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Draw bounding boxes on the image and optionally save to disk.

    Draws:
      - Object boxes in yellow with label text above
      - Hazard boxes colour-coded by type (first keyword match in hazard_colors)

    Args:
        image_input:   Path string or PIL Image.
        output:        Result dict from HazardousSceneAnalyzer.analyze().
        hazard_colors: {keyword: "#RRGGBB"} from hazard_config.yaml.
        save_path:     If provided, saves the annotated image here.
    """
    image = (
        Image.open(image_input).convert("RGB")
        if isinstance(image_input, str)
        else image_input.copy()
    )
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()

    # ── Objects — yellow ──────────────────────────────────────────────────────
    for obj in output.get("objects_detected_detail", []):
        bb = obj["bounding_box"]
        draw.rectangle(
            [bb["x1"], bb["y1"], bb["x2"], bb["y2"]], outline="yellow", width=2
        )
        draw.text((bb["x1"], bb["y1"] - 15), obj["label"], fill="yellow", font=font)

    # ── Hazards — colour-coded ────────────────────────────────────────────────
    for hazard in output.get("hazards_detected_detail", []):
        bb    = hazard["bounding_box"]
        label = hazard["label"]
        color = next(
            (c for kw, c in hazard_colors.items() if kw in label.lower()),
            "#FF0000",
        )
        draw.rectangle(
            [bb["x1"], bb["y1"], bb["x2"], bb["y2"]], outline=color, width=4
        )
        label_y = max(0, bb["y1"] - 18)
        draw.rectangle(
            [bb["x1"], label_y, bb["x1"] + len(label) * 8, label_y + 16], fill=color
        )
        draw.text((bb["x1"] + 2, label_y + 1), label, fill="white", font=font)

    if save_path:
        image.save(save_path)
        print(f"  💾 Saved: {save_path}")

    return image


def print_report(output: dict) -> None:
    """
    Print a human-readable operator report to stdout.

    Args:
        output: Result dict from HazardousSceneAnalyzer.analyze().
    """
    ss  = output.get("scene_summary", {})
    sep = "=" * 65

    print(f"\n{sep}")
    print("🤖 MOBILE ROBOT HAZARD ANALYSIS — OPERATOR REPORT")
    print(sep)
    print(f"\n📁 Image         : {output['image_path']}")
    print(f"⚙️  Reasoning      : {output.get('reasoning_source', 'unknown').upper()}")

    # Objects
    print(f"\n{'─'*65}")
    print("📦 OBJECTS DETECTED")
    print(f"{'─'*65}")
    for label in output.get("objects_detected", []):
        print(f"   • {label}")

    # Hazard bboxes
    print(f"\n🎯 HAZARD REGIONS (with bounding boxes)")
    for h in output.get("hazards_detected_detail", []):
        bb = h["bounding_box"]
        print(f"   • {h['label']}  [{bb['x1']},{bb['y1']} → {bb['x2']},{bb['y2']}]")

    # Scene summary
    print(f"\n{'─'*65}")
    print("📝 SCENE SUMMARY")
    print(f"{'─'*65}")
    print(f"   {ss.get('description', '')}")
    print(f"\n⚠️  Possible Hazards : {', '.join(ss.get('possible_hazards', [])) or 'None detected'}")
    print(f"   Severity         : {ss.get('severity', '?').upper()}")

    if ss.get("hazard_details"):
        print(f"\n   Hazard Details:")
        for h in ss["hazard_details"]:
            print(f"   • [{h.get('severity','?').upper()}] {h.get('type','?').upper()}")
            print(f"     Location : {h.get('location', 'unknown')}")
            print(f"     Detail   : {h.get('description', '')}")

    # Explanation
    print(f"\n{'─'*65}")
    print("💡 DECISION-SUPPORT EXPLANATION  (for operator)")
    print(f"{'─'*65}")
    print(f"   {output.get('explanation', '')}")

    if output.get("full_briefing") and output["full_briefing"] != output.get("explanation"):
        print(f"\n📋 Full Team Briefing:")
        print(f"   {output['full_briefing']}")

    if output.get("recommendations"):
        print(f"\n🚨 Recommendations:")
        for i, rec in enumerate(output["recommendations"], 1):
            print(f"   {i}. {rec}")

    # Confidence bar
    print(f"\n{'─'*65}")
    conf = output.get("confidence", 0.5)
    bar  = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    print(f"📊 CONFIDENCE : {conf:.2f}  [{bar}]")

    if output.get("clarifying_question"):
        print(f"\n❓ CLARIFYING QUESTION (ask operator):")
        print(f"   {output['clarifying_question']}")
    else:
        print(f"\n   (No clarifying question — sufficient evidence to act)")

    print(f"\n{sep}\n")
