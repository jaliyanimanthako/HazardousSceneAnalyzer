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
    for obj in output.get("_objects_detected_detail", []):
        bb = obj["bounding_box"]
        draw.rectangle(
            [bb["x1"], bb["y1"], bb["x2"], bb["y2"]], outline="yellow", width=2
        )
        draw.text((bb["x1"], bb["y1"] - 15), obj["label"], fill="yellow", font=font)

    # ── Hazards — colour-coded ────────────────────────────────────────────────
    for hazard in output.get("_hazards_detected_detail", []):
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
        print(f"  Saved: {save_path}")

    return image


def print_report(output: dict) -> None:
    """
    Print a concise operator report to stdout.

    Args:
        output: Result dict from HazardousSceneAnalyzer.analyze().
    """
    sep = "=" * 65

    print(f"\n{sep}")
    print("HAZARD ANALYSIS REPORT")
    print(sep)

    print(f"\nObjects Detected : {', '.join(output.get('objects_detected', [])) or 'None'}")

    hazards = output.get("possible_hazards", [])
    severity = output.get("severity", "unknown").upper()
    print(f"Possible Hazards : {', '.join(hazards) or 'None detected'}  [{severity}]")

    print(f"\nExplanation :")
    print(f"   {output.get('explanation', '')}")

    conf = output.get("confidence", 0.5)
    bar  = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    print(f"\nConfidence : {conf:.2f}  [{bar}]")

    if output.get("clarifying_question"):
        print(f"\nClarifying Question :")
        print(f"   {output['clarifying_question']}")

    print(f"\n{sep}\n")
