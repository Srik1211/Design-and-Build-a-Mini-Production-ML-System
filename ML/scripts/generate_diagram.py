"""Generates architecture_diagram.png for the design doc. Pure stdlib +
matplotlib, no network/graphviz dependency so it reproduces anywhere.

Usage: python scripts/generate_diagram.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve

# Brand-neutral, colorblind-safe-ish palette, one hue per pipeline stage group.
COLORS = {
    "data": "#8AA9D6",
    "pipeline": "#7FB88A",
    "model": "#C79FE0",
    "serving": "#E6A85C",
    "monitor": "#E08585",
}
TEXT_COLOR = "#1F2430"
BG_COLOR = "#FFFFFF"
ARROW_COLOR = "#5A6270"


def box(ax, xy, w, h, text, color, fontsize=10.5, sub=None):
    x, y = xy
    fb = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=TEXT_COLOR,
        facecolor=color,
        alpha=0.92,
        zorder=2,
    )
    ax.add_patch(fb)
    cy = y + h / 2 + (0.08 if sub else 0)
    ax.text(x + w / 2, cy, text, ha="center", va="center",
             fontsize=fontsize, color=TEXT_COLOR, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.16, sub, ha="center", va="center",
                 fontsize=8.3, color=TEXT_COLOR, zorder=3, style="italic")
    return (x, y, w, h)


def arrow(ax, start, end, style="-|>", curve=0.0, color=ARROW_COLOR, lw=1.6, ls="-"):
    fa = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=14,
        connectionstyle=f"arc3,rad={curve}",
        linewidth=lw,
        linestyle=ls,
        color=color,
        zorder=1,
    )
    ax.add_patch(fa)


def edge_point(b, side):
    x, y, w, h = b
    return {
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
    }[side]


def main():
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.text(7.5, 6.05, "Churn Prediction — Production ML Architecture",
             ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT_COLOR)

    # --- Top row: data -> training pipeline ---
    y_top = 4.15
    b_source = box(ax, (0.3, y_top), 2.0, 1.1, "Data Sources", COLORS["data"], sub="CRM / billing CSV drops")
    b_ingest = box(ax, (2.7, y_top), 2.0, 1.1, "Ingestion", COLORS["pipeline"], sub="batch / micro-batch\n+ schema check")
    b_store = box(ax, (5.1, y_top), 2.0, 1.1, "Processed\nData Store", COLORS["pipeline"], sub="training_data.csv")
    b_feat = box(ax, (7.5, y_top), 2.3, 1.1, "Feature\nEngineering", COLORS["pipeline"], sub="shared module")
    b_train = box(ax, (10.2, y_top), 2.1, 1.1, "Training\nPipeline", COLORS["model"], sub="baseline vs candidate")
    b_registry = box(ax, (12.7, y_top), 2.0, 1.1, "Model\nRegistry", COLORS["model"], sub="promotion rule\n+ versioning")

    for a, b in [(b_source, b_ingest), (b_ingest, b_store), (b_store, b_feat), (b_feat, b_train), (b_train, b_registry)]:
        arrow(ax, edge_point(a, "right"), edge_point(b, "left"))

    # --- Bottom row: serving -> monitoring -> retrain, all one row so
    # nothing has to route through the top row's boxes ---
    y_bot = 1.55
    b_retrain = box(ax, (0.3, y_bot), 2.5, 1.1, "Retraining\nTrigger", COLORS["monitor"],
                     sub="new data / AUC drop /\ndrift score", fontsize=10)
    b_monitor = box(ax, (3.2, y_bot), 2.5, 1.1, "Monitoring", COLORS["monitor"],
                     sub="drift + quality + latency")
    b_batch = box(ax, (6.1, y_bot), 2.3, 1.1, "Batch Scorer", COLORS["serving"], sub="nightly bulk scoring")
    b_online = box(ax, (8.8, y_bot), 2.3, 1.1, "Online API", COLORS["serving"], sub="FastAPI /predict")

    # Straight lines (no curvature) from a shared origin only diverge, they
    # never cross -- unlike curved arcs fanned out from one point, which can
    # visually cross each other.
    arrow(ax, edge_point(b_registry, "bottom"), edge_point(b_online, "top"), curve=0.0)
    arrow(ax, edge_point(b_registry, "bottom"), edge_point(b_batch, "top"), curve=0.0)
    arrow(ax, edge_point(b_online, "left"), edge_point(b_batch, "right"))
    arrow(ax, edge_point(b_batch, "left"), edge_point(b_monitor, "right"))
    arrow(ax, edge_point(b_monitor, "left"), edge_point(b_retrain, "right"))

    # Retraining Trigger feeds back into Training Pipeline. Routed as an
    # explicit step path through the empty gap between the two rows (never
    # a single arc, which would bulge up into the boxes above it).
    x_start, y_start = edge_point(b_retrain, "top")
    x_end, y_end = edge_point(b_train, "bottom")
    y_mid = (y_start + y_end) / 2
    feedback_color = "#B0453E"
    ax.plot([x_start, x_start], [y_start, y_mid], color=feedback_color, lw=1.8, ls="--", zorder=1)
    ax.plot([x_start, x_end], [y_mid, y_mid], color=feedback_color, lw=1.8, ls="--", zorder=1)
    arrow(ax, (x_end, y_mid), (x_end, y_end), color=feedback_color, lw=1.8, ls="--")
    ax.text((x_start + x_end) / 2, y_mid + 0.14, "triggers retrain",
             fontsize=9, color=feedback_color, style="italic", ha="center")

    # Feature Engineering is shared by Training + both serving paths.
    arrow(ax, edge_point(b_feat, "bottom"), edge_point(b_online, "top"), curve=0.0, color="#6B7280", lw=1.2, ls=":")
    arrow(ax, edge_point(b_feat, "bottom"), edge_point(b_batch, "top"), curve=0.0, color="#6B7280", lw=1.2, ls=":")

    # legend / footnote
    ax.text(0.3, 0.65,
             "Dotted lines: Feature Engineering is one shared module imported by Training, the Online API and the Batch\n"
             "Scorer — this is what prevents training/serving skew. Monitoring reads a frozen training-time reference\n"
             "snapshot, independent of how the Processed Data Store grows afterward.",
             fontsize=9.3, color=TEXT_COLOR, va="top")

    out_path = resolve("architecture_diagram.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor=BG_COLOR, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
