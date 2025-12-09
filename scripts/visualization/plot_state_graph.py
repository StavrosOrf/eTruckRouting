"""
Generate a clean state-graph figure for the event-driven truck environment.

States:
- ready, routing, waiting_to_charge, charging, unloading, complete, failed

Transitions are labeled with the event/action that triggers them.
Output: docs/figures/state_graph.pdf
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def draw_node(ax, pos, label):
    ax.scatter(*pos, s=1400, color="#f5f8ff", edgecolor="#2c3e50", zorder=3, linewidth=1.6)
    ax.text(*pos, label, ha="center", va="center", fontsize=11, weight="bold", color="#2c3e50", zorder=4)


def draw_arrow(ax, src, dst, text, rad=0.0, offset=0.6, shrink=0.2):
    """Draw curved arrow with a readable label and a pointer to the edge."""
    arrow = FancyArrowPatch(
        posA=src,
        posB=dst,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=14,
        linewidth=1.8,
        color="#34495e",
        alpha=0.9,
        zorder=2,
        shrinkA=shrink,
        shrinkB=shrink,
    )
    ax.add_patch(arrow)

    dx, dy = dst[0] - src[0], dst[1] - src[1]
    norm = (dx**2 + dy**2) ** 0.5 or 1.0
    px, py = -dy / norm, dx / norm  # perpendicular
    mx, my = (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2
    lx, ly = mx + px * offset, my + py * offset

    ax.annotate(
        text,
        xy=(mx, my),
        xytext=(lx, ly),
        ha="center",
        va="center",
        fontsize=9,
        color="#2c3e50",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#d5d8dd"),
        arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2, alpha=0.8),
        zorder=5,
    )


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    # Node positions with ready at center
    nodes = {
        "ready": (0.0, 0.0),
        "routing": (0.0, 3.0),
        "unloading": (2.9, 1.9),
        "complete": (3.6, 0.0),
        "charging": (2.9, -1.9),
        "waiting_to_charge": (-2.9, -1.9),
        "failed": (-3.6, 0.0),
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw nodes
    for name, pos in nodes.items():
        draw_node(ax, pos, name)

    # Arrows with curvature and perpendicular label offsets to avoid collisions
    draw_arrow(ax, nodes["ready"], nodes["routing"], "nav action\n(TRUCK_ROUTING)", rad=0.25, offset=0.65, shrink=0.35)
    draw_arrow(ax, nodes["routing"], nodes["ready"], "arrival\nTRUCK_READY", rad=-0.25, offset=-0.65, shrink=0.35)

    draw_arrow(ax, nodes["routing"], nodes["waiting_to_charge"], "charger full\nFCFS gating", rad=0.18, offset=-0.65, shrink=0.3)
    draw_arrow(ax, nodes["waiting_to_charge"], nodes["charging"], "port freed\nwake + start", rad=0.1, offset=-0.6, shrink=0.3)
    draw_arrow(ax, nodes["waiting_to_charge"], nodes["ready"], "leave queue /\nreroute", rad=-0.35, offset=-0.7, shrink=0.35)

    draw_arrow(ax, nodes["ready"], nodes["charging"], "charge action", rad=-0.18, offset=-0.7, shrink=0.35)
    draw_arrow(ax, nodes["charging"], nodes["ready"], "charge_complete\nTRUCK_READY", rad=0.35, offset=0.75, shrink=0.35)

    draw_arrow(ax, nodes["ready"], nodes["unloading"], "deliver action\n(start service)", rad=0.15, offset=0.65, shrink=0.35)
    draw_arrow(ax, nodes["unloading"], nodes["ready"], "service done\nTRUCK_READY", rad=-0.32, offset=-0.7, shrink=0.35)

    draw_arrow(ax, nodes["ready"], nodes["complete"], "all deliveries done", rad=0.12, offset=0.6, shrink=0.3)
    draw_arrow(ax, nodes["routing"], nodes["failed"], "battery depletion /\ninfeasible path", rad=-0.12, offset=-0.6, shrink=0.3)
    draw_arrow(ax, nodes["ready"], nodes["failed"], "invalid /\ninfeasible action", rad=-0.2, offset=-0.7, shrink=0.3)

    ax.set_title("Event-Driven Truck State Graph (single active truck)", pad=10)

    outdir = Path("docs/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    outfile_pdf = outdir / "state_graph.pdf"
    outfile_png = outdir / "state_graph.png"
    plt.savefig(outfile_pdf, bbox_inches="tight")
    plt.savefig(outfile_png, bbox_inches="tight", dpi=300)
    print(f"Saved state graph figure to {outfile_pdf} and {outfile_png}")


if __name__ == "__main__":
    main()
