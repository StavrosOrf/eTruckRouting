"""
Create an illustrative example of the heterogeneous GNN state space (action-oriented view).

Depicts:
- Active truck node (ready) and another truck (routing)
- Delivery nodes (green) and charger nodes (blue)
- Feasible edges from the active truck with energy/time annotations (solid)
- Masked/blocked edge (dashed red) to illustrate action masking

Outputs:
- docs/figures/gnn_state_space_example.pdf
- docs/figures/gnn_state_space_example.png
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle


def draw_node(ax, xy, radius, facecolor, edgecolor, label, subtitle=None, z=3):
    """Draw a circular node with optional subtitle."""
    circle = Circle(xy, radius=radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=2.4, zorder=z)
    ax.add_patch(circle)
    ax.text(
        xy[0],
        xy[1],
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=edgecolor,
        zorder=z + 1,
    )
    if subtitle:
        ax.text(
            xy[0],
            xy[1] - radius - 0.14,
            subtitle,
            ha="center",
            va="top",
            fontsize=9,
            color="#34495e",
            zorder=z + 1,
        )


def draw_edge(ax, src, dst, text, color="#555555", rad=0.0, style="-|>", lw=2.0, alpha=0.95):
    """Draw a curved arrow with a label at its midpoint."""
    arrow = FancyArrowPatch(
        posA=src,
        posB=dst,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(arrow)

    # Midpoint for label
    mx = (src[0] + dst[0]) / 2
    my = (src[1] + dst[1]) / 2
    ax.text(
        mx,
        my + 0.14,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#0f172a",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        zorder=4,
    )


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # Node positions
    pos = {
        "truck_active": (0.0, 0.0),
        "truck_other": (-3.0, -0.3),
        "del_1": (3.2, 1.9),
        "del_2": (3.5, 0.2),
        "del_3": (3.0, -1.6),
        "ch_1": (-1.7, 2.2),
        "ch_2": (-1.7, -2.1),
    }

    # Draw nodes
    draw_node(ax, pos["truck_active"], 0.45, facecolor="#fff4e6", edgecolor="#d35400", label="Truck A", subtitle="active / ready")
    draw_node(ax, pos["truck_other"], 0.38, facecolor="#f2f4f8", edgecolor="#7f8c8d", label="Truck B", subtitle="routing")
    draw_node(ax, pos["del_1"], 0.36, facecolor="#e6f6ec", edgecolor="#1e8449", label="Delivery 1")
    draw_node(ax, pos["del_2"], 0.36, facecolor="#e6f6ec", edgecolor="#1e8449", label="Delivery 2")
    draw_node(ax, pos["del_3"], 0.36, facecolor="#e6f6ec", edgecolor="#1e8449", label="Delivery 3")
    draw_node(ax, pos["ch_1"], 0.36, facecolor="#e8eeff", edgecolor="#2c5fd5", label="Charger A", subtitle="DC fast")
    draw_node(ax, pos["ch_2"], 0.36, facecolor="#e8eeff", edgecolor="#2c5fd5", label="Charger B", subtitle="Level 2")

    # Feasible edges from active truck
    draw_edge(ax, pos["truck_active"], pos["del_1"], "energy 38 kWh\n0.7 h", rad=0.12, color="#1e8449")
    draw_edge(ax, pos["truck_active"], pos["del_2"], "energy 46 kWh\n0.9 h", rad=0.04, color="#1e8449")
    draw_edge(ax, pos["truck_active"], pos["del_3"], "energy 32 kWh\n0.6 h", rad=-0.06, color="#1e8449")
    draw_edge(ax, pos["truck_active"], pos["ch_1"], "energy 18 kWh\n0.3 h", rad=0.15, color="#2c5fd5")
    draw_edge(ax, pos["truck_active"], pos["ch_2"], "energy 20 kWh\n0.4 h", rad=-0.15, color="#2c5fd5")

    # Masked/blocked edge example (insufficient battery)
    draw_edge(ax, pos["truck_active"], (pos["ch_2"][0] + 0.2, pos["ch_2"][1] - 0.2), "masked (energy > battery)", rad=-0.2, color="#c0392b", style="-[", lw=2.2, alpha=0.8)

    # Destination edge if routing (dashed to indicate routing-only edge)
    dash_arrow = FancyArrowPatch(
        posA=pos["truck_other"],
        posB=pos["ch_1"],
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.05",
        mutation_scale=11,
        linewidth=1.6,
        color="#7f8c8d",
        alpha=0.75,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(dash_arrow)
    ax.text(
        (pos["truck_other"][0] + pos["ch_1"][0]) / 2 - 0.05,
        (pos["truck_other"][1] + pos["ch_1"][1]) / 2 + 0.16,
        "routing edge (time-only)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4a4a4a",
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
        zorder=4,
    )

    # Legend / annotations
    ax.text(
        -4.4,
        2.5,
        "Node types:\n- Truck (orange)\n- Delivery (green)\n- Charger (blue)\n\nEdge features:\n- energy_distance (kWh)\n- time_distance (hours)\n\nState-dependent edges:\n- Feasible edges: solid, labeled energy/time\n- Masked edges: dashed red\n- Routing trucks: dashed gray to destination (time-only)",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#0f172a",
        bbox=dict(facecolor="white", alpha=0.94, edgecolor="#cfd4dd"),
        zorder=5,
    )

    ax.set_title("Example GNN State Space (heterogeneous graph)", pad=12)
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-3.0, 3.0)

    outdir = Path("docs/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "gnn_state_space_example.pdf"
    png_path = outdir / "gnn_state_space_example.png"
    plt.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved GNN state space example to {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
