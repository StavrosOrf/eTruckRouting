#!/usr/bin/env python3
"""
Create a transportation graph visualization with charging stations,
example delivery points, and example routes.
"""
import argparse
import os
import sys
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


class _ParallelLineHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x1 = xdescent
        x2 = xdescent + width
        center = ydescent + height * 0.5
        offsets = (-height * 0.2, 0.0, height * 0.2)
        artists = []
        for line, dy in zip(orig_handle, offsets):
            y = center + dy
            artists.append(
                Line2D(
                    [x1, x2],
                    [y, y],
                    color=line.get_color(),
                    linewidth=line.get_linewidth(),
                    alpha=line.get_alpha(),
                    linestyle=line.get_linestyle(),
                    transform=trans,
                )
            )
        return artists
import numpy as np

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.utils.plotter import EnvironmentPlotter


def _build_node_coord_map(graph, osm_coords):
    """Map internal node ids to (lat, lon) using original_id."""
    node_coords = {}
    for node_id, data in graph.nodes(data=True):
        osm_id = data.get("original_id")
        if osm_id in osm_coords:
            node_coords[node_id] = osm_coords[osm_id]
    return node_coords


def _add_charger_coords_from_file(graph, node_coords, charger_coords):
    """Fill missing node coordinates for chargers using station file coords."""
    for node_id, data in graph.nodes(data=True):
        if not data.get("has_charger", False):
            continue
        if node_id in node_coords:
            continue
        osm_id = data.get("original_id")
        if osm_id in charger_coords:
            node_coords[node_id] = charger_coords[osm_id]


# def _plot_road_segments(ax, segments, max_segments=None, seed=0):
#     """Draw faint road segments, optionally subsampled."""
#     if max_segments is not None and len(segments) > max_segments:
#         rng = random.Random(seed)
#         segments = rng.sample(segments, max_segments)

#     for seg in segments:
#         ax.plot(
#             [seg["start_lon"], seg["end_lon"]],
#             [seg["start_lat"], seg["end_lat"]],
#             color="#c7c7c7",
#             linewidth=0.4,
#             alpha=0.05,
#             zorder=1,
#         )


def _plot_stations(ax, charger_coords):
    """Plot all charging stations (assumed DC)."""
    if not charger_coords:
        return
    lats = [lat for lat, _lon in charger_coords.values()]
    lons = [lon for _lat, lon in charger_coords.values()]
    ax.scatter(
        lons,
        lats,
        s=75,
        c="#419B0C",
        #use a square marker for chargers
        marker="s",        
        alpha=0.9,
        linewidths=0.5,
        edgecolors="#0b3d2e",
        label="Charging Stations",
        zorder=4,
    )


def _plot_delivery_points(ax, delivery_points, label):
    """Plot delivery points for a single truck."""
    if not delivery_points:
        return
    lats = [lat for lat, _lon in delivery_points]
    lons = [lon for _lat, lon in delivery_points]
    ax.scatter(
        lons,
        lats,
        s=52,
        c="#d95f02",
        alpha=0.9,
        linewidths=0.4,
        edgecolors="#4a2a00",
        label=label,
        zorder=5,
    )


def _plot_connectivity_edges(
    ax,
    transport_graph,
    node_coords,
    charger_nodes,
    delivery_nodes,
    max_link_energy,
    max_links_per_charger,
):
    """Plot edges between chargers and deliveries, and between chargers."""
    edge_segments = []

    for charger in charger_nodes:
        if charger not in node_coords:
            print(f"Warning: Charger node {charger} missing coordinates, skipping connectivity edges.")
            continue

        delivery_candidates = []
        for delivery in delivery_nodes:
            if delivery not in node_coords:
                continue
            energy = transport_graph.get_path_energy(charger, delivery)
            if np.isfinite(energy) and energy <= max_link_energy:
                delivery_candidates.append((energy, delivery))

        delivery_candidates.sort(key=lambda x: x[0])
        for energy, delivery in delivery_candidates[:max_links_per_charger]:
            edge_segments.append(
                (
                    node_coords[charger][1],
                    node_coords[charger][0],
                    node_coords[delivery][1],
                    node_coords[delivery][0],
                    energy,
                )
            )

        charger_candidates = []
        for other_charger in charger_nodes:
            if other_charger <= charger:
                continue
            if other_charger not in node_coords:
                continue
            energy = transport_graph.get_path_energy(charger, other_charger)
            if np.isfinite(energy) and energy <= max_link_energy:
                charger_candidates.append((energy, other_charger))

        charger_candidates.sort(key=lambda x: x[0])
        for energy, other_charger in charger_candidates[:max_links_per_charger]:
            edge_segments.append(
                (
                    node_coords[charger][1],
                    node_coords[charger][0],
                    node_coords[other_charger][1],
                    node_coords[other_charger][0],
                    energy,
                )
            )

    if not edge_segments:
        return None

    energies = [seg[4] for seg in edge_segments]
    vmin = min(energies)
    vmax = max(energies)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for x1, y1, x2, y2, energy in edge_segments:
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=cmap(norm(energy)),
            linewidth=0.8,
            alpha=0.2,
            zorder=2,
        )

    return {"cmap": cmap, "norm": norm, "edge_count": len(edge_segments)}


def _get_potential_delivery_nodes(transport_graph):
    """Return all candidate delivery nodes (non-chargers, non-sinks)."""
    graph = transport_graph.graph
    charging_nodes = set(transport_graph.get_charging_nodes())
    return [
        node_id
        for node_id in transport_graph.get_all_nodes()
        if node_id not in charging_nodes and graph.out_degree(node_id) > 0
    ]


def _configure_axes(ax, all_coords, pad_fraction=0.02):
    """Set axes limits and cosmetics."""
    if all_coords:
        lats = [lat for lat, _lon in all_coords]
        lons = [lon for _lat, lon in all_coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        lat_pad = (max_lat - min_lat) * pad_fraction
        lon_pad = (max_lon - min_lon) * pad_fraction
        ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
        ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
        
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_facecolor("#f7f5f2")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#c7c7c7", alpha=0.7)


def _add_osm_basemap(ax, all_coords):
    """Add an OpenStreetMap basemap to the axes if contextily is available."""
    try:
        import contextily as ctx
    except ImportError:
        return

    if not all_coords:
        return

    lats = [lat for lat, _lon in all_coords]
    lons = [lon for _lat, lon in all_coords]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_pad = (max_lat - min_lat) * 0.02
    lon_pad = (max_lon - min_lon) * 0.02

    ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)

    provider = ctx.providers.OpenStreetMap.Mapnik
    try:
        provider = ctx.providers.CartoDB.Positron
    except AttributeError:
        provider = ctx.providers.OpenStreetMap.Mapnik

    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=provider,
        zoom=7,
        alpha=1,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize transport graph with routes.")
    parser.add_argument("--config", default="EVRoutingEnv/config_files/config.yaml")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-trucks", type=int, default=10)
    parser.add_argument("--num-stops", type=int, default=3)
    parser.add_argument("--output", default="results/visualization/transport_graph.png")
    parser.add_argument("--max-road-segments", type=int, default=100000)
    parser.add_argument("--max-link-energy", type=float, default=350)
    parser.add_argument("--max-links-per-charger", type=int, default=3000)
    args = parser.parse_args()

    config = load_config(args.config)
    config["environment"]["num_trucks"] = args.num_trucks
    config["environment"]["num_stops"] = args.num_stops

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = EnvironmentPlotter(output_dir=str(output_path.parent), verbose=False, use_osm=False)

    env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=args.seed)
        graph = env.transport_graph.graph
        node_coords = _build_node_coord_map(graph, plotter.node_coords)
        _add_charger_coords_from_file(graph, node_coords, plotter.charger_coords)
        delivery_node_ids = _get_potential_delivery_nodes(env.transport_graph)
        charger_node_ids = env.transport_graph.get_charging_nodes()

        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        })

        fig, ax = plt.subplots(figsize=(16, 10), dpi=300)

        delivery_points_all = [
            node_coords[n] for n in delivery_node_ids if n in node_coords
        ]
        charger_points_all = [
            node_coords[n] for n in charger_node_ids if n in node_coords
        ]

        basemap_coords = delivery_points_all + list(plotter.charger_coords.values())
        _add_osm_basemap(ax, basemap_coords)

        # if plotter.road_segments:
        #     _plot_road_segments(
        #         ax,
        #         plotter.road_segments,
        #         max_segments=args.max_road_segments,
        #         seed=args.seed,
        #     )

        max_link_energy = args.max_link_energy
        if max_link_energy is None:
            max_link_energy = config["truck"]["battery_capacity"]

        edge_color_info = _plot_connectivity_edges(
            ax,
            env.transport_graph,
            node_coords,
            charger_node_ids,
            delivery_node_ids,
            max_link_energy,
            args.max_links_per_charger,
        )

        if edge_color_info:
            sm = plt.cm.ScalarMappable(
                norm=edge_color_info["norm"],
                cmap=edge_color_info["cmap"],
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
            
            #the ticks of the cbar should be from 0-max_link_energy with ticks every 50 kWh, includign max link energy
            tick_interval = 50
            ticks = np.arange(0, max_link_energy + tick_interval, tick_interval)
            cbar.set_ticks(ticks)
            
            #pad the ticks and label of the cbar            cbar.ax.tick_params(pad=5)
            cbar.ax.tick_params(pad=10)
            cbar.set_label("Average Energy Needed (kWh)")

        _plot_stations(ax, plotter.charger_coords)

        _plot_delivery_points(ax, delivery_points_all, label="Potential Delivery Points")

        all_coords = delivery_points_all + list(plotter.charger_coords.values())
        _configure_axes(ax, all_coords)

        x = 20
        handles, labels = ax.get_legend_handles_labels()
        if edge_color_info:
            cmap = plt.cm.viridis
            edge_lines = (
                Line2D([0], [0], color=cmap(0.2), linewidth=1.5, alpha=0.6),
                Line2D([0], [0], color=cmap(0.5), linewidth=1.5, alpha=0.6),
                Line2D([0], [0], color=cmap(0.8), linewidth=1.5, alpha=0.6),
            )
            handles.append(edge_lines)
            labels.append("Transportation Links")

        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            markerscale=2.5,
            fontsize=x + 3,
            handler_map={tuple: _ParallelLineHandler()},
        )
        
        #change fontsie of ticks
        ax.tick_params(axis='both', which='major', labelsize=x)
        ax.tick_params(axis='both', which='minor', labelsize=x)
        #change size of axis labels
        ax.xaxis.label.set_size(x+4)
        ax.yaxis.label.set_size(x+4)
        #also  change the size of the cbar ticks and label
        if edge_color_info:
            cbar.ax.tick_params(labelsize=x)
            cbar.set_label("Average Energy Needed (kWh)", fontsize=x+5)

        edge_count = edge_color_info["edge_count"] if edge_color_info else 0
        print(f"Total delivery points: {len(delivery_points_all)}")
        print(f"Total chargers: {len(charger_points_all)}")
        print(f"Total connectivity edges: {edge_count}")

        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        #save as PDF as well        
        pdf_output_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_output_path, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
