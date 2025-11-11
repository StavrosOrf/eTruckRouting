"""
Visualization and plotting utilities for the event-driven truck environment.

Uses OpenStreetMap tiles and real geographic coordinates for visualization.
"""

import os
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class EnvironmentPlotter:
    """
    Handles all plotting and visualization for the truck routing environment using OSM coordinates.
    """

    def __init__(self, output_dir: str, verbose: bool = False, use_osm: bool = True):
        """
        Initialize the plotter.

        Args:
            output_dir: Directory to save plots
            verbose: Print verbose messages
            use_osm: Whether to include OSM basemap in plots
        """
        self.output_dir = output_dir
        self.verbose = verbose
        self.use_osm = use_osm
        self.node_coords = {}
        self.charger_coords = {}
        self.road_segments = []

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._load_visualization_data()

    def _load_visualization_data(self):
        """Load visualization data from JSON files."""
        try:
            data_dir = Path(__file__).parent.parent / "data" / "vis_data"

            with open(data_dir / "node_CA_df.json", "r") as f:
                route_data = json.load(f)

            with open(data_dir / "station_CA_df.json", "r") as f:
                station_data = json.load(f)

            self.node_coords = self._create_node_coordinate_map_from_routes(route_data)
            self.charger_coords = self._create_charger_coordinate_map(station_data)
            self.road_segments = self._extract_road_segments_from_routes(route_data)

            if self.verbose:
                print(f"  ✓ Loaded {len(self.node_coords)} OSM nodes")
                print(f"  ✓ Loaded {len(self.charger_coords)} charger stations")
                print(f"  ✓ Loaded {len(self.road_segments)} road segments")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Warning: Could not load visualization data: {e}")

    def _extract_road_segments_from_routes(self, route_data):
        """Extract road segments from route data."""
        segments = []
        for route in route_data:
            if all(
                key in route
                for key in [
                    "Start_Latitude",
                    "Start_Longitude",
                    "End_Latitude",
                    "End_Longitude",
                ]
            ):
                segment = {
                    "start_lat": float(route["Start_Latitude"]),
                    "start_lon": float(route["Start_Longitude"]),
                    "end_lat": float(route["End_Latitude"]),
                    "end_lon": float(route["End_Longitude"]),
                }
                segments.append(segment)
        return segments

    def _create_node_coordinate_map_from_routes(self, route_data):
        """Create a mapping from OSM node ID to (latitude, longitude)."""
        node_coords = {}
        for route in route_data:
            if (
                "origin_node" in route
                and "Start_Latitude" in route
                and "Start_Longitude" in route
            ):
                node_id = int(route["origin_node"])
                lat = float(route["Start_Latitude"])
                lon = float(route["Start_Longitude"])
                if node_id not in node_coords:
                    node_coords[node_id] = (lat, lon)

            if (
                "destination_node" in route
                and "End_Latitude" in route
                and "End_Longitude" in route
            ):
                node_id = int(route["destination_node"])
                lat = float(route["End_Latitude"])
                lon = float(route["End_Longitude"])
                if node_id not in node_coords:
                    node_coords[node_id] = (lat, lon)
        return node_coords

    def _create_charger_coordinate_map(self, station_data):
        """Create a mapping from charger node ID to (latitude, longitude)."""
        charger_coords = {}
        for station in station_data:
            node_id = int(station["node"])
            lat = float(station["Latitude"])
            lon = float(station["Longitude"])
            charger_coords[node_id] = (lat, lon)
        return charger_coords

    def _create_node_id_to_osm_map(self, transport_graph):
        """Create mapping from graph node indices to OSM coordinates."""
        node_coords = {}
        for node_idx in transport_graph.graph.nodes():
            node_data = transport_graph.graph.nodes[node_idx]
            if "original_id" in node_data:
                osm_id = node_data["original_id"]
                if osm_id in self.node_coords:
                    node_coords[node_idx] = self.node_coords[osm_id]
        return node_coords
    
    def _create_charger_id_to_osm_map(self, transport_graph):
        """Create mapping from charger node indices to OSM coordinates."""
        charger_coords = {}
        for node_idx in transport_graph.graph.nodes():
            node_data = transport_graph.graph.nodes[node_idx]
            if node_data.get("has_charger", False):
                osm_id = node_data["original_id"]
                if osm_id in self.charger_coords:
                    charger_coords[node_idx] = self.charger_coords[osm_id]
        return charger_coords

    def _add_osm_background(self, ax, node_coords):
        """Add OpenStreetMap background to axes."""
        if not self.use_osm:
            return

        try:
            import contextily as ctx

            if node_coords:
                all_lats = [coord[0] for coord in node_coords.values()]
                all_lons = [coord[1] for coord in node_coords.values()]
                min_lat, max_lat = min(all_lats), max(all_lats)
                min_lon, max_lon = min(all_lons), max(all_lons)

                lat_margin = (max_lat - min_lat) * 0.1
                lon_margin = (max_lon - min_lon) * 0.1

                ax.set_xlim(min_lon - lon_margin, max_lon + lon_margin)
                ax.set_ylim(min_lat - lat_margin, max_lat + lat_margin)

            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=10,
                alpha=0.4,
            )
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Warning: Could not add OSM background: {e}")

    def plot_initial_state(
        self,
        transport_graph: Any,
        truck_initial_plans: Dict[int, Dict],
        charging_nodes: List[int],
        num_trucks: int,
    ):
        """Plot initial truck positions and planned delivery routes."""
        try:
            fig, ax = plt.subplots(figsize=(18, 14), dpi=150)

            node_coords = self._create_node_id_to_osm_map(transport_graph)
            if not node_coords:
                print("Warning: No node coordinates found.")
                return

            self._add_osm_background(ax, node_coords)

            if self.road_segments:
                for segment in self.road_segments:
                    ax.plot(
                        [segment["start_lon"], segment["end_lon"]],
                        [segment["start_lat"], segment["end_lat"]],
                        c="#cccccc",
                        linewidth=0.5,
                        alpha=0.4,
                        zorder=2,
                    )

            if node_coords:
                all_lats = [coord[0] for coord in node_coords.values()]
                all_lons = [coord[1] for coord in node_coords.values()]
                ax.scatter(
                    all_lons,
                    all_lats,
                    c="#888888",
                    s=15,
                    alpha=0.5,
                    label="Network Nodes",
                    zorder=3,
                    edgecolors="none",
                )

            truck_colors = plt.cm.tab10(range(num_trucks))

            for truck_id in range(num_trucks):
                truck_color = truck_colors[truck_id]
                plan = truck_initial_plans[truck_id]

                start_node = plan["start"]
                if start_node in node_coords:
                    start_lat, start_lon = node_coords[start_node]
                    ax.scatter(
                        start_lon,
                        start_lat,
                        c=[truck_color],
                        s=250,
                        marker="^",
                        edgecolors="black",
                        linewidths=2,
                        label=f"Truck {truck_id} Start",
                        zorder=7,
                    )

                delivery_nodes = plan["deliveries"]
                delivery_lats, delivery_lons, delivery_coords_list = [], [], []

                for delivery_idx, delivery_node in enumerate(delivery_nodes):
                    if delivery_node in node_coords:
                        lat, lon = node_coords[delivery_node]
                        delivery_lats.append(lat)
                        delivery_lons.append(lon)
                        delivery_coords_list.append((lat, lon, delivery_idx + 1))

                if delivery_lats:
                    ax.scatter(
                        delivery_lons,
                        delivery_lats,
                        c=[truck_color] * len(delivery_lats),
                        s=100,
                        marker="o",
                        alpha=0.9,
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=6,
                    )

                    for lat, lon, delivery_num in delivery_coords_list:
                        ax.text(
                            lon,
                            lat,
                            str(delivery_num),
                            ha="center",
                            va="center",
                            fontsize=10,
                            fontweight="bold",
                            color="white",
                            zorder=8,
                            bbox=dict(
                                boxstyle="circle,pad=0.1",
                                facecolor="black",
                                alpha=0.5,
                                edgecolor="none",
                            ),
                        )

                    ax.plot(
                        delivery_lons,
                        delivery_lats,
                        c="gray",
                        alpha=0.3,
                        linewidth=1.5,
                        linestyle="--",
                        zorder=4,
                    )

            charger_lats = [lat for _, (lat, lon) in self.charger_coords.items()]
            charger_lons = [lon for _, (lat, lon) in self.charger_coords.items()]

            if charger_lats:
                ax.scatter(
                    charger_lons,
                    charger_lats,
                    c="red",
                    s=80,
                    marker="s",
                    edgecolors="darkred",
                    linewidths=1,
                    alpha=0.9,
                    label="Charging Stations",
                    zorder=6,
                )

            ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
            ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
            ax.set_title(
                f"Initial Truck Positions\n({num_trucks} Trucks, {len(self.charger_coords)} Stations)",
                fontsize=15,
                fontweight="bold",
                pad=20,
            )
            ax.set_facecolor("#f5f5f5")
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
            ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="black")

            filepath = os.path.join(self.output_dir, "initial_state.png")
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()

            if self.verbose:
                print(f"  ✓ Initial state plot saved to: {filepath}")

        except Exception as e:
            print(f"Error creating initial state plot: {e}")
            import traceback

            traceback.print_exc()

    def plot_final_routes(
        self,
        transport_graph: Any,
        truck_routes: Dict[int, List[Tuple]],
        charging_nodes: List[int],
        num_trucks: int,
        final_time: float,
        truck_initial_plans: Dict[int, Dict],
    ):
        """Plot actual routes followed by trucks during simulation."""
        
        fig, ax = plt.subplots(figsize=(18, 14), dpi=150)

        node_coords = self._create_node_id_to_osm_map(transport_graph)
        if not node_coords:
            print("Warning: No node coordinates found.")
            return

        self._add_osm_background(ax, node_coords)

        charger_coords = self._create_charger_id_to_osm_map(transport_graph)

        if self.road_segments:
            for segment in self.road_segments:
                ax.plot(
                    [segment["start_lon"], segment["end_lon"]],
                    [segment["start_lat"], segment["end_lat"]],
                    c="#cccccc",
                    linewidth=0.5,
                    alpha=0.4,
                    zorder=2,
                )

        if node_coords:
            all_lats = [coord[0] for coord in node_coords.values()]
            all_lons = [coord[1] for coord in node_coords.values()]
            ax.scatter(
                all_lons,
                all_lats,
                c="#888888",
                s=15,
                alpha=0.5,
                label="Network Nodes",
                zorder=3,
                edgecolors="none",
            )

        truck_colors = plt.cm.tab10(range(num_trucks))

        for truck_id in range(num_trucks):
            route = truck_routes[truck_id]
            
            #get route nodes and coordinates
            route_nodes = [int(entry[0]) for entry in route]
            event_types = [entry[2] for entry in route]
            route_coords = []
            for node in route_nodes:
                if node in node_coords:
                    route_coords.append(node_coords[node])
                elif node in charger_coords:
                    route_coords.append(charger_coords[node])
                else:
                    print(f"Warning: No coordinates found for node {node}")

            truck_color = truck_colors[truck_id]
            
            # Draw colored edges for each consecutive pair of nodes in the route with numbering
            for i in range(len(route_coords) - 1):
                lat1, lon1 = route_coords[i]
                lat2, lon2 = route_coords[i + 1]
                ax.plot([lon1, lon2], [lat1, lat2], c=truck_color, alpha=0.7,
                       linewidth=2.5, zorder=4)
                
                # Add edge number to show direction
                mid_lat = (lat1 + lat2) / 2
                mid_lon = (lon1 + lon2) / 2
                ax.text(mid_lon, mid_lat, str(i + 1), ha="center", va="center",
                       fontsize=8, fontweight="bold", color=truck_color, zorder=5,
                       bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", alpha=0.8, edgecolor=truck_color, linewidth=1.5))
            
            # Plot truck start position
            if route_coords:
                start_lat, start_lon = route_coords[0]
                ax.scatter(start_lon, start_lat, c=[truck_color], s=250, marker="^",
                          edgecolors="black", linewidths=2,
                          label=f"Truck {truck_id} Start", zorder=7)
            
            # Plot all planned delivery nodes (not just completed ones)
            # Find delivery nodes from the initial plans

            
            plan = truck_initial_plans[truck_id]
            
            delivery_nodes = plan["deliveries"]
            delivery_lats, delivery_lons, delivery_coords_list = [], [], []

            for delivery_idx, delivery_node in enumerate(delivery_nodes):
                if delivery_node in node_coords:
                    lat, lon = node_coords[delivery_node]
                    delivery_lats.append(lat)
                    delivery_lons.append(lon)
                    delivery_coords_list.append((lat, lon, delivery_idx + 1))

            if delivery_lats:
                ax.scatter(
                    delivery_lons,
                    delivery_lats,
                    c=[truck_color] * len(delivery_lats),
                    s=100,
                    marker="o",
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=6,
                )

                for lat, lon, delivery_num in delivery_coords_list:
                    ax.text(
                        lon,
                        lat,
                        str(delivery_num),
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="white",
                        zorder=8,
                        bbox=dict(
                            boxstyle="circle,pad=0.1",
                            facecolor="black",
                            alpha=0.5,
                            edgecolor="none",
                        ),
                    )

                ax.plot(
                    delivery_lons,
                    delivery_lats,
                    c="gray",
                    alpha=0.3,
                    linewidth=1.5,
                    linestyle="--",
                    zorder=4,
                )
                
                
            # Plot charging events
            charger_indices = [i for i, et in enumerate(event_types) if et == 'charger']
            if charger_indices:
                charger_locs = [route_coords[i] for i in charger_indices if i < len(route_coords)]
                if charger_locs:
                    charger_lats = [coord[0] for coord in charger_locs]
                    charger_lons = [coord[1] for coord in charger_locs]
                    ax.scatter(charger_lons, charger_lats, c=[truck_color]*len(charger_lons),
                              s=80, marker="D", alpha=0.7, edgecolors="black", linewidths=1, zorder=5)
        
        # Plot all charging stations
        charger_lats = [lat for _, (lat, lon) in self.charger_coords.items()]
        charger_lons = [lon for _, (lat, lon) in self.charger_coords.items()]
        
        if charger_lats:
            ax.scatter(charger_lons, charger_lats, c="red", s=80, marker="s",
                      edgecolors="darkred", linewidths=1, alpha=0.9,
                      label="Charging Stations", zorder=6)
        
        ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
        ax.set_title(f"Final Truck Routes\n({num_trucks} Trucks, Time: {final_time:.1f}h)",
                    fontsize=15, fontweight="bold", pad=20)
        ax.set_facecolor("#f5f5f5")
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="black")
        
        filepath = os.path.join(self.output_dir, "final_routes.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        if self.verbose:
            print(f"  ✓ Final routes plot saved to: {filepath}")
    
    def plot_charger_queue_dynamics(
        self,
        charging_station,
        transport_graph: Any,
        final_time: float,
    ):
        """
        Plot queue dynamics (occupancy and waitlist) for each charging station visited.
        
        Args:
            charging_station: ChargingStation instance with queue_history
            transport_graph: TransportationGraph instance
            final_time: Final simulation time
        """
        try:
            # Get chargers that were actually visited
            visited_chargers = []
            for node in charging_station.charging_nodes:
                history = charging_station.queue_history[node]
                if len(history["truck_events"]) > 0:  # Only plot chargers with activity
                    visited_chargers.append(node)
            
            if not visited_chargers:
                if self.verbose:
                    print("  ⚠ No charging stations were visited during simulation")
                return
            
            # Create subplots: one row per charger
            n_chargers = len(visited_chargers)
            fig, axes = plt.subplots(n_chargers, 1, figsize=(16, 4 * n_chargers), dpi=150)
            
            # Ensure axes is always a list
            if n_chargers == 1:
                axes = [axes]
            
            for idx, charger_node in enumerate(visited_chargers):
                ax = axes[idx]
                history = charging_station.queue_history[charger_node]
                
                # Get charger info
                charger_type = charging_station.charger_type[charger_node]
                capacity = int(charging_station.charger_capacity[charger_node])
                
                # Extract time series data
                times = np.array(history["times"])
                occupancy = np.array(history["occupancy"])
                waitlist = np.array(history["waitlist"])
                truck_events = history["truck_events"]
                
                # Plot occupancy and waitlist over time
                ax.plot(times, occupancy, 'b-', linewidth=2, label='Occupancy (Charging)', marker='o', markersize=4)
                ax.plot(times, waitlist, 'r--', linewidth=2, label='Waitlist (Waiting)', marker='s', markersize=4)
                ax.axhline(y=capacity, color='green', linestyle=':', linewidth=2, label=f'Capacity ({capacity})')
                
                # Add event markers
                event_colors = {'arrive': 'orange', 'start': 'blue', 'finish': 'purple'}
                event_markers = {'arrive': '^', 'start': 'o', 'finish': 'v'}
                event_labels = {'arrive': 'Arrive', 'start': 'Start Charging', 'finish': 'Finish Charging'}
                
                plotted_events = set()
                for event_time, truck_id, event_type in truck_events:
                    label = event_labels[event_type] if event_type not in plotted_events else None
                    ax.scatter(event_time, -0.3, c=event_colors[event_type], 
                             marker=event_markers[event_type], s=100, 
                             label=label, zorder=5, edgecolors='black', linewidths=0.5)
                    # Add truck ID annotation
                    ax.text(event_time, -0.5, f'T{truck_id}', ha='center', va='top', 
                           fontsize=8, fontweight='bold')
                    plotted_events.add(event_type)
                
                # Styling
                ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Number of Trucks', fontsize=11, fontweight='bold')
                ax.set_title(f'Charger Node {charger_node} ({charger_type}, Capacity: {capacity})',
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
                ax.set_xlim(-0.5, final_time + 0.5)
                ax.set_ylim(-1, max(capacity + 1, max(occupancy.max() if len(occupancy) > 0 else 0,
                                                     waitlist.max() if len(waitlist) > 0 else 0) + 1))
                
                # Fill areas
                ax.fill_between(times, 0, occupancy, alpha=0.2, color='blue', label='_nolegend_')
                ax.fill_between(times, 0, waitlist, alpha=0.2, color='red', label='_nolegend_')
            
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, "charger_queue_dynamics.png")
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            
            if self.verbose:
                print(f"  ✓ Charger queue dynamics plot saved to: {filepath}")
        
        except Exception as e:
            print(f"Error creating charger queue dynamics plot: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_charger_utilization_heatmap(
        self,
        charging_station,
        transport_graph: Any,
        final_time: float,
    ):
        """
        Plot a heatmap showing when each charging station was in use.
        
        Args:
            charging_station: ChargingStation instance
            transport_graph: TransportationGraph instance
            final_time: Final simulation time
        """
        try:
            # Get visited chargers
            visited_chargers = []
            for node in charging_station.charging_nodes:
                history = charging_station.queue_history[node]
                if len(history["truck_events"]) > 0:
                    visited_chargers.append(node)
            
            if not visited_chargers:
                return
            
            # Create time bins (e.g., 0.5 hour intervals)
            time_resolution = 0.5  # hours
            n_bins = int(np.ceil(final_time / time_resolution))
            time_bins = np.arange(0, final_time + time_resolution, time_resolution)
            
            # Create occupancy matrix
            occupancy_matrix = np.zeros((len(visited_chargers), n_bins))
            
            for i, charger_node in enumerate(visited_chargers):
                history = charging_station.queue_history[charger_node]
                times = np.array(history["times"])
                occupancy = np.array(history["occupancy"])
                
                # Interpolate occupancy for each time bin
                for j in range(n_bins):
                    bin_start = j * time_resolution
                    bin_end = (j + 1) * time_resolution
                    
                    # Find occupancy values in this time range
                    mask = (times >= bin_start) & (times < bin_end)
                    if mask.any():
                        occupancy_matrix[i, j] = occupancy[mask].mean()
                    elif len(times) > 0 and times[0] > bin_end:
                        occupancy_matrix[i, j] = 0
                    elif len(times) > 0:
                        # Use last known value
                        last_idx = np.where(times < bin_start)[0]
                        if len(last_idx) > 0:
                            occupancy_matrix[i, j] = occupancy[last_idx[-1]]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(16, max(6, len(visited_chargers) * 0.5)), dpi=150)
            
            im = ax.imshow(occupancy_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            
            # Set ticks and labels
            ax.set_yticks(range(len(visited_chargers)))
            charger_labels = []
            for node in visited_chargers:
                c_type = charging_station.charger_type[node]
                capacity = int(charging_station.charger_capacity[node])
                charger_labels.append(f'Node {node}\n({c_type}, Cap:{capacity})')
            ax.set_yticklabels(charger_labels, fontsize=9)
            
            # X-axis: time bins
            x_tick_interval = max(1, n_bins // 20)  # Show ~20 ticks max
            x_ticks = range(0, n_bins, x_tick_interval)
            x_labels = [f'{i * time_resolution:.1f}h' for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            
            ax.set_xlabel('Simulation Time', fontsize=12, fontweight='bold')
            ax.set_ylabel('Charging Station', fontsize=12, fontweight='bold')
            ax.set_title('Charging Station Utilization Heatmap\n(Average Occupancy per Time Bin)',
                        fontsize=14, fontweight='bold', pad=15)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Average Occupancy', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, "charger_utilization_heatmap.png")
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            
            if self.verbose:
                print(f"  ✓ Charger utilization heatmap saved to: {filepath}")
        
        except Exception as e:
            print(f"Error creating charger utilization heatmap: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_charger_statistics_summary(
        self,
        charging_station,
        transport_graph: Any,
    ):
        """
        Plot summary statistics for each charging station.
        
        Args:
            charging_station: ChargingStation instance
            transport_graph: TransportationGraph instance
        """
        try:
            # Get visited chargers and their stats
            visited_chargers = []
            for node in charging_station.charging_nodes:
                stats = charging_station.charger_stats[node]
                if stats["total_charge_sessions"] > 0:
                    visited_chargers.append(node)
            
            if not visited_chargers:
                return
            
            # Extract statistics
            charger_labels = []
            sessions = []
            charge_times = []
            trucks_served = []
            
            for node in visited_chargers:
                stats = charging_station.charger_stats[node]
                c_type = charging_station.charger_type[node]
                capacity = int(charging_station.charger_capacity[node])
                
                charger_labels.append(f'Node {node}\n({c_type[:4]})')
                sessions.append(stats["total_charge_sessions"])
                charge_times.append(stats["total_charge_time"])
                trucks_served.append(len(stats["total_trucks_served"]))
            
            # Create bar charts
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
            
            x = np.arange(len(visited_chargers))
            width = 0.6
            
            # Sessions
            axes[0].bar(x, sessions, width, color='steelblue', edgecolor='black', linewidth=1)
            axes[0].set_ylabel('Number of Sessions', fontsize=11, fontweight='bold')
            axes[0].set_title('Total Charge Sessions', fontsize=12, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(charger_labels, fontsize=9)
            axes[0].grid(axis='y', alpha=0.3)
            
            # Charge time
            axes[1].bar(x, charge_times, width, color='coral', edgecolor='black', linewidth=1)
            axes[1].set_ylabel('Total Time (hours)', fontsize=11, fontweight='bold')
            axes[1].set_title('Total Charge Time', fontsize=12, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(charger_labels, fontsize=9)
            axes[1].grid(axis='y', alpha=0.3)
            
            # Trucks served
            axes[2].bar(x, trucks_served, width, color='mediumseagreen', edgecolor='black', linewidth=1)
            axes[2].set_ylabel('Number of Trucks', fontsize=11, fontweight='bold')
            axes[2].set_title('Unique Trucks Served', fontsize=12, fontweight='bold')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(charger_labels, fontsize=9)
            axes[2].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for ax in axes:
                for i, bar in enumerate(ax.patches):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.suptitle('Charging Station Statistics Summary', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, "charger_statistics_summary.png")
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            
            if self.verbose:
                print(f"  ✓ Charger statistics summary saved to: {filepath}")
        
        except Exception as e:
            print(f"Error creating charger statistics summary: {e}")
            import traceback
            traceback.print_exc()

