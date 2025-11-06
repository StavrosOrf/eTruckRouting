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
            if all(key in route for key in ["Start_Latitude", "Start_Longitude", "End_Latitude", "End_Longitude"]):
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
            if "origin_node" in route and "Start_Latitude" in route and "Start_Longitude" in route:
                node_id = int(route["origin_node"])
                lat = float(route["Start_Latitude"])
                lon = float(route["Start_Longitude"])
                if node_id not in node_coords:
                    node_coords[node_id] = (lat, lon)
            
            if "destination_node" in route and "End_Latitude" in route and "End_Longitude" in route:
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
            if 'original_id' in node_data:
                osm_id = node_data['original_id']
                if osm_id in self.node_coords:
                    node_coords[node_idx] = self.node_coords[osm_id]
        return node_coords
    
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
            
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, 
                          zoom=10, alpha=0.4)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Warning: Could not add OSM background: {e}")
    
    def plot_initial_state(self, transport_graph: Any, truck_initial_plans: Dict[int, Dict],
                          charging_nodes: List[int], num_trucks: int):
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
                        c="#cccccc", linewidth=0.5, alpha=0.4, zorder=2,
                    )
            
            if node_coords:
                all_lats = [coord[0] for coord in node_coords.values()]
                all_lons = [coord[1] for coord in node_coords.values()]
                ax.scatter(all_lons, all_lats, c="#888888", s=15, alpha=0.5,
                          label="Network Nodes", zorder=3, edgecolors="none")
            
            truck_colors = plt.cm.tab10(range(num_trucks))
            
            for truck_id in range(num_trucks):
                truck_color = truck_colors[truck_id]
                plan = truck_initial_plans[truck_id]
                
                start_node = plan["start"]
                if start_node in node_coords:
                    start_lat, start_lon = node_coords[start_node]
                    ax.scatter(start_lon, start_lat, c=[truck_color], s=250, marker="^",
                              edgecolors="black", linewidths=2,
                              label=f"Truck {truck_id} Start", zorder=7)
                
                delivery_nodes = plan["deliveries"]
                delivery_lats, delivery_lons, delivery_coords_list = [], [], []
                
                for delivery_idx, delivery_node in enumerate(delivery_nodes):
                    if delivery_node in node_coords:
                        lat, lon = node_coords[delivery_node]
                        delivery_lats.append(lat)
                        delivery_lons.append(lon)
                        delivery_coords_list.append((lat, lon, delivery_idx + 1))
                
                if delivery_lats:
                    ax.scatter(delivery_lons, delivery_lats, c=[truck_color]*len(delivery_lats),
                              s=100, marker="o", alpha=0.9, edgecolors="black", linewidths=1.5, zorder=6)
                    
                    for lat, lon, delivery_num in delivery_coords_list:
                        ax.text(lon, lat, str(delivery_num), ha="center", va="center",
                               fontsize=10, fontweight="bold", color="white", zorder=8,
                               bbox=dict(boxstyle="circle,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))
                    
                    ax.plot(delivery_lons, delivery_lats, c="gray", alpha=0.3,
                           linewidth=1.5, linestyle="--", zorder=4)
            
            charger_lats = [lat for _, (lat, lon) in self.charger_coords.items()]
            charger_lons = [lon for _, (lat, lon) in self.charger_coords.items()]
            
            if charger_lats:
                ax.scatter(charger_lons, charger_lats, c="red", s=80, marker="s",
                          edgecolors="darkred", linewidths=1, alpha=0.9,
                          label="Charging Stations", zorder=6)
            
            ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
            ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
            ax.set_title(f"Initial Truck Positions\n({num_trucks} Trucks, {len(self.charger_coords)} Stations)",
                        fontsize=15, fontweight="bold", pad=20)
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
    
    def plot_final_routes(self, transport_graph: Any, truck_routes: Dict[int, List[Tuple]],
                         charging_nodes: List[int], num_trucks: int, final_time: float):
        """Plot actual routes followed by trucks during simulation."""
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
                        c="#cccccc", linewidth=0.5, alpha=0.4, zorder=2,
                    )
            
            if node_coords:
                all_lats = [coord[0] for coord in node_coords.values()]
                all_lons = [coord[1] for coord in node_coords.values()]
                ax.scatter(all_lons, all_lats, c="#888888", s=15, alpha=0.5,
                          label="Network Nodes", zorder=3, edgecolors="none")
            
            truck_colors = plt.cm.tab10(range(num_trucks))
            
            for truck_id, route in truck_routes.items():
                if not route:
                    continue
                
                truck_color = truck_colors[truck_id]
                
                route_nodes = [r[0] for r in route]
                event_types = [r[2] if len(r) > 2 else 'travel' for r in route]
                
                route_coords = [node_coords[node] for node in route_nodes if node in node_coords]
                if route_coords:
                    route_lats = [coord[0] for coord in route_coords]
                    route_lons = [coord[1] for coord in route_coords]
                    ax.plot(route_lons, route_lats, c=truck_color, alpha=0.7,
                           linewidth=2.5, zorder=3)
                
                if route_nodes and route_nodes[0] in node_coords:
                    start_lat, start_lon = node_coords[route_nodes[0]]
                    ax.scatter(start_lon, start_lat, c=[truck_color], s=250, marker="^",
                              edgecolors="black", linewidths=2,
                              label=f"Truck {truck_id} Start", zorder=7)
                
                delivery_indices = [i for i, et in enumerate(event_types) if et == 'delivery']
                if delivery_indices:
                    delivery_lats = [node_coords[route_nodes[i]][0] for i in delivery_indices if route_nodes[i] in node_coords]
                    delivery_lons = [node_coords[route_nodes[i]][1] for i in delivery_indices if route_nodes[i] in node_coords]
                    
                    ax.scatter(delivery_lons, delivery_lats, c=[truck_color]*len(delivery_lons),
                              s=100, marker="o", alpha=0.9, edgecolors="black", linewidths=1.5, zorder=6)
                    
                    for idx, delivery_idx in enumerate(delivery_indices, 1):
                        if route_nodes[delivery_idx] in node_coords:
                            lat, lon = node_coords[route_nodes[delivery_idx]]
                            ax.text(lon, lat, str(idx), ha="center", va="center",
                                   fontsize=10, fontweight="bold", color="white", zorder=8,
                                   bbox=dict(boxstyle="circle,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))
                
                charger_indices = [i for i, et in enumerate(event_types) if et == 'charger']
                if charger_indices:
                    charger_lats = [node_coords[route_nodes[i]][0] for i in charger_indices if route_nodes[i] in node_coords]
                    charger_lons = [node_coords[route_nodes[i]][1] for i in charger_indices if route_nodes[i] in node_coords]
                    ax.scatter(charger_lons, charger_lats, c=[truck_color]*len(charger_lons),
                              s=80, marker="D", alpha=0.7, edgecolors="black", linewidths=1, zorder=5)
            
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
        
        except Exception as e:
            print(f"Error creating final routes plot: {e}")
            import traceback
            traceback.print_exc()
