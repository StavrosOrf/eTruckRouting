"""
Transportation Graph class for managing the road network.
"""
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set


class TransportationGraph:
    """Manages the transportation network graph and routing operations."""
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the transportation graph.
        
        Args:
            graph: NetworkX directed graph with nodes and edges representing the road network
        """
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.charging_nodes = self._extract_charging_nodes()
        
    def _extract_charging_nodes(self) -> List[int]:
        """Extract all nodes that have charging stations."""
        return [
            node for node, data in self.graph.nodes(data=True)
            if data.get("has_charger", False)
        ]
    
    def get_charging_nodes(self) -> List[int]:
        """Return list of all charging station nodes."""
        return self.charging_nodes.copy()
    
    def get_all_nodes(self) -> List[int]:
        """Return list of all nodes in the graph."""
        return list(self.graph.nodes())
    
    def get_distance(self, from_node: int, to_node: int) -> float:
        """
        Get the distance between two nodes.
        
        Args:
            from_node: Starting node
            to_node: Destination node
            
        Returns:
            Distance in km, or float('inf') if no path exists
        """
        try:
            return nx.shortest_path_length(
                self.graph, from_node, to_node, weight="distance"
            )
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_shortest_path(self, from_node: int, to_node: int) -> List[int]:
        """
        Get the shortest path between two nodes.
        
        Args:
            from_node: Starting node
            to_node: Destination node
            
        Returns:
            List of nodes in the shortest path, or empty list if no path exists
        """
        try:
            return nx.shortest_path(
                self.graph, from_node, to_node, weight="distance"
            )
        except nx.NetworkXNoPath:
            return []
    
    def get_edge_data(self, from_node: int, to_node: int) -> Dict:
        """
        Get edge data between two nodes.
        
        Args:
            from_node: Starting node
            to_node: Destination node
            
        Returns:
            Dictionary containing edge attributes (distance, time, terrain_factor)
        """
        if self.graph.has_edge(from_node, to_node):
            return self.graph[from_node][to_node]
        return {}
    
    def get_neighbors(self, node: int) -> List[int]:
        """
        Get all neighboring nodes (direct connections).
        
        Args:
            node: Node to get neighbors for
            
        Returns:
            List of neighboring node IDs
        """
        return list(self.graph.neighbors(node))
    
    def generate_delivery_sequence(
        self, 
        start_node: int, 
        num_stops: int, 
        min_hop_distance: float = 10.0,
        max_hop_distance: float = 100.0,
        exclude_charging_nodes: bool = False
    ) -> List[int]:
        """
        Generate a random delivery sequence with constrained hop distances.
        
        Args:
            start_node: Starting node for the delivery sequence
            num_stops: Number of delivery stops (excluding start)
            min_hop_distance: Minimum distance between consecutive stops (km)
            max_hop_distance: Maximum distance between consecutive stops (km)
            exclude_charging_nodes: If True, avoid charging nodes as delivery stops
            
        Returns:
            List of nodes representing the delivery sequence [start, stop1, stop2, ...]
        """
        sequence = [start_node]
        current_node = start_node
        
        # Get candidate nodes (excluding charging nodes if requested)
        if exclude_charging_nodes:
            candidate_nodes = [
                n for n in self.get_all_nodes() 
                if n not in self.charging_nodes
            ]
        else:
            candidate_nodes = self.get_all_nodes()
        
        attempts = 0
        max_attempts = 1000
        
        while len(sequence) < num_stops + 1 and attempts < max_attempts:
            attempts += 1
            
            # Find nodes within distance range from current node
            valid_next_nodes = []
            for node in candidate_nodes:
                if node in sequence:  # Skip already visited nodes
                    continue
                
                distance = self.get_distance(current_node, node)
                if min_hop_distance <= distance <= max_hop_distance:
                    valid_next_nodes.append(node)
            
            if not valid_next_nodes:
                # Relax constraints if no valid nodes found
                for node in candidate_nodes:
                    if node not in sequence:
                        valid_next_nodes.append(node)
                
                if not valid_next_nodes:
                    break
            
            # Randomly select next node
            next_node = np.random.choice(valid_next_nodes)
            sequence.append(next_node)
            current_node = next_node
        
        # If we couldn't generate enough stops, fill with random nodes
        while len(sequence) < num_stops + 1:
            remaining_nodes = [n for n in candidate_nodes if n not in sequence]
            if not remaining_nodes:
                break
            sequence.append(np.random.choice(remaining_nodes))
        
        return sequence
    
    def get_nearest_charging_node(self, from_node: int) -> Tuple[int, float]:
        """
        Find the nearest charging station from a given node.
        
        Args:
            from_node: Starting node
            
        Returns:
            Tuple of (nearest_charging_node, distance) or (None, float('inf')) if none found
        """
        min_distance = float('inf')
        nearest_node = None
        
        for charging_node in self.charging_nodes:
            if charging_node == from_node:
                return charging_node, 0.0
            
            distance = self.get_distance(from_node, charging_node)
            if distance < min_distance:
                min_distance = distance
                nearest_node = charging_node
        
        return nearest_node, min_distance
    
    def get_charger_info(self, node: int) -> Dict:
        """
        Get charging station information for a node.
        
        Args:
            node: Node to check
            
        Returns:
            Dictionary with 'station_type' and 'total_capacity', or empty dict if no charger
        """
        if node in self.charging_nodes:
            node_data = self.graph.nodes[node]
            return {
                'station_type': node_data.get('station_type', 'Level2'),
                'total_capacity': node_data.get('total_capacity', 1.0)
            }
        return {}
    
    def get_charger_type(self, node: int) -> str:
        """
        Get the charger type for a charging station node.
        
        Args:
            node: Charging station node
            
        Returns:
            'Level2' or 'DCFast', or None if not a charging station
        """
        info = self.get_charger_info(node)
        return info.get('station_type', None)
    
    def get_charger_capacity(self, node: int) -> float:
        """
        Get the total capacity (number of charging slots) for a charging station.
        
        Args:
            node: Charging station node
            
        Returns:
            Number of charging slots, or 0 if not a charging station
        """
        info = self.get_charger_info(node)
        return info.get('total_capacity', 0.0)
    
    def has_charger(self, node: int) -> bool:
        """Check if a node has a charging station."""
        return node in self.charging_nodes
    
    def calculate_total_distance(self, sequence: List[int]) -> float:
        """
        Calculate total distance for a sequence of nodes.
        
        Args:
            sequence: List of nodes in order
            
        Returns:
            Total distance in km
        """
        total = 0.0
        for i in range(len(sequence) - 1):
            total += self.get_distance(sequence[i], sequence[i + 1])
        return total
