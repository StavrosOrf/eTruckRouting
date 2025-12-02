"""
Truck class representing individual delivery vehicles.
"""
from typing import List, Dict, Optional
import numpy as np


class Truck:
    """Represents a delivery truck with battery, route, and state information."""
    
    def __init__(
        self,
        truck_id: int,
        truck_type: str,
        delivery_sequence: List[int],
        initial_battery: float,
        battery_capacity: float,
        base_speed: float,
    ):
        """
        Initialize a truck.
        
        Args:
            truck_id: Unique identifier for the truck
            truck_type: Type of truck ("standard" or "heavy")
            delivery_sequence: List of nodes to visit in order [start, stop1, stop2, ...]
            initial_battery: Starting battery level (kWh)
            battery_capacity: Maximum battery capacity (kWh)
            base_speed: Base speed of truck (km/h)
        """
        self.truck_id = truck_id
        self.truck_type = truck_type
        self.delivery_sequence = delivery_sequence.copy()
        self.battery_capacity = battery_capacity
        self.base_speed = base_speed
        
        # Current state
        self.current_battery = initial_battery
        self.current_node = delivery_sequence[0]
        self.current_sequence_index = 0  # Index in delivery_sequence
        
        # Statistics
        self.total_distance_traveled = 0.0
        self.total_time_elapsed = 0.0
        self.total_charging_time = 0.0
        self.num_charging_sessions = 0
        self.waiting_time = 0.0
        self.is_charging = False
        self.charge_start_time = None
        
        # Completion tracking
        self.is_complete = False
        self.failed = False  # True if ran out of battery
        
        # Route tracking (for GNN state representation)
        self.route_destination = None  # Next destination when on route
        self.route_arrival_time = None  # Event time when truck will arrive at destination
        
        # Charging policy: truck must leave after charging
        self.must_leave_charger = False  # True if truck just finished charging and must leave
    
    def get_next_delivery_target(self) -> Optional[int]:
        """
        Get the next delivery destination in the sequence.
        
        Returns:
            Next node ID, or None if sequence is complete
        """
        if self.current_sequence_index + 1 < len(self.delivery_sequence):
            return self.delivery_sequence[self.current_sequence_index + 1]
        return None
    
    def get_remaining_deliveries(self) -> List[int]:
        """Get list of remaining delivery nodes."""
        return self.delivery_sequence[self.current_sequence_index + 1:]
    
    def advance_to_next_delivery(self):
        """Mark current delivery as complete and advance to next."""
        if self.current_sequence_index + 1 < len(self.delivery_sequence):
            self.current_sequence_index += 1
            self.current_node = self.delivery_sequence[self.current_sequence_index]
            
            # Check if all deliveries complete
            if self.current_sequence_index == len(self.delivery_sequence) - 1:
                self.is_complete = True
    
    def move_to_node(self, node: int, distance: float, travel_time: float, discharge: float):
        """
        Update truck state after moving to a new node.
        
        Args:
            node: Destination node
            distance: Distance traveled (km)
            travel_time: Time taken (hours)
            discharge: Battery consumed (kWh)
        """
        self.current_node = node
        self.current_battery -= discharge
        self.total_distance_traveled += distance
        self.total_time_elapsed += travel_time
        self.must_leave_charger = False  # Reset flag when moving away
        
        # Check if this was a delivery target
        if node == self.get_next_delivery_target():
            self.advance_to_next_delivery()
        
        # Check if out of battery
        if self.current_battery <= 0:
            self.current_battery = 0
            self.failed = True
    
    def start_charging(self, current_time: float):
        """Mark truck as starting to charge."""
        self.is_charging = True
        self.charge_start_time = current_time
        self.must_leave_charger = False  # Reset flag when starting new charge session
    
    def finish_charging(self, charge_amount: float, charge_duration: float):
        """
        Update truck state after charging.
        
        Args:
            charge_amount: Amount charged (kWh)
            charge_duration: Time spent charging (hours)
        """
        self.current_battery = min(self.battery_capacity, self.current_battery + charge_amount)
        self.total_charging_time += charge_duration
        self.total_time_elapsed += charge_duration
        self.num_charging_sessions += 1
        self.is_charging = False
        self.charge_start_time = None
        self.must_leave_charger = True  # Force truck to leave after charging
    
    def add_waiting_time(self, wait_duration: float):
        """Add waiting time at a charging station."""
        self.waiting_time += wait_duration
        self.total_time_elapsed += wait_duration
    
    def get_battery_percentage(self) -> float:
        """Get current battery level as percentage."""
        return 100.0 * self.current_battery / self.battery_capacity
    

    def get_state_dict(self) -> Dict:
        """
        Get truck state as a dictionary.
        
        Returns:
            Dictionary containing all relevant truck state information
        """
        next_target = self.get_next_delivery_target()
        return {
            "truck_id": self.truck_id,
            "truck_type": self.truck_type,
            "delivery_sequence": self.delivery_sequence,
            "current_node": self.current_node,
            "next_delivery_target": next_target if next_target is not None else -1,
            "current_battery": self.current_battery,
            "battery_capacity": self.battery_capacity,
            "battery_percentage": self.get_battery_percentage(),
            "base_speed": self.base_speed,
            "is_charging": self.is_charging,
            "must_leave_charger": self.must_leave_charger,
            "is_complete": self.is_complete,
            "failed": self.failed,
            "deliveries_remaining": len(self.get_remaining_deliveries()),
            "total_distance": self.total_distance_traveled,
            "total_time": self.total_time_elapsed,
            "total_charging_time": self.total_charging_time,
            "waiting_time": self.waiting_time,
            "num_charging_sessions": self.num_charging_sessions,
            "total_distance_to_travel": sum(
                self.delivery_sequence[i+1] - self.delivery_sequence[i] 
                for i in range(len(self.delivery_sequence) - 1)
            ) if len(self.delivery_sequence) > 1 else 0,
        }
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get truck state as a numpy array (for RL observation).
        
        Returns:
            Numpy array containing normalized state values
        """
        state = self.get_state_dict()
        return np.array([
            float(state["current_node"]),
            float(state["next_delivery_target"]),
            state["current_battery"],
            state["battery_percentage"] / 100.0,  # Normalize to [0, 1]
            float(state["is_charging"]),
            float(state["is_complete"]),
            float(state["deliveries_remaining"]),
            state["total_distance"],
            state["total_time"],
        ], dtype=np.float32)
    
    def __repr__(self) -> str:
        """String representation of truck."""
        status = "COMPLETE" if self.is_complete else ("FAILED" if self.failed else "ACTIVE")
        return (
            f"Truck(id={self.truck_id}, type={self.truck_type}, "
            f"battery={self.current_battery:.1f}/{self.battery_capacity:.1f} kWh, "
            f"at_node={self.current_node}, status={status})"
        )
