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
        enable_flexible_delivery_order: bool = False,
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
            enable_flexible_delivery_order: If True, allow flexible delivery order selection
        """
        self.truck_id = truck_id
        self.truck_type = truck_type
        self.delivery_sequence = delivery_sequence.copy()
        self.battery_capacity = battery_capacity
        self.base_speed = base_speed
        self.enable_flexible_delivery_order = enable_flexible_delivery_order
        
        # Current state - clamp initial battery to capacity
        self.current_battery = min(battery_capacity, initial_battery)
        self.current_node = delivery_sequence[0]
        self.current_sequence_index = 0  # Index in delivery_sequence
        
        # For flexible delivery order: track completed deliveries as set
        self.delivered_nodes = set()  # Set of delivered node IDs
        
        # Statistics
        self.total_distance_traveled = 0.0
        self.total_time_elapsed = 0.0
        self.total_charging_time = 0.0
        self.num_charging_sessions = 0
        self.waiting_time = 0.0
        self.total_unloading_time = 0.0  # Track cumulative unloading time at deliveries
        self.is_charging = False
        self.charge_start_time = None
        
        # Completion tracking
        self.is_complete = False
        self.failed = False  # True if ran out of battery
        self.battery_at_completion = None  # Store battery level when completing last delivery
        # VRP: require return to depot after last delivery in flexible mode
        self.return_to_depot_pending = False #True if enable_flexible_delivery_order else False
        
        # Route tracking (for GNN state representation)
        self.route_destination = None  # Next destination when on route
        self.route_arrival_time = None  # Event time when truck will arrive at destination
        
        # Charging policy: truck must leave after charging
        self.must_leave_charger = False  # True if truck just finished charging and must leave

        # Detour loop guard tracking (sequential mode)
        self.detour_last_action_was_charge = False
        self.detour_charger_hops_since_delivery = 0
        
        # Event monitoring system
        self.event_history: List[Dict] = []  # Log of all truck events with timestamps
        self.current_state: str = "initial"  # Track current state for event logging
        self.unloading_start_time: Optional[float] = None  # Track when unloading started
        self.waiting_start_time: Optional[float] = None  # Track when waiting started
        self.routing_start_time: Optional[float] = None  # Track when routing started
    
    def _record_event(
        self,
        event_type: str,
        timestamp: float,
        location: Optional[int] = None,
        details: Optional[Dict] = None
    ):
        """
        Record an event in the truck's event history.
        
        Args:
            event_type: Type of event (e.g., 'ROUTING_START', 'CHARGING_END')
            timestamp: Simulation time when event occurred
            location: Node ID where event occurred (defaults to current_node)
            details: Additional event-specific data
        """
        if location is None:
            location = self.current_node
        
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "location": location,
            "state_before": self.current_state,
            "battery_soc": self.get_battery_percentage(),
            "battery_kwh": self.current_battery,
            "details": details or {}
        }
        
        self.event_history.append(event)
    
    def get_next_delivery_target(self):
        """
        Get the next delivery destination(s).
        
        Returns:
            - If flexible delivery order is disabled: Single int (next node ID) or None
            - If flexible delivery order is enabled: List of remaining delivery node IDs (may be empty)
        """
        if self.enable_flexible_delivery_order:
            # Return all undelivered nodes (excluding depot at index 0)
            remaining = [node for node in self.delivery_sequence[1:] if node not in self.delivered_nodes]
            if self.return_to_depot_pending:
                depot_node = self.delivery_sequence[0]
                if depot_node != self.current_node and depot_node not in remaining:
                    remaining.append(depot_node)
            return remaining
        else:
            # Sequential mode: return next in sequence
            if self.current_sequence_index + 1 < len(self.delivery_sequence):
                return self.delivery_sequence[self.current_sequence_index + 1]
            return None
    
    def get_remaining_deliveries(self) -> List[int]:
        """Get list of remaining delivery nodes."""
        if self.enable_flexible_delivery_order:
            # Return all undelivered nodes (excluding depot at index 0)
            remaining = [node for node in self.delivery_sequence[1:] if node not in self.delivered_nodes]
            if self.return_to_depot_pending:
                depot_node = self.delivery_sequence[0]
                if depot_node != self.current_node and depot_node not in remaining:
                    remaining.append(depot_node)
            return remaining
        else:
            # Sequential mode: return remaining in sequence
            return self.delivery_sequence[self.current_sequence_index + 1:]
    
    def advance_to_next_delivery(self, delivered_node: Optional[int] = None):
        """
        Mark delivery as complete and advance.
        
        Args:
            delivered_node: Specific node to mark as delivered (for flexible order mode).
                           If None, advances to next in sequence (sequential mode).
        """
        if self.enable_flexible_delivery_order:
            # Flexible mode: mark specific node as delivered
            if delivered_node is not None:
                self.delivered_nodes.add(delivered_node)
                self.current_node = delivered_node

                # Reset detour loop counters on delivery progress
                self.detour_last_action_was_charge = False
                self.detour_charger_hops_since_delivery = 0
            
            # Check if all deliveries complete (excluding depot at index 0)
            all_delivery_nodes = set(self.delivery_sequence[1:])
            if self.delivered_nodes == all_delivery_nodes:
                # Mark return-to-depot leg as pending; completion happens on depot arrival
                self.return_to_depot_pending = True
        else:
            # Sequential mode: advance to next in sequence
            if self.current_sequence_index + 1 < len(self.delivery_sequence):
                self.current_sequence_index += 1
                self.current_node = self.delivery_sequence[self.current_sequence_index]

                # Reset detour loop counters on delivery progress
                self.detour_last_action_was_charge = False
                self.detour_charger_hops_since_delivery = 0
                
                # Check if all deliveries complete
                if self.current_sequence_index == len(self.delivery_sequence) - 1:
                    self.is_complete = True
                    # Store battery level at completion for penalty calculation
                    self.battery_at_completion = self.current_battery
    
    def start_routing(self, destination: int, timestamp: float):
        """
        Mark truck as starting to route to a destination.
        
        Args:
            destination: Target node ID
            timestamp: Current simulation time
        """
        self.route_destination = destination
        self.routing_start_time = timestamp
        
        self._record_event(
            event_type="ROUTING_START",
            timestamp=timestamp,
            details={
                "destination": destination,
                "origin": self.current_node
            }
        )
        self.current_state = "routing"
    
    def move_to_node(
        self,
        node: int,
        distance: float,
        travel_time: float,
        discharge: float,
        timestamp: Optional[float] = None
    ):
        """
        Update truck state after moving to a new node.
        
        Args:
            node: Destination node
            distance: Distance traveled (km)
            travel_time: Time taken (hours)
            discharge: Battery consumed (kWh)
            timestamp: Current simulation time (for event logging)
        """
        origin = self.current_node
        
        self.current_node = node
        self.current_battery -= discharge
        # Clamp in case of negative discharge (regen/uncertainty) or precision drift
        self.current_battery = min(self.battery_capacity, max(0.0, self.current_battery))
        self.total_distance_traveled += distance
        self.total_time_elapsed += travel_time
        self.must_leave_charger = False  # Reset flag when moving away
        
        # Record routing end event
        if timestamp is not None:
            self._record_event(
                event_type="ROUTING_END",
                timestamp=timestamp,
                location=node,
                details={
                    "origin": origin,
                    "distance_km": distance,
                    "travel_time_hours": travel_time,
                    "energy_consumed_kwh": discharge,
                    "routing_start_time": self.routing_start_time
                }
            )
        
        # Check if this was a delivery target
        if self.enable_flexible_delivery_order:
            # Flexible mode: check if node is any remaining delivery
            remaining_deliveries = self.get_next_delivery_target()
            if node in remaining_deliveries:
                self.advance_to_next_delivery(delivered_node=node)
        else:
            # Sequential mode: check if node is next delivery
            if node == self.get_next_delivery_target():
                self.advance_to_next_delivery()
        
        # Check if out of battery
        if self.current_battery <= 0:
            self.current_battery = 0
            self.failed = True
            if timestamp is not None:
                self._record_event(
                    event_type="FAILED",
                    timestamp=timestamp,
                    location=node,
                    details={"reason": "battery_depleted"}
                )
                self.current_state = "failed"
        elif self.enable_flexible_delivery_order and self.return_to_depot_pending:
            depot_node = self.delivery_sequence[0]
            if node == depot_node:
                self.is_complete = True
                self.return_to_depot_pending = False
                # Store battery level at completion for penalty calculation
                self.battery_at_completion = self.current_battery
    
    def start_charging(self, current_time: float):
        """Mark truck as starting to charge."""
        self.is_charging = True
        self.charge_start_time = current_time
        self.must_leave_charger = False  # Reset flag when starting new charge session

        # Track that the last decision was a charge
        self.detour_last_action_was_charge = True
        
        self._record_event(
            event_type="CHARGING_START",
            timestamp=current_time,
            details={
                "initial_soc": self.get_battery_percentage(),
                "initial_battery_kwh": self.current_battery
            }
        )
        self.current_state = "charging"
    
    def finish_charging(self, charge_amount: float, charge_duration: float, timestamp: Optional[float] = None):
        """
        Update truck state after charging.
        
        Args:
            charge_amount: Amount charged (kWh)
            charge_duration: Time spent charging (hours)
            timestamp: Current simulation time (for event logging)
        """
        initial_soc = self.get_battery_percentage()
        # Clamp to prevent exceeding capacity (handle floating point precision errors)
        self.current_battery = min(self.battery_capacity, max(0.0, self.current_battery + charge_amount))
        final_soc = self.get_battery_percentage()
        
        self.total_charging_time += charge_duration
        self.total_time_elapsed += charge_duration
        self.num_charging_sessions += 1
        self.is_charging = False
        self.must_leave_charger = True  # Force truck to leave after charging
        
        if timestamp is not None:
            self._record_event(
                event_type="CHARGING_END",
                timestamp=timestamp,
                details={
                    "charge_amount_kwh": charge_amount,
                    "charge_duration_hours": charge_duration,
                    "initial_soc": initial_soc,
                    "final_soc": final_soc,
                    "charge_start_time": self.charge_start_time
                }
            )
        
        self.charge_start_time = None
    
    def start_waiting(self, timestamp: float, reason: str = "charger_queue"):
        """
        Mark truck as starting to wait.
        
        Args:
            timestamp: Current simulation time
            reason: Reason for waiting (e.g., 'charger_queue', 'charger_gating')
        """
        self.waiting_start_time = timestamp
        
        self._record_event(
            event_type="WAITING_START",
            timestamp=timestamp,
            details={"reason": reason}
        )
        self.current_state = "waiting_to_charge"
    
    def finish_waiting(self, timestamp: float):
        """
        Mark truck as finishing waiting period.
        
        Args:
            timestamp: Current simulation time
        """
        if self.waiting_start_time is not None:
            wait_duration = timestamp - self.waiting_start_time
            
            self._record_event(
                event_type="WAITING_END",
                timestamp=timestamp,
                details={
                    "wait_duration_hours": wait_duration,
                    "wait_start_time": self.waiting_start_time
                }
            )
            self.waiting_start_time = None
    
    def add_waiting_time(self, wait_duration: float, timestamp: Optional[float] = None):
        """
        Add waiting time at a charging station.
        
        Args:
            wait_duration: Duration of wait (hours)
            timestamp: Current simulation time (for event logging)
        """
        self.waiting_time += wait_duration
        self.total_time_elapsed += wait_duration
        
        # If timestamp provided and we have a start time, record the waiting event
        if timestamp is not None and self.waiting_start_time is not None:
            self._record_event(
                event_type="WAITING_END",
                timestamp=timestamp,
                details={
                    "wait_duration_hours": wait_duration,
                    "wait_start_time": self.waiting_start_time
                }
            )
            self.waiting_start_time = None
    
    def start_unloading(self, timestamp: float, delivery_node: int):
        """
        Mark truck as starting to unload at a delivery.
        
        Args:
            timestamp: Current simulation time
            delivery_node: Node ID where unloading
        """
        self.unloading_start_time = timestamp
        
        self._record_event(
            event_type="UNLOADING_START",
            timestamp=timestamp,
            location=delivery_node,
            details={"delivery_node": delivery_node}
        )
        self.current_state = "unloading"
    
    def finish_unloading(self, unloading_duration: float, timestamp: Optional[float] = None):
        """
        Update truck state after unloading at a delivery.
        
        Args:
            unloading_duration: Time spent unloading (hours)
            timestamp: Current simulation time (for event logging)
        """
        self.total_unloading_time += unloading_duration
        self.total_time_elapsed += unloading_duration
        
        if timestamp is not None:
            self._record_event(
                event_type="UNLOADING_END",
                timestamp=timestamp,
                details={
                    "unloading_duration_hours": unloading_duration,
                    "unloading_start_time": self.unloading_start_time
                }
            )
        
        self.unloading_start_time = None
    
    def mark_ready(self, timestamp: float, reason: str = "unknown"):
        """
        Mark truck as ready to take an action.
        
        Args:
            timestamp: Current simulation time
            reason: Reason for becoming ready (e.g., 'initial', 'charge_complete', 'unloading_complete')
        """
        self._record_event(
            event_type="READY_STATE",
            timestamp=timestamp,
            details={"reason": reason}
        )
        self.current_state = "ready"
    
    def mark_complete(self, timestamp: float):
        """
        Mark truck as having completed all deliveries.
        
        Args:
            timestamp: Current simulation time
        """
        self.is_complete = True
        
        self._record_event(
            event_type="COMPLETE",
            timestamp=timestamp,
            details={
                "total_distance_km": self.total_distance_traveled,
                "total_time_hours": self.total_time_elapsed,
                "total_charging_time_hours": self.total_charging_time,
                "total_unloading_time_hours": self.total_unloading_time,
                "total_waiting_time_hours": self.waiting_time,
                "num_charging_sessions": self.num_charging_sessions,
                "final_battery_soc": self.get_battery_percentage()
            }
        )
        self.current_state = "complete"
    
    def get_battery_percentage(self) -> float:
        """Get current battery level as percentage."""
        # Fix battery if it exceeds capacity (handle floating point precision errors)
        if self.current_battery > self.battery_capacity:
            import os
            from datetime import datetime
            warning_msg = (
                f"[{datetime.now().isoformat()}] "
                f"Truck {self.truck_id}: Battery exceeds capacity! "
                f"current_battery={self.current_battery:.4f} kWh, "
                f"battery_capacity={self.battery_capacity:.4f} kWh, "
                f"percentage={100.0 * self.current_battery / self.battery_capacity:.4f}%. "
                f"Clamping to capacity.\n"
            )
            # Write to battery_warnings.log in root folder
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "battery_warnings.log")
            with open(log_path, "a") as f:
                f.write(warning_msg)
            # Fix the battery level
            self.current_battery = self.battery_capacity
        
        percentage = 100.0 * self.current_battery / self.battery_capacity
        # Clamp to [0, 100] to handle any remaining floating point precision issues
        return min(100.0, max(0.0, percentage))
    

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
            "total_unloading_time": self.total_unloading_time,
            "waiting_time": self.waiting_time,
            "num_charging_sessions": self.num_charging_sessions,
            "detour_last_action_was_charge": self.detour_last_action_was_charge,
            "detour_charger_hops_since_delivery": self.detour_charger_hops_since_delivery,
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
    
    def get_event_history(self) -> List[Dict]:
        """
        Get complete event history for this truck.
        
        Returns:
            List of event dictionaries with timestamps and details
        """
        return self.event_history.copy()
    
    def get_activity_timeline(self) -> List[Dict]:
        """
        Get timeline of activities with durations.
        
        Returns:
            List of activities with start/end times and durations
        """
        timeline = []
        
        # Pair up start/end events
        i = 0
        while i < len(self.event_history):
            event = self.event_history[i]
            
            if event["event_type"] in ["ROUTING_START", "CHARGING_START", "UNLOADING_START", "WAITING_START"]:
                activity_type = event["event_type"].replace("_START", "")
                end_type = event["event_type"].replace("START", "END")
                
                # Find matching end event
                end_event = None
                for j in range(i + 1, len(self.event_history)):
                    if self.event_history[j]["event_type"] == end_type:
                        end_event = self.event_history[j]
                        break
                
                if end_event:
                    timeline.append({
                        "activity": activity_type.lower(),
                        "start_time": event["timestamp"],
                        "end_time": end_event["timestamp"],
                        "duration": end_event["timestamp"] - event["timestamp"],
                        "location": event["location"],
                        "details": {**event["details"], **end_event["details"]}
                    })
            
            i += 1
        
        return timeline
    
    def get_activity_breakdown(self) -> Dict[str, float]:
        """
        Get breakdown of time spent in each activity type.
        
        Returns:
            Dictionary mapping activity type to total time (hours)
        """
        breakdown = {
            "routing": 0.0,
            "charging": 0.0,
            "unloading": 0.0,
            "waiting": 0.0,
            "ready": 0.0,
            "total": 0.0
        }
        
        timeline = self.get_activity_timeline()
        for activity in timeline:
            activity_type = activity["activity"]
            if activity_type in breakdown:
                breakdown[activity_type] += activity["duration"]
        
        # Calculate ready time (time in ready state between activities)
        ready_events = [e for e in self.event_history if e["event_type"] == "READY_STATE"]
        if ready_events:
            for i, ready_event in enumerate(ready_events):
                # Find next activity start
                ready_time = ready_event["timestamp"]
                next_activity_time = None
                
                for event in self.event_history:
                    if (event["timestamp"] > ready_time and 
                        event["event_type"] in ["ROUTING_START", "CHARGING_START"]):
                        next_activity_time = event["timestamp"]
                        break
                
                if next_activity_time:
                    breakdown["ready"] += next_activity_time - ready_time
        
        breakdown["total"] = self.total_time_elapsed
        return breakdown
    
    def export_event_log(self, format: str = "dict") -> Dict:
        """
        Export event log in structured format.
        
        Args:
            format: Output format ('dict' or 'json')
        
        Returns:
            Structured event log with metadata
        """
        log = {
            "truck_id": self.truck_id,
            "truck_type": self.truck_type,
            "battery_capacity": self.battery_capacity,
            "delivery_sequence": self.delivery_sequence,
            "episode_summary": {
                "is_complete": self.is_complete,
                "failed": self.failed,
                "total_time_hours": self.total_time_elapsed,
                "total_distance_km": self.total_distance_traveled,
                "total_charging_time_hours": self.total_charging_time,
                "total_unloading_time_hours": self.total_unloading_time,
                "total_waiting_time_hours": self.waiting_time,
                "num_charging_sessions": self.num_charging_sessions,
                "final_battery_soc": self.get_battery_percentage(),
            },
            "events": self.event_history,
            "activity_timeline": self.get_activity_timeline(),
            "activity_breakdown": self.get_activity_breakdown()
        }
        
        if format == "json":
            import json
            return json.dumps(log, indent=2)
        
        return log
    
    def __repr__(self) -> str:
        """String representation of truck."""
        status = "COMPLETE" if self.is_complete else ("FAILED" if self.failed else "ACTIVE")
        return (
            f"Truck(id={self.truck_id}, type={self.truck_type}, "
            f"battery={self.current_battery:.1f}/{self.battery_capacity:.1f} kWh, "
            f"at_node={self.current_node}, status={status})"
        )
