"""
Truck class representing individual delivery vehicles.
"""
import math

import numpy as np


class Truck:
    """Represents a delivery truck with battery, route, and state information."""
    
    def __init__(
        self,
        truck_id: int,
        truck_type: str,
        delivery_sequence: list[int],
        initial_battery: float,
        battery_capacity: float,
        base_speed: float,
        enable_flexible_delivery_order: bool = False,
        payload_capacity: float | None = None,
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
            payload_capacity: Maximum delivery demand carried by the truck. Legacy
                route-execution instances may omit this constraint.
        """
        if int(truck_id) < 0:
            raise ValueError("truck_id must be non-negative")
        if not delivery_sequence:
            raise ValueError("delivery_sequence cannot be empty")
        if any(int(node) < 0 for node in delivery_sequence):
            raise ValueError("delivery_sequence nodes must be non-negative")
        if not math.isfinite(battery_capacity) or battery_capacity <= 0.0:
            raise ValueError("battery_capacity must be finite and positive")
        if not math.isfinite(initial_battery) or initial_battery < 0.0:
            raise ValueError("initial_battery must be finite and non-negative")
        if initial_battery > battery_capacity:
            raise ValueError("initial_battery cannot exceed battery_capacity")
        if not math.isfinite(base_speed) or base_speed <= 0.0:
            raise ValueError("base_speed must be finite and positive")

        self.truck_id = int(truck_id)
        self.truck_type = truck_type
        self.delivery_sequence = delivery_sequence.copy()
        self.battery_capacity = battery_capacity
        self.base_speed = base_speed
        self.enable_flexible_delivery_order = enable_flexible_delivery_order
        if payload_capacity is not None and (
            not math.isfinite(payload_capacity) or payload_capacity <= 0
        ):
            raise ValueError(
                "payload_capacity must be finite and positive when specified"
            )
        self.payload_capacity = (
            float(payload_capacity) if payload_capacity is not None else None
        )
        self.remaining_payload = self.payload_capacity
        self.served_task_ids: list[int] = []
        
        self.current_battery = initial_battery
        self.current_node = delivery_sequence[0]
        self.current_sequence_index = 0  # Index in delivery_sequence
        
        # For flexible delivery order: track completed deliveries as set
        self.delivered_nodes = set()  # Set of delivered node IDs
        
        # Statistics
        self.total_distance_traveled = 0.0
        self.total_time_elapsed = 0.0
        self.total_routing_time = 0.0
        self.total_energy_consumed = 0.0
        self.total_energy_charged = 0.0
        self.total_charging_time = 0.0
        self.num_charging_sessions = 0
        self.waiting_time = 0.0
        self.time_window_waiting_time = 0.0
        self.total_unloading_time = 0.0  # Track cumulative unloading time at deliveries
        self.is_charging = False
        self.charge_start_time = None
        
        # Completion tracking
        self.is_complete = False
        self.failed = False  # True if ran out of battery
        self.failure_reason = None
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
        self.event_history: list[dict] = []  # Log of all truck events with timestamps
        self.current_state: str = "initial"  # Track current state for event logging
        self.unloading_start_time: float | None = None  # Track when unloading started
        self.waiting_start_time: float | None = None  # Track when waiting started
        self.routing_start_time: float | None = None  # Track when routing started
    
    def _record_event(
        self,
        event_type: str,
        timestamp: float,
        location: int | None = None,
        details: dict | None = None
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
    
    def get_remaining_deliveries(self) -> list[int]:
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

    def can_accept_demand(self, demand: float) -> bool:
        """Return whether the truck has enough remaining payload for a task."""
        if not math.isfinite(demand) or demand <= 0:
            return False
        if self.remaining_payload is None:
            return True
        return float(demand) <= self.remaining_payload + 1e-9

    def complete_customer_service(
        self,
        task_id: int,
        demand: float,
        timestamp: float,
        node_id: int,
    ) -> None:
        """Consume payload and record service of one fleet-owned task."""
        if int(task_id) < 0:
            raise ValueError("task_id must be non-negative")
        if int(node_id) < 0:
            raise ValueError("node_id must be non-negative")
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        if int(task_id) in self.served_task_ids:
            raise ValueError(f"truck {self.truck_id} already served task {task_id}")
        if not self.can_accept_demand(demand):
            raise ValueError(
                f"truck {self.truck_id} lacks payload for demand {demand}"
            )
        if self.remaining_payload is not None:
            self.remaining_payload = max(0.0, self.remaining_payload - float(demand))
        self.served_task_ids.append(int(task_id))
        self._record_event(
            event_type="CUSTOMER_SERVICE_COMPLETE",
            timestamp=float(timestamp),
            location=int(node_id),
            details={
                "task_id": int(task_id),
                "demand": float(demand),
                "remaining_payload": self.remaining_payload,
            },
        )
    
    def advance_to_next_delivery(self, delivered_node: int | None = None):
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
        destination = int(destination)
        timestamp = float(timestamp)
        if destination < 0:
            raise ValueError("routing destination must be non-negative")
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("routing timestamp must be finite and non-negative")
        if self.route_destination is not None:
            raise RuntimeError("truck is already routing")
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
        timestamp: float | None = None,
        mark_delivery_on_arrival: bool = True,
    ):
        """
        Update truck state after moving to a new node.
        
        Args:
            node: Destination node
            distance: Distance traveled (km)
            travel_time: Time taken (hours)
            discharge: Battery consumed (kWh)
            timestamp: Current simulation time (for event logging)
            mark_delivery_on_arrival: Preserve legacy route-execution semantics
                when true. Joint-routing episodes set this to false and commit
                service through the fleet task registry after unloading.
        """
        node = int(node)
        distance = float(distance)
        travel_time = float(travel_time)
        discharge = float(discharge)
        if node < 0:
            raise ValueError("destination node must be non-negative")
        for label, value in (
            ("distance", distance),
            ("travel_time", travel_time),
            ("discharge", discharge),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if discharge > self.current_battery + 1e-9:
            raise ValueError("discharge cannot exceed current battery")
        if timestamp is not None and (
            not math.isfinite(timestamp) or timestamp < 0.0
        ):
            raise ValueError("routing timestamp must be finite and non-negative")

        origin = self.current_node
        
        self.current_node = node
        self.current_battery -= discharge
        # Clamp only for floating-point precision at exact depletion.
        self.current_battery = min(self.battery_capacity, max(0.0, self.current_battery))
        self.total_distance_traveled += distance
        self.total_time_elapsed += travel_time
        self.total_routing_time += travel_time
        self.total_energy_consumed += discharge
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
        
        if mark_delivery_on_arrival:
            # Check if this was a delivery target. This is the historical
            # route-execution behavior; the joint model completes service only
            # after its unloading event.
            if self.enable_flexible_delivery_order:
                remaining_deliveries = self.get_next_delivery_target()
                if node in remaining_deliveries:
                    self.advance_to_next_delivery(delivered_node=node)
            elif node == self.get_next_delivery_target():
                self.advance_to_next_delivery()
        
        # Check if out of battery
        if self.current_battery <= 0:
            self.current_battery = 0
            self.mark_failed(
                reason="battery_depleted",
                timestamp=timestamp if timestamp is not None else 0.0,
            )
        elif self.enable_flexible_delivery_order and self.return_to_depot_pending:
            depot_node = self.delivery_sequence[0]
            if node == depot_node:
                self.is_complete = True
                self.return_to_depot_pending = False
                # Store battery level at completion for penalty calculation
                self.battery_at_completion = self.current_battery
    
    def start_charging(self, current_time: float):
        """Mark truck as starting to charge."""
        current_time = float(current_time)
        if not math.isfinite(current_time) or current_time < 0.0:
            raise ValueError("charging start time must be finite and non-negative")
        if self.failed or self.is_complete:
            raise RuntimeError("failed or complete truck cannot start charging")
        if self.is_charging:
            raise RuntimeError("truck is already charging")
        if self.current_battery >= self.battery_capacity - 1e-9:
            raise RuntimeError("truck battery is already full")
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
    
    def finish_charging(self, charge_amount: float, charge_duration: float, timestamp: float | None = None):
        """
        Update truck state after charging.
        
        Args:
            charge_amount: Amount charged (kWh)
            charge_duration: Time spent charging (hours)
            timestamp: Current simulation time (for event logging)
        """
        charge_amount = float(charge_amount)
        charge_duration = float(charge_duration)
        if not self.is_charging:
            raise RuntimeError("truck is not charging")
        for label, value in (
            ("charge_amount", charge_amount),
            ("charge_duration", charge_duration),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        available_capacity = self.battery_capacity - self.current_battery
        if charge_amount > available_capacity + 1e-8:
            raise ValueError("charge_amount exceeds available battery capacity")
        if timestamp is not None and (
            not math.isfinite(timestamp) or timestamp < 0.0
        ):
            raise ValueError("charging timestamp must be finite and non-negative")
        initial_soc = self.get_battery_percentage()
        # Clamp only for floating-point precision at an exact target.
        self.current_battery = min(self.battery_capacity, max(0.0, self.current_battery + charge_amount))
        final_soc = self.get_battery_percentage()
        
        self.total_charging_time += charge_duration
        self.total_energy_charged += charge_amount
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
    
    def add_waiting_time(self, wait_duration: float, timestamp: float | None = None):
        """
        Add waiting time at a charging station.
        
        Args:
            wait_duration: Duration of wait (hours)
            timestamp: Current simulation time (for event logging)
        """
        wait_duration = float(wait_duration)
        if not math.isfinite(wait_duration) or wait_duration < 0.0:
            raise ValueError("wait_duration must be finite and non-negative")
        if timestamp is not None and (
            not math.isfinite(timestamp) or timestamp < 0.0
        ):
            raise ValueError("waiting timestamp must be finite and non-negative")
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

    def add_time_window_waiting(self, duration: float, timestamp: float) -> None:
        """Record waiting caused by arrival before a customer's opening time."""
        duration = float(duration)
        timestamp = float(timestamp)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("time-window waiting duration must be non-negative")
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("time-window waiting timestamp must be non-negative")
        if duration == 0.0:
            return
        self.time_window_waiting_time += duration
        self.total_time_elapsed += duration
        self._record_event(
            event_type="TIME_WINDOW_WAIT",
            timestamp=timestamp,
            details={"wait_duration_hours": duration},
        )
    
    def start_unloading(self, timestamp: float, delivery_node: int):
        """
        Mark truck as starting to unload at a delivery.
        
        Args:
            timestamp: Current simulation time
            delivery_node: Node ID where unloading
        """
        timestamp = float(timestamp)
        delivery_node = int(delivery_node)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("unloading timestamp must be finite and non-negative")
        if delivery_node < 0:
            raise ValueError("delivery_node must be non-negative")
        if self.unloading_start_time is not None:
            raise RuntimeError("truck is already unloading")
        self.unloading_start_time = timestamp
        
        self._record_event(
            event_type="UNLOADING_START",
            timestamp=timestamp,
            location=delivery_node,
            details={"delivery_node": delivery_node}
        )
        self.current_state = "unloading"
    
    def finish_unloading(self, unloading_duration: float, timestamp: float | None = None):
        """
        Update truck state after unloading at a delivery.
        
        Args:
            unloading_duration: Time spent unloading (hours)
            timestamp: Current simulation time (for event logging)
        """
        unloading_duration = float(unloading_duration)
        if not math.isfinite(unloading_duration) or unloading_duration < 0.0:
            raise ValueError(
                "unloading_duration must be finite and non-negative"
            )
        if self.unloading_start_time is None:
            raise RuntimeError("truck is not unloading")
        if timestamp is not None and (
            not math.isfinite(timestamp) or timestamp < 0.0
        ):
            raise ValueError("unloading timestamp must be finite and non-negative")
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

    def mark_failed(self, reason: str, timestamp: float) -> None:
        """Mark the truck failed once and retain a stable cause code."""
        if self.failed:
            return
        self.failed = True
        self.failure_reason = str(reason)
        self._record_event(
            event_type="FAILED",
            timestamp=float(timestamp),
            details={"reason": self.failure_reason},
        )
        self.current_state = "failed"
    
    def get_battery_percentage(self) -> float:
        """Get current battery level as percentage."""
        # Fix battery if it exceeds capacity (handle floating point precision errors)
        if self.current_battery > self.battery_capacity:
            import os
            from datetime import UTC, datetime
            warning_msg = (
                f"[{datetime.now(UTC).isoformat()}] "
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
    

    def get_state_dict(self) -> dict:
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
            "payload_capacity": self.payload_capacity,
            "remaining_payload": self.remaining_payload,
            "served_task_ids": self.served_task_ids.copy(),
            "base_speed": self.base_speed,
            "is_charging": self.is_charging,
            "must_leave_charger": self.must_leave_charger,
            "is_complete": self.is_complete,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "deliveries_remaining": len(self.get_remaining_deliveries()),
            "total_distance": self.total_distance_traveled,
            "total_time": self.total_time_elapsed,
            "total_routing_time": self.total_routing_time,
            "total_energy_consumed": self.total_energy_consumed,
            "total_energy_charged": self.total_energy_charged,
            "total_charging_time": self.total_charging_time,
            "total_unloading_time": self.total_unloading_time,
            "waiting_time": self.waiting_time,
            "time_window_waiting_time": self.time_window_waiting_time,
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
    
    def get_event_history(self) -> list[dict]:
        """
        Get complete event history for this truck.
        
        Returns:
            List of event dictionaries with timestamps and details
        """
        return self.event_history.copy()
    
    def get_activity_timeline(self) -> list[dict]:
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
    
    def get_activity_breakdown(self) -> dict[str, float]:
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
            for ready_event in ready_events:
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
    
    def export_event_log(self, format: str = "dict") -> dict:
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
            "payload_capacity": self.payload_capacity,
            "delivery_sequence": self.delivery_sequence,
            "episode_summary": {
                "is_complete": self.is_complete,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
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
