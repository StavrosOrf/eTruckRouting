"""
Charging station management for the event-driven truck environment.
"""

import heapq
from typing import Dict, List, Optional, Tuple
import json
import os


class ChargingStation:
    """
    Manages all charging station related logic including queues, occupancy,
    waiting times, and statistics.
    """

    def __init__(
        self,
        charging_nodes: List[int],
        transport_graph,
        waiting_time_lookup_path: str,
        verbose: bool = False,
    ):
        """
        Initialize the charging station manager.

        Args:
            charging_nodes: List of charging node IDs
            transport_graph: TransportationGraph instance
            waiting_time_lookup_path: Path to waiting time lookup JSON file
            verbose: Print detailed information
        """
        self.charging_nodes = charging_nodes
        self.transport_graph = transport_graph
        self.verbose = verbose

        # Load waiting time lookup table for queue simulation
        with open(waiting_time_lookup_path, "r") as f:
            self.waiting_time_lookup = json.load(f)

        # Charger properties (capacity, type)
        self.charger_capacity = {
            node: transport_graph.get_charger_capacity(node)
            for node in charging_nodes
        }
        self.charger_type = {
            node: transport_graph.get_charger_type(node) for node in charging_nodes
        }

        # Charging station occupancy tracking
        self.charger_occupancy = {
            node: [] for node in charging_nodes
        }  # List of truck IDs currently charging

        # Charging queue (trucks that have started charging)
        self.charger_queue = {
            node: [] for node in charging_nodes
        }  # List of (truck_id, scheduled_start_time, charge_duration)

        # Track when each truck will finish charging (for queue management)
        self.truck_charge_end_time = {}  # truck_id -> expected charge completion time

        # Simplified FCFS waitlist per charging station (feeds all ports)
        # Each entry: {"truck_id": int, "planned_plug_time": Optional[float]}
        self.charger_waitlist = {node: [] for node in charging_nodes}

        # Charging station utilization tracking
        self.charger_stats = {
            node: {
                "total_charge_sessions": 0,
                "total_charge_time": 0.0,
                "total_trucks_served": set(),
                "occupancy_time": 0.0,  # Total time with at least one truck
                "last_update_time": 0.0,
                "queue_length": 0,  # Current queue length
            }
            for node in charging_nodes
        }
        
        # Time-series tracking for queue visualization
        self.queue_history = {
            node: {
                "times": [],
                "occupancy": [],
                "waitlist": [],
                "truck_events": [],  # (time, truck_id, event_type: 'arrive'/'start'/'finish')
            }
            for node in charging_nodes
        }

    def reset(self):
        """Reset all charging station state for a new episode."""
        self.charger_occupancy = {node: [] for node in self.charging_nodes}
        self.charger_queue = {node: [] for node in self.charging_nodes}
        self.truck_charge_end_time = {}
        self.charger_waitlist = {node: [] for node in self.charging_nodes}
        self.charger_stats = {
            node: {
                "total_charge_sessions": 0,
                "total_charge_time": 0.0,
                "total_trucks_served": set(),
                "occupancy_time": 0.0,
                "last_update_time": 0.0,
                "queue_length": 0,
            }
            for node in self.charging_nodes
        }
        self.queue_history = {
            node: {
                "times": [],
                "occupancy": [],
                "waitlist": [],
                "truck_events": [],
            }
            for node in self.charging_nodes
        }

    def get_waiting_time(self, charger_node: int, current_utilization: float) -> float:
        """
        Get expected waiting time at a charger based on current utilization.

        Args:
            charger_node: The charging station node
            current_utilization: Current utilization rate (0-1)

        Returns:
            Expected waiting time in hours
        """
        charger_type = self.charger_type[charger_node]
        capacity = int(self.charger_capacity[charger_node])

        # Get lookup table for this charger type and capacity
        if charger_type not in self.waiting_time_lookup:
            return 0.0

        capacity_str = str(capacity)
        if capacity_str not in self.waiting_time_lookup[charger_type]:
            # Use closest available capacity
            available_capacities = sorted(
                [int(c) for c in self.waiting_time_lookup[charger_type].keys()]
            )
            closest_capacity = min(
                available_capacities, key=lambda x: abs(x - capacity)
            )
            capacity_str = str(closest_capacity)

        # Round utilization to nearest 0.05
        util_rounded = round(current_utilization / 0.05) * 0.05
        util_rounded = max(0.05, min(0.95, util_rounded))  # Clamp to available range
        util_str = f"{util_rounded:.2f}"

        # Get waiting time in minutes and convert to hours
        waiting_minutes = self.waiting_time_lookup[charger_type][capacity_str].get(
            util_str, 0.0
        )
        waiting_hours = waiting_minutes / 60.0

        return waiting_hours

    def _record_queue_state(self, charger_node: int, global_clock: float, truck_id: int = None, event_type: str = None):
        """
        Record current queue state for visualization.
        
        Args:
            charger_node: Charging station node
            global_clock: Current simulation time
            truck_id: Optional truck ID for event tracking
            event_type: Optional event type ('arrive', 'start', 'finish')
        """
        history = self.queue_history[charger_node]
        history["times"].append(global_clock)
        history["occupancy"].append(len(self.charger_occupancy[charger_node]))
        history["waitlist"].append(len(self.charger_waitlist[charger_node]))
        
        if truck_id is not None and event_type is not None:
            history["truck_events"].append((global_clock, truck_id, event_type))

    def check_charger_gating(
        self, truck_id: int, charger_node: int, global_clock: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if a truck can proceed with an action at a charging station.
        Enforces FCFS waitlist with capacity ports.

        Args:
            truck_id: ID of the truck
            charger_node: Charging station node
            global_clock: Current simulation time

        Returns:
            Tuple of (can_proceed, next_check_time)
            - can_proceed: True if truck can act now, False otherwise
            - next_check_time: When to check again (None if can proceed)
        """
        capacity = int(self.charger_capacity[charger_node])
        occupancy = len(self.charger_occupancy[charger_node])
        free_slots = max(0, capacity - occupancy)
        waitlist = self.charger_waitlist[charger_node]

        def ensure_in_waitlist(tid: int, planned: Optional[float]):
            """Add truck to waitlist if not already present."""
            if not any(e["truck_id"] == tid for e in waitlist):
                waitlist.append({"truck_id": tid, "planned_plug_time": planned})
                # Record arrival event
                self._record_queue_state(charger_node, global_clock, tid, 'arrive')
            else:
                if planned is not None:
                    for e in waitlist:
                        if e["truck_id"] == tid:
                            e["planned_plug_time"] = planned
                            break

        idx = next(
            (i for i, e in enumerate(waitlist) if e["truck_id"] == truck_id), None
        )

        if idx is None:
            # New arrival at charger
            if free_slots > 0 and len(waitlist) == 0:
                # Scenario 1: No other truck is waiting at any port
                # Use lookup table to predict wait time
                util = occupancy / float(capacity) if capacity > 0 else 0.0
                wait_h = self.get_waiting_time(charger_node, util)
                if wait_h > 0:
                    plug_time = global_clock + wait_h
                    ensure_in_waitlist(truck_id, planned=plug_time)
                    return False, plug_time  # Schedule event at predicted time
                else:
                    ensure_in_waitlist(truck_id, planned=global_clock)
                    return True, None  # Can proceed immediately
            elif capacity == 1 and occupancy > 0:
                # Scenario 2: Single-port charger with truck already charging
                # Truck will ONLY be woken when the charging truck finishes
                ensure_in_waitlist(truck_id, planned=None)
                return False, None  # No self-scheduled event - wait for wake
            elif capacity > 1 and (occupancy > 0 or len(waitlist) > 0):
                # Scenario 3: Multi-port charger with trucks already charging/waiting
                # Sample wait time from lookup table
                util = occupancy / float(capacity) if capacity > 0 else 0.0
                wait_h = self.get_waiting_time(charger_node, util)
                plug_time = global_clock + max(wait_h, 0.1)  # At least 6 minutes
                ensure_in_waitlist(truck_id, planned=plug_time)
                return False, plug_time  # Schedule event at predicted time
            else:
                # Fallback: wait for wake
                ensure_in_waitlist(truck_id, planned=None)
                return False, None
        else:
            # Already waiting in line
            planned = waitlist[idx]["planned_plug_time"]
            if (
                (free_slots > 0)
                and (idx < free_slots)
                and (planned is None or planned <= global_clock)
            ):
                # Eligible to act now - at front of queue with available slot
                return True, None
            else:
                # Can't proceed yet
                # Only schedule recheck if this truck has a planned time (first in queue)
                if planned is not None and planned > global_clock:
                    # Use the planned plug time
                    return False, planned
                else:
                    # No planned time - truck will be woken by wake_waiting_trucks
                    return False, None

    def start_charging(
        self, truck_id: int, charger_node: int, charge_hours: float, global_clock: float
    ):
        """
        Begin charging for a truck at a charging station.

        Args:
            truck_id: ID of the truck
            charger_node: Charging station node
            charge_hours: Duration of charging in hours
            global_clock: Current simulation time
        """
        # Calculate when this truck will finish charging
        charge_end_time = global_clock + charge_hours
        self.truck_charge_end_time[truck_id] = charge_end_time

        # Remove from waitlist (truck is now charging)
        waitlist = self.charger_waitlist[charger_node]
        idx = next(
            (i for i, e in enumerate(waitlist) if e["truck_id"] == truck_id), None
        )
        if idx is not None:
            waitlist.pop(idx)

        # Update occupancy
        if truck_id not in self.charger_occupancy[charger_node]:
            self.charger_occupancy[charger_node].append(truck_id)
        
        # Record start charging event
        self._record_queue_state(charger_node, global_clock, truck_id, 'start')

        # Update queue with actual start/duration
        already_in_queue = any(
            tid == truck_id for tid, _, _ in self.charger_queue[charger_node]
        )
        if not already_in_queue:
            self.charger_queue[charger_node].append(
                (truck_id, global_clock, charge_hours)
            )
        else:
            # Update existing placeholder entry if present
            updated = []
            for tid, start_time, duration in self.charger_queue[charger_node]:
                if tid == truck_id:
                    updated.append((tid, global_clock, charge_hours))
                else:
                    updated.append((tid, start_time, duration))
            self.charger_queue[charger_node] = updated

        # Update utilization stats
        stats = self.charger_stats[charger_node]
        if len(self.charger_occupancy[charger_node]) == 1:  # First truck at charger
            if stats["last_update_time"] > 0:
                stats["occupancy_time"] += global_clock - stats["last_update_time"]
        stats["last_update_time"] = global_clock
        stats["total_charge_sessions"] += 1
        stats["total_trucks_served"].add(truck_id)
        stats["total_charge_time"] += charge_hours
        stats["queue_length"] = len(self.charger_queue[charger_node])

    def finish_charging(self, truck_id: int, charger_node: int, global_clock: float):
        """
        Complete charging for a truck at a charging station.

        Args:
            truck_id: ID of the truck
            charger_node: Charging station node
            global_clock: Current simulation time
        """
        # Remove from charger occupancy
        if truck_id in self.charger_occupancy[charger_node]:
            self.charger_occupancy[charger_node].remove(truck_id)

        # Remove from queue
        self.charger_queue[charger_node] = [
            (tid, start, dur)
            for tid, start, dur in self.charger_queue[charger_node]
            if tid != truck_id
        ]

        # Update queue length stat
        self.charger_stats[charger_node]["queue_length"] = len(
            self.charger_queue[charger_node]
        )

        # Update occupancy statistics
        stats = self.charger_stats[charger_node]
        if len(self.charger_occupancy[charger_node]) == 0:
            # Charger became empty
            stats["occupancy_time"] += global_clock - stats["last_update_time"]
            stats["last_update_time"] = global_clock

        # Clean up charge end time tracking
        if truck_id in self.truck_charge_end_time:
            del self.truck_charge_end_time[truck_id]
        
        # Record finish charging event
        self._record_queue_state(charger_node, global_clock, truck_id, 'finish')

    def wake_waiting_trucks(
        self, charger_node: int, global_clock: float, event_queue: List, EventType, Event
    ):
        """
        Wake trucks waiting at a charging station when a port becomes available.

        Args:
            charger_node: Charging station node
            global_clock: Current simulation time
            event_queue: Event queue for scheduling wake events
            EventType: EventType enum
            Event: Event class
        """
        capacity = int(self.charger_capacity[charger_node])
        occupancy = len(self.charger_occupancy[charger_node])
        free_slots = max(0, capacity - occupancy)
        waitlist = self.charger_waitlist[charger_node]

        if free_slots > 0 and waitlist:
            # Wake up to free_slots number of trucks immediately
            k = min(free_slots, len(waitlist))
            for i in range(k):
                tid = waitlist[i]["truck_id"]
                # Wake truck immediately, regardless of planned time
                # This handles scenario 3: truck finishes earlier than predicted
                # Note: If truck already has a scheduled event, it will be processed but ignored
                # if the truck is no longer in waiting_to_charge state
                heapq.heappush(
                    event_queue,
                    Event(
                        time=global_clock,
                        event_type=EventType.TRUCK_READY,
                        truck_id=tid,
                        data={"reason": "port_freed_early"},
                    ),
                )

    def remove_from_waitlist(self, truck_id: int, charger_node: int):
        """
        Remove a truck from the waitlist (e.g., when leaving to navigate elsewhere).

        Args:
            truck_id: ID of the truck
            charger_node: Charging station node
        """
        wl = self.charger_waitlist[charger_node]
        if any(e["truck_id"] == truck_id for e in wl):
            self.charger_waitlist[charger_node] = [
                e for e in wl if e["truck_id"] != truck_id
            ]

    def get_charger_info(self, charger_node: int, global_clock: float) -> Dict:
        """
        Get current information about a charging station.

        Args:
            charger_node: Charging station node
            global_clock: Current simulation time

        Returns:
            Dictionary with charger information
        """
        return {
            "node": int(charger_node),
            "type": self.charger_type[charger_node],
            "capacity": int(self.charger_capacity[charger_node]),
            "current_occupancy": len(self.charger_occupancy[charger_node]),
            "waitlist_length": len(self.charger_waitlist[charger_node]),
            "queue_length": len(self.charger_queue[charger_node]),
            "sessions": self.charger_stats[charger_node]["total_charge_sessions"],
            "trucks_served": len(
                self.charger_stats[charger_node]["total_trucks_served"]
            ),
        }

    def get_utilization_stats(self, global_clock: float) -> Dict:
        """
        Calculate charging station utilization statistics.

        Args:
            global_clock: Current simulation time

        Returns:
            Dictionary with utilization statistics
        """
        # Update occupancy time for currently occupied chargers
        for node in self.charging_nodes:
            if len(self.charger_occupancy[node]) > 0:
                stats = self.charger_stats[node]
                stats["occupancy_time"] += global_clock - stats["last_update_time"]
                stats["last_update_time"] = global_clock

        # Compile statistics by charger type
        level2_stats = {
            "nodes": [],
            "utilization_rates": [],
            "total_sessions": 0,
            "total_charge_time": 0,
        }
        dcfast_stats = {
            "nodes": [],
            "utilization_rates": [],
            "total_sessions": 0,
            "total_charge_time": 0,
        }

        all_chargers = []

        for node in self.charging_nodes:
            stats = self.charger_stats[node]
            c_type = self.charger_type[node]
            capacity = self.charger_capacity[node]

            # Calculate utilization rate
            utilization_rate = (
                stats["occupancy_time"] / global_clock if global_clock > 0 else 0.0
            )

            charger_info = {
                "node": int(node),
                "type": c_type,
                "capacity": int(capacity),
                "utilization_rate": utilization_rate,
                "sessions": stats["total_charge_sessions"],
                "charge_time": stats["total_charge_time"],
                "trucks_served": len(stats["total_trucks_served"]),
                "current_occupancy": len(self.charger_occupancy[node]),
            }

            all_chargers.append(charger_info)

            # Aggregate by type
            if c_type == "Level2":
                level2_stats["nodes"].append(int(node))
                level2_stats["utilization_rates"].append(utilization_rate)
                level2_stats["total_sessions"] += stats["total_charge_sessions"]
                level2_stats["total_charge_time"] += stats["total_charge_time"]
            else:  # DCFast
                dcfast_stats["nodes"].append(int(node))
                dcfast_stats["utilization_rates"].append(utilization_rate)
                dcfast_stats["total_sessions"] += stats["total_charge_sessions"]
                dcfast_stats["total_charge_time"] += stats["total_charge_time"]

        # Calculate average utilization by type
        if level2_stats["nodes"]:
            level2_stats["avg_utilization"] = sum(
                level2_stats["utilization_rates"]
            ) / len(level2_stats["utilization_rates"])
        else:
            level2_stats["avg_utilization"] = 0.0

        if dcfast_stats["nodes"]:
            dcfast_stats["avg_utilization"] = sum(
                dcfast_stats["utilization_rates"]
            ) / len(dcfast_stats["utilization_rates"])
        else:
            dcfast_stats["avg_utilization"] = 0.0

        # Calculate overall statistics
        total_chargers = len(self.charging_nodes)
        if total_chargers > 0:
            overall_avg_util = (
                level2_stats["avg_utilization"] * len(level2_stats["nodes"])
                + dcfast_stats["avg_utilization"] * len(dcfast_stats["nodes"])
            ) / total_chargers
        else:
            overall_avg_util = 0.0

        return {
            "all_chargers": all_chargers,
            "level2": {
                "avg_utilization": level2_stats["avg_utilization"],
                "total_sessions": level2_stats["total_sessions"],
                "total_charge_time": level2_stats["total_charge_time"],
                "num_chargers": len(level2_stats["nodes"]),
            },
            "dcfast": {
                "avg_utilization": dcfast_stats["avg_utilization"],
                "total_sessions": dcfast_stats["total_sessions"],
                "total_charge_time": dcfast_stats["total_charge_time"],
                "num_chargers": len(dcfast_stats["nodes"]),
            },
            "overall": {
                "avg_utilization": overall_avg_util,
                "total_sessions": level2_stats["total_sessions"]
                + dcfast_stats["total_sessions"],
                "total_charge_time": level2_stats["total_charge_time"]
                + dcfast_stats["total_charge_time"],
            },
        }
