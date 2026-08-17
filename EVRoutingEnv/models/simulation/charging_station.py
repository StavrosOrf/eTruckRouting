"""
Charging station management for the event-driven truck environment.
"""

import heapq
import json
import math


class ChargingStation:
    """
    Manages all charging station related logic including queues, occupancy,
    waiting times, and statistics.
    """

    def __init__(
        self,
        charging_nodes: list[int],
        transport_graph,
        waiting_time_lookup_path: str,
        verbose: bool = False,
        charging_config: dict | None = None,
    ):
        """
        Initialize the charging station manager.

        Args:
            charging_nodes: List of charging node IDs
            transport_graph: TransportationGraph instance
            waiting_time_lookup_path: Path to waiting time lookup JSON file
            verbose: Print detailed information
            charging_config: Optional station-power assignment configuration
        """
        self.charging_nodes = charging_nodes
        self.transport_graph = transport_graph
        self.verbose = verbose

        # Load waiting time lookup table for queue simulation
        with open(waiting_time_lookup_path) as f:
            self.waiting_time_lookup = json.load(f)

        # Charger properties (capacity, type). Port counts are station-specific
        # and come from the network data; the scale factor exists so a
        # congestion sensitivity can make ports scarce or plentiful without
        # editing the data file, and it never drops a station below one port.
        charging_config = charging_config or {}
        self.charger_capacity = {
            node: transport_graph.get_charger_capacity(node)
            for node in charging_nodes
        }
        # Bad data still has to fail loudly; the scaling below is applied only
        # to capacities that were valid to begin with.
        invalid_capacity = {
            node: capacity
            for node, capacity in self.charger_capacity.items()
            if int(capacity) <= 0
        }
        if invalid_capacity:
            raise ValueError(
                f"charger capacities must be positive: {invalid_capacity}"
            )

        port_scale = float(charging_config.get("port_capacity_scale", 1.0))
        if not math.isfinite(port_scale) or port_scale <= 0.0:
            raise ValueError("charging.port_capacity_scale must be positive")
        if port_scale != 1.0:
            # Congestion sensitivity: make ports scarce or plentiful without
            # editing the network data, never dropping a station below one port.
            self.charger_capacity = {
                node: max(1.0, float(round(capacity * port_scale)))
                for node, capacity in self.charger_capacity.items()
            }
        self.charger_type = {
            node: transport_graph.get_charger_type(node) for node in charging_nodes
        }
        configured_classes = [
            float(value)
            for value in charging_config.get("station_power_classes_kw", [])
        ]
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in configured_classes
        ):
            raise ValueError("station power classes must be positive")
        raw_overrides = charging_config.get("station_power_overrides_kw", {})
        power_overrides = {
            int(node): float(power) for node, power in raw_overrides.items()
        }
        unknown_overrides = set(power_overrides) - set(charging_nodes)
        if unknown_overrides:
            raise ValueError(
                "station power overrides reference unknown nodes: "
                f"{sorted(unknown_overrides)}"
            )
        if any(
            not math.isfinite(power) or power <= 0.0
            for power in power_overrides.values()
        ):
            raise ValueError("station power overrides must be positive")

        self.charger_power_kw: dict[int, float] = {}
        for index, node in enumerate(sorted(charging_nodes)):
            if node in power_overrides:
                power = power_overrides[node]
            elif configured_classes:
                power = configured_classes[index % len(configured_classes)]
            else:
                charger_type = self.charger_type[node]
                type_key = "dcfast" if charger_type == "DCFast" else "level2"
                legacy_default = 50.0 if charger_type == "DCFast" else 7.2
                power = float(
                    charging_config.get(type_key, {}).get(
                        "charge_rate",
                        legacy_default,
                    )
                )
            if power <= 0.0:
                raise ValueError(
                    f"charger {node} has no positive configured power"
                )
            self.charger_power_kw[node] = power

        # Conversion efficiency was type-level only, so two stations of the same
        # class necessarily charged at the same efficiency.  Empirical station
        # data distinguishes them, and the review asks for station-specific
        # values to be preserved rather than averaged into the class.
        raw_efficiency = charging_config.get("station_efficiency_overrides", {})
        efficiency_overrides = {
            int(node): float(value) for node, value in raw_efficiency.items()
        }
        unknown_efficiency = set(efficiency_overrides) - set(charging_nodes)
        if unknown_efficiency:
            raise ValueError(
                "station efficiency overrides reference unknown nodes: "
                f"{sorted(unknown_efficiency)}"
            )
        if any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in efficiency_overrides.values()
        ):
            raise ValueError("station efficiency overrides must be in (0, 1]")

        self.charger_efficiency: dict[int, float] = {}
        for node in sorted(charging_nodes):
            if node in efficiency_overrides:
                self.charger_efficiency[node] = efficiency_overrides[node]
                continue
            charger_type = self.charger_type[node]
            type_key = "dcfast" if charger_type == "DCFast" else "level2"
            self.charger_efficiency[node] = float(
                charging_config.get(type_key, {}).get("efficiency", 0.90)
            )

        self.station_available = dict.fromkeys(charging_nodes, True)

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
        # Each entry: {"truck_id": int, "sequence": int}
        self.charger_waitlist = {node: [] for node in charging_nodes}
        
        # Global sequence counter for strict FCFS ordering
        self.waitlist_sequence_counter = 0
        self.pending_wake_trucks = {node: set() for node in charging_nodes}

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
        self.waitlist_sequence_counter = 0
        self.pending_wake_trucks = {
            node: set() for node in self.charging_nodes
        }
        self.station_available = dict.fromkeys(self.charging_nodes, True)
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
        if (
            not math.isfinite(current_utilization)
            or not 0.0 <= current_utilization <= 1.0
        ):
            raise ValueError(
                "current_utilization must be finite and in [0, 1]"
            )
        charger_type = self.charger_type[charger_node]
        capacity = int(self.charger_capacity[charger_node])
        if not self.station_available[charger_node]:
            return math.inf

        # Get lookup table for this charger type and capacity
        if charger_type not in self.waiting_time_lookup:
            return 0.0

        capacity_str = str(capacity)
        if capacity_str not in self.waiting_time_lookup[charger_type]:
            # Use closest available capacity
            available_capacities = sorted(
                [int(c) for c in self.waiting_time_lookup[charger_type]]
            )
            if not available_capacities:
                return 0.0
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

    def _record_queue_state(
        self,
        charger_node: int,
        global_clock: float,
        truck_id: int | None = None,
        event_type: str | None = None,
    ):
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
    ) -> tuple[bool, float | None]:
        """
        Check if a truck can proceed with an action at a charging station.
        Pure event-driven FCFS with strict ordering via sequence numbers.

        Args:
            truck_id: ID of the truck
            charger_node: Charging station node
            global_clock: Current simulation time

        Returns:
            Tuple of (can_proceed, next_check_time)
            - can_proceed: True if truck can act now, False otherwise
            - next_check_time: Always None (no time-based predictions)
        """
        capacity = int(self.charger_capacity[charger_node])
        occupancy = len(self.charger_occupancy[charger_node])
        free_slots = max(0, capacity - occupancy)
        waitlist = self.charger_waitlist[charger_node]

        # Check if truck is already in waitlist
        idx = next(
            (i for i, e in enumerate(waitlist) if e["truck_id"] == truck_id), None
        )

        if idx is None:
            # NEW ARRIVAL: Truck not in waitlist yet
            
            # Case 1: Free slots available AND no one waiting ahead
            # Truck can charge immediately - don't add to waitlist
            if free_slots > 0 and len(waitlist) == 0:
                # Record arrival event (but not added to waitlist)
                self._record_queue_state(charger_node, global_clock, truck_id, 'arrive')
                return True, None  # Can proceed immediately
            
            # Case 2: All ports occupied OR someone is already waiting
            # Add to waitlist with sequence number for strict FCFS ordering
            self.waitlist_sequence_counter += 1
            waitlist.append({
                "truck_id": truck_id, 
                "sequence": self.waitlist_sequence_counter
            })
            self._record_queue_state(charger_node, global_clock, truck_id, 'arrive')
            
            # Truck will be woken by wake_waiting_trucks when port becomes available
            return False, None  # Always return None - no time predictions
        
        else:
            # ALREADY IN WAITLIST: Check if truck is eligible based on strict FCFS
            
            # Truck can proceed if:
            # 1. There's a free slot available
            # 2. Truck is within the first free_slots positions in the waitlist
            # 3. All trucks ahead in sequence order have been processed
            
            if free_slots > 0 and idx < free_slots:
                # Eligible based on position
                self.pending_wake_trucks[charger_node].discard(truck_id)
                return True, None
            
            # Can't proceed yet - will be woken by wake_waiting_trucks
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
        charge_hours = float(charge_hours)
        global_clock = float(global_clock)
        if not math.isfinite(charge_hours) or charge_hours <= 0.0:
            raise ValueError("charge_hours must be positive")
        if not math.isfinite(global_clock) or global_clock < 0.0:
            raise ValueError("global_clock must be finite and non-negative")
        if int(truck_id) < 0:
            raise ValueError("truck_id must be non-negative")
        if not self.station_available[charger_node]:
            raise RuntimeError(f"charger {charger_node} is unavailable")

        capacity = int(self.charger_capacity[charger_node])
        occupancy = self.charger_occupancy[charger_node]
        if truck_id in occupancy:
            raise RuntimeError(
                f"truck {truck_id} is already charging at {charger_node}"
            )
        if len(occupancy) >= capacity:
            raise RuntimeError(
                f"charger {charger_node} has no free port for truck {truck_id}"
            )
        occupied_elsewhere = [
            node
            for node, truck_ids in self.charger_occupancy.items()
            if node != charger_node and truck_id in truck_ids
        ]
        if occupied_elsewhere:
            raise RuntimeError(
                f"truck {truck_id} is already charging at {occupied_elsewhere[0]}"
            )

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
        self.pending_wake_trucks[charger_node].discard(truck_id)

        # Update occupancy
        was_empty = len(occupancy) == 0
        if truck_id not in occupancy:
            occupancy.append(truck_id)
        
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
        if was_empty:
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
        global_clock = float(global_clock)
        if not math.isfinite(global_clock) or global_clock < 0.0:
            raise ValueError("global_clock must be finite and non-negative")
        expected_end = self.truck_charge_end_time.get(truck_id)
        if expected_end is not None and global_clock < expected_end - 1e-9:
            raise ValueError("charging cannot finish before its scheduled end")
        # Remove from charger occupancy
        if truck_id not in self.charger_occupancy[charger_node]:
            raise RuntimeError(
                f"truck {truck_id} is not charging at {charger_node}"
            )
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
        self.pending_wake_trucks[charger_node].discard(truck_id)
        
        # Record finish charging event
        self._record_queue_state(charger_node, global_clock, truck_id, 'finish')

    def wake_waiting_trucks(
        self,
        charger_node: int,
        global_clock: float,
        event_queue: list,
        EventType,
        Event,
        truck_states: dict | None = None,
    ):
        """
        Wake trucks waiting at a charging station when a port becomes available.
        Uses strict FCFS ordering based on sequence numbers.
        Skips trucks that are currently routing.

        Args:
            charger_node: Charging station node
            global_clock: Current simulation time
            event_queue: Event queue for scheduling wake events
            EventType: EventType enum
            Event: Event class
            truck_states: Optional dict mapping truck_id to state (to skip routing trucks)
        """
        capacity = int(self.charger_capacity[charger_node])
        occupancy = len(self.charger_occupancy[charger_node])
        free_slots = max(0, capacity - occupancy)
        waitlist = self.charger_waitlist[charger_node]

        if free_slots > 0 and waitlist:
            if truck_states is not None:
                stale_ids = {
                    entry["truck_id"]
                    for entry in waitlist
                    if truck_states.get(entry["truck_id"])
                    in {"routing", "failed", "complete"}
                }
                if stale_ids:
                    waitlist[:] = [
                        entry
                        for entry in waitlist
                        if entry["truck_id"] not in stale_ids
                    ]
                    self.pending_wake_trucks[charger_node].difference_update(
                        stale_ids
                    )

            # Sort waitlist by sequence number to ensure strict FCFS
            # (should already be sorted, but this guarantees it)
            waitlist.sort(key=lambda x: x["sequence"])

            # Wake up to free_slots number of trucks in strict FCFS order
            unreserved_slots = max(
                0,
                free_slots - len(self.pending_wake_trucks[charger_node]),
            )
            num_to_wake = min(unreserved_slots, len(waitlist))
            woken = 0

            for entry in waitlist:
                if woken >= num_to_wake:
                    break
                tid = entry["truck_id"]
                sequence = entry["sequence"]
                
                # Skip trucks that are currently routing (to any destination)
                if truck_states is not None and truck_states.get(tid) == "routing":
                    if self.verbose:
                        print(f"    Skipping truck {tid} (sequence {sequence}) - currently routing")
                    continue

                if tid in self.pending_wake_trucks[charger_node]:
                    continue
                
                if self.verbose:
                    print(f"    Waking truck {tid} (sequence {sequence}) at charger {charger_node}")
                
                # Schedule immediate wake event
                heapq.heappush(
                    event_queue,
                    Event(
                        time=global_clock,
                        event_type=EventType.TRUCK_READY,
                        truck_id=tid,
                        data={
                            "reason": "charger_port_available",
                            "charger_node": charger_node,
                            "sequence": sequence
                        },
                    ),
                )
                self.pending_wake_trucks[charger_node].add(tid)
                woken += 1

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
        self.pending_wake_trucks[charger_node].discard(truck_id)

    def set_station_available(
        self,
        charger_node: int,
        available: bool,
    ) -> list[int]:
        """Set station availability and release queued trucks on closure."""
        if charger_node not in self.station_available:
            raise KeyError(f"node {charger_node} is not a charging station")
        available = bool(available)
        if self.station_available[charger_node] == available:
            return []
        self.station_available[charger_node] = available
        if available:
            return []

        released = [
            entry["truck_id"]
            for entry in self.charger_waitlist[charger_node]
        ]
        self.charger_waitlist[charger_node] = []
        self.pending_wake_trucks[charger_node].difference_update(released)
        return released

    def get_charger_info(self, charger_node: int, global_clock: float) -> dict:
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
            "power_kw": self.charger_power_kw[charger_node],
            "available": self.station_available[charger_node],
            "capacity": int(self.charger_capacity[charger_node]),
            "current_occupancy": len(self.charger_occupancy[charger_node]),
            "waitlist_length": len(self.charger_waitlist[charger_node]),
            "queue_length": len(self.charger_queue[charger_node]),
            "sessions": self.charger_stats[charger_node]["total_charge_sessions"],
            "trucks_served": len(
                self.charger_stats[charger_node]["total_trucks_served"]
            ),
        }

    def get_utilization_stats(self, global_clock: float) -> dict:
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
                "power_kw": self.charger_power_kw[node],
                "available": self.station_available[node],
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

    def print_queues(self):
        """
        Print current status of all charger queues showing which trucks are charging
        and which are waiting.
        """
        print("\n" + "=" * 80)
        print("CHARGER QUEUE STATUS")
        print("=" * 80)
        
        has_activity = False
        
        for charger_node in sorted(self.charging_nodes):
            charging = self.charger_occupancy.get(charger_node, [])
            waiting = self.charger_waitlist.get(charger_node, [])
            capacity = self.charger_capacity[charger_node]
            charger_type = self.charger_type[charger_node]
            
            # Only print chargers with activity
            if charging or waiting:
                has_activity = True
                print(f"\nCharger Node {charger_node} ({charger_type}, Capacity: {capacity})")
                print("-" * 80)
                
                # Print charging trucks
                if charging:
                    print(f"  Charging ({len(charging)}/{capacity} ports occupied):")
                    for truck_id in charging:
                        end_time = self.truck_charge_end_time.get(truck_id, "unknown")
                        if end_time != "unknown":
                            print(f"    • Truck {truck_id} (finishes at t={end_time:.2f}h)")
                        else:
                            print(f"    • Truck {truck_id}")
                else:
                    print(f"  Charging: None (0/{capacity} ports occupied)")
                
                # Print waiting trucks
                if waiting:
                    print(f"  Waiting ({len(waiting)} trucks in queue):")
                    for i, truck_id in enumerate(waiting, 1):
                        print(f"    {i}. Truck {truck_id}")
                else:
                    print("  Waiting: None")
        
        if not has_activity:
            print("  No trucks currently charging or waiting at any charger")
        
        print("=" * 80 + "\n")

    def print_charger_queue(self, charger_node: int):
        """
        Print current status of a specific charger showing which trucks are charging
        and which are waiting.
        
        Args:
            charger_node: The charger node ID to print status for
        """
        if charger_node not in self.charging_nodes:
            print(f"  Node {charger_node} is not a charger")
            return
        
        charging = self.charger_occupancy[charger_node]
        waiting = self.charger_waitlist[charger_node]
        capacity = self.charger_capacity[charger_node]
        charger_type = self.charger_type[charger_node]
        
        print(f"  Charger Node {charger_node} ({charger_type}, Capacity: {capacity})")
        print("  " + "-" * 76)
        
        # Print charging trucks
        if charging:
            print(f"    Charging ({len(charging)}/{capacity} ports occupied):")
            for truck_id in charging:
                end_time = self.truck_charge_end_time.get(truck_id, "unknown")
                if end_time != "unknown":
                    print(f"      • Truck {truck_id} (finishes at t={end_time:.2f}h)")
                else:
                    print(f"      • Truck {truck_id}")
        else:
            print(f"    Charging: None (0/{capacity} ports occupied)")
        
        # Print waiting trucks
        if waiting:
            print(f"    Waiting ({len(waiting)} trucks in queue):")
            for i, entry in enumerate(waiting, 1):
                truck_id = entry["truck_id"]
                sequence = entry["sequence"]
                print(f"      {i}. Truck {truck_id} (seq #{sequence})")
        else:
            print("    Waiting: None")
