"""
Event handling logic for the event-driven truck environment.
"""

import heapq
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    """Types of events in the simulation."""

    TRUCK_READY = "truck_ready"  # Truck is ready to take an action (initial, after route, after charge, after wait)
    TRUCK_ROUTING = "truck_routing"  # Truck completed routing to a node (arrival event)


@dataclass(order=True)
class Event:
    """Represents a simulation event."""

    time: float  # When the event occurs
    event_type: EventType = field(compare=False)
    truck_id: int = field(compare=False)
    data: Dict = field(default_factory=dict, compare=False)

    def __repr__(self):
        return f"Event(time={self.time:.2f}, type={self.event_type.value}, truck={self.truck_id})"


class EventHandler:
    """
    Handles processing of events in the simulation.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the event handler.

        Args:
            verbose: Print verbose messages
        """
        self.verbose = verbose

    def handle_truck_routing(
        self,
        event: Event,
        trucks: List[Any],
        truck_states: Dict[int, str],
        truck_routes: Dict[int, List],
        event_queue: List,
        global_clock: float,
        enable_plotting: bool,
    ):
        """
        Handle truck arrival at a node (after routing).

        Args:
            event: The truck routing event
            trucks: List of Truck objects
            truck_states: Dictionary of truck states
            truck_routes: Dictionary of truck routes
            event_queue: Priority queue of events
            global_clock: Current simulation time
            enable_plotting: Whether plotting is enabled
        """
        truck = trucks[event.truck_id]
        data = event.data

        # Check if this will be a delivery event BEFORE updating truck state
        destination = data["destination"]
        next_delivery_target = truck.get_next_delivery_target()
        is_delivery = destination == next_delivery_target

        # Update truck position and state
        truck.move_to_node(
            node=data["destination"],
            distance=data["distance"],
            travel_time=data["travel_time"],
            discharge=data["discharge"],
        )

        # Clear route tracking information
        truck.route_destination = None
        truck.route_arrival_time = None

        # Track route for visualization
        if enable_plotting:
            event_label = "delivery" if is_delivery else "charger"
            # Store the full path if available, otherwise just the destination
            path = data.get("path", [destination])
            # Add all intermediate nodes from the path (excluding start which is already in route)
            if len(path) > 1:
                for node in path[1:]:  # Skip first node (already in previous route)
                    # Only label the final destination
                    node_label = event_label if node == destination else "travel"
                    truck_routes[truck.truck_id].append(
                        (node, global_clock, node_label)
                    )
            else:
                # No path available, just add destination
                truck_routes[truck.truck_id].append(
                    (destination, global_clock, event_label)
                )

        if self.verbose:
            print(f"  Truck {truck.truck_id} arrived at node {data['destination']} at time {global_clock:.2f}")
            print(
                f"    Battery: {truck.current_battery:.1f} kWh ({truck.get_battery_percentage():.1f}%)"
            )
            # Debug: Show delivery progress
            remaining_deliveries = truck.get_remaining_deliveries()
            next_delivery = truck.get_next_delivery_target()
            total_deliveries = len(truck.delivery_sequence) - 1  # Exclude depot
            completed_deliveries = truck.current_sequence_index
            print(
                f"    Delivery progress: {completed_deliveries}/{total_deliveries} complete, "
                f"{len(remaining_deliveries)} remaining"
            )
            print(f"    Current sequence index: {truck.current_sequence_index}/{len(truck.delivery_sequence)-1}")
            print(f"    Next delivery target: {next_delivery}")
            print(f"    Remaining deliveries: {remaining_deliveries}")
            print(f"    is_complete flag: {truck.is_complete}")

        # Check if truck failed
        if truck.failed:
            truck_states[truck.truck_id] = "failed"
            if self.verbose:
                print(f"  Truck {truck.truck_id} FAILED: battery depleted")
        # Check if truck completed all deliveries
        elif truck.is_complete:
            truck_states[truck.truck_id] = "complete"
            if self.verbose:
                print(f"  Truck {truck.truck_id} COMPLETED all deliveries")
        else:
            # Truck is ready for next action - update state
            # Note: TRUCK_READY event will be scheduled by the main event loop
            truck_states[truck.truck_id] = "ready"


