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
    truck_id: int  # Tie-breaker: lower truck_id gets priority when times are equal
    event_type: EventType = field(compare=False)
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
        delivery_simulator: Any = None,
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
            delivery_simulator: DeliverySimulator instance for stochastic unloading time
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
            timestamp=global_clock,
        )

        # Clear route tracking information
        truck.route_destination = None
        truck.route_arrival_time = None

        # Track route for visualization
        if enable_plotting:
            event_label = "delivery" if is_delivery else "charger"
            # Get SoC at arrival (after battery discharge from travel)
            soc_at_arrival = truck.get_battery_percentage()
            
            # Store the full path if available, otherwise just the destination
            path = data.get("path", [destination])
            # Add all intermediate nodes from the path (excluding start which is already in route)
            if len(path) > 1:
                for node in path[1:]:  # Skip first node (already in previous route)
                    # Only label the final destination
                    node_label = event_label if node == destination else "travel"
                    truck_routes[truck.truck_id].append(
                        (node, global_clock, node_label, soc_at_arrival)
                    )
            else:
                # No path available, just add destination
                truck_routes[truck.truck_id].append(
                    (destination, global_clock, event_label, soc_at_arrival)
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

        # Check if truck failed (already logged in move_to_node)
        if truck.failed:
            truck_states[truck.truck_id] = "failed"
            if self.verbose:
                print(f"  Truck {truck.truck_id} FAILED: battery depleted")
        # Check if truck completed all deliveries
        elif truck.is_complete:
            truck.mark_complete(timestamp=global_clock)
            truck_states[truck.truck_id] = "complete"
            if self.verbose:
                print(f"  Truck {truck.truck_id} COMPLETED all deliveries")
        # If this was a delivery (and not complete/failed), apply unloading time
        elif is_delivery and delivery_simulator is not None:
            # Apply stochastic unloading time
            unloading_time = delivery_simulator.apply_unloading_time(
                delivery_node=destination,
                current_time=global_clock
            )
            
            # Start unloading event
            truck.start_unloading(timestamp=global_clock, delivery_node=destination)
            
            # Schedule TRUCK_READY event after unloading completes
            heapq.heappush(
                event_queue,
                Event(
                    time=global_clock + unloading_time,
                    event_type=EventType.TRUCK_READY,
                    truck_id=truck.truck_id,
                    data={
                        "reason": "unloading_complete",
                        "unloading_duration": unloading_time
                    }
                )
            )
            
            # Mark truck as "unloading" state
            truck_states[truck.truck_id] = "unloading"
            
            if self.verbose:
                print(f"  Truck {truck.truck_id} unloading at delivery node {destination}")
                print(f"    Unloading time: {unloading_time:.3f}h ({unloading_time*60:.1f} min)")
                print(f"    Will be ready at: {global_clock + unloading_time:.2f}h")
        else:
            # Truck is ready for next action - update state
            # Note: TRUCK_READY event will be scheduled by the main event loop
            truck.mark_ready(timestamp=global_clock, reason="arrived_at_charger")
            truck_states[truck.truck_id] = "ready"


