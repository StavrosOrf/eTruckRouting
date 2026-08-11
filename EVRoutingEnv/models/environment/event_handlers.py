"""
Event handling logic for the event-driven truck environment.
"""

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """Types of events in the simulation."""

    TRUCK_READY = "truck_ready"  # Truck is ready to take an action (initial, after route, after charge, after wait)
    TRUCK_ROUTING = "truck_routing"  # Truck completed routing to a node (arrival event)


@dataclass(order=True)
class Event:
    """Represents a simulation event."""

    time: float  # When the event occurs
    priority: int = field(init=False, repr=False)
    truck_id: int  # Tie-breaker: lower truck_id gets priority when times are equal
    event_type: EventType = field(compare=False)
    data: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.time) or self.time < 0.0:
            raise ValueError("event time must be finite and non-negative")
        if self.truck_id < 0:
            raise ValueError("event truck_id must be non-negative")
        if self.event_type is EventType.TRUCK_ROUTING:
            self.priority = 0
        elif self.data.get("reason") in {
            "charge_complete",
            "unloading_complete",
            "service_window_open",
        }:
            self.priority = 1
        else:
            self.priority = 2

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
        trucks: list[Any],
        truck_states: dict[int, str],
        truck_routes: dict[int, list],
        event_queue: list,
        global_clock: float,
        enable_plotting: bool,
        delivery_simulator: Any = None,
        task_registry: Any = None,
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

        # Joint-routing events carry an explicit task identity. Legacy events
        # infer delivery status from the truck-owned sequence.
        destination = data["destination"]
        task_id = data.get("task_id")
        is_joint_delivery = task_registry is not None and task_id is not None
        next_delivery_target = truck.get_next_delivery_target()

        if is_joint_delivery:
            is_delivery = True
        elif truck.enable_flexible_delivery_order:
            remaining_deliveries = (
                next_delivery_target
                if isinstance(next_delivery_target, list)
                else []
            )
            is_delivery = destination in remaining_deliveries
        else:
            # Sequential mode: check if destination is next delivery
            is_delivery = destination == next_delivery_target

        # Update truck position and state
        truck.move_to_node(
            node=data["destination"],
            distance=data["distance"],
            travel_time=data["travel_time"],
            discharge=data["discharge"],
            timestamp=global_clock,
            mark_delivery_on_arrival=not is_joint_delivery,
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
            
            if truck.enable_flexible_delivery_order:
                completed_deliveries = len(truck.delivered_nodes)
                print(
                    f"    Delivery progress: {completed_deliveries}/{total_deliveries} complete, "
                    f"{len(remaining_deliveries)} remaining (flexible mode)"
                )
                print(f"    Delivered nodes: {truck.delivered_nodes}")
                print(f"    Next delivery targets: {next_delivery}")
                print(f"    Remaining deliveries: {remaining_deliveries}")
            else:
                completed_deliveries = truck.current_sequence_index
                print(
                    f"    Delivery progress: {completed_deliveries}/{total_deliveries} complete, "
                    f"{len(remaining_deliveries)} remaining (sequential mode)"
                )
                print(f"    Current sequence index: {truck.current_sequence_index}/{len(truck.delivery_sequence)-1}")
                print(f"    Next delivery target: {next_delivery}")
                print(f"    Remaining deliveries: {remaining_deliveries}")
            
            print(f"    is_complete flag: {truck.is_complete}")

        if is_joint_delivery:
            waiting_for_service = False
            if truck.failed:
                task_registry.release_claim(destination, truck.truck_id)
            else:
                task = task_registry.task_for_node(destination)
                if global_clock > task.latest_service + 1e-9:
                    task_registry.release_claim(destination, truck.truck_id)
                    truck.mark_failed(
                        reason="time_window_violation_after_realization",
                        timestamp=global_clock,
                    )
                    truck_states[truck.truck_id] = "failed"
                elif global_clock < task.earliest_service - 1e-9:
                    waiting_for_service = True
                    heapq.heappush(
                        event_queue,
                        Event(
                            time=task.earliest_service,
                            event_type=EventType.TRUCK_READY,
                            truck_id=truck.truck_id,
                            data={
                                "reason": "service_window_open",
                                "customer_node": destination,
                                "task_id": task_id,
                                "unloading_duration": data.get(
                                    "unloading_time",
                                    0.0,
                                ),
                                "wait_duration": (
                                    task.earliest_service - global_clock
                                ),
                            },
                        ),
                    )
                    truck_states[truck.truck_id] = "waiting_for_service"
                else:
                    task_registry.start_service(
                        destination,
                        truck_id=truck.truck_id,
                        timestamp=global_clock,
                    )
        else:
            waiting_for_service = False

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
        elif waiting_for_service:
            # The opening-time event starts service and schedules completion.
            pass
        elif is_delivery and delivery_simulator is not None:
            # The navigation action samples service time once and stores it on
            # the arrival event so reward accounting and event timing use the
            # same realization. Fall back for legacy/external events.
            unloading_time = data.get("unloading_time")
            if unloading_time is None:
                unloading_time = delivery_simulator.apply_unloading_time(
                    delivery_node=destination,
                    current_time=global_clock,
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
                        "unloading_duration": unloading_time,
                        "task_id": task_id,
                        "customer_node": destination,
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
