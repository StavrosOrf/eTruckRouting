"""
Event handling logic for the event-driven truck environment.
"""
import heapq
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    """Types of events in the simulation."""
    TRUCK_READY = "truck_ready"  # Truck is ready to take an action
    ROUTE_COMPLETE = "route_complete"  # Truck completed routing to a node
    CHARGE_COMPLETE = "charge_complete"  # Truck completed charging
    TRUCK_TERMINATED = "truck_terminated"  # Truck finished or failed


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
    
    def handle_route_complete(
        self,
        event: Event,
        trucks: List[Any],
        truck_states: Dict[int, str],
        truck_routes: Dict[int, List],
        event_queue: List,
        global_clock: float,
        enable_plotting: bool
    ):
        """
        Handle completion of routing to a node.
        
        Args:
            event: The route complete event
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
        destination = data['destination']
        next_delivery_target = truck.get_next_delivery_target()
        is_delivery = destination == next_delivery_target
        
        # Update truck position and state
        truck.move_to_node(
            node=data['destination'],
            distance=data['distance'],
            travel_time=data['travel_time'],
            discharge=data['discharge']
        )
        
        # Track route for visualization
        if enable_plotting:
            event_label = 'delivery' if is_delivery else 'charger'
            # Store the full path if available, otherwise just the destination
            path = data.get('path', [destination])
            # Add all intermediate nodes from the path (excluding start which is already in route)
            if len(path) > 1:
                for node in path[1:]:  # Skip first node (already in previous route)
                    # Only label the final destination
                    node_label = event_label if node == destination else 'travel'
                    truck_routes[truck.truck_id].append(
                        (node, global_clock, node_label)
                    )
            else:
                # No path available, just add destination
                truck_routes[truck.truck_id].append(
                    (destination, global_clock, event_label)
                )
        
        if self.verbose:
            print(f"  Truck {truck.truck_id} arrived at node {data['destination']}")
            print(f"    Battery: {truck.current_battery:.1f} kWh ({truck.get_battery_percentage():.1f}%)")
        
        # Check if truck failed
        if truck.failed:
            truck_states[truck.truck_id] = "failed"
            heapq.heappush(event_queue, Event(
                time=global_clock,
                event_type=EventType.TRUCK_TERMINATED,
                truck_id=truck.truck_id,
                data={"reason": "battery_depleted"}
            ))
        # Check if truck completed all deliveries
        elif truck.is_complete:
            truck_states[truck.truck_id] = "complete"
            heapq.heappush(event_queue, Event(
                time=global_clock,
                event_type=EventType.TRUCK_TERMINATED,
                truck_id=truck.truck_id,
                data={"reason": "deliveries_complete"}
            ))
        else:
            # Truck is ready for next action
            truck_states[truck.truck_id] = "active"
            heapq.heappush(event_queue, Event(
                time=global_clock,
                event_type=EventType.TRUCK_READY,
                truck_id=truck.truck_id,
                data={"reason": "route_complete"}
            ))
    
    def handle_charge_complete(
        self,
        event: Event,
        trucks: List[Any],
        truck_states: Dict[int, str],
        charger_occupancy: Dict[int, List],
        charger_queue: Dict[int, List],
        charger_stats: Dict,
        event_queue: List,
        global_clock: float
    ):
        """
        Handle completion of charging.
        
        Args:
            event: The charge complete event
            trucks: List of Truck objects
            truck_states: Dictionary of truck states
            charger_occupancy: Dictionary of charger occupancy
            charger_queue: Dictionary of charger queues (new parameter)
            charger_stats: Dictionary of charger statistics
            event_queue: Priority queue of events
            global_clock: Current simulation time
        """
        truck = trucks[event.truck_id]
        data = event.data
        
        # Complete charging
        truck.finish_charging(
            charge_amount=data['charge_amount'],
            charge_duration=data['charge_duration']
        )
        
        # Remove from charger occupancy and queue
        charger_node = truck.current_node
        if charger_node in charger_occupancy:
            if truck.truck_id in charger_occupancy[charger_node]:
                charger_occupancy[charger_node].remove(truck.truck_id)
            
            # Remove from queue as well
            if charger_node in charger_queue:
                charger_queue[charger_node] = [
                    (tid, start, dur) for tid, start, dur in charger_queue[charger_node]
                    if tid != truck.truck_id
                ]
                # Update queue length stat
                if charger_node in charger_stats:
                    charger_stats[charger_node]["queue_length"] = len(charger_queue[charger_node])
            
            # Update occupancy statistics
            if charger_node in charger_stats:
                stats = charger_stats[charger_node]
                if len(charger_occupancy[charger_node]) == 0:
                    # Charger became empty
                    stats['occupancy_time'] += (global_clock - stats['last_update_time'])
                    stats['last_update_time'] = global_clock
        
        if self.verbose:
            print(f"  Truck {truck.truck_id} finished charging")
            print(f"    Battery: {truck.current_battery:.1f} kWh ({truck.get_battery_percentage():.1f}%)")
            print(f"    Charged: {data['charge_amount']:.1f} kWh in {data['charge_duration']:.2f}h")
        
        # Truck is ready for next action
        truck_states[truck.truck_id] = "active"
        heapq.heappush(event_queue, Event(
            time=global_clock,
            event_type=EventType.TRUCK_READY,
            truck_id=truck.truck_id,
            data={"reason": "charge_complete"}
        ))
    
    def handle_truck_terminated(
        self,
        event: Event,
        trucks: List[Any]
    ):
        """
        Handle truck termination.
        
        Args:
            event: The truck terminated event
            trucks: List of Truck objects
        """
        truck = trucks[event.truck_id]
        reason = event.data.get('reason', 'unknown')
        
        if self.verbose:
            print(f"  Truck {truck.truck_id} TERMINATED: {reason}")
            print(f"    Total time: {truck.total_time_elapsed:.2f}h")
            print(f"    Total distance: {truck.total_distance_traveled:.2f} km")
