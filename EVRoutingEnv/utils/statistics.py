"""
Statistics collection and reporting for the event-driven truck environment.
"""
import os
from typing import Dict, List, Any


class EnvironmentStatistics:
    """
    Handles statistics collection, calculation, and reporting.
    """
    
    def __init__(self, output_dir: str, verbose: bool = False):
        """
        Initialize the statistics collector.
        
        Args:
            output_dir: Directory to save statistics files
            verbose: Print verbose messages
        """
        self.output_dir = output_dir
        self.verbose = verbose
    
    def print_statistics(
        self,
        trucks: List[Any],
        truck_states: Dict[int, str],
        truck_routes: Dict[int, List],
        charger_util: Dict,
        global_clock: float,
        num_trucks: int
    ):
        """
        Print comprehensive simulation statistics.
        
        Args:
            trucks: List of Truck objects
            truck_states: Dictionary of truck states
            truck_routes: Dictionary of truck routes
            charger_util: Charger utilization statistics
            global_clock: Current simulation time
            num_trucks: Number of trucks
        """
        print("\n" + "="*80)
        print("SIMULATION STATISTICS")
        print("="*80)
        
        # Overall statistics
        print(f"\nTotal Simulation Time: {global_clock:.2f} hours")
        print(f"Number of Trucks: {num_trucks}")
        
        # Truck-specific statistics
        print(f"\n{'Truck Statistics:':-^80}")
        total_distance = 0.0
        total_deliveries = 0
        successful_trucks = 0
        failed_trucks = 0
        
        for truck in trucks:
            print(f"\nTruck {truck.truck_id}:")
            print(f"  Status: {truck_states.get(truck.truck_id, 'unknown')}")
            print(f"  Total Distance: {truck.total_distance_traveled:.2f} km")
            print(f"  Total Time: {truck.total_time_elapsed:.2f} hours")
            print(f"  Deliveries Completed: {len(truck.delivery_sequence) - len(list(truck.get_remaining_deliveries()))}/{len(truck.delivery_sequence)}")
            print(f"  Final Battery: {truck.current_battery:.1f}/{truck.battery_capacity:.1f} kWh ({truck.get_battery_percentage():.1f}%)")
            print(f"  Total Charge Sessions: {truck.num_charging_sessions}")
            print(f"  Total Charge Time: {truck.total_charging_time:.2f} hours")
            
            if truck.truck_id in truck_routes:
                route_length = len(truck_routes[truck.truck_id])
                charger_visits = len([r for r in truck_routes[truck.truck_id] if r[2] == 'charger'])
                print(f"  Route Nodes Visited: {route_length}")
                print(f"  Charging Station Visits: {charger_visits}")
            
            total_distance += truck.total_distance_traveled
            total_deliveries += len(truck.delivery_sequence) - len(list(truck.get_remaining_deliveries()))
            
            if truck_states.get(truck.truck_id) == 'complete':
                successful_trucks += 1
            elif truck_states.get(truck.truck_id) == 'failed':
                failed_trucks += 1
        
        # Aggregate statistics
        print(f"\n{'Aggregate Statistics:':-^80}")
        print(f"Successful Trucks: {successful_trucks}/{num_trucks}")
        print(f"Failed Trucks: {failed_trucks}/{num_trucks}")
        print(f"Total Distance Traveled: {total_distance:.2f} km")
        print(f"Total Deliveries Completed: {total_deliveries}")
        print(f"Average Distance per Truck: {total_distance/num_trucks:.2f} km")
        
        # Charging infrastructure statistics
        # self._print_charger_statistics(charger_util)
        
        # Save statistics to file
        self._save_statistics_file(
            global_clock, num_trucks, successful_trucks, failed_trucks,
            total_distance, total_deliveries, charger_util
        )
        
        print("="*80 + "\n")
    
    def _print_charger_statistics(self, charger_util: Dict):
        """Print charging infrastructure statistics."""
        print(f"\n{'Charging Infrastructure Statistics:':-^80}")
        print(f"\nLevel 2 Chargers:")
        print(f"  Average Utilization: {charger_util['level2']['avg_utilization']*100:.1f}%")
        print(f"  Total Sessions: {charger_util['level2']['total_sessions']}")
        print(f"  Total Charge Time: {charger_util['level2']['total_charge_time']:.2f} hours")
        print(f"  Number of Chargers: {charger_util['level2']['num_chargers']}")
        
        print(f"\nDC Fast Chargers:")
        print(f"  Average Utilization: {charger_util['dcfast']['avg_utilization']*100:.1f}%")
        print(f"  Total Sessions: {charger_util['dcfast']['total_sessions']}")
        print(f"  Total Charge Time: {charger_util['dcfast']['total_charge_time']:.2f} hours")
        print(f"  Number of Chargers: {charger_util['dcfast']['num_chargers']}")
        
        print(f"\nOverall Charging:")
        print(f"  Average Utilization: {charger_util['overall']['avg_utilization']*100:.1f}%")
        print(f"  Total Sessions: {charger_util['overall']['total_sessions']}")
        print(f"  Total Charge Time: {charger_util['overall']['total_charge_time']:.2f} hours")
        
        # Top utilized chargers
        sorted_chargers = sorted(charger_util['all_chargers'], 
                                key=lambda x: x['utilization_rate'], reverse=True)
        print(f"\nTop 5 Most Utilized Chargers:")
        for i, charger in enumerate(sorted_chargers[:5], 1):
            print(f"  {i}. Node {charger['node']} ({charger['type']}):")
            print(f"     Utilization: {charger['utilization_rate']*100:.1f}%")
            print(f"     Sessions: {charger['sessions']}")
            print(f"     Capacity: {charger['capacity']}")
    
    def _save_statistics_file(
        self,
        global_clock: float,
        num_trucks: int,
        successful_trucks: int,
        failed_trucks: int,
        total_distance: float,
        total_deliveries: int,
        charger_util: Dict
    ):
        """Save statistics to a text file."""
        stats_file = os.path.join(self.output_dir, 'statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SIMULATION STATISTICS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total Simulation Time: {global_clock:.2f} hours\n")
            f.write(f"Number of Trucks: {num_trucks}\n")
            f.write(f"Successful Trucks: {successful_trucks}/{num_trucks}\n")
            f.write(f"Failed Trucks: {failed_trucks}/{num_trucks}\n")
            f.write(f"Total Distance Traveled: {total_distance:.2f} km\n")
            f.write(f"Total Deliveries Completed: {total_deliveries}\n")
            f.write(f"\nCharging Infrastructure:\n")
            f.write(f"  Overall Utilization: {charger_util['overall']['avg_utilization']*100:.1f}%\n")
            f.write(f"  Total Charge Sessions: {charger_util['overall']['total_sessions']}\n")
            f.write(f"  Total Charge Time: {charger_util['overall']['total_charge_time']:.2f} hours\n")
        
        print(f"\nStatistics saved to: {stats_file}")
    
    @staticmethod
    def get_charger_utilization_stats(
        charging_nodes: List[int],
        charger_stats: Dict,
        charger_type: Dict,
        charger_capacity: Dict,
        charger_occupancy: Dict,
        global_clock: float
    ) -> Dict:
        """
        Calculate charging station utilization statistics.
        
        Args:
            charging_nodes: List of charging node IDs
            charger_stats: Dictionary of charger statistics
            charger_type: Dictionary mapping node to charger type
            charger_capacity: Dictionary mapping node to capacity
            charger_occupancy: Dictionary of current charger occupancy
            global_clock: Current simulation time
        
        Returns:
            Dictionary with utilization statistics
        """
        # Update occupancy time for currently occupied chargers
        for node in charging_nodes:
            if len(charger_occupancy[node]) > 0:
                stats = charger_stats[node]
                stats['occupancy_time'] += (global_clock - stats['last_update_time'])
                stats['last_update_time'] = global_clock
        
        # Compile statistics by charger type
        level2_stats = {'nodes': [], 'utilization_rates': [], 'total_sessions': 0, 'total_charge_time': 0}
        dcfast_stats = {'nodes': [], 'utilization_rates': [], 'total_sessions': 0, 'total_charge_time': 0}
        
        all_chargers = []
        
        for node in charging_nodes:
            stats = charger_stats[node]
            c_type = charger_type[node]
            capacity = charger_capacity[node]
            
            # Calculate utilization rate (time with at least one truck / total time)
            utilization_rate = stats['occupancy_time'] / global_clock if global_clock > 0 else 0.0
            
            charger_info = {
                'node': int(node),
                'type': c_type,
                'capacity': int(capacity),
                'utilization_rate': utilization_rate,
                'sessions': stats['total_charge_sessions'],
                'charge_time': stats['total_charge_time'],
                'trucks_served': len(stats['total_trucks_served']),
                'current_occupancy': len(charger_occupancy[node])
            }
            
            all_chargers.append(charger_info)
            
            # Aggregate by type
            if c_type == 'Level2':
                level2_stats['nodes'].append(int(node))
                level2_stats['utilization_rates'].append(utilization_rate)
                level2_stats['total_sessions'] += stats['total_charge_sessions']
                level2_stats['total_charge_time'] += stats['total_charge_time']
            else:  # DCFast
                dcfast_stats['nodes'].append(int(node))
                dcfast_stats['utilization_rates'].append(utilization_rate)
                dcfast_stats['total_sessions'] += stats['total_charge_sessions']
                dcfast_stats['total_charge_time'] += stats['total_charge_time']
        
        # Calculate average utilization by type
        level2_avg = sum(level2_stats['utilization_rates']) / len(level2_stats['utilization_rates']) if level2_stats['utilization_rates'] else 0.0
        dcfast_avg = sum(dcfast_stats['utilization_rates']) / len(dcfast_stats['utilization_rates']) if dcfast_stats['utilization_rates'] else 0.0
        
        return {
            'all_chargers': all_chargers,
            'level2': {
                'avg_utilization': level2_avg,
                'total_sessions': level2_stats['total_sessions'],
                'total_charge_time': level2_stats['total_charge_time'],
                'num_chargers': len(level2_stats['nodes'])
            },
            'dcfast': {
                'avg_utilization': dcfast_avg,
                'total_sessions': dcfast_stats['total_sessions'],
                'total_charge_time': dcfast_stats['total_charge_time'],
                'num_chargers': len(dcfast_stats['nodes'])
            },
            'overall': {
                'avg_utilization': (level2_avg * len(level2_stats['nodes']) + dcfast_avg * len(dcfast_stats['nodes'])) / len(charging_nodes) if charging_nodes else 0.0,
                'total_sessions': level2_stats['total_sessions'] + dcfast_stats['total_sessions'],
                'total_charge_time': level2_stats['total_charge_time'] + dcfast_stats['total_charge_time']
            }
        }
