"""
Test script to visualize the action graph for trucks.

Shows the feasible actions for each truck based on their current state and battery level.
"""

import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/sorfanouda/EVPR')

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.baselines.heuristic_policy import HeuristicPolicy
from visualize_gnn_state import GNNVisualizer


def test_action_graph():
    """Test the action graph visualization."""
    print("\n" + "="*80)
    print("ACTION GRAPH VISUALIZATION TEST")
    print("="*80)
    
    # Initialize environment
    config_file = "truck_env/config_files/config.yaml"
    env = EventDrivenTruckEnv(
        config=config_file,
        run_id="action_graph_test",
        verbose=True,
        enable_plotting=False
    )
    
    # Create GNN state space
    gnn_state = GNNStateSpace(
        num_trucks=env.num_trucks,
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
    )
    
    # Create visualizer
    visualizer = GNNVisualizer()
    
    # Create policy for taking actions
    policy = HeuristicPolicy(verbose=False)
    
    # Reset environment
    env.reset(seed=42)
    
    # Test 1: Initial state (trucks at depot, READY)
    print("\n" + "="*80)
    print("TEST 1: INITIAL STATE - Trucks at Depot (READY)")
    print("="*80)
    
    data = gnn_state.get_state_GNN(env)
    fig1 = visualizer.plot_action_graph(data, env, title="Initial State - READY Trucks")
    if fig1:
        fig1.savefig('gnn_plots/action_graph_1_ready.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: gnn_plots/action_graph_1_ready.png")
        plt.close(fig1)
    
    # Test 2: After taking action (truck ROUTING)
    print("\n" + "="*80)
    print("TEST 2: AFTER TAKING ACTION - Truck Routing")
    print("="*80)
    
    action = policy.get_action(env)
    env.step(action)
    
    data = gnn_state.get_state_GNN(env)
    fig2 = visualizer.plot_action_graph(data, env, title="After Action - ROUTING Truck")
    if fig2:
        fig2.savefig('gnn_plots/action_graph_2_routing.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: gnn_plots/action_graph_2_routing.png")
        plt.close(fig2)
    
    # Test 3: Take a few more steps
    print("\n" + "="*80)
    print("TEST 3: MULTIPLE STEPS - Various States")
    print("="*80)
    
    for step in range(3, 6):
        action = policy.get_action(env)
        if action is None:
            print("No action available")
            break
        
        env.step(action)
        
        data = gnn_state.get_state_GNN(env)
        fig = visualizer.plot_action_graph(data, env, title=f"Step {step} - Mixed States")
        if fig:
            fig.savefig(f'gnn_plots/action_graph_{step}_mixed.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: gnn_plots/action_graph_{step}_mixed.png")
            plt.close(fig)
    
    print("\n" + "="*80)
    print("✅ ACTION GRAPH TESTS COMPLETE")
    print("="*80)
    print("\nGenerated action graph visualizations:")
    print("  - gnn_plots/action_graph_1_ready.png (READY state)")
    print("  - gnn_plots/action_graph_2_routing.png (ROUTING state)")
    print("  - gnn_plots/action_graph_3_mixed.png (Mixed states)")
    print("  - gnn_plots/action_graph_4_mixed.png (Mixed states)")
    print("  - gnn_plots/action_graph_5_mixed.png (Mixed states)")
    print("\nAction graphs show:")
    print("  • Truck at center with current battery level")
    print("  • Next delivery node (if in READY state)")
    print("  • All reachable chargers with current energy")
    print("  • Edge labels showing energy (kWh) and time (hours)")
    print("  • Current truck state (READY/ROUTING/CHARGING/WAITING)")
    

if __name__ == "__main__":
    test_action_graph()
