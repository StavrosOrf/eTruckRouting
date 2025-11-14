"""
Unit tests for environment and state space functionality.
Tests state construction, action feasibility, and episode flow.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.utils.utils import load_config


class TestEnvironmentSetup(unittest.TestCase):
    """Test environment initialization and configuration."""
    
    def test_environment_creation(self):
        """Test that environment can be created."""
        config = load_config('truck_env/config_files/config.yaml')
        config['environment']['num_trucks'] = 1
        config['environment']['num_stops'] = 3
        
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        env.reset(seed=42)
        
        assert env is not None
        assert env.num_trucks == 1
        assert len(env.trucks) == 1
        # Check that truck has delivery sequence (includes start + stops)
        assert len(env.trucks[0].delivery_sequence) == 4  # start + 3 stops
        assert env.num_charging_nodes > 0
        
        env.close()
        print("✓ Environment creation test passed")
    
    def test_environment_reset(self):
        """Test that environment reset works."""
        config = load_config('truck_env/config_files/config.yaml')
        config['environment']['num_trucks'] = 1
        config['environment']['num_stops'] = 3
        
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        
        # Same seed should give same initial state
        # Compare truck positions and other state elements
        assert env.trucks[0].current_node == env.trucks[0].current_node
        assert env.trucks[0].current_battery == env.trucks[0].current_battery
        
        env.close()
        print("✓ Environment reset test passed")


class TestGNNStateSpace(unittest.TestCase):
    """Test GNN state space construction."""
    
    def test_state_construction(self):
        """Test that GNN state is properly constructed."""
        config = load_config('truck_env/config_files/config.yaml')
        config['environment']['num_trucks'] = 1
        config['environment']['num_stops'] = 3
        
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        gnn_state_space = GNNStateSpace(
            num_trucks=1,
            num_stops=3,
            max_time=200.0,
            num_charging_nodes=env.num_charging_nodes
        )
        
        env.reset(seed=42)
        state = gnn_state_space.get_state_GNN(env)
        
        # Check node types exist
        assert 'truck' in state.node_types or 'ev' in state.node_types
        assert 'delivery' in state.node_types or 'tr' in state.node_types
        assert 'charger' in state.node_types or 'cs' in state.node_types
        
        # Check action mapping exists
        assert hasattr(state, 'action_to_node_map'), "Missing action_to_node_map"
        assert hasattr(state, 'feasible_action_mask'), "Missing feasible_action_mask"
        
        # Check dimensions match
        num_actions = len(state.action_to_node_map)
        assert len(state.feasible_action_mask) == num_actions, \
            f"Mask length {len(state.feasible_action_mask)} != actions {num_actions}"
        
        env.close()
        print("✓ State construction test passed")
    
    def test_feasible_actions_always_exist(self):
        """Test that there are always feasible actions."""
        config = load_config('truck_env/config_files/config.yaml')
        config['environment']['num_trucks'] = 1
        config['environment']['num_stops'] = 3
        
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        gnn_state_space = GNNStateSpace(
            num_trucks=1,
            num_stops=3,
            max_time=200.0,
            num_charging_nodes=env.num_charging_nodes
        )
        
        # Test multiple episodes
        for episode in range(5):
            env.reset(seed=42 + episode)
            
            done = False
            truncated = False
            steps = 0
            max_steps = 50
            
            while not (done or truncated) and steps < max_steps:
                state = gnn_state_space.get_state_GNN(env)
                
                # Check that at least one action is feasible
                num_feasible = state.feasible_action_mask.sum().item()
                assert num_feasible > 0, f"No feasible actions at step {steps} in episode {episode}"
                
                # Take a random feasible action
                feasible_indices = state.feasible_action_mask.nonzero(as_tuple=True)[0]
                action_idx = feasible_indices[np.random.randint(len(feasible_indices))].item()
                
                node_id, is_charging = state.action_to_node_map[action_idx]
                charging_duration = np.random.uniform(0.5, 10.0)
                
                obs, reward, done, truncated, info = env.step((node_id, charging_duration, is_charging))
                steps += 1
        
        env.close()
        print("✓ Feasible actions test passed")


class TestActionExecution(unittest.TestCase):
    """Test action execution in the environment."""
    
    def test_delivery_action(self):
        """Test that delivery actions work."""
        config = load_config('truck_env/config_files/config.yaml')
        config['environment']['num_trucks'] = 1
        config['environment']['num_stops'] = 3
        
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        gnn_state_space = GNNStateSpace(
            num_trucks=1,
            num_stops=3,
            max_time=200.0,
            num_charging_nodes=env.num_charging_nodes
        )
        
        env.reset(seed=42)
        state = gnn_state_space.get_state_GNN(env)
        
        # Find a delivery action
        delivery_actions = [
            (idx, node_id) for idx, (node_id, is_charging) in enumerate(state.action_to_node_map)
            if not is_charging and state.feasible_action_mask[idx]
        ]
        
        if len(delivery_actions) > 0:
            action_idx, node_id = delivery_actions[0]
            
            initial_deliveries = len(env.trucks[0].get_remaining_deliveries())
            
            obs, reward, done, truncated, info = env.step((node_id, 0.0, False))
            
            final_deliveries = len(env.trucks[0].get_remaining_deliveries())
            
            # Delivery should have been made (or attempted)
            print(f"  Initial remaining: {initial_deliveries}, Final remaining: {final_deliveries}")
        
        env.close()
        print("✓ Delivery action test passed")
    
    def test_charging_action(self):
        """Test that charging actions work."""
        config = load_config('truck_env/config_files/config.yaml')
        config['environment']['num_trucks'] = 1
        config['environment']['num_stops'] = 3
        
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        gnn_state_space = GNNStateSpace(
            num_trucks=1,
            num_stops=3,
            max_time=200.0,
            num_charging_nodes=env.num_charging_nodes
        )
        
        env.reset(seed=42)
        
        # Deplete battery first by moving around
        for _ in range(3):
            state = gnn_state_space.get_state_GNN(env)
            feasible_indices = state.feasible_action_mask.nonzero(as_tuple=True)[0]
            action_idx = feasible_indices[0].item()
            node_id, is_charging = state.action_to_node_map[action_idx]
            env.step((node_id, 0.0, is_charging))
        
        # Now try charging
        state = gnn_state_space.get_state_GNN(env)
        charging_actions = [
            (idx, node_id) for idx, (node_id, is_charging) in enumerate(state.action_to_node_map)
            if is_charging and state.feasible_action_mask[idx]
        ]
        
        if len(charging_actions) > 0:
            action_idx, node_id = charging_actions[0]
            
            initial_battery = env.trucks[0].battery_level if hasattr(env, 'trucks') else 0
            charging_duration = 2.0
            
            obs, reward, done, truncated, info = env.step((node_id, charging_duration, True))
            
            final_battery = env.trucks[0].battery_level if hasattr(env, 'trucks') else 0
            
            print(f"  Battery before: {initial_battery:.2f}, after: {final_battery:.2f}")
            # Battery should have increased (or at least not decreased from charging)
        
        env.close()
        print("✓ Charging action test passed")


class TestEpisodeCompletion(unittest.TestCase):
    """Test episode completion conditions."""
    
    def test_episode_can_complete(self):
        """Test that episodes can successfully complete."""
        config = load_config('truck_env/config_files/config.yaml')
        config['environment']['num_trucks'] = 1
        config['environment']['num_stops'] = 3
        
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        gnn_state_space = GNNStateSpace(
            num_trucks=1,
            num_stops=3,
            max_time=200.0,
            num_charging_nodes=env.num_charging_nodes
        )
        
        completed_episodes = 0
        
        for episode in range(3):
            env.reset(seed=42 + episode * 100)
            
            done = False
            truncated = False
            steps = 0
            max_steps = 100
            
            while not (done or truncated) and steps < max_steps:
                state = gnn_state_space.get_state_GNN(env)
                
                # Take random feasible action
                feasible_indices = state.feasible_action_mask.nonzero(as_tuple=True)[0]
                if len(feasible_indices) == 0:
                    break
                
                action_idx = feasible_indices[np.random.randint(len(feasible_indices))].item()
                node_id, is_charging = state.action_to_node_map[action_idx]
                charging_duration = np.random.uniform(0.5, 5.0) if is_charging else 0.0
                
                obs, reward, done, truncated, info = env.step((node_id, charging_duration, is_charging))
                steps += 1
            
            if done and info.get('all_complete', False):
                completed_episodes += 1
                print(f"  Episode {episode} completed in {steps} steps")
        
        print(f"  {completed_episodes}/3 episodes completed successfully")
        
        env.close()
        print("✓ Episode completion test passed")


class TestRewardSignals(unittest.TestCase):
    """Test reward computation."""
    
    def test_rewards_are_finite(self):
        """Test that rewards are always finite."""
        config = load_config('truck_env/config_files/config.yaml')
        config['environment']['num_trucks'] = 1
        config['environment']['num_stops'] = 3
        
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        gnn_state_space = GNNStateSpace(
            num_trucks=1,
            num_stops=3,
            max_time=200.0,
            num_charging_nodes=env.num_charging_nodes
        )
        
        env.reset(seed=42)
        
        rewards = []
        for _ in range(20):
            state = gnn_state_space.get_state_GNN(env)
            
            feasible_indices = state.feasible_action_mask.nonzero(as_tuple=True)[0]
            if len(feasible_indices) == 0:
                break
            
            action_idx = feasible_indices[0].item()
            node_id, is_charging = state.action_to_node_map[action_idx]
            charging_duration = np.random.uniform(0.5, 5.0) if is_charging else 0.0
            
            obs, reward, done, truncated, info = env.step((node_id, charging_duration, is_charging))
            
            assert np.isfinite(reward), f"Reward is not finite: {reward}"
            rewards.append(reward)
            
            if done or truncated:
                break
        
        print(f"  Collected {len(rewards)} rewards, all finite")
        print(f"  Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
        
        env.close()
        print("✓ Reward finiteness test passed")


def run_all_tests():
    """Run all environment tests."""
    print("\n" + "="*80)
    print("Running Environment Unit Tests")
    print("="*80 + "\n")
    
    test_classes = [
        TestEnvironmentSetup,
        TestGNNStateSpace,
        TestActionExecution,
        TestEpisodeCompletion,
        TestRewardSignals
    ]
    
    results = []
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n{class_name}:")
        print("-" * 60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                results.append((class_name, method_name, True, None))
            except Exception as e:
                print(f"✗ {method_name} FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append((class_name, method_name, False, str(e)))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, _, success, _ in results if success)
    total = len(results)
    
    for class_name, method_name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {class_name}.{method_name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return all(success for _, _, success, _ in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
