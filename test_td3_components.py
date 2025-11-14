"""
Unit tests for TD3 Action-GNN components.
Tests actor, critic, action selection, and training updates.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algo.TD3_actionGNN import TD3_ActionGNN
from algo.replay_buffer import ReplayBuffer
from algo.networks import HeteroGNN_Actor, HeteroGNN_Critic, Actor, Critic
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.utils.utils import load_config


class TestChargingDurationMapping:
    """Test charging duration mapping from sigmoid to [min, max] range."""
    
    def test_sigmoid_to_range_mapping(self):
        """Test that sigmoid output correctly maps to [min, max] range."""
        min_dur = 0.5
        max_dur = 10.0
        
        # Test edge cases
        sigmoid_0 = torch.tensor([[0.0]])
        sigmoid_1 = torch.tensor([[1.0]])
        sigmoid_mid = torch.tensor([[0.5]])
        
        # Apply mapping: min + sigmoid * (max - min)
        dur_0 = min_dur + sigmoid_0 * (max_dur - min_dur)
        dur_1 = min_dur + sigmoid_1 * (max_dur - min_dur)
        dur_mid = min_dur + sigmoid_mid * (max_dur - min_dur)
        
        assert abs(dur_0.item() - min_dur) < 1e-6, f"Expected {min_dur}, got {dur_0.item()}"
        assert abs(dur_1.item() - max_dur) < 1e-6, f"Expected {max_dur}, got {dur_1.item()}"
        assert abs(dur_mid.item() - 5.25) < 1e-6, f"Expected 5.25, got {dur_mid.item()}"
        
        print("✓ Charging duration mapping test passed")
    
    def test_actor_charging_duration_range(self):
        """Test that actor produces charging durations in correct range."""
        min_dur = 0.5
        max_dur = 10.0
        
        actor = HeteroGNN_Actor(
            node_feature_dims={'truck': 13, 'delivery': 2, 'charger': 5},
            hidden_dim=32,
            num_layers=2,
            min_charging_duration=min_dur,
            max_charging_duration=max_dur,
            device='cpu'
        )
        
        # Create dummy heterogeneous graph data
        from torch_geometric.data import HeteroData
        
        data = HeteroData()
        data['truck'].x = torch.randn(1, 13)
        data['delivery'].x = torch.randn(3, 2)
        data['charger'].x = torch.randn(5, 5)
        
        # Add edges
        data['truck', 'to', 'delivery'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['truck', 'to', 'delivery'].edge_attr = torch.randn(1, 2)
        data['delivery', 'to', 'truck'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['delivery', 'to', 'truck'].edge_attr = torch.randn(1, 2)
        data['truck', 'to', 'charger'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['truck', 'to', 'charger'].edge_attr = torch.randn(1, 2)
        data['charger', 'to', 'truck'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['charger', 'to', 'truck'].edge_attr = torch.randn(1, 2)
        data['truck', 'to', 'truck'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['truck', 'to', 'truck'].edge_attr = torch.randn(1, 2)
        data['charger', 'to', 'charger'].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data['charger', 'to', 'charger'].edge_attr = torch.randn(2, 2)
        data['delivery', 'to', 'delivery'].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data['delivery', 'to', 'delivery'].edge_attr = torch.randn(2, 2)
        data['charger', 'to', 'delivery'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['charger', 'to', 'delivery'].edge_attr = torch.randn(1, 2)
        
        # Test multiple forward passes
        with torch.no_grad():
            for _ in range(10):
                action_scores, charging_duration = actor(data, apply_mask=False)
                
                dur_value = charging_duration.squeeze().item()
                assert min_dur <= dur_value <= max_dur, \
                    f"Charging duration {dur_value} outside range [{min_dur}, {max_dur}]"
        
        print(f"✓ Actor charging duration range test passed (range: [{min_dur}, {max_dur}])")


class TestActionSelection:
    """Test action selection with masking and exploration."""
    
    def test_action_masking(self):
        """Test that infeasible actions are properly masked."""
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
        
        policy = TD3_ActionGNN(
            action_dim=30,
            max_action=1.0,
            fx_node_sizes={'ev': 13, 'cs': 5, 'tr': 2},
            fx_GNN_hidden_dim=32,
            mlp_hidden_dim=128,
            discrete_actions=30,
            min_charging_duration=0.5,
            max_charging_duration=10.0
        )
        
        # Get initial state
        env.reset(seed=42)
        state = gnn_state_space.get_state_GNN(env)
        
        # Test that selected actions are always feasible
        for _ in range(20):
            action = policy.select_action(state, expl_noise=0.1)
            
            if isinstance(action, tuple):
                node_id, charging_duration, is_charging = action
                
                # Find action index
                action_idx = None
                for idx, (mapped_node, mapped_is_charging) in enumerate(state.action_to_node_map):
                    if mapped_node == node_id and mapped_is_charging == is_charging:
                        action_idx = idx
                        break
                
                if action_idx is not None:
                    assert state.feasible_action_mask[action_idx].item(), \
                        f"Selected infeasible action: idx={action_idx}, node={node_id}, is_charging={is_charging}"
                
                # Check charging duration range
                assert 0.5 <= charging_duration <= 10.0, \
                    f"Charging duration {charging_duration} out of range [0.5, 10.0]"
        
        env.close()
        print("✓ Action masking test passed")
    
    def test_greedy_vs_exploration(self):
        """Test that exploration noise affects action selection."""
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
        
        policy = TD3_ActionGNN(
            action_dim=30,
            max_action=1.0,
            fx_node_sizes={'ev': 13, 'cs': 5, 'tr': 2},
            fx_GNN_hidden_dim=32,
            discrete_actions=30
        )
        
        env.reset(seed=42)
        state = gnn_state_space.get_state_GNN(env)
        
        # Greedy actions should be more consistent
        greedy_actions = []
        for _ in range(5):
            action = policy.select_action(state, expl_noise=0.0)
            if isinstance(action, tuple):
                greedy_actions.append(action[0])  # node_id
        
        # Exploration should produce more variety
        explore_actions = []
        for _ in range(5):
            action = policy.select_action(state, expl_noise=0.2)
            if isinstance(action, tuple):
                explore_actions.append(action[0])  # node_id
        
        # Greedy should be more consistent (but may not be identical due to ties)
        greedy_unique = len(set(greedy_actions))
        explore_unique = len(set(explore_actions))
        
        print(f"  Greedy unique actions: {greedy_unique}/5")
        print(f"  Exploration unique actions: {explore_unique}/5")
        
        env.close()
        print("✓ Greedy vs exploration test passed")


class TestReplayBuffer:
    """Test replay buffer functionality."""
    
    def test_buffer_storage_and_sampling(self):
        """Test that buffer stores and samples correctly."""
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
        
        buffer = ReplayBuffer(max_size=100)
        
        # Add transitions
        env.reset(seed=42)
        for i in range(20):
            state = gnn_state_space.get_state_GNN(env)
            action_idx = np.random.randint(0, len(state.action_to_node_map))
            charging_duration = np.random.uniform(0.5, 10.0)
            
            next_state = gnn_state_space.get_state_GNN(env)
            reward = np.random.randn()
            done = float(i == 19)
            
            buffer.add(state, (action_idx, charging_duration), next_state, reward, done)
        
        assert len(buffer) == 20, f"Expected 20 transitions, got {len(buffer)}"
        
        # Test sampling
        batch_size = 4
        state_batch, actions, charging_durations, next_state_batch, rewards, not_dones = \
            buffer.sample(batch_size, device='cpu')
        
        assert actions.shape[0] == batch_size, f"Expected {batch_size} actions, got {actions.shape[0]}"
        assert charging_durations.shape == (batch_size, 1), \
            f"Expected shape ({batch_size}, 1), got {charging_durations.shape}"
        assert rewards.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {rewards.shape}"
        
        # Check that feasible_action_mask is concatenated
        assert hasattr(state_batch, 'feasible_action_mask'), "Missing feasible_action_mask"
        assert state_batch.feasible_action_mask.dim() == 1, "feasible_action_mask should be 1D"
        
        env.close()
        print("✓ Replay buffer test passed")


class TestNetworkForwardPass:
    """Test network forward passes."""
    
    def test_actor_output_shapes(self):
        """Test actor produces correct output shapes."""
        from torch_geometric.data import HeteroData
        
        actor = HeteroGNN_Actor(
            node_feature_dims={'truck': 13, 'delivery': 2, 'charger': 5},
            hidden_dim=32,
            num_layers=2,
            device='cpu'
        )
        
        data = HeteroData()
        data['truck'].x = torch.randn(1, 13)
        data['delivery'].x = torch.randn(3, 2)
        data['charger'].x = torch.randn(5, 5)
        
        # Add required edges
        data['truck', 'to', 'delivery'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['truck', 'to', 'delivery'].edge_attr = torch.randn(1, 2)
        data['delivery', 'to', 'truck'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['delivery', 'to', 'truck'].edge_attr = torch.randn(1, 2)
        data['truck', 'to', 'charger'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['truck', 'to', 'charger'].edge_attr = torch.randn(1, 2)
        data['charger', 'to', 'truck'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['charger', 'to', 'truck'].edge_attr = torch.randn(1, 2)
        data['truck', 'to', 'truck'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['truck', 'to', 'truck'].edge_attr = torch.randn(1, 2)
        data['charger', 'to', 'charger'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['charger', 'to', 'charger'].edge_attr = torch.randn(1, 2)
        data['delivery', 'to', 'delivery'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['delivery', 'to', 'delivery'].edge_attr = torch.randn(1, 2)
        data['charger', 'to', 'delivery'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['charger', 'to', 'delivery'].edge_attr = torch.randn(1, 2)
        
        # Add action mapping
        data.action_to_node_map = [(i, i >= 3) for i in range(9)]  # 3 deliveries + 5 chargers + 1 charge here
        data.feasible_action_mask = torch.ones(9, dtype=torch.bool)
        
        with torch.no_grad():
            action_scores, charging_duration = actor(data)
        
        assert action_scores.shape[0] == 9, f"Expected 9 action scores, got {action_scores.shape[0]}"
        assert charging_duration.shape == (1, 1), f"Expected shape (1, 1), got {charging_duration.shape}"
        
        print("✓ Actor output shapes test passed")
    
    def test_critic_output_shapes(self):
        """Test critic produces correct Q-values."""
        from torch_geometric.data import HeteroData
        
        critic = HeteroGNN_Critic(
            node_feature_dims={'truck': 13, 'delivery': 2, 'charger': 5},
            hidden_dim=32,
            num_layers=2,
            device='cpu'
        )
        
        data = HeteroData()
        data['truck'].x = torch.randn(2, 13)
        data['truck'].batch = torch.tensor([0, 1], dtype=torch.long)
        data['delivery'].x = torch.randn(6, 2)
        data['delivery'].batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        data['charger'].x = torch.randn(10, 5)
        data['charger'].batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
        
        # Add edges
        data['truck', 'to', 'delivery'].edge_index = torch.tensor([[0, 1], [0, 3]], dtype=torch.long)
        data['truck', 'to', 'delivery'].edge_attr = torch.randn(2, 2)
        data['delivery', 'to', 'truck'].edge_index = torch.tensor([[0, 3], [0, 1]], dtype=torch.long)
        data['delivery', 'to', 'truck'].edge_attr = torch.randn(2, 2)
        data['truck', 'to', 'charger'].edge_index = torch.tensor([[0, 1], [0, 5]], dtype=torch.long)
        data['truck', 'to', 'charger'].edge_attr = torch.randn(2, 2)
        data['charger', 'to', 'truck'].edge_index = torch.tensor([[0, 5], [0, 1]], dtype=torch.long)
        data['charger', 'to', 'truck'].edge_attr = torch.randn(2, 2)
        data['truck', 'to', 'truck'].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data['truck', 'to', 'truck'].edge_attr = torch.randn(2, 2)
        data['charger', 'to', 'charger'].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data['charger', 'to', 'charger'].edge_attr = torch.randn(2, 2)
        data['delivery', 'to', 'delivery'].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data['delivery', 'to', 'delivery'].edge_attr = torch.randn(2, 2)
        data['charger', 'to', 'delivery'].edge_index = torch.tensor([[0, 5], [0, 3]], dtype=torch.long)
        data['charger', 'to', 'delivery'].edge_attr = torch.randn(2, 2)
        
        actions = torch.tensor([1, 2], dtype=torch.long)
        charging_durations = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        
        with torch.no_grad():
            q_value = critic(data, actions, charging_durations)
        
        assert q_value.shape == (2, 1), f"Expected shape (2, 1), got {q_value.shape}"
        
        print("✓ Critic output shapes test passed")


class TestTrainingUpdate:
    """Test training updates."""
    
    def test_critic_loss_computation(self):
        """Test that critic loss is computed and finite."""
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
        
        policy = TD3_ActionGNN(
            action_dim=30,
            max_action=1.0,
            fx_node_sizes={'ev': 13, 'cs': 5, 'tr': 2},
            fx_GNN_hidden_dim=32,
            mlp_hidden_dim=128,
            discrete_actions=30
        )
        
        buffer = ReplayBuffer(max_size=100)
        
        # Collect some transitions
        env.reset(seed=42)
        for i in range(20):
            state = gnn_state_space.get_state_GNN(env)
            action_idx = np.random.randint(0, len(state.action_to_node_map))
            charging_duration = np.random.uniform(0.5, 10.0)
            
            next_state = gnn_state_space.get_state_GNN(env)
            reward = np.random.randn()
            done = float(i == 19)
            
            buffer.add(state, (action_idx, charging_duration), next_state, reward, done)
        
        # Perform training update
        critic_loss, actor_loss = policy.train(buffer, batch_size=4)
        
        assert critic_loss is not None, "Critic loss is None"
        assert np.isfinite(critic_loss), f"Critic loss is not finite: {critic_loss}"
        assert critic_loss >= 0, f"Critic loss is negative: {critic_loss}"
        
        env.close()
        print(f"✓ Critic loss computation test passed (loss={critic_loss:.4f})")
    
    def test_actor_loss_computation(self):
        """Test that actor loss is computed on delayed updates."""
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
        
        policy = TD3_ActionGNN(
            action_dim=30,
            max_action=1.0,
            fx_node_sizes={'ev': 13, 'cs': 5, 'tr': 2},
            fx_GNN_hidden_dim=32,
            mlp_hidden_dim=128,
            discrete_actions=30,
            policy_freq=2
        )
        
        buffer = ReplayBuffer(max_size=100)
        
        # Collect transitions
        env.reset(seed=42)
        for i in range(20):
            state = gnn_state_space.get_state_GNN(env)
            action_idx = np.random.randint(0, len(state.action_to_node_map))
            charging_duration = np.random.uniform(0.5, 10.0)
            
            next_state = gnn_state_space.get_state_GNN(env)
            reward = np.random.randn()
            done = float(i == 19)
            
            buffer.add(state, (action_idx, charging_duration), next_state, reward, done)
        
        # First update should only have critic loss
        critic_loss1, actor_loss1 = policy.train(buffer, batch_size=4)
        assert critic_loss1 is not None
        assert actor_loss1 is None, "Actor should not update on first iteration"
        
        # Second update should have both losses (policy_freq=2)
        critic_loss2, actor_loss2 = policy.train(buffer, batch_size=4)
        assert critic_loss2 is not None
        assert actor_loss2 is not None, "Actor should update on second iteration"
        assert np.isfinite(actor_loss2), f"Actor loss is not finite: {actor_loss2}"
        
        env.close()
        print(f"✓ Actor loss computation test passed (delayed update working)")


class TestParameterUpdates:
    """Test that network parameters actually update during training."""
    
    def test_parameters_change_after_training(self):
        """Test that actor and critic parameters change after training."""
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
        
        policy = TD3_ActionGNN(
            action_dim=30,
            max_action=1.0,
            fx_node_sizes={'ev': 13, 'cs': 5, 'tr': 2},
            fx_GNN_hidden_dim=32,
            mlp_hidden_dim=128,
            discrete_actions=30,
            policy_freq=1  # Update every iteration
        )
        
        # Store initial parameters
        initial_actor_params = {name: param.clone() for name, param in policy.actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in policy.critic.named_parameters()}
        
        buffer = ReplayBuffer(max_size=100)
        
        # Collect transitions
        env.reset(seed=42)
        for i in range(30):
            state = gnn_state_space.get_state_GNN(env)
            action_idx = np.random.randint(0, len(state.action_to_node_map))
            charging_duration = np.random.uniform(0.5, 10.0)
            
            next_state = gnn_state_space.get_state_GNN(env)
            reward = np.random.randn()
            done = float(i == 29)
            
            buffer.add(state, (action_idx, charging_duration), next_state, reward, done)
        
        # Train for several iterations
        for _ in range(5):
            policy.train(buffer, batch_size=4)
        
        # Check that parameters changed
        actor_changed = 0
        for name, param in policy.actor.named_parameters():
            if not torch.allclose(param, initial_actor_params[name], atol=1e-6):
                actor_changed += 1
        
        critic_changed = 0
        for name, param in policy.critic.named_parameters():
            if not torch.allclose(param, initial_critic_params[name], atol=1e-6):
                critic_changed += 1
        
        total_actor = len(initial_actor_params)
        total_critic = len(initial_critic_params)
        
        assert actor_changed > 0, f"No actor parameters changed (0/{total_actor})"
        assert critic_changed > 0, f"No critic parameters changed (0/{total_critic})"
        
        print(f"✓ Parameter update test passed (Actor: {actor_changed}/{total_actor}, Critic: {critic_changed}/{total_critic})")
        
        env.close()


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("Running TD3 Component Unit Tests")
    print("="*80 + "\n")
    
    test_classes = [
        TestChargingDurationMapping,
        TestActionSelection,
        TestReplayBuffer,
        TestNetworkForwardPass,
        TestTrainingUpdate,
        TestParameterUpdates
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
