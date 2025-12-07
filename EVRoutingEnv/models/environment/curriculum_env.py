"""
Curriculum Learning Environment Wrapper for Variable Problem Sizes.

This wrapper enables training on episodes with varying numbers of trucks and stops,
supporting multiple curriculum strategies for robust generalization.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
import copy
from collections import defaultdict

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


class CurriculumStrategy:
    """Base class for curriculum sampling strategies."""
    
    def sample(self, episode: int) -> Tuple[int, int]:
        """Sample (num_trucks, num_stops) for the given episode number."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        """Return statistics about the curriculum."""
        return {}


class UniformRandomStrategy(CurriculumStrategy):
    """Uniformly sample from specified ranges every episode."""
    
    def __init__(
        self,
        truck_range: Tuple[int, int],
        stop_range: Tuple[int, int],
        seed: Optional[int] = None
    ):
        """
        Args:
            truck_range: (min_trucks, max_trucks) inclusive
            stop_range: (min_stops, max_stops) inclusive
            seed: Random seed for reproducibility
        """
        self.truck_range = truck_range
        self.stop_range = stop_range
        self.rng = np.random.RandomState(seed)
        
        # Track sampling history
        self.sample_history = defaultdict(int)
    
    def sample(self, episode: int) -> Tuple[int, int]:
        """Sample uniformly from ranges."""
        num_trucks = self.rng.randint(self.truck_range[0], self.truck_range[1] + 1)
        num_stops = self.rng.randint(self.stop_range[0], self.stop_range[1] + 1)
        
        # Track this sample
        self.sample_history[(num_trucks, num_stops)] += 1
        
        return num_trucks, num_stops
    
    def get_stats(self) -> Dict:
        """Return sampling statistics."""
        return {
            'truck_range': self.truck_range,
            'stop_range': self.stop_range,
            'unique_configs': len(self.sample_history),
            'total_samples': sum(self.sample_history.values()),
        }


class StagedCurriculumStrategy(CurriculumStrategy):
    """Gradually increase problem difficulty over training."""
    
    def __init__(
        self,
        stages: List[Dict],
        seed: Optional[int] = None
    ):
        """
        Args:
            stages: List of stage configs, each with:
                - 'episodes': number of episodes for this stage
                - 'truck_range': (min, max) trucks
                - 'stop_range': (min, max) stops
            seed: Random seed
        
        Example:
            stages = [
                {'episodes': 1000, 'truck_range': (1, 3), 'stop_range': (3, 5)},
                {'episodes': 2000, 'truck_range': (3, 6), 'stop_range': (5, 8)},
                {'episodes': -1, 'truck_range': (5, 10), 'stop_range': (8, 10)},  # -1 = forever
            ]
        """
        self.stages = stages
        self.rng = np.random.RandomState(seed)
        self.current_stage = 0
        self.stage_episode_count = 0
        
        # Validate stages
        for stage in stages:
            assert 'episodes' in stage and 'truck_range' in stage and 'stop_range' in stage
    
    def sample(self, episode: int) -> Tuple[int, int]:
        """Sample based on current curriculum stage."""
        # Determine current stage
        if self.stage_episode_count >= self.stages[self.current_stage]['episodes'] > 0:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.stage_episode_count = 0
        
        stage = self.stages[self.current_stage]
        num_trucks = self.rng.randint(stage['truck_range'][0], stage['truck_range'][1] + 1)
        num_stops = self.rng.randint(stage['stop_range'][0], stage['stop_range'][1] + 1)
        
        self.stage_episode_count += 1
        
        return num_trucks, num_stops
    
    def get_stats(self) -> Dict:
        """Return curriculum stage information."""
        return {
            'current_stage': self.current_stage,
            'stage_episode_count': self.stage_episode_count,
            'total_stages': len(self.stages),
        }


class MixedCurriculumStrategy(CurriculumStrategy):
    """Sample from multiple difficulty levels with specified weights."""
    
    def __init__(
        self,
        difficulty_levels: List[Dict],
        seed: Optional[int] = None
    ):
        """
        Args:
            difficulty_levels: List of difficulty configs, each with:
                - 'truck_range': (min, max) trucks
                - 'stop_range': (min, max) stops
                - 'weight': sampling weight (will be normalized)
            seed: Random seed
        
        Example:
            difficulty_levels = [
                {'truck_range': (1, 3), 'stop_range': (3, 5), 'weight': 0.3},  # Easy
                {'truck_range': (4, 7), 'stop_range': (5, 8), 'weight': 0.5},  # Medium
                {'truck_range': (8, 10), 'stop_range': (8, 10), 'weight': 0.2},  # Hard
            ]
        """
        self.difficulty_levels = difficulty_levels
        self.rng = np.random.RandomState(seed)
        
        # Normalize weights
        total_weight = sum(level['weight'] for level in difficulty_levels)
        self.probabilities = [level['weight'] / total_weight for level in difficulty_levels]
        
        # Track sampling per difficulty
        self.difficulty_counts = [0] * len(difficulty_levels)
    
    def sample(self, episode: int) -> Tuple[int, int]:
        """Sample from weighted difficulty levels."""
        # Select difficulty level
        level_idx = self.rng.choice(len(self.difficulty_levels), p=self.probabilities)
        level = self.difficulty_levels[level_idx]
        
        # Sample within that level
        num_trucks = self.rng.randint(level['truck_range'][0], level['truck_range'][1] + 1)
        num_stops = self.rng.randint(level['stop_range'][0], level['stop_range'][1] + 1)
        
        self.difficulty_counts[level_idx] += 1
        
        return num_trucks, num_stops
    
    def get_stats(self) -> Dict:
        """Return sampling statistics per difficulty level."""
        return {
            'difficulty_counts': self.difficulty_counts,
            'probabilities': self.probabilities,
        }


class CurriculumEnvWrapper:
    """
    Environment wrapper that samples varying problem sizes each episode.
    
    This wrapper recreates the base environment with new num_trucks/num_stops
    at each reset, enabling curriculum learning across problem sizes.
    """
    
    def __init__(
        self,
        base_config: Union[str, Dict],
        curriculum_strategy: CurriculumStrategy,
        verbose: bool = False,
        enable_plotting: bool = False,
    ):
        """
        Args:
            base_config: Base configuration file or dict
            curriculum_strategy: Strategy for sampling problem sizes
            verbose: Print detailed information
            enable_plotting: Enable visualization
        """
        self.base_config = base_config if isinstance(base_config, dict) else None
        self.config_path = base_config if isinstance(base_config, str) else None
        self.curriculum_strategy = curriculum_strategy
        self.verbose = verbose
        self.enable_plotting = enable_plotting
        
        # Track curriculum statistics
        self.episode_count = 0
        self.size_performance = defaultdict(lambda: {'rewards': [], 'success': []})
        
        # Current environment and its configuration
        self.env = None
        self.current_num_trucks = None
        self.current_num_stops = None
        
        # Create initial environment with first sampled size
        self._create_new_env(0, 0)
    
    def _create_new_env(self, num_trucks: int, num_stops: int):
        """Create environment with specified problem size."""
        # Load or copy base config
        if self.config_path:
            from EVRoutingEnv.utils.utils import load_config
            config = load_config(self.config_path)
        else:
            config = copy.deepcopy(self.base_config)
        
        # Override with curriculum-sampled sizes
        config['environment']['num_trucks'] = num_trucks
        config['environment']['num_stops'] = num_stops
        
        # Update max_episode_steps based on problem size
        config['environment']['max_episode_steps'] = int(num_trucks * num_stops * 7.5)
        
        # Close old environment if it exists
        if self.env is not None:
            self.env.close()
        
        # Create new environment
        run_id = f"curriculum_ep{self.episode_count}_{num_trucks}t_{num_stops}s"
        self.env = EventDrivenTruckEnv(
            config=config,
            verbose=self.verbose,
            enable_plotting=self.enable_plotting,
            run_id=run_id
        )
        
        self.current_num_trucks = num_trucks
        self.current_num_stops = num_stops
        
        if self.verbose:
            print(f"\n[Curriculum] Episode {self.episode_count}: "
                  f"{num_trucks} trucks, {num_stops} stops")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple:
        """Reset with new problem size sampled from curriculum."""
        # Sample new problem size
        num_trucks, num_stops = self.curriculum_strategy.sample(self.episode_count)
        
        # Create new environment if size changed
        if (num_trucks != self.current_num_trucks or 
            num_stops != self.current_num_stops):
            self._create_new_env(num_trucks, num_stops)
        
        # Reset the environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Add curriculum info
        info['curriculum'] = {
            'num_trucks': num_trucks,
            'num_stops': num_stops,
            'episode': self.episode_count,
        }
        
        return obs, info
    
    def step(self, action):
        """Forward step to underlying environment."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Track performance by problem size
        if done or truncated:
            size_key = (self.current_num_trucks, self.current_num_stops)
            self.size_performance[size_key]['rewards'].append(
                info.get('episode_reward', reward)
            )
            self.size_performance[size_key]['success'].append(
                1.0 if info.get('all_complete', False) else 0.0
            )
            
            self.episode_count += 1
        
        # Add curriculum info
        info['curriculum'] = {
            'num_trucks': self.current_num_trucks,
            'num_stops': self.current_num_stops,
            'episode': self.episode_count,
        }
        
        return obs, reward, done, truncated, info
    
    def get_curriculum_stats(self) -> Dict:
        """Get comprehensive curriculum learning statistics."""
        stats = {
            'total_episodes': self.episode_count,
            'strategy_stats': self.curriculum_strategy.get_stats(),
            'performance_by_size': {}
        }
        
        # Aggregate performance by problem size
        for size_key, perf in self.size_performance.items():
            num_trucks, num_stops = size_key
            if perf['rewards']:
                stats['performance_by_size'][f"{num_trucks}t_{num_stops}s"] = {
                    'mean_reward': np.mean(perf['rewards']),
                    'std_reward': np.std(perf['rewards']),
                    'success_rate': np.mean(perf['success']),
                    'num_episodes': len(perf['rewards']),
                }
        
        return stats
    
    def close(self):
        """Close the underlying environment."""
        if self.env is not None:
            self.env.close()
    
    # Proxy properties to underlying environment
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def num_charging_nodes(self):
        return self.env.num_charging_nodes
    
    @property
    def trucks(self):
        return self.env.trucks
    
    @property
    def active_truck_id(self):
        return self.env.active_truck_id
    
    @property
    def transport_graph(self):
        return self.env.transport_graph
    
    @property
    def charging_nodes(self):
        return self.env.charging_nodes
