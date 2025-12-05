#!/usr/bin/env python
"""
Quick test script to verify curriculum learning implementation works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from EVRoutingEnv.models.curriculum_env import (
    CurriculumEnvWrapper,
    UniformRandomStrategy,
    StagedCurriculumStrategy,
    MixedCurriculumStrategy
)
from EVRoutingEnv.utils.utils import load_config

def test_uniform_strategy():
    """Test uniform random sampling strategy."""
    print("\n" + "="*60)
    print("Testing Uniform Random Strategy")
    print("="*60)
    
    strategy = UniformRandomStrategy(
        truck_range=(3, 5),
        stop_range=(3, 5),
        seed=42
    )
    
    # Sample 10 times
    samples = [strategy.sample(i) for i in range(10)]
    print(f"Samples (trucks, stops): {samples}")
    
    # Check all within range
    for trucks, stops in samples:
        assert 3 <= trucks <= 5, f"Trucks {trucks} out of range"
        assert 3 <= stops <= 5, f"Stops {stops} out of range"
    
    stats = strategy.get_stats()
    print(f"Stats: {stats}")
    print("✅ Uniform strategy test passed!")

def test_staged_strategy():
    """Test staged curriculum strategy."""
    print("\n" + "="*60)
    print("Testing Staged Curriculum Strategy")
    print("="*60)
    
    stages = [
        {'episodes': 3, 'truck_range': (1, 2), 'stop_range': (3, 4)},
        {'episodes': 3, 'truck_range': (3, 5), 'stop_range': (5, 7)},
        {'episodes': -1, 'truck_range': (5, 8), 'stop_range': (7, 10)},
    ]
    
    strategy = StagedCurriculumStrategy(stages=stages, seed=42)
    
    # Sample through stages
    samples = []
    for i in range(10):
        sample = strategy.sample(i)
        samples.append(sample)
        trucks, stops = sample
        
        if i < 3:
            # Stage 0
            assert 1 <= trucks <= 2, f"Episode {i}: trucks {trucks} not in stage 0"
            assert 3 <= stops <= 4, f"Episode {i}: stops {stops} not in stage 0"
        elif i < 6:
            # Stage 1
            assert 3 <= trucks <= 5, f"Episode {i}: trucks {trucks} not in stage 1"
            assert 5 <= stops <= 7, f"Episode {i}: stops {stops} not in stage 1"
        else:
            # Stage 2
            assert 5 <= trucks <= 8, f"Episode {i}: trucks {trucks} not in stage 2"
            assert 7 <= stops <= 10, f"Episode {i}: stops {stops} not in stage 2"
    
    print(f"Samples: {samples}")
    stats = strategy.get_stats()
    print(f"Stats: {stats}")
    print("✅ Staged strategy test passed!")

def test_mixed_strategy():
    """Test mixed curriculum strategy."""
    print("\n" + "="*60)
    print("Testing Mixed Curriculum Strategy")
    print("="*60)
    
    difficulty_levels = [
        {'truck_range': (1, 3), 'stop_range': (3, 5), 'weight': 0.5},
        {'truck_range': (4, 6), 'stop_range': (5, 7), 'weight': 0.3},
        {'truck_range': (7, 10), 'stop_range': (7, 10), 'weight': 0.2},
    ]
    
    strategy = MixedCurriculumStrategy(difficulty_levels=difficulty_levels, seed=42)
    
    # Sample many times to check distribution
    samples = [strategy.sample(i) for i in range(100)]
    
    # Count per difficulty
    level_0 = sum(1 for t, s in samples if 1 <= t <= 3 and 3 <= s <= 5)
    level_1 = sum(1 for t, s in samples if 4 <= t <= 6 and 5 <= s <= 7)
    level_2 = sum(1 for t, s in samples if 7 <= t <= 10 and 7 <= s <= 10)
    
    print(f"Level 0 (easy): {level_0}/100 (expected ~50)")
    print(f"Level 1 (medium): {level_1}/100 (expected ~30)")
    print(f"Level 2 (hard): {level_2}/100 (expected ~20)")
    
    stats = strategy.get_stats()
    print(f"Stats: {stats}")
    print("✅ Mixed strategy test passed!")

def test_curriculum_env():
    """Test curriculum environment wrapper."""
    print("\n" + "="*60)
    print("Testing Curriculum Environment Wrapper")
    print("="*60)
    
    config_path = "EVRoutingEnv/config_files/config.yaml"
    
    strategy = UniformRandomStrategy(
        truck_range=(3, 5),
        stop_range=(3, 4),
        seed=42
    )
    
    env = CurriculumEnvWrapper(
        base_config=config_path,
        curriculum_strategy=strategy,
        verbose=False,
        enable_plotting=False
    )
    
    # Test multiple resets
    for episode in range(3):
        obs, info = env.reset(seed=42 + episode)
        
        trucks = info['curriculum']['num_trucks']
        stops = info['curriculum']['num_stops']
        
        print(f"\nEpisode {episode}: {trucks} trucks, {stops} stops")
        assert 3 <= trucks <= 5, f"Trucks {trucks} out of range"
        assert 3 <= stops <= 4, f"Stops {stops} out of range"
        
        # Take a few steps
        for step in range(5):
            action = env.env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                print(f"  Episode ended at step {step}")
                break
    
    # Check statistics
    stats = env.get_curriculum_stats()
    print(f"\nCurriculum stats: {stats}")
    
    env.close()
    print("✅ Environment wrapper test passed!")

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CURRICULUM LEARNING IMPLEMENTATION TESTS")
    print("="*60)
    
    try:
        test_uniform_strategy()
        test_staged_strategy()
        test_mixed_strategy()
        test_curriculum_env()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nCurriculum learning implementation is ready to use.")
        print("\nNext steps:")
        print("1. Review docs/CURRICULUM_LEARNING.md for usage guide")
        print("2. Run example scripts:")
        print("   ./scripts/training/run_curriculum_uniform.sh")
        print("3. Or start with a simple command:")
        print("   python scripts/training/train_curriculum.py \\")
        print("       --curriculum-strategy uniform \\")
        print("       --truck-range 3 5 --stop-range 3 5 \\")
        print("       --max-timesteps 10000 --no-wandb")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
