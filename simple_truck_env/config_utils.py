"""
Configuration utilities for SimpleTruckEnv
"""
import yaml
import os
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        # Use default config in same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract environment initialization parameters from config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with parameters for SimpleTruckEnv.__init__()
    """
    env_config = config.get('environment', {})
    
    return {
        'num_stops': env_config.get('num_stops', 5),
        'min_hop_distance': env_config.get('min_hop_distance', 30.0),
        'max_hop_distance': env_config.get('max_hop_distance', 120.0),
        'max_steps': env_config.get('max_steps', 300),
        'verbose': env_config.get('verbose', False),
    }


def get_reward_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract reward function parameters from config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with reward parameters
    """
    return config.get('rewards', {})


def get_truck_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract truck configuration parameters.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with truck parameters
    """
    return config.get('truck', {})


def get_charging_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract charging configuration parameters.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with charging parameters
    """
    return config.get('charging', {})


def apply_preset(config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """
    Apply a preset configuration.
    
    Args:
        config: Base configuration dictionary
        preset_name: Name of preset to apply ('easy', 'medium', 'hard', 'extreme')
        
    Returns:
        Modified configuration dictionary
    """
    presets = config.get('presets', {})
    
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")
    
    preset = presets[preset_name]
    
    # Update environment section with preset values
    if 'environment' not in config:
        config['environment'] = {}
    
    config['environment'].update(preset)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def print_config_summary(config: Dict[str, Any]):
    """
    Print a human-readable summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("="*70)
    print("SimpleTruckEnv Configuration Summary")
    print("="*70)
    
    # Environment settings
    env = config.get('environment', {})
    print("\n📦 Environment:")
    print(f"   Delivery stops: {env.get('num_stops', 'N/A')}")
    print(f"   Hop distance: {env.get('min_hop_distance', 'N/A')}-{env.get('max_hop_distance', 'N/A')} km")
    print(f"   Max steps: {env.get('max_steps', 'N/A')}")
    print(f"   Verbose: {env.get('verbose', False)}")
    
    # Truck settings
    truck = config.get('truck', {})
    print("\n🚛 Truck:")
    print(f"   Type selection: {truck.get('type_selection', 'N/A')}")
    print(f"   Initial battery: {truck.get('initial_battery', 'N/A')}")
    
    standard = truck.get('standard', {})
    print(f"   Standard: {standard.get('battery_capacity', 'N/A')} kWh, {standard.get('base_speed', 'N/A')} km/h")
    
    heavy = truck.get('heavy', {})
    print(f"   Heavy: {heavy.get('battery_capacity', 'N/A')} kWh, {heavy.get('base_speed', 'N/A')} km/h")
    
    # Charging settings
    charging = config.get('charging', {})
    print("\n🔌 Charging:")
    print(f"   Charge rate: {charging.get('charge_rate', 'N/A')} kWh/h")
    print(f"   Durations: {charging.get('charge_durations', 'N/A')} hours")
    print(f"   Efficiency: {charging.get('efficiency', 'N/A')}")
    
    # Reward settings
    rewards = config.get('rewards', {})
    print("\n🎯 Rewards:")
    print(f"   Time penalty: {rewards.get('time_penalty', 'N/A')}")
    print(f"   Delivery bonus: {rewards.get('delivery_bonus', 'N/A')}")
    print(f"   Completion bonus: {rewards.get('completion_bonus', 'N/A')}")
    print(f"   Failure penalty: {rewards.get('failure_penalty', 'N/A')}")
    
    print("\n" + "="*70)


# Example usage functions
def create_env_from_config(config_path: Optional[str] = None, preset: Optional[str] = None):
    """
    Create a SimpleTruckEnv instance from configuration file.
    
    Args:
        config_path: Path to config file (None for default)
        preset: Optional preset name to apply
        
    Returns:
        Configured SimpleTruckEnv instance
    """
    from simple_truck_env import SimpleTruckEnv
    
    # Load config
    config = load_config(config_path)
    
    # Apply preset if specified
    if preset:
        config = apply_preset(config, preset)
    
    # Extract environment parameters
    env_params = get_env_config(config)
    
    # Create environment
    env = SimpleTruckEnv(**env_params)
    
    # Store full config in environment for reference
    env.config = config
    
    return env


if __name__ == "__main__":
    # Demo: Load and print config
    config = load_config()
    print_config_summary(config)
    
    print("\n\nEnvironment parameters:")
    print(get_env_config(config))
