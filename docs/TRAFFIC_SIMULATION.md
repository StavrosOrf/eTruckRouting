# Traffic Simulation Models

## Overview

The EV routing environment now supports realistic traffic uncertainty through multiple configurable models. Traffic simulation adds stochastic variation to travel times, making the environment more realistic and challenging for RL agents to learn robust routing policies.

## Configuration

Traffic simulation is configured in `EVRoutingEnv/config_files/config.yaml` under the `traffic` section:

```yaml
traffic:
  # Enable/disable traffic simulation
  enable_traffic: false  # Set to true to enable
  
  # Select traffic model
  traffic_model: "gaussian"  # Options: gaussian, time_of_day, distance_dependent, correlated
  
  # Base uncertainty parameters
  std_dev_factor: 0.15  # Standard deviation as fraction of base travel time (15%)
  max_std_dev_hours: 1.0  # Maximum std dev cap (1 hour)
  
  # Model-specific parameters
  rush_hour_multiplier: 2.0  # For time_of_day model
  distance_variance_scale: 1.0  # For distance_dependent model
  correlation_weight: 0.7  # For correlated model
```

## Traffic Model

### Time-of-Day Dependent Model

**Description**: Gaussian distribution with higher variance during rush hours (7-9am, 5-7pm).

**Characteristics**:
- Travel time ~ N(mean_time, std_dev)
- Off-peak std_dev = std_dev_factor × mean_time
- Rush hour std_dev = std_dev_factor × mean_time × rush_hour_multiplier
- **Rush hour effects apply during travel, not just at departure**
- Variance is interpolated based on fraction of journey during rush hours
- Captures realistic daily traffic patterns
- Based on global simulation clock modulo 24h
- Bounded to [50%, 250%] of base travel time

**Example**: Base travel time of 10h with `std_dev_factor=0.15`, `rush_hour_multiplier=2.0`:
- **Off-peak** (most hours): Std Dev = 1.5h (15% of mean)
- **Rush hour** (7-9am, 5-7pm): Std Dev = 3.0h (30% of mean)

**Implementation**:
```python
# Calculate what fraction of the journey occurs during rush hours
departure_time = self.global_clock
arrival_time = departure_time + travel_time
rush_hour_fraction = self._calculate_rush_hour_fraction(departure_time, arrival_time)

# Interpolate std_dev based on rush hour exposure
base_std_dev = travel_time * self.traffic_std_factor
rush_std_dev = base_std_dev * self.rush_hour_multiplier
std_dev = base_std_dev + rush_hour_fraction * (rush_std_dev - base_std_dev)

# Sample from normal distribution
actual_travel_time = np.random.normal(loc=travel_time, scale=std_dev)

# Apply bounds [50%, 250%]
actual_travel_time = np.clip(actual_travel_time, travel_time * 0.5, travel_time * 2.5)
```

The `_calculate_rush_hour_fraction()` method samples the journey at regular intervals to determine what percentage occurs during rush hours (7-9am or 5-7pm).

**Verbose Output Examples**:

Off-peak travel (0% in rush hour):
```
Traffic simulation: 10.00h → 10.82h (+8.2%)
```

Partial rush hour travel:
```
Traffic simulation [PARTIAL RUSH 36%]: 5.50h → 6.24h (+13.5%)
```

Majority rush hour travel (>50%):
```
Traffic simulation [RUSH HOUR 67%]: 2.00h → 2.45h (+22.5%)
```

## Implementation Details

### Code Location

**Main Implementation**: `EVRoutingEnv/models/traffic_simulation.py`

**Class**: `TrafficSimulator`
- `apply_traffic()`: Applies traffic variation to travel time
- `_calculate_rush_hour_fraction()`: Calculates rush hour overlap

**Integration**: `EVRoutingEnv/models/event_driven_env.py`
- `_apply_traffic_simulation()`: Wrapper method that delegates to `TrafficSimulator`
- Called by `_execute_navigation_action()` and `_execute_navigation_action_gnn()`

**Initialization**: 
- `TrafficSimulator` instantiated in `EventDrivenTruckEnv.__init__()`
- Loads traffic config parameters from config.yaml

### Bounds and Safety

All traffic models apply consistent bounds:
- **Lower bound**: 50% of base travel time (prevents negative/zero times)
- **Upper bound**: 250% of base travel time (prevents extreme delays)
- **Std dev cap**: `max_std_dev_hours` parameter (prevents excessive variance)

```python
actual_travel_time = np.clip(actual_travel_time, travel_time * 0.5, travel_time * 2.5)
```

### Random Seed Control

Traffic simulation respects the environment's random seed:

```python
def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
    if seed is not None:
        np.random.seed(seed)
```

This ensures **reproducible** episodes for evaluation and debugging.

## Impact on Learning

### Reward Signal

Traffic uncertainty affects the time penalty in the reward function:

```python
time_penalty = -actual_travel_time * self.reward_config["time_multiplier"]
```

The agent receives penalty based on **actual travel time** (with traffic), not base time.

### Action Masking

Travel time uncertainty does NOT affect action masking - feasibility checks use deterministic base travel times. This design choice:
- ✅ Prevents infeasible actions that would cause battery failure
- ✅ Allows agents to plan with worst-case scenarios in mind
- ⚠️ May lead to conservative policies (agents avoid risky routes)

### GNN State Representation

Edge features in the GNN still show **deterministic** base travel times:
```python
edge_features[:, 1] = time_distance / max_time  # Normalized base time
```

**Future Enhancement**: Could add expected traffic variance as additional edge feature to help agent learn risk-aware policies.

## Usage Examples

### Basic Traffic (Moderate Variance)

```yaml
traffic:
  enable_traffic: true
  std_dev_factor: 0.15  # 15% base variance
  max_std_dev_hours: 1.0
  rush_hour_multiplier: 2.0  # 30% variance during rush hours
```

### High Traffic Variability

```yaml
traffic:
  enable_traffic: true
  std_dev_factor: 0.20  # 20% base variance
  max_std_dev_hours: 2.0
  rush_hour_multiplier: 2.5  # 50% variance during rush hours
```

### Conservative Traffic (Low Variance)

```yaml
traffic:
  enable_traffic: true
  std_dev_factor: 0.10  # 10% base variance
  max_std_dev_hours: 0.5
  rush_hour_multiplier: 1.5  # 15% variance during rush hours
```

### Extreme Rush Hour Effects

```yaml
traffic:
  enable_traffic: true
  std_dev_factor: 0.15
  max_std_dev_hours: 3.0
  rush_hour_multiplier: 3.0  # 45% variance during rush hours
```

## Testing and Validation

### Test Script

Run the comprehensive test suite:

```bash
python scripts/test_traffic_simulation.py
```

This script:
1. Tests traffic disabled (baseline)
2. Tests traffic enabled with time-of-day model
3. Runs multiple episodes
4. Prints verbose output showing traffic variations with rush hour labels

### Expected Output

With `verbose=True`, you'll see traffic simulation logs:

Off-peak:
```
Traffic simulation: 7.05h → 8.45h (+19.8%)
    Routing to node 83
    Distance: 282.18 km, Time: 8.45h (base: 7.05h)
```

Rush hour (7-9am or 5-7pm):
```
Traffic simulation [RUSH HOUR]: 7.05h → 9.87h (+40.0%)
    Routing to node 83
    Distance: 282.18 km, Time: 9.87h (base: 7.05h)
```

### Validation Metrics

To validate realistic behavior:

1. **Mean Preservation**: Average actual_time ≈ base_time across many samples
2. **Variance Check**: Std dev of actual_time ≈ std_dev_factor × base_time
3. **Rush Hour Effect**: Higher variance during 7-9am and 5-7pm (time_of_day model)
4. **Correlation Test**: Same edges show similar delays in correlated model

### Performance Considerations

### Computational Overhead

Traffic simulation adds minimal overhead:
- Single `np.random.normal()` call per navigation action
- Simple time-of-day check (modulo operation)
- No significant impact on training speed

### Memory Usage

No additional memory required - all operations use local variables.

## Troubleshooting

### Issue: Traffic Not Applying

**Check**:
1. `enable_traffic: true` in config.yaml
2. Verbose output shows "Traffic simulation" logs
3. No errors in environment initialization

### Issue: Extreme Travel Times

**Solution**: Adjust bounds or std_dev cap:
```yaml
std_dev_factor: 0.10  # Reduce variance
max_std_dev_hours: 0.5  # Tighter cap
```

### Issue: No Rush Hour Effect Visible

**Check**:
1. `rush_hour_multiplier` is set > 1.0
2. Simulation clock reaches rush hour times (7-9am or 5-7pm)
3. Verbose output shows `[RUSH HOUR]` label during those times

## Future Enhancements

### Potential Improvements

1. **Traffic-Aware State Features**: Add expected variance to GNN edge features
2. **Weather Integration**: Combine with weather conditions for compound effects
3. **Incident Modeling**: Add discrete random events (accidents, road closures)
4. **Historical Traffic Data**: Learn realistic traffic patterns from real data
5. **Agent-Induced Congestion**: Multiple agents affect each other's travel times
6. **Day-of-Week Effects**: Different patterns for weekdays vs weekends
7. **Spatial Correlation**: Traffic patterns propagate across nearby edges

### Research Directions

1. **Robust RL**: How well do agents trained with traffic generalize without it?
2. **Risk-Aware Policies**: Can agents learn to avoid high-variance routes?
3. **Stochastic Planning**: Compare RL vs optimization under uncertainty
4. **Multi-Agent Competition**: Traffic increases with fleet density

## References

### Related Documentation

- `docs/STATE_SPACE_FEATURES.md` - State representation details
- `docs/REALISTIC_CHARGING_VALIDATION.md` - Charging curve validation
- `docs/GNN_STATE_SPACE_PRESENTATION.md` - GNN architecture

### Code References

- Traffic simulator: `EVRoutingEnv/models/traffic_simulation.py`
- Event-driven environment: `EVRoutingEnv/models/event_driven_env.py`
- Configuration schema: `EVRoutingEnv/config_files/config.yaml`
- Test suite: `scripts/test_traffic_simulation.py`

---

**Last Updated**: December 7, 2024  
**Author**: GitHub Copilot  
**Status**: Production Ready ✅
