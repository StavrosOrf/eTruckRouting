# Electric Truck Routing with Charging - Research Overview

## Research Question

### How can we efficiently route hundreds of electric trucks to make multiple deliveries while managing battery constraints and charging infrastructure?

## Sources of Uncertainty

- **Battery Depletion**: Actual discharge rates vary with terrain, weather, and load conditions
- **Charging Availability**: Charging stations may be occupied when trucks arrive, causing unpredictable wait times
- **Traffic Conditions**: Travel times fluctuate based on congestion and road conditions
- **Dynamic Demand**: New delivery requests may arrive during execution, requiring route adjustments
- **Infrastructure Failures**: Charging stations or road segments may become unavailable unexpectedly

## Challenges

- **Combinatorial Complexity**: Routing hundreds of trucks with multiple stops creates an exponentially large search space
- **Range Anxiety**: Trucks must reach charging stations before battery depletion, limiting routing flexibility
- **Temporal Coordination**: Multiple trucks competing for limited charging infrastructure requires scheduling
- **Multi-Objective Tradeoff**: Balancing delivery speed, energy efficiency, and infrastructure utilization
- **Scalability**: Solutions must work for large fleets (100+ trucks) in real-time

## Motivation

Electric trucks are essential for reducing transportation emissions, but their limited range and charging requirements create complex operational challenges that, if solved efficiently, can make them economically viable alternatives to diesel fleets while informing infrastructure deployment strategies.

## Goal

**Develop a reinforcement learning framework that:**

- **Optimizes** multi-truck routing decisions under battery and charging constraints
- **Minimizes** total delivery time while ensuring all trucks complete their routes
- **Coordinates** charging station usage to avoid congestion and reduce wait times
- **Generalizes** across different fleet sizes, delivery patterns, and network topologies
- **Provides** insights for charging infrastructure planning and fleet management

## Approach

- **Event-Driven Simulation**: Global clock with continuous time progression for realistic modeling
- **Single-Agent RL**: Control active trucks sequentially, compatible with standard RL algorithms (PPO, DQN, A2C)
- **State Representation**: Battery level, location, remaining deliveries, global time, and system state
- **Action Space**: Navigate to delivery/charger or charge for variable durations (1-4 hours)
- **Reward Shaping**: Penalize time/distance, reward successful deliveries, heavily penalize failures

## Contributions

- **Event-Driven Environment**: Novel Gymnasium-compatible simulation with global clock and continuous time progression for realistic truck routing
- **Single-Agent Formulation**: Reformulation of multi-truck coordination as single-agent sequential decision-making, enabling use of standard RL algorithms
- **Scalable Architecture**: Event queue design that efficiently handles 100+ trucks without action space dimensionality explosion
- **Benchmark Framework**: Comprehensive test suite and baseline policies for evaluating routing strategies across different scenarios


