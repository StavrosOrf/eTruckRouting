# Benchmark Instances

This directory holds the 30 large-scale EVRPTW benchmark instances
(60–100 customers) used in Section 5.1 of the paper.

## Instance Specification

Instances extend the Solomon benchmark suite
([schneider2014electric](https://doi.org/10.1287/trsc.2014.0549))
with the following additions:

| Parameter | Value |
|---|---|
| Customer range | 60–100 per instance |
| Charging stations | 12 per instance (0.15 stations/km²) |
| Charging rates | 50–150 kW (heterogeneous) |
| Battery capacity | 150 kWh |
| Vehicle payload | 3,000 kg |
| Service period | 8 hours |
| Service time | 15 min/customer |
| Energy CV | 30–40% (Gaussian noise) |
| Grid area | 100 × 100 km² |

## Downloading the Instances

The original Solomon instances are publicly available at:
> Schneider, M., Stenger, A., & Goeke, D. (2014).
> *The electric vehicle-routing problem with time windows and recharging stations.*
> Transportation Science, 48(4), 500–520.
> https://doi.org/10.1287/trsc.2014.0549

The extended instances used in this paper (with stochastic energy models
and heterogeneous charging rates) are available upon request from the
corresponding author: **ayman.ja90@uoanbar.edu.iq**

## Instance JSON Format

Each instance file follows this schema:

```json
{
  "instance_id":   "inst_001",
  "n_customers":   80,
  "n_chargers":    12,
  "N":             93,
  "coords":        [[x_0, y_0], ...],
  "demands":       [0, q_1, ..., q_n, 0, ...],
  "service_times": [0, s_1, ..., s_n, 0, ...],
  "time_windows":  [[e_0, l_0], ...],
  "dist":          [[d_00, d_01, ...], ...],
  "travel_times":  [[t_00, t_01, ...], ...],
  "mu_energy":     [[mu_00, ...], ...],
  "var_energy":    [[var_00, ...], ...],
  "charging_rates": [0, ..., r_1, ..., r_12]
}
```

## Generating Synthetic Instances

To generate synthetic instances for testing without the original data:

```bash
python -c "
from environment.evrptw_env import EVRPTWEnv, EVRPTWConfig
import json, pathlib
env = EVRPTWEnv(EVRPTWConfig(n_customers_min=80, n_customers_max=80))
env.reset()
inst = env._instance
pathlib.Path('data/benchmarks').mkdir(exist_ok=True)
with open('data/benchmarks/synthetic_080.json', 'w') as f:
    json.dump({k: v.tolist() if hasattr(v,'tolist') else v
               for k,v in inst.items()}, f, indent=2)
print('Synthetic instance created.')
"
```
