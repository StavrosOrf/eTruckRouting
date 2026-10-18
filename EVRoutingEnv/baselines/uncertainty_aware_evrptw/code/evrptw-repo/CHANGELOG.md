# Changelog

All notable changes to this project are documented here.

## [1.0.0] — 2025

### Initial Release

- Full implementation of hybrid DRL framework for EVRPTW under uncertainty
- GATEncoder: 4-layer, 8-head Graph Attention Network with edge feature attention
- LSTMPredictor: bidirectional LSTM with heteroscedastic NLL loss (MAPE 3.8%, coverage 94.7%)
- PPOPolicy: Proximal Policy Optimisation with clipped surrogate objective
- ChanceConstraintChecker: analytical Φ⁻¹(α) battery feasibility (α = 0.95)
- MILPSolver: Gurobi warm-start refinement (1% MIP gap, 300–1200 s budget)
- EVRPTWEnv: Gymnasium-style environment with curriculum learning (10→100 customers)
- Three-stage transfer learning protocol (72h → 3–5h adaptation)
- Complete results: 12 figures, 9 data tables, JSON archive
- Pre-trained checkpoint available on request
