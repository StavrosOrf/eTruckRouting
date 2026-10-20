# California GraphPPO progress and available performance

Updated 2026-10-20T08:09:51.686812+00:00. Scheduler observation: 2026-10-20T10:09:50.745606+02:00.

Goal active and unmet: at least 99% successful depot-return completion and at most 5% mean paired operating-hours excess against independent feasible BKS, five training seeds and protected final evaluation.

## Latest findings

- The completed three-seed action-edge comparison gives routing gates 98.95833% completion, 129.37 successful operating hours and 16.92% paired BKS excess. This is the lowest excess among the evaluated primary GraphPPO recipes, but misses both targets.
- Its matched standard action-GCN control achieves 99.21875%, 131.36 hours and 18.04% excess. Critic-encoder detachment gives 100% completion but 161.92 hours and 46.75% excess. Standard control and routing gates were promoted; critic detachment was not.
- For routing gates, travel contributes 16.59 excess percentage points and charging 0.32. On successful pairs, the policy uses 5.72 trucks versus 4.27 for the references. Routing gates improve cost without yet solving fleet allocation or travel efficiency.
- The policies use CanonicalActorCritic with a heterogeneous state GNN and ragged complete-GCN action head, optimized end to end by PPO. They do not instantiate the original PPOVariableActionGNN class. This retains the graph/variable-action/PPO structure, but is a different implementation; exact implementation parity has not been established.
- The current input adapter has EVRP-specific node/action schemas, saved batching ceilings and a station-connectivity cache. Variable candidate counts do not establish support for arbitrary new graph schemas or action semantics. Train-only physical-node relabeling changed 22.40–25.78% of frozen PPO greedy choices, exposing ID sensitivity; this does not establish its contribution to the California cost gap.
- All current stronger recipes use behavior-cloning and warm-start initialization followed by PPO. All encoder/action/value parameters remain trainable. No teacher or solver chooses actions at neural deployment. Fine-tuning seeds share one fixed parent; inherited training budgets differ from fresh recipes.

## Completed performance comparisons

All rows use the same opened 128 development cases. Neural recipes have three evaluated training seeds. Costs and paired excess condition on success; completion retains failures. Paired excess averages per-case ratios. The reference pool is a per-case feasible oracle, not a deployed single policy or certified optimum. These reused development results are not final validation or test evidence.

| Approach | Stage | Completion | Successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- | --- |
| Independent BKS pool (per-case oracle) | Reference coverage, not one policy | 100.0000% | 113.20 | 0.00% |
| heuristic | Independent reference | 96.8750% | 277.30 | 150.19% |
| mpc | Independent reference | 94.5312% | 316.57 | 179.34% |
| alns | Independent reference | 100.0000% | 263.17 | 137.98% |
| cpsat | Independent reference | 100.0000% | 264.60 | 138.75% |
| alns_planned | Independent reference | 98.4375% | 112.11 | 1.73% |
| cpsat_planned | Independent reference | 99.2188% | 116.27 | 5.15% |
| bc0_large_fresh_cost40 | 20 BC epochs; zero new PPO steps | 92.9688% | 141.29 | 28.00% |
| bc0_large_warm_cost40 | 20 BC epochs; zero new PPO steps | 90.6250% | 136.21 | 25.56% |
| bc0_large_warm_mc_cost40 | 20 BC epochs; zero new PPO steps | 91.6667% | 139.41 | 27.29% |
| large_fresh_cost40 | Selected checkpoint through 1M additional PPO decisions | 98.4375% | 139.62 | 25.52% |
| large_warm_cost40 | Selected checkpoint through 1M additional PPO decisions | 98.4375% | 131.12 | 18.14% |
| large_warm_mc_cost40 | Selected checkpoint through 1M additional PPO decisions | 97.9167% | 134.60 | 20.28% |
| ppo1m_final_large_fresh_cost40 | Final at 1M additional PPO decisions | 98.1771% | 138.77 | 25.40% |
| ppo1m_final_large_warm_cost40 | Final at 1M additional PPO decisions | 98.6979% | 132.18 | 19.65% |
| ppo1m_final_large_warm_mc_cost40 | Final at 1M additional PPO decisions | 98.4375% | 135.97 | 21.38% |
| bc_links_progress | Selected checkpoint after completed 12M run budgets | 97.9167% | 151.23 | 37.06% |
| bc_progress | Selected checkpoint after completed 12M run budgets | 97.6562% | 145.94 | 33.91% |
| links_shield | Selected checkpoint after completed 12M run budgets | 96.6146% | 155.16 | 39.30% |
| progress_links_shield | Selected checkpoint after completed 12M run budgets | 98.6979% | 157.82 | 41.40% |
| shield | Selected checkpoint after completed 12M run budgets | 97.6562% | 152.66 | 40.18% |
| v2_shield | Selected checkpoint after completed 12M run budgets | 95.0521% | 147.99 | 34.86% |
| sampling_0p25 | Same frozen warm-cost actors; 3 policy RNG repetitions per actor | 98.7847% | 132.64 | 18.90% |
| sampling_0p5 | Same frozen warm-cost actors; 3 policy RNG repetitions per actor | 98.7847% | 131.95 | 18.93% |
| sampling_1p0 | Same frozen warm-cost actors; 3 policy RNG repetitions per actor | 98.1771% | 135.10 | 21.58% |
| bc_nonlinear_cost40 | Selected checkpoint after completed 12M run budgets | 96.8750% | 138.77 | 26.39% |
| bc_nonlinear_gamma1 | Selected checkpoint after completed 12M run budgets | 98.9583% | 146.55 | 32.76% |
| bc_capacity_links | Selected checkpoint after completed 12M run budgets | 98.1771% | 151.88 | 36.90% |
| bc_capacity_nonlinear_links | Selected checkpoint after completed 12M run budgets | 97.9167% | 142.09 | 28.97% |
| bc0_edge_control_cost40 | BC-only or selected/final actor after 1M additional PPO | 93.4896% | 140.35 | 26.76% |
| bc0_edge_route_cost40 | BC-only or selected/final actor after 1M additional PPO | 93.7500% | 141.02 | 28.52% |
| bc0_edge_route_criticstop | BC-only or selected/final actor after 1M additional PPO | 94.2708% | 140.46 | 27.79% |
| edge_control_cost40 | BC-only or selected/final actor after 1M additional PPO | 99.2188% | 131.36 | 18.04% |
| edge_route_cost40 | BC-only or selected/final actor after 1M additional PPO | 98.9583% | 129.37 | 16.92% |
| edge_route_criticstop | BC-only or selected/final actor after 1M additional PPO | 100.0000% | 161.92 | 46.75% |
| ppo1m_final_edge_control_cost40 | BC-only or selected/final actor after 1M additional PPO | 98.9583% | 132.96 | 19.59% |
| ppo1m_final_edge_route_cost40 | BC-only or selected/final actor after 1M additional PPO | 98.6979% | 130.67 | 17.98% |
| ppo1m_final_edge_route_criticstop | BC-only or selected/final actor after 1M additional PPO | 99.7396% | 165.77 | 50.12% |

## Current campaign

Training queue entries: 6; states: {'PENDING': 6}. Current maximum simultaneous training allocations: 6, below the global limit of 30. New submissions permit only gpu_mig and gpu_a100. H100 and cbuild are excluded.

| Family | Training array | Allocation layout | Remaining milestone |
| --- | --- | --- | --- |
| Larger-data warm cost / MC | 28360521 | One model per MIG | Five actors completed 4M; last MC seed resumes 1M → 4M |
| Recovery-data mixtures | 28360534 | One model per MIG | Eight actors completed 1M; last teacher50 seed BC20 + 1M |
| Standard action GCN / routing gates | 28360721 | Three models per A100 | Six actors continue 1M → 4M |
| Objective controls | 28360730 | Two models per A100 | Six actors completed 12M; four full states resume ~8M → 12M |

Every model retains nine CPUs and seven rollout workers. The allocation change preserves model, data, masks, rewards, interaction budgets and saved RNG/optimizer states. Immediate continuation failures came from node-local scratch quota exhaustion before training resumed; runtime caches now use shared workspace storage. Previously queued H100 jobs had already terminated when the new restriction was applied. Their completed results are retained.

## Observed training speed

Completed initial single-model MIG BC20 + 1M stages took 93–112 minutes, excluding queue wait and the separate paired evaluation. PPO-only logs previously measured roughly 190–240 decisions/s. Packed A100 throughput is not yet measured for this stage. The owned-MIG experiment completed its first recovery actor; its second failed before creating a resumable state when the parent ended. No steady two-recovery-model packing throughput claim is supported.

## Next steps

1. Confirm resumed training starts with shared runtime caches; measure packed A100 throughput and memory on the live jobs.
2. Complete the three-seed recovery BC-only/selected/final 1M comparison and freeze selected/final 4M actors from the warm and action-edge cohorts before further training.
3. Document architecture fidelity explicitly: separate the generic graph/candidate-action interface from the EVRP adapter; remove ordinal ID dependence and preserve ownership/destination information through explicit relations in a controlled future ablation. Do not claim arbitrary-schema support or exact original-class parity from current results.
4. Promote at most two qualifying recipes per family, expand to five training seeds and 12M, then perform shared selection on 512 disjoint final-validation cases after all registered cohorts finish or are screened out.
5. Open the locked 1,000-case test only after both final-validation targets pass. Report certified optimality only with valid independent bounds. Final validation and test remain unopened.

Sources: [comparison CSV](../../results/california_graphppo/performance_comparison_20261019.csv), [scheduler status](california_graphppo_status.md), [implementation audit](../../results/california_graphppo/GPU_policy_recovery_20261020/implementation_audit.json).
