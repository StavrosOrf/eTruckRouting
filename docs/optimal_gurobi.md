# Deterministic Optimal Charging Model (Gurobi)

This note documents the mixed-integer linear program used in `EVRoutingEnv/baselines/optimal_gurobi.py` to compute per-truck optimal charging and navigation decisions. The formulation is intentionally compact and per-truck (no charger contention), matching the environment’s deterministic travel/energy model and fixed delivery order.

> VS Code math rendering: enable `"markdown.math.enabled": true` in Settings, or use the **Markdown+Math** / **Markdown Preview Enhanced** extension. A plain-text fallback is included below each equation block.

## Problem setting and assumptions
- One truck, fixed delivery order: nodes \(d_0, d_1, \dots, d_K\) where \(d_0\) is the current node.
- Network is directed with deterministic travel time \(t(i,j)\) and energy \(e(i,j)\) for any nodes \(i,j\).
- A set of charger nodes \(C\) with charging rate \(r_c\) (kW) and efficiency \(\eta_c\).
- Battery capacity \(B^{\max}\); initial energy \(B^{0}\).
- Charging duration can be any real value in \([0,24]\) hours (continuous).
- At most one charger can be visited between two consecutive deliveries. No queueing/waiting costs.

## Sets and indices
- Deliveries: \(k = 0,\dots,K-1\) denote segments \((d_k \rightarrow d_{k+1})\).
- Chargers for segment \(k\): \(c \in C\).
- (No discrete duration set; durations are continuous.)

## Parameters
- \(e_k = e(d_k, d_{k+1})\): energy for direct travel on segment \(k\).
- \(t_k = t(d_k, d_{k+1})\): time for direct travel on segment \(k\).
- \(e^{\to}_{k,c} = e(d_k, c)\), \(e^{\gets}_{k,c} = e(c, d_{k+1})\): energy to/from charger \(c\).
- \(t^{\to}_{k,c} = t(d_k, c)\), \(t^{\gets}_{k,c} = t(c, d_{k+1})\): time to/from charger \(c\).
- \(r_c\): charge rate (kW) at charger \(c\).
- \(\eta_c\): charging efficiency at charger \(c\).
- Battery capacity \(B^{\max}\) and initial battery \(B^{0}\).
- Big-\(M\) constant: \(M\) large enough to deactivate battery constraints when an option is unused.

## Decision variables
- \(B_k \in [0, B^{\max}]\): battery just before starting segment \(k\) (after any charge on the start node or prior charger).
- \(B_{k+1}\): battery after completing segment \(k\).
- \(x_k \in \{0,1\}\): 1 if segment \(k\) is driven directly.
- \(y_{k,c} \in \{0,1\}\): 1 if segment \(k\) detours via charger \(c\).
- \(q_{k,c} \in [0,24]\): charging time (hours) if charger \(c\) is chosen on segment \(k\).
- Optional start charge (if \(d_0\) is a charger):
  - \(q^{\text{start}} \in [0,24]\): time charged at start node.

## Constraints
1) **Choice per segment**  
   $$
   x_k + \sum_{c} y_{k,c} = 1 \quad \forall\,k
   $$

   Plain text: exactly one of direct or any charger is chosen per segment.

2) **Charging duration bounds (segment chargers)**  
   $$
   0 \le q_{k,c} \le 24 \quad \forall\,k,c
   $$

   Plain text: if a charger is selected, its charge time is continuous between 0 and 24 hours (unused chargers can have \(q=0\)).

3) **Battery propagation – direct**  
   $$
   B_k - e_k \ge -M\bigl(1 - x_k\bigr)
   $$
   $$
   B_{k+1} = B_k - e_k \qquad \text{if } x_k = 1
   $$

   Plain text: if direct is chosen, battery drops by the segment energy.

4) **Battery propagation – via charger \(c\)**  
   Arrival feasibility:
   $$
   B_k - e^{\rightarrow}_{k,c} \ge -M\bigl(1 - y_{k,c}\bigr)
   $$
   Charge added: \(r_c \eta_c q_{k,c}\). Capacity cap (relaxed):
   $$
   B_k - e^{\rightarrow}_{k,c} + r_c \eta_c q_{k,c} \le B^{\max} + M\bigl(1 - y_{k,c}\bigr)
   $$
   Battery after segment:
   $$
   B_{k+1} = B_k - e^{\rightarrow}_{k,c} - e^{\leftarrow}_{k,c} + r_c \eta_c q_{k,c} \qquad \text{if } y_{k,c} = 1
   $$

   Plain text: if a charger is used, arrive with enough energy, add charged energy (bounded), then subtract the outbound energy.

5) **Start charging (optional, if charger at \(d_0\))**  
   $$
   0 \le q^{\text{start}} \le 24
   $$
   $$
   B_0 = B^{0} + r_{d_0}\eta_{d_0} q^{\text{start}} \quad \text{if charger exists; else } \quad B_0 = B^{0}
   $$

   Plain text: optionally charge once at the start node if it is a charger.

6) **Battery bounds**  
   $$
   0 \le B_k, \; B_{k+1} \le B^{\max}
   $$

   Plain text: standard bounds; big-\(M\) relaxes constraints when an option is unused.

## Objective
Minimize total elapsed time (travel + charging):
$$
\min \sum_{k} x_k t_k
     + \sum_{k,c} y_{k,c}\left(t^{\to}_{k,c} + t^{\gets}_{k,c}\right)
     + \sum_{k,c} q_{k,c}
     + q^{\text{start}}
$$
Plain text: total travel time (direct or via charger legs) plus all charging time (including optional start charge).

## Interpretation and execution
- The model chooses, per delivery segment, either direct travel or a single charger detour with a discrete charge duration.
- The resulting plan is converted to environment actions:
  - `nav_delivery` \(\rightarrow\) go to next delivery action.
  - `nav_charger(c)` \(\rightarrow\) go-to-charger action for node \(c\).
  - `charge(h)` \(\rightarrow\) charge-duration action; the MILP’s continuous \(h\) is rounded **up** to the nearest discrete duration allowed by the environment to preserve feasibility.
- Because the environment is deterministic and chargers are assumed uncongested, solving per truck is sufficient and produces the time-optimal policy under these assumptions.

## Simplifications and limitations
- No charger queueing or shared capacity: trucks are optimized independently.
- At most one charger per leg; allowing multiple chargers would require expanding the path space or adding additional binary stages.
- Charging durations are restricted to the discrete set in the environment config.
- Travel times/energy are taken as fixed; no stochasticity or traffic variations are modeled in the MILP.
