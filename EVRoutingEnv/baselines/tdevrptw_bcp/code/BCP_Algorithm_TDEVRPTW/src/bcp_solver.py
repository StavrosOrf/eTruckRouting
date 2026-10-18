import time
import os
from multiprocessing import Pool, TimeoutError
import numpy as np

from utils import load_instance, compute_mbr, PiecewiseLinearFunction, get_travel_time_and_consumption
from model import RMP_Model
from cutting import find_and_add_violated_cuts
from branching import select_branching_variable, create_branches
from pricing_worker import solve_pricing_wrapper

EPS_TIME = 1e-6
EPS_BATT = 1e-6
EPS_VAL  = 1e-6
BIG_M    = 1e6


class BCPSolver:
    def __init__(self, instance_path: str, num_processes: int | None = None, pricing_timeout_s: int = 120,
                 max_pricing_loops_per_node: int = 25):
        print("--- Initializing Parallel BCP Solver ---")
        self.instance = load_instance(instance_path)
        self.instance['mbr'] = compute_mbr(self.instance)

        # service must finish within TW if parameters.tw_end_enforced=True
        self.tw_end_enforced: bool = bool(self.instance.get("parameters", {}).get("tw_end_enforced", False))

        self.best_solution_cost: float = float('inf')
        self.best_solution: list | None = None
        self.best_solution_feas_report: dict | None = None

        self.node_count = 0
        self.bbtree: list[tuple[float, dict]] = [(0.0, {})]
        self.num_processes = int(num_processes) if num_processes else os.cpu_count()
        self.pricing_timeout_s = int(pricing_timeout_s)
        self.max_pricing_loops_per_node = int(max_pricing_loops_per_node)

        # to avoid revisiting the same constraint sets
        self.visited_signatures = set()

        self.debug_nodes: list[dict] = []
        print(f"Using {self.num_processes} processes for pricing.")

    # =============================== helpers: feasibility ===================================

    def _is_artificial_route(self, route: dict) -> bool:
        return bool(route.get('is_artificial', False) or route.get('cost', 0.0) >= 1e5)

    def _route_respects_branch_constraints(self, path: list[int], cons: dict) -> bool:
        if not cons:
            return True
        inst = self.instance
        customers = inst['customer_ids']
        forbidden_arcs = cons.get('forbidden_arcs', set())
        req_next = cons.get('required_next_customer', {})
        forb_next = cons.get('forbidden_next_customer', {})

        # explicitly forbidden arcs (applied as-is)
        for i in range(len(path)-1):
            if (path[i], path[i+1]) in forbidden_arcs:
                return False

        # required/forbidden next-customer (customers only)
        for i in range(len(path)):
            u = path[i]
            if u not in customers:
                continue
            nxt_cust = None
            for j in range(i+1, len(path)):
                v = path[j]
                if v in customers:
                    nxt_cust = v
                    break
            if u in req_next:
                if nxt_cust is None or nxt_cust != req_next[u]:
                    return False
            if u in forb_next and nxt_cust in forb_next[u]:
                return False
        return True

    def _simulate_route(self, route: list[int]) -> tuple[bool, str, list[dict]]:
        inst = self.instance
        nodes_by_id = inst['nodes_by_id']
        params = inst['parameters']
        T = float(params['planning_horizon_T'])

        dist_matrix = inst['distance_matrix']
        speed_defs = inst["time_dependent_definitions"]["speed_profiles"]
        tb = np.array(speed_defs["time_intervals_factors"], dtype=float) * T

        wf = inst["time_dependent_definitions"].get("waiting_function", {})
        waiting_fn = PiecewiseLinearFunction(wf["points"]) if wf and wf.get("points") else None

        recharge_modes = inst["time_dependent_definitions"].get('recharge_functions', {})
        station_modes = inst["time_dependent_assignments"].get('station_recharge_modes', {})

        time_now = 0.0
        batt = params['battery_capacity']
        step_log: list[dict] = []

        for k in range(len(route) - 1):
            u, v = route[k], route[k + 1]
            nu, nv = nodes_by_id[u], nodes_by_id[v]

            profile = inst["time_dependent_assignments"]["arc_speed_profiles"][u][v]
            multipliers = np.array(speed_defs["profiles"].get(profile, [1.0]), dtype=float)
            dist = float(dist_matrix[u][v])
            tt, cons = get_travel_time_and_consumption(time_now, dist, tb, multipliers, params)
            if not np.isfinite(tt):
                return False, f"Arco intransitabile {nu['string_id']}→{nv['string_id']}.", step_log

            arr_t = time_now + tt
            arr_b = batt - cons
            if arr_b < -EPS_BATT:
                return False, f"Batteria negativa all'arrivo su {nv['string_id']}.", step_log
            if arr_t > nv['time_window']['end'] + EPS_TIME or arr_t > T + EPS_TIME:
                return False, f"Arrivo fuori TW/T su {nv['string_id']} (arr={arr_t:.3f}, TW_end={nv['time_window']['end']:.3f}).", step_log

            eff_arr_t = arr_t
            wait_station = 0.0
            if nv['type'] == 1 and waiting_fn is not None:
                wait_station = float(waiting_fn.evaluate(arr_t))
                if wait_station < 0:
                    wait_station = 0.0
                eff_arr_t += wait_station

            wait_passive = max(0.0, nv['time_window']['start'] - eff_arr_t)
            eff_arr_t += wait_passive

            leave_t = eff_arr_t
            leave_b = arr_b
            srv = 0.0

            if nv['type'] == 0:
                srv = nv.get('service_time', 0.0)
                if self.tw_end_enforced and (eff_arr_t + srv > nv['time_window']['end'] + EPS_TIME):
                    return False, f"Fine servizio oltre TW su {nv['string_id']} (leave={eff_arr_t+srv:.3f} > TW_end={nv['time_window']['end']:.3f}).", step_log
                leave_t += srv
            elif nv['type'] == 1:
                mode = station_modes.get(str(v))
                pwl_data = recharge_modes.get(mode, [[0, 0], [params['battery_capacity'], params['battery_capacity']]])
                pwl = PiecewiseLinearFunction(pwl_data)
                try:
                    t_start = pwl.evaluate_inverse(max(0.0, arr_b)); t_end = pwl.domain()[1]
                    rech = max(0.0, t_end - t_start)
                except Exception:
                    rech = 0.0
                srv = rech
                leave_t += srv
                leave_b = params['battery_capacity']

            if leave_t > T + EPS_TIME:
                return False, f"Fine attività oltre T su {nv['string_id']} (leave={leave_t:.3f} > T={T:.3f}).", step_log

            step_log.append({
                "from": nu['string_id'], "to": nv['string_id'],
                "dep_t": time_now, "arr_t": arr_t, "eff_arr_t": eff_arr_t,
                "wait_station": wait_station, "wait_passive": wait_passive,
                "service/recharge": srv, "leave_t": leave_t,
                "batt_arr": arr_b, "batt_leave": leave_b
            })

            time_now, batt = leave_t, leave_b

        return True, "OK", step_log

    def _validate_solution(self, solution: list) -> dict:
        inst = self.instance
        if not solution:
            return {"feasible": False, "reason": "Soluzione vuota.", "per_route": [], "duplicated": set(), "uncovered": set()}

        depot = inst['depot_id']
        customers = set(inst['customer_ids'])
        active = [item for item in solution if item.get('value', 0.0) > EPS_VAL]

        cover_count = {c: 0 for c in customers}
        routes_paths = []
        has_artificial_active = False

        for item in active:
            r = item['route']
            if self._is_artificial_route(r):
                has_artificial_active = True
            path = r['path']
            routes_paths.append(path)
            for c in (r.get('customers_visited') or []):
                if c in cover_count:
                    cover_count[c] += 1

        duplicated = {c for c, cnt in cover_count.items() if cnt > 1}
        uncovered  = {c for c, cnt in cover_count.items() if cnt != 1}

        per_route = []
        all_ok = True
        first_bad_reason = None

        for path in routes_paths:
            if not path or path[0] != depot or path[-1] != depot:
                reason = "La rotta non parte/termina al depot."
                per_route.append({"path": path, "feasible": False, "reason": reason})
                if first_bad_reason is None:
                    first_bad_reason = reason
                all_ok = False
                continue

            ok, rreason, rlog = self._simulate_route(path)
            per_route.append({"path": path, "feasible": ok, "reason": rreason, "log": rlog})
            if not ok:
                all_ok = False
                if first_bad_reason is None:
                    first_bad_reason = rreason

        top_reason_parts = []
        if duplicated:
            top_reason_parts.append(f"Clienti ripetuti: {sorted(list(duplicated))}")
        if uncovered:
            top_reason_parts.append(f"Clienti non coperti: {sorted(list(uncovered))}")
        if first_bad_reason:
            top_reason_parts.append(first_bad_reason)

        overall_feasible = all_ok and (not duplicated) and (not uncovered)
        reason = "OK" if overall_feasible else (" | ".join(top_reason_parts) if top_reason_parts else "Infeasible")

        return {
            "feasible": overall_feasible,
            "reason": reason,
            "per_route": per_route,
            "duplicated": duplicated,
            "uncovered": uncovered,
            "has_artificial_active": has_artificial_active
        }

    # =============================== artificial columns: respect  branch ===================================

    def _add_initial_artificials_respecting_branch(self, rmp: RMP_Model, cons: dict):
        inst = self.instance
        depot = inst['depot_id']
        customers = sorted(list(inst['customer_ids']))
        req_next = (cons or {}).get('required_next_customer', {})

        for u in customers:
            if u in req_next:
                v = req_next[u]
                rmp.add_column({
                    'path': [depot, u, v, depot],
                    'cost': BIG_M,
                    'customers_visited': {u, v},
                    'is_artificial': True
                })
            else:
                rmp.add_column({
                    'path': [depot, u, depot],
                    'cost': BIG_M,
                    'customers_visited': {u},
                    'is_artificial': True
                })

    # =============================== solve node  ===================================

    def solve_node(self, branching_constraints: dict) -> tuple:
        rmp = RMP_Model(self.instance)
        self._add_initial_artificials_respecting_branch(rmp, branching_constraints)

        loop_counter = 0
        any_real_added = False  # at least one real column entered the node
        pricing_saturated = False  # we tried heuristics+exact without getting any column in

        while True:
            loop_counter += 1
            if loop_counter > self.max_pricing_loops_per_node:
                print(f"[Node] Reached pricing loop budget ({self.max_pricing_loops_per_node}). Returning current LP.")
                status, obj_val, _ = rmp.solve()
                sol = rmp.get_solution()
                return obj_val, sol, pricing_saturated, any_real_added

            status, obj_val, duals = rmp.solve()
            if status != 2:
                return None, None, True, any_real_added

            print(f"[RMP] Obj={obj_val:.6f} | cols={len(rmp.route_vars)}")
            new_routes = []

            heuristic_args = [
                (self.instance, duals, 'relax-s', 0, branching_constraints, True),
                (self.instance, duals, 'k-shrink', 3, branching_constraints, True),
                (self.instance, duals, 'k-shrink', 7, branching_constraints, True),
                (self.instance, duals, 'k-shrink', 12, branching_constraints, True),
                (self.instance, duals, 'relax-b', 0, branching_constraints, True),
                (self.instance, duals, 'relax-s', 0, branching_constraints, True),
                (self.instance, duals, 'k-shrink', 7, branching_constraints, True)
            ]

            print("Running pricing heuristics in parallel...")
            pool = Pool(processes=min(len(heuristic_args), self.num_processes))
            try:
                results = [pool.apply_async(solve_pricing_wrapper, args=(args,)) for args in heuristic_args]
                for res in results:
                    routes = res.get(timeout=30)
                    if routes:
                        new_routes.extend(routes)
                        pool.terminate(); pool.join()
                        break
                else:
                    pool.terminate(); pool.join()
            except (TimeoutError, Exception) as e:
                print(f"Heuristic pricing failed or timed out: {e}")
                pool.terminate(); pool.join()

            exact_tried = False
            if not new_routes:
                exact_tried = True
                print("Heuristics failed, running exact pricing solver...")
                exact_args = (self.instance, duals, 'exact', 0, branching_constraints, False)
                pool_e = Pool(processes=1)
                try:
                    res = pool_e.apply_async(solve_pricing_wrapper, args=(exact_args,))
                    routes = res.get(timeout=self.pricing_timeout_s)
                    if routes:
                        new_routes.extend(routes)
                    pool_e.terminate(); pool_e.join()
                except (TimeoutError, Exception) as e:
                    print(f"Exact pricing failed or timed out: {e}")
                    pool_e.terminate(); pool_e.join()

            # Gate branch-constraints + physical feasibility gate
            if new_routes:
                filtered = []
                dropped_branch = dropped_feas = 0
                for r in new_routes:
                    path = r['path']
                    if not self._route_respects_branch_constraints(path, branching_constraints):
                        dropped_branch += 1
                        continue
                    ok, _, _ = self._simulate_route(path)
                    if not ok:
                        dropped_feas += 1
                        continue
                    filtered.append(r)
                if dropped_branch or dropped_feas:
                    print(f"Filtered out {dropped_branch} by branch-constraints, {dropped_feas} infeasible by physics.")
                new_routes = filtered

            if not new_routes:
                # Nothing to add: if we also tried exact, consider saturated
                if exact_tried:
                    pricing_saturated = True
                solution = rmp.get_solution()
                cuts = find_and_add_violated_cuts(rmp, solution, self.instance)
                if cuts > 0:
                    print(f"Added {cuts} cuts. Re-solving RMP...")
                    continue
                else:
                    return obj_val, solution, pricing_saturated, any_real_added

            # add new columns (all real) and set flag
            vars_before = len(rmp.route_vars)
            for route in new_routes:
                # safety: explicitly mark as non-artificial
                route['is_artificial'] = False
                rmp.add_column(route)
            vars_after = len(rmp.route_vars)
            added_cols = vars_after - vars_before
            if added_cols > 0:
                any_real_added = True
                print(f"Added {added_cols} new columns from heuristics.")
            else:
                # Try a last-chance exact (already attempted above if empty); if still 0, continue
                print("No new columns were added (likely duplicates).")
                continue

    # =============================== main BCP ===================================

    def _constraints_signature(self, cons: dict) -> tuple:
        """Canonical signature of constraints for node deduplication."""
        forb = cons.get('forbidden_arcs', set())
        req  = cons.get('required_next_customer', {})
        fnext = cons.get('forbidden_next_customer', {})

        forb_sig = tuple(sorted((int(u), int(v)) for (u, v) in forb))
        req_sig  = tuple(sorted((int(u), int(v)) for u, v in req.items()))
        fnext_sig = tuple(sorted((int(u), tuple(sorted(int(x) for x in vs))) for u, vs in fnext.items()))
        return (forb_sig, req_sig, fnext_sig)

    def solve(self):
        start_time = time.time()
        while self.bbtree:
            self.node_count += 1
            # best-first on the lower bound
            self.bbtree.sort(key=lambda x: x[0], reverse=True)
            current_lb, current_constraints = self.bbtree.pop()

            # deduplicate nodes with same constraints
            sig = self._constraints_signature(current_constraints)
            if sig in self.visited_signatures:
                # already explored
                continue
            self.visited_signatures.add(sig)

            if current_lb >= self.best_solution_cost - 1e-9:
                continue

            print(f"\n--- Exploring B&B Node #{self.node_count} ---")
            print(f"  Constraints: { {k: list(v) if isinstance(v, set) else v for k,v in current_constraints.items()} }")

            obj_val, solution, pricing_saturated, any_real_added = self.solve_node(current_constraints)
            node_info = {
                "node_id": self.node_count,
                "lb": current_lb,
                "obj_lp": obj_val,
                "constraints": current_constraints,
                "accepted_as_incumbent": False,
                "branched": False,
                "pricing_saturated": pricing_saturated,
                "any_real_added": any_real_added
            }

            if solution is None or obj_val is None:
                self.debug_nodes.append(node_info); continue
            if obj_val >= self.best_solution_cost - 1e-9:
                self.debug_nodes.append(node_info); continue

            branch_var = select_branching_variable(solution, self.instance)

            if branch_var is None:
                feas_report = self._validate_solution(solution)

                # case: LP is integer and feasible -> new incumbent
                if feas_report["feasible"]:
                    print(f"Integer & feasible solution found with cost {obj_val:.4f}")
                    self.best_solution_cost = obj_val
                    self.best_solution = solution
                    self.best_solution_feas_report = feas_report
                    node_info["accepted_as_incumbent"] = True
                    self.debug_nodes.append(node_info)
                    continue

                # case: LP integer but with artificials, pricing saturated and no real columns ever added -> node infeasible ⇒ fathom
                active = [it for it in solution if it.get('value', 0.0) > EPS_VAL]
                only_artificials = all(self._is_artificial_route(it['route']) for it in active) if active else True
                if only_artificials and pricing_saturated and (not any_real_added):
                    print("LP integer with only artificial columns and pricing saturated -> fathom node (prune).")
                    self.debug_nodes.append(node_info)
                    continue

                # otherwise, we could attempt a fallback branching (but avoid depot→customer to prevent cycles)
                print(f"Integer solution rejected (infeasible): {feas_report['reason']}")
                self.debug_nodes.append(node_info)
                # no fallback branching on (0,c): risks loops. Prune if saturated, otherwise the node will return with obj>=UB or without progress.

            else:
                print(f"  Branching on {branch_var}")
                node_info["branched"] = True
                self.debug_nodes.append(node_info)
                for new_constraints in create_branches(branch_var, current_constraints, self.instance):
                    self.bbtree.append((obj_val, new_constraints))

        end_time = time.time()
        print("\n" + "=" * 40 + "\nBCP Algorithm Finished\n" + "=" * 40)
        print(f"Nodes explored: {self.node_count}")
        print(f"Best solution cost: {self.best_solution_cost}")
        print(f"Total time: {end_time - start_time:.2f} seconds")

        if self.best_solution:
            print("Best solution (routes with value > 0):")
            for item in self.best_solution:
                if item.get('value', 0.0) > EPS_VAL:
                    print(f"  - Path: {item['route']['path']}, Value: {item['value']:.3f}, c_r={item['route'].get('cost')}")

        return {
            "best_cost": self.best_solution_cost,
            "best_solution": self.best_solution,
            "feasibility_report": self.best_solution_feas_report,
            "nodes_explored": self.node_count,
            "node_logs": self.debug_nodes
        }


