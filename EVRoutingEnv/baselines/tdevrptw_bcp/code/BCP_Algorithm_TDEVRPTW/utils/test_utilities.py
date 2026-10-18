from __future__ import annotations
from typing import List, Dict, Set, Any
import numpy as np
import matplotlib.pyplot as plt
import os, sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils import PiecewiseLinearFunction, get_travel_time_and_consumption

EPS = 1e-6

def simple_external_feasibility_check(instance: dict, routes_solution: list, verbose: bool = True) -> dict:
    """
    Minimal, solver-consistent external checker:
      - each route must start/end at the depot,
      - every customer must be covered exactly once (using 'customers_visited'),
      - basic time/battery simulation (consistent with the solver):
        time-dependent travel, station waiting pattern, TW checks, and full recharge via PWL inverse.
    Returns a dict with:
      { 'feasible': bool, 'reasons': [str], 'routes': [{'idx', 'path', 'feasible'}] }
    """
    if not routes_solution:
        return {"feasible": False, "reason": "No routes provided."}

    depot = instance["depot_id"]
    customers = set(instance["customer_ids"])
    cover = {c: 0 for c in customers}

    T = float(instance["parameters"]["planning_horizon_T"])
    speed_defs = instance["time_dependent_definitions"]["speed_profiles"]
    tb = np.array(speed_defs["time_intervals_factors"], dtype=float) * T
    D = instance["distance_matrix"]

    wf = instance["time_dependent_definitions"].get("waiting_function", {})
    waiting_fn = PiecewiseLinearFunction(wf["points"]) if wf and wf.get("points") else None

    recharge_modes = instance["time_dependent_definitions"].get('recharge_functions', {})
    station_modes = instance["time_dependent_assignments"].get('station_recharge_modes', {})

    ok_all = True
    reasons: List[str] = []
    route_summaries: List[dict] = []

    for it, item in enumerate(routes_solution, 1):
        path = item["route"]["path"]
        custs = item["route"]["customers_visited"]
        for c in custs:
            if c in cover:
                cover[c] += 1

        if path[0] != depot or path[-1] != depot:
            ok_all = False
            reasons.append(f"Route {it}: not depot→depot.")
            route_summaries.append({"idx": it, "path": path, "feasible": False})
            continue

        time_now = 0.0
        batt = instance["parameters"]["battery_capacity"]
        feasible = True
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            nu, nv = instance["nodes_by_id"][u], instance["nodes_by_id"][v]

            profile = instance["time_dependent_assignments"]["arc_speed_profiles"][u][v]
            mult = np.array(speed_defs["profiles"][profile], dtype=float)
            dist = float(D[u, v])
            tt, cons = get_travel_time_and_consumption(time_now, dist, tb, mult, instance["parameters"])
            if not np.isfinite(tt):
                feasible = False
                reasons.append(f"Route {it}: non-traversable arc {nu['string_id']}→{nv['string_id']}.")
                break

            arr_t = time_now + tt
            arr_b = batt - cons
            if arr_b < -EPS:
                feasible = False
                reasons.append(f"Route {it}: battery < 0 at {nv['string_id']}.")
                break
            if arr_t > nv["time_window"]["end"] + EPS or arr_t > T + EPS:
                feasible = False
                reasons.append(f"Route {it}: arrival outside TW/T at {nv['string_id']}.")
                break

            eff_arr_t = arr_t
            if nv["type"] == 1 and waiting_fn is not None:
                add_wait = max(0.0, waiting_fn.evaluate(arr_t))
                eff_arr_t += add_wait

            # Passive wait to meet TW start
            eff_arr_t = max(eff_arr_t, nv["time_window"]["start"])
            leave_t = eff_arr_t
            leave_b = arr_b

            if nv["type"] == 0:
                # Customer service
                leave_t += nv.get("service_time", 0.0)
            elif nv["type"] == 1:
                # Station: recharge to full using PWL inverse
                mode = station_modes.get(str(v))
                pwl = PiecewiseLinearFunction(
                    recharge_modes.get(
                        mode,
                        [[0, 0], [instance["parameters"]["battery_capacity"], instance["parameters"]["battery_capacity"]]],
                    )
                )
                try:
                    t_start = pwl.evaluate_inverse(max(0.0, arr_b))
                    t_end = pwl.domain()[1]
                    rech = max(0.0, t_end - t_start)
                except Exception:
                    rech = 0.0
                leave_t += rech
                leave_b = instance["parameters"]["battery_capacity"]

            # Ensure TW end and horizon
            if leave_t > nv["time_window"]["end"] + EPS or leave_t > T + EPS:
                feasible = False
                reasons.append(f"Route {it}: end of operation outside TW/T at {nv['string_id']}.")
                break

            time_now, batt = leave_t, leave_b

        route_summaries.append({"idx": it, "path": path, "feasible": feasible})
        ok_all = ok_all and feasible

    dup = [c for c, v in cover.items() if v > 1]
    miss = [c for c, v in cover.items() if v != 1]
    if dup:
        ok_all = False
        reasons.append(f"Repeated customers: {sorted(dup)}")
    if miss:
        ok_all = False
        reasons.append(f"Uncovered customers: {sorted(miss)}")

    return {"feasible": ok_all, "reasons": reasons, "routes": route_summaries}


def plot_instance_and_routes(instance: dict, routes_solution: list | None, best_cost: float | None, save_path: str) -> None:
    """
    Simple 2D plot of the instance and (fractional) routes:
      - depot as a square, stations as triangles, customers as circles,
      - each route path is drawn with a distinct color and labeled with its value.
    """
    nodes = instance["nodes"]
    depot_id = instance["depot_id"]
    customer_ids = set(instance["customer_ids"])
    station_ids = set(instance["station_ids"])

    fig, ax = plt.subplots(figsize=(8, 7))

    # Depot
    dx, dy = nodes[depot_id]["coords"]["x"], nodes[depot_id]["coords"]["y"]
    ax.scatter([dx], [dy], marker="s", s=120, edgecolor="k", facecolor="none", label="Depot")

    # Stations
    sx = [nodes[i]["coords"]["x"] for i in station_ids]
    sy = [nodes[i]["coords"]["y"] for i in station_ids]
    if sx:
        ax.scatter(sx, sy, marker="^", s=70, alpha=0.9, label="Stations")

    # Customers
    cx = [nodes[i]["coords"]["x"] for i in customer_ids]
    cy = [nodes[i]["coords"]["y"] for i in customer_ids]
    if cx:
        ax.scatter(cx, cy, marker="o", s=45, alpha=0.9, label="Customers")

    # Routes
    if routes_solution:
        cmap = plt.cm.get_cmap('tab10', max(1, len(routes_solution)))
        for ridx, item in enumerate(routes_solution, 1):
            path = item["route"]["path"]
            px = [nodes[i]["coords"]["x"] for i in path]
            py = [nodes[i]["coords"]["y"] for i in path]
            ax.plot(px, py, linewidth=2.0, label=f"Route {ridx} (val={item['value']:.2f})", color=cmap((ridx - 1) % 10))

    ax.set_title(
        f"Instance: {instance.get('name','?')}  |  Routes: {len(routes_solution or [])}  |  Cost: {best_cost}"
    )
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

    # Legend to the side
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{instance.get('name', 'instance')}.png"), bbox_inches='tight', dpi=200)
    plt.show()
