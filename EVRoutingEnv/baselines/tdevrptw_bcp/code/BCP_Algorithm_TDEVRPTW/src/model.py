"""Restricted Master Problem wrapper using Gurobi (robust version, paper-consistent)."""
from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Set, Tuple, Optional


class RMP_Model:
    def __init__(self, instance: dict):
        self.instance = instance
        self.customers: List[int] = sorted(list(instance["customer_ids"]))

        self.model = gp.Model("RMP")
        # Quiet output, with slightly tighter tolerances for numerical robustness
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('FeasibilityTol', 1e-8)
        self.model.setParam('OptimalityTol', 1e-8)
        # (ModelSense = minimize for clarity; it is already the default)
        self.model.ModelSense = GRB.MINIMIZE

        # Pools for variables/columns
        self.route_vars: List[gp.Var] = []
        self.routes: List[dict] = []
        self._route_keys: Set[Tuple[int, ...]] = set()  # to avoid exact duplicates of the path

        # Partition constraints: one per customer
        self.partition_constrs: Dict[int, gp.Constr] = {}
        for cust_id in self.customers:
            c = self.model.addConstr(gp.LinExpr() == 1.0, name=f"part_{cust_id}")
            self.partition_constrs[cust_id] = c

        self.model.update()

    # =============================== Handling Columns, Variables ===================================

    def _is_duplicate(self, route: dict) -> bool:
        """True if the path has already been added (exact duplicate)."""
        key = tuple(route.get('path', []))
        if not key:
            return False
        return key in self._route_keys

    def _validate_route(self, route: dict) -> bool:
        """
        Minimal checks:
          - path exists and has length >= 2
          - customers_visited is a subset of the instance's customers
          - at least one customer in the column (otherwise it doesn't help partitioning)
          - finite/valid cost
        """
        path = route.get('path')
        custs = route.get('customers_visited')
        cost = route.get('cost', None)

        if not isinstance(path, list) or len(path) < 2:
            return False
        if custs is None:
            return False
        if not isinstance(custs, (set, frozenset)):
            try:
                custs = set(custs)
                route['customers_visited'] = custs
            except Exception:
                return False
        # must be a subset of known customers
        if not custs.issubset(set(self.customers)):
            # allow columns that include non-customer nodes (stations/depot) but
            # 'customers_visited' must contain ONLY valid customers
            invalid = [c for c in custs if c not in self.customers]
            if invalid:
                return False
        # at least one customer (columns with no customers don't help the RMP cover)
        if len(custs) == 0:
            return False
        # numerically valid cost
        try:
            _ = float(cost)
        except Exception:
            return False
        return True

    def add_artificial_column(self):
        """High-cost artificial columns to start with a feasible LP."""
        depot = self.instance['depot_id']
        for cust_id in self.customers:
            self.add_column({
                'path': [depot, cust_id, depot],
                'cost': 1e6,
                'customers_visited': {cust_id}
            })

    def add_column(self, route: dict):
        """Add a route as a column. route['customers_visited'] is a set of customer IDs."""
        if not self._validate_route(route):
            return  # ignore malformed columns
        if self._is_duplicate(route):
            return  # avoid duplicates

        path = route['path']
        custs = route['customers_visited']
        cost = float(route['cost'])

        # Coefficients for partition constraints
        constrs = [self.partition_constrs[c] for c in custs if c in self.partition_constrs]
        coeffs = [1.0] * len(constrs)
        col = gp.Column(coeffs, constrs) if constrs else None

        var = self.model.addVar(obj=cost, vtype=GRB.CONTINUOUS,
                                name=f"route_{len(self.route_vars)}", column=col)
        self.route_vars.append(var)
        self.routes.append({'path': path, 'cost': cost, 'customers_visited': set(custs)})
        self._route_keys.add(tuple(path))

        self.model.update()

    def add_columns_bulk(self, routes: List[dict]):
        """Bulk insertion with duplicate/invalid filtering."""
        for r in routes:
            self.add_column(r)

   # =============================== Solver ===================================

    def solve(self):
        self.model.optimize()
        status = self.model.status
        if status == GRB.OPTIMAL:
            duals = {cid: self.partition_constrs[cid].Pi for cid in self.customers}
            return status, float(self.model.ObjVal), duals
        # If infeasible/unbounded, do not return duals
        return status, None, None

    def get_solution(self, tol: float = 1e-6):
        """List of active columns: [{'route':..., 'value': x}] with x > tol."""
        if self.model.status != GRB.OPTIMAL:
            return None
        sol = []
        for i, var in enumerate(self.route_vars):
            val = float(var.X)
            if val > tol:
                sol.append({'route': self.routes[i], 'value': val})
        return sol
