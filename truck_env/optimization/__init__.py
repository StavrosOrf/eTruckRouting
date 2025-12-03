"""
Optimization module for truck routing problems.
"""

from .gurobi_solver import GurobiOptimalPlanner

__all__ = ["GurobiOptimalPlanner"]
