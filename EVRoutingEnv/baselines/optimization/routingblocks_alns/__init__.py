"""RoutingBlocks + exact charging-pricing baseline for the nominal joint routing problem.

The package splits into two sides:

* the *solver* (``problem``, ``search``, ``run_solver``) runs in a dedicated
  Python 3.11 interpreter that has ``routingblocks`` installed, and talks JSON;
* the *bridge* (``bridge``, ``policy``) runs in the project interpreter and
  launches the solver as a subprocess, like the Julia branch-and-price baseline.

Import only ``bridge``/``policy`` from the project interpreter.
"""
