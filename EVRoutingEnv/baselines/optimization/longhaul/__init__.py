"""Static long-haul planners and executors used as best-known references.

The three static long-haul problems of the generalist suite (``tx_longhaul``,
``tx_longhaul_congested`` and ``pnw_longhaul_hos``) send every route across a
state, so every route must recharge. The day-ahead ALNS and CP-SAT planners of
``exact_optimization`` price charging as energy over a nominal power and their
executors charge greedily; neither knows the public-station queues or the
hours-of-service block. This package adds, without changing those modules:

* ``network``: nominal leg tables, the charging curve and a charger-waypoint
  closure of the leg costs (the shortest-path tables break the triangle
  inequality, so passing through a station can be faster than the direct leg);
* ``continuation``: a label-setting search over waypoint chargers, discrete
  target-SoC charges, expected station waits and US hours-of-service rests at
  stops, which completes a fixed customer sequence from any state;
* ``planner``: an ALNS over customer routes with route-duration, payload and
  hours-of-service-aware costs;
* ``policy``: closed-loop executors that follow a plan and re-solve the
  continuation at every decision, optionally re-planning the assignment.
"""
