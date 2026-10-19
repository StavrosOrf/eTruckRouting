"""Parallel ALNS with a shared route pool.

Each worker is a persistent process running its own ``Search`` (own operators,
penalties, route cache). In rounds the master collects every worker's new elite
routes into one pool, solves the set-partitioning problem over it, and sends the
result back to half of the workers; the others continue their own trajectory, so
the portfolio stays diverse while good routes found anywhere get recombined.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from dataclasses import fields

from .pool import solve_partition
from .problem import Problem
from .search import Search, SearchParameters


def _worker_main(conn, instance: dict, parameters: SearchParameters, objective, seeds) -> None:
    try:
        problem = Problem(instance, objective=objective)
        search = Search(problem, parameters)
        search.initialize(seeds)
        conn.send(("ready", search.best_cost, [list(r) for r in search.best_routes], search.export_pool()))
        while True:
            command = conn.recv()
            if command[0] == "stop":
                conn.send(("log", search.log[-100:], search.history[-50:], search.iterations,
                           search.pricer.calls, search.pricer.hits))
                break
            _, seconds, progress, span, adopt = command
            if adopt:
                search.set_current(adopt)
            search.advance(seconds, progress_at_start=progress, progress_span=span)
            conn.send(("done", search.best_cost, [list(r) for r in search.best_routes], search.export_pool()))
    except BaseException as error:  # surface the failure to the master instead of hanging it
        conn.send(("error", repr(error)))
        raise


def with_seed(parameters: SearchParameters, seed: int, verbose: bool) -> SearchParameters:
    values = {f.name: getattr(parameters, f.name) for f in fields(parameters)}
    values.update(seed=seed, verbose=verbose)
    return SearchParameters(**values)


def solve_parallel(instance: dict, parameters: SearchParameters, workers: int, objective, seeds) -> dict:
    started = time.time()
    context = mp.get_context("spawn")
    pipes, processes = [], []
    for k in range(workers):
        parent, child = context.Pipe()
        process = context.Process(
            target=_worker_main,
            args=(child, instance, with_seed(parameters, parameters.seed + k, parameters.verbose and k == 0),
                  objective, seeds),
            daemon=True,
        )
        process.start()
        pipes.append(parent)
        processes.append(process)

    def receive(pipe):
        message = pipe.recv()
        if message[0] == "error":
            raise RuntimeError(f"worker failed: {message[1]}")
        return message

    problem = Problem(instance, objective=objective)
    pool: dict[frozenset, tuple[float, tuple]] = {}
    best_cost, best_routes = math.inf, []
    history = []

    def absorb(messages) -> None:
        nonlocal best_cost, best_routes
        for message in messages:
            _, cost, routes, fresh = message
            for key, entry in fresh.items():
                held = pool.get(key)
                if held is None or entry[0] < held[0] - 1e-12:
                    pool[key] = entry
            if cost < best_cost - 1e-9:
                best_cost, best_routes = cost, [tuple(r) for r in routes]
                history.append((time.time() - started, best_cost))

    absorb([receive(pipe) for pipe in pipes])
    p = parameters
    reserve = min(p.pool_seconds, 0.15 * p.time_limit_seconds) if p.pool else 0.0
    budget = max(p.time_limit_seconds - reserve, 1.0)
    round_seconds = max(5.0, p.pool_interval_seconds)
    round_index = 0
    while time.time() - started < budget:
        span = min(round_seconds, budget - (time.time() - started))
        if span < 1.0:
            break
        progress = (time.time() - started) / budget
        adopt_all = round_index == 0 or not p.pool
        for k, pipe in enumerate(pipes):
            adopt = best_routes if (not adopt_all and k >= workers * (1.0 - p.adopt_fraction) - 1e-9) else None
            pipe.send(("advance", span, progress, span / budget, adopt))
        absorb([receive(pipe) for pipe in pipes])
        round_index += 1
        if p.pool:
            merged = _partition(problem, pool, best_routes, best_cost, p, workers)
            if merged is not None:
                best_routes, best_cost = merged
                history.append((time.time() - started, best_cost))
    if p.pool:
        merged = _partition(problem, pool, best_routes, best_cost, p, workers)
        if merged is not None:
            best_routes, best_cost = merged
            history.append((time.time() - started, best_cost))

    summaries = []
    for pipe in pipes:
        pipe.send(("stop",))
        summaries.append(pipe.recv())
    for process in processes:
        process.join(timeout=10)
    return {
        "routes": best_routes,
        "cost": best_cost,
        "iterations": sum(s[3] for s in summaries),
        "seconds": time.time() - started,
        "history": history,
        "pricer_calls": sum(s[4] for s in summaries),
        "pricer_hits": sum(s[5] for s in summaries),
        "pool_columns": len(pool),
        "log": summaries[0][1],
        "per_worker_iterations": [s[3] for s in summaries],
    }


def _partition(problem: Problem, pool, best_routes, best_cost, p: SearchParameters, workers: int):
    for route in best_routes:  # the incumbent is always a column
        key = frozenset(route)
        if key not in pool:
            pool[key] = (problem.price(route), tuple(route))
    limit = max(1.0, min(p.pool_seconds, 0.15 * p.time_limit_seconds))
    found = solve_partition(pool, problem.customers, problem.trucks, best_routes, limit, threads=workers)
    if found is None:
        return None
    routes, _ = found
    exact = sum(problem.price(r) for r in routes)  # re-price: the native core is authoritative
    if math.isfinite(exact) and exact < best_cost - 1e-9:
        return [tuple(r) for r in routes], exact
    return None
