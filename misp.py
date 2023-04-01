from __future__ import annotations
import random
import numpy as np
from typing import Set
from itertools import islice
from graph import Graph, MispEnumQueue
from scipy.optimize import LinearConstraint, milp, Bounds


class IndependenceError(Exception):
    "Solver returned a dependent solution. Check values passed."


def lin_solve_misp(graph: Graph, p: int=0) -> Set[int]:
    """
    Uses linear programming to solve maximum independent subset problem.

    For (Graph) graph, explores set of nodes to find independent subset of
        size p. If p==0, finds maximum size independent set.

    Args:
        graph (Graph): Graph object initialized for MISP search.
        p (int): Size of desired independent graph to return. If 0,
            maximum sized graph is returned instead. (Default: 0)

    Returns:
        Set comprised of integers representing vertices in the maximum
            independent subset of size p in graph. Maximum size if p is
            not specified.
    """
    A = []
    lb_ineq = []
    ub_ineq = []

    n = len(graph.explore)  # We only need to find independence with uncertain nodes.
    subset = graph.indep | set() # Copying solved nodes from graph.
    p_g = p - len(subset) if p else 0

    # Quick checks to avoid unecessary work.
    if p:
        if len(subset) >= p:
            return islice(subset, p)
        if p_g > n:  # Can't solve
            return set()
        # Adding inequality constraint for performance improvement.
        A.append([1 for _ in range(n)])
        lb_ineq.append(p_g)
        ub_ineq.append(p_g)

    valid_node = list(graph.explore)
    codebook = {val: idx for idx, val in enumerate(valid_node)}

    obj = [-1 for _ in range(n)]
    for i, v1 in enumerate(valid_node):
        connected = graph.is_edge[v1]
        for v2 in connected:
            if v2 in graph.explore and v1 != v2:
                ineq = [0 for _ in range(n)] # all zeros
                ineq[codebook[v1]] = 1
                ineq[codebook[v2]] = 1
                # Imposes no connection constraint.
                A.append(ineq)
                ub_ineq.append(1)
                lb_ineq.append(0)
    constraint = LinearConstraint(A=A, lb=lb_ineq, ub=ub_ineq)

    solution = milp(c=obj, constraints=constraint, bounds=(0,1), integrality=1)
    xs = solution['x']

    if xs is not None:
        xs = [valid_node[idx] for idx, val in enumerate(xs) if val > 0.5]
        subset = set(xs) | subset

    independent, violations = graph.is_independent_subset(subset)
    if independent:
        return subset
    else:
        raise IndependenceError(subset, violations)


def enum_solve_misp(graph: Graph, p: float=float("inf"))-> Set:
    best, v = set(), 0
    queue = MispEnumQueue(graph)
    while queue.has_entry:
        node = queue.peek()
        if node.size > v: # Better node, we add
            v = node.size
            best = node.subset
            if v >= p : # we have a breaking condition
                break
        elif node.bound <= v: # we will never break the constraints
                queue.pop()
                continue
        queue.update()
    independent, violations = graph.is_independent_subset(best)
    if independent:
        return best
    else:
        raise IndependenceError(best, violations)


misp_factory = {
    "enum": enum_solve_misp,
    "lin": lin_solve_misp,
}

