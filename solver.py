import logging
import numpy as np
import random
from typing import Set, Tuple, List

from scipy.optimize import Bounds, LinearConstraint, milp

from graph import Graph
from misp import misp_factory, IndependenceError

logging.basicConfig(level = logging.INFO)


def get_dispersion_sum(subset: List, matrix: np.array) -> float:
    total = 0
    for i, v1 in enumerate(subset):
        for v2 in subset[i+1:]:
            total += matrix[v1,v2]
    return total


def get_dispersion_min(subset: List, matrix: np.array) -> float:
    arg_min = -1
    minimum = np.inf
    for idx, v1 in enumerate(subset):
        total = 0
        for v2 in subset:
            if v1 != v2:
                total += matrix[v1,v2]
        if total < minimum:
            arg_min = v1
            minimum = total
    return arg_min, minimum


def binary_max_min_dispersion(d_matrix: np.array,  p: int,  method: str = "enum") -> Set[int]:
    misp = misp_factory[method]
    n = len(d_matrix)
    if n < p or p < 2:
        raise ValueError("p must be non-arbitrary")
    if n == p:
        return set([_ for _ in range(n)])
    # Sort distances monotonically.
    ds = np.unique(d_matrix.flatten())[1:]  # 0 distance is arbitrary.
    # Initializing values.
    low, high = 0, len(ds) - 1
    l_bound, u_bound = False, False
    best_subset, best_bound = set(), -1
    logging.info(f"Solving MaxMin Diversity for p={p}")
    logging.info(f"Beginning binary search over distances: {ds}.")
    while high - low > 1 or best_bound < 0:  # Need at least one search. 
        idx = random.randint(low, high)  # Use random indexing.
        logging.info(f"Searching with lower bound: {ds[idx]}")
        graph = Graph(d_matrix < ds[idx])
        subset = misp(graph, p=p)
        v = len(subset)
        if v >= p:
            l_bound = True
            low = idx
            best_subset, best_bound = subset, ds[idx]
            logging.info(f"Found p={v}. Raising lower bound.")
        else:
            u_bound = True
            high = idx
            logging.info(f"Found p={v}. Lowering upper bound.")
    if not u_bound:  # Never established upper bound.
        logging.info(f"Finalizing upper bound: {ds[high]}")
        graph = Graph(d_matrix < ds[high])
        max_subset, max_bound = misp(graph, p=p), ds[high]
        logging.info(f"Established maximum bound with p = {len(max_subset)}")
        if len(max_subset) >= len(best_subset):  # We can use max possible distance.
            logging.info(f"Search complete with bound: [{max_bound}]")
            return max_subset, max_bound
    if not l_bound:
        logging.info(f"Finalizing lower bound: {ds[low]}")
        graph = Graph(d_matrix < ds[low])
        min_subset, min_bound = misp(graph, p=p), ds[low]
        logging.info(f"Established minimum bound with p = {len(min_subset)}")
        best_subset, best_bound = min_subset, min_bound
    logging.info(f"Solution complete with bounds: [{ds[low]},{ds[high]})")
    return best_subset, best_bound
        

def max_min_dispersion_max_min_sum(d_matrix: np.array, p: int, L: float = 0.0, method: str = 'lin') -> Tuple[Set[int], float]:
    """
    Solves constrained Maximum Minimum Dispersion Problem. 

    Specifically, this solves two problems: the maximum-minimum
        dispersion problem under the constraint of L, followed by
        a constrained max-min sum problem. (The latter is actually
        a dual problem of finding the maximum independent set bounded
        by L then the max-min sum problem). 

    Args:
        d_matrix (np.array): Pairwise distance measures between each
            point.
        p (int): Constraint on subset size to satisfy. (Default: 0).
        L (float): Bound on minimum distance between nodes to limit
            selection of nodes. If none indicated, performs search
            using search. (Default: 0)
        search: Search function to find bound for distance constraint.
            Only binary_max_min supported. (Default: binary_max_min_search)
        method (str): String indicating whether to use linear or
            enumerative solver to find L.

    Returns:
        Set[int] of vertex indices satisfying the optimization problem.
    """
    A = []
    lb_ineq = []
    ub_ineq = []

    # Reducing problem size to only nodes fullfilling max-min constaint.
    graph = Graph(d_matrix < L)
    valid_node = list(graph.explore | graph.indep)
    n = len(valid_node)
    
    bounded_matrix = np.where(d_matrix >= L, d_matrix, 0.0)

    ##### Independence constraints #####
    for i in range(n-1):
        v1 = valid_node[i]
        for j in range(i+1, n):
            v2 = valid_node[j]
            if bounded_matrix[v1][v2] == 0:
                ineq = [0 for _ in range(n+1)] # all zeros
                ineq[i] = 1
                ineq[j] = 1
                A.append(ineq)
                lb_ineq.append(0)
                ub_ineq.append(1)

    ##### Max-Min Sum #####
    U = 1 + bounded_matrix.sum(axis=1).max()
    for i, v1 in enumerate(valid_node):
        equa = [0 for _ in range(n)] + [1] # S-var
        equa[i] = U
        for j, v2 in enumerate(valid_node):
            equa[j] = -1 * bounded_matrix[v1][v2]
        A.append(equa)
        lb_ineq.append(L-U-1)
        ub_ineq.append(U)

    # p-constraint
    A.append([1 for _ in range(n)] + [0])  # var
    lb_ineq.append(p)
    ub_ineq.append(p)

    # Obj
    constraint = LinearConstraint(A=A, lb=lb_ineq, ub=ub_ineq)

    # Bounds
    l_bound = [0 for _ in range(n)] + [L]
    u_bound = [1 for _ in range(n)] + [U-1]
    bounds = Bounds(lb=l_bound, ub=u_bound)

    integrality = [3 for _ in range(n)] + [0]

    obj = [0 for _ in range(n)] + [-1]
    solution = milp(c=obj, constraints=constraint, bounds=bounds, integrality=integrality)
    
    xs = solution.x
    if xs is not None:
        xs = [valid_node[idx] for idx, val in enumerate(xs[:-1]) if val > 0.5]
    subset = set(xs)

    independent, violations = graph.is_independent_subset(subset)

    if independent:
        return subset, get_dispersion_min(subset, d_matrix)[1]
    else:
        raise IndependenceError(subset, violations) 