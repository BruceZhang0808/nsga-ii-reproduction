import numpy as np
from nsga2.core.individual import Individual


def dominate(p: Individual, q: Individual) -> bool:
    """Return True if p dominates q else False"""
    if np.all(p.f <= q.f) and np.any(p.f < q.f):
        return True 
    return False


def fast_nondominated_sort(pop: list[Individual]) -> list[list[Individual]]:
    """Fast Nondominated Sorting Approach, returns all the Pareto-optimal fronts."""
    N = len(pop)
    n, S = [0 for _ in range(N)], [[] for _ in range(N)]
    first_front = []
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominate(pop[p], pop[q]):
                S[p].append(q)
            elif dominate(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            pop[p].rank = 0
            first_front.append(p)

    fronts = [first_front]
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    pop[q].rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()

    return [[pop[i] for i in front] for front in fronts]