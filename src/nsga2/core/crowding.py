import numpy as np
from nsga2.core.individual import Individual


def crowding_distance_assignment(fronts: list[list[Individual]]) -> None:
    """
    Set all the individuals' crowding-distance **in place** according to the Pareto-optimal fronts.

    """
    for front in fronts:
        l = len(front)
        if l <= 2:
            for ind in front:
                ind.crowding_dist = np.inf
            continue

        F = np.array([ind.f for ind in front])
        dist = np.zeros(l)

        n_obj = F.shape[1]
        for m in range(n_obj):
            f = F[:, m]
            order = np.argsort(f)
            f_sorted = F[order, m]

            f_max, f_min = np.max(f), np.min(f)
            if f_max == f_min:
                continue

            dist[order[0]] = dist[order[l-1]] = np.inf
            dist[order[1:-1]] += (f_sorted[2:] - f_sorted[:-2]) / (f_max - f_min)

        for ind, d in zip(front, dist):
            ind.crowding_dist = d
    return 

