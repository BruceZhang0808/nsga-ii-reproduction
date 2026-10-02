import numpy as np
from nsga2.core.individual import Individual


def binary_tournament(pop: list[Individual], rng: np.random.Generator):
    i1, i2 = rng.choice(len(pop), size=2, replace=False)
    ind1, ind2 = pop[i1], pop[i2]

    if ind1.rank < ind2.rank or (ind1.rank == ind2.rank and ind1.crowding_dist > ind2.crowding_dist):
        return ind1
    else:
        return ind2