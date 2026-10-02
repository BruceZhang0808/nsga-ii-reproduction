import numpy as np
from tqdm import trange

from nsga2.core.individual import Individual
from nsga2.problems.base import Problem
from nsga2.core.dominance import fast_nondominated_sort
from nsga2.core.crowding import crowding_distance_assignment
from nsga2.operators.selection import binary_tournament

class NSGA2:

    def __init__(
        self,
        problem: Problem,
        pop_size: int,
        n_gen: int,
        seed: int, 
        crossover,
        mutation
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.rng = np.random.default_rng(seed)
        self.crossover = crossover
        self.mutation = mutation

    def initialize_population(self) -> list[Individual]:
        """Initialize the population and set all the individuals' `x` and `f`."""
        pop = []
        for _ in range(self.pop_size):
            x = self.rng.uniform(self.problem.xl, self.problem.xu, size=self.problem.n_var)
            f = self.problem.evaluate(x)
            ind = Individual(x, f)
            pop.append(ind)
        return pop

    def evaluate(self, pop: list[Individual]):
        """Compute all the individuals' fitness by setting their `rank` and `crowding_dist`."""
        fronts = fast_nondominated_sort(pop)
        crowding_distance_assignment(fronts)

        return fronts

    def evolve(self, pop: list[Individual]):
        """The NSGA-II procedure for selecting the next generation."""
        assert all(ind.rank is not None and ind.crowding_dist is not None for ind in pop)

        offspring = []
        for _ in range(0, self.pop_size, 2):
            p1 = binary_tournament(pop, rng=self.rng)
            p2 = binary_tournament(pop, rng=self.rng)

            o1, o2 = self.crossover(p1, p2, rng=self.rng)

            self.mutation(o1, rng=self.rng)
            self.mutation(o2, rng=self.rng)

            o1.f = self.problem.evaluate(o1.x)
            o2.f = self.problem.evaluate(o2.x)
            offspring.extend([o1, o2])

        fronts = self.evaluate(pop + offspring)

        for front in fronts:
            front.sort(key=lambda ind: ind.crowding_dist, reverse=True)
        pop_new = [ind for front in fronts for ind in front]

        return pop_new[:self.pop_size]


    def run(self):
        """Run NSGA-II and return the final population."""
        pop = self.initialize_population()
        self.evaluate(pop)

        for _ in trange(self.n_gen):
            pop = self.evolve(pop)

        return pop
