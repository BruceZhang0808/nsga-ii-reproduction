import numpy as np
from nsga2.core.individual import Individual
from nsga2.problems.base import Problem


def bitwise_mutation():
    pass


def polynomial_mutation(
        eta_m: int, 
        xl: np.ndarray,
        xu: np.ndarray,
        pm: float | None = None
    ):
    """The mutation operator in place"""
    n_var = xl.size
    if pm is None:
        pm = 1 / n_var

    def mutation(ind: Individual, rng: np.random.Generator):
        mask = rng.random(n_var) < pm
        if not mask.any():
            return
        
        u = rng.random(n_var)
        delta1 = (ind.x - xl) / (xu - xl)
        delta2 = (xu - ind.x) / (xu - xl)

        delta_tilde = np.where(u <= 0.5,
                               (2*u + (1-2*u)*(1-delta1)**(eta_m+1))**(1/(eta_m+1)) - 1,
                               1 - (2*(1-u) + (2*u-1)*(1-delta2)**(eta_m+1))**(1/(eta_m+1)))
        ind.x[mask] += delta_tilde[mask] * (xu - xl)[mask]

    return mutation