import numpy as np
from dataclasses import dataclass

@dataclass
class Individual:
    x: np.ndarray                           # a vector represents all the variables, shape: (n_var, )
    f: np.ndarray | None = None             # the values of objective functions, shape: (n_obj, )
    rank: int | None = None                 # rank in "fast nondominated sort"
    crowding_dist: float | None = None  