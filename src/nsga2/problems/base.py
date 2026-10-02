import numpy as np
from abc import ABC, abstractmethod


class Problem(ABC):
    n_var: int
    n_obj: int
    xl: np.ndarray      # the lower bound of x, shape: (n_var,)
    xu: np.ndarray

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray: ...   # shape: (n_obj,)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.empty(0)
