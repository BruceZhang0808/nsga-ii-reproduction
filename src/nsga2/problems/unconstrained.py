"""Test problems of Deb et al. (2002), Table I.

All objectives are minimized. Each class transcribes one row of the table;
docstrings cite the formulas so the code can be checked line by line against
the paper. Problems with a known closed-form Pareto front also provide
``pareto_front(n_points)`` (used later by the gamma / delta metrics of
Section IV-B). POL and KUR have no closed-form front (the paper refers to
their original studies), so they do not define it.
"""

import numpy as np
from abc import abstractmethod

from .base import Problem


class SCH(Problem):
    """SCH (Schaffer): n=1, x in [-10^3, 10^3].

    f1 = x^2, f2 = (x-2)^2. Convex front, optimal x in [0, 2].
    """

    n_var = 1
    n_obj = 2
    xl = np.array([-1.0e3])
    xu = np.array([1.0e3])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2, (x[0] - 2.0) ** 2])

    def pareto_front(self, n_points: int) -> np.ndarray:
        x = np.linspace(0.0, 2.0, n_points)
        return np.column_stack([x ** 2, (x - 2.0) ** 2])


class FON(Problem):
    """FON (Fonseca & Fleming): n=3, x_i in [-4, 4].

    f1 = 1 - exp(-sum_i (x_i - 1/sqrt(3))^2)
    f2 = 1 - exp(-sum_i (x_i + 1/sqrt(3))^2)
    Nonconvex front, optimal x1 = x2 = x3 in [-1/sqrt(3), 1/sqrt(3)].
    """

    n_var = 3
    n_obj = 2
    xl = np.full(3, -4.0)
    xu = np.full(3, 4.0)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = 1.0 - np.exp(-np.sum((x - 1.0 / np.sqrt(3.0)) ** 2))
        f2 = 1.0 - np.exp(-np.sum((x + 1.0 / np.sqrt(3.0)) ** 2))
        return np.array([f1, f2])

    def pareto_front(self, n_points: int) -> np.ndarray:
        t = np.linspace(-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), n_points)
        f1 = 1.0 - np.exp(-3.0 * (t - 1.0 / np.sqrt(3.0)) ** 2)
        f2 = 1.0 - np.exp(-3.0 * (t + 1.0 / np.sqrt(3.0)) ** 2)
        return np.column_stack([f1, f2])



class POL(Problem):
    """POL (Poloni): n=2, x_i in [-pi, pi]. Nonconvex, disconnected front.

    f1 = 1 + (A1 - B1)^2 + (A2 - B2)^2
    f2 = (x1 + 3)^2 + (x2 + 1)^2
    No closed-form Pareto front.
    """

    n_var = 2
    n_obj = 2
    xl = np.full(2, -np.pi)
    xu = np.full(2, np.pi)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        a1 = 0.5 * np.sin(1.0) - 2.0 * np.cos(1.0) + np.sin(2.0) - 1.5 * np.cos(2.0)
        a2 = 1.5 * np.sin(1.0) - np.cos(1.0) + 2.0 * np.sin(2.0) - 0.5 * np.cos(2.0)
        b1 = 0.5 * np.sin(x[0]) - 2.0 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
        b2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2.0 * np.sin(x[1]) - 0.5 * np.cos(x[1])

        f1 = 1.0 + (a1 - b1) ** 2 + (a2 - b2) ** 2
        f2 = (x[0] + 3.0) ** 2 + (x[1] + 1.0) ** 2
        return np.array([f1, f2])



class KUR(Problem):
    """KUR (Kursawe): n=3, x_i in [-5, 5]. Nonconvex, disconnected front.

    f1 = sum_{i=1}^{n-1} -10 exp(-0.2 sqrt(x_i^2 + x_{i+1}^2))
    f2 = sum_{i=1}^{n} (|x_i|^0.8 + 5 sin(x_i^3))
    The cube is on x_i, inside the sine (Table I).
    No closed-form Pareto front.
    """

    n_var = 3
    n_obj = 2
    xl = np.full(3, -5.0)
    xu = np.full(3, 5.0)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = np.sum(-10.0 * np.exp(-0.2 * np.sqrt(x[:-1] ** 2 + x[1:] ** 2)))
        f2 = np.sum(np.abs(x) ** 0.8 + 5.0 * np.sin(x ** 3))
        return np.array([f1, f2])



# =============================================================================================
class ZDT(Problem):
    """Shared skeleton of the ZDT suite: f1 = x1, g(x2..xn), h(f1, g)."""

    n_var = 30
    n_obj = 2
    xl = np.zeros(n_var)
    xu = np.ones(n_var)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = self._g(x)
        return np.array([f1, g * self._h(f1, g)])

    def _g(self, x: np.ndarray) -> float:
        # g(x) = 1 + 9 (sum_{i=2}^{n} x_i) / (n - 1)    [ZDT1-ZDT3]
        return 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)

    @abstractmethod
    def _h(self, f1: float | np.ndarray, g: float) -> float | np.ndarray: ...

    def pareto_front(self, n_points: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, self._h(f1, 1.0)])


class ZDT1(ZDT):
    """ZDT1: n=30, x_i in [0, 1]. Convex front.

    f2 = g(x) [1 - sqrt(x1 / g(x))]
    """

    def _h(self, f1: float | np.ndarray, g: float) -> float | np.ndarray:
        return 1.0 - np.sqrt(f1 / g)


class ZDT2(ZDT):
    """ZDT2: n=30, x_i in [0, 1]. Nonconvex front.

    f2 = g(x) [1 - (x1 / g(x))^2]
    """

    def _h(self, f1: float | np.ndarray, g: float) -> float | np.ndarray:
        return 1.0 - (f1 / g) ** 2


class ZDT3(ZDT):
    """ZDT3: n=30, x_i in [0, 1]. Convex, disconnected front (5 segments).

    f2 = g(x) [1 - sqrt(x1 / g(x)) - (x1 / g(x)) sin(10 pi x1)]
    pareto_front samples the whole g=1 curve; the true front is its
    nondominated subset.
    """

    def _h(self, f1: float | np.ndarray, g: float) -> float | np.ndarray:
        # the paper writes sin(10 pi x1); f1 = x1 on the front
        return 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)


class ZDT4(ZDT):
    """ZDT4: n=10, x1 in [0, 1], x_i in [-5, 5] for i = 2..n. Nonconvex front
    with ~21 local fronts; same front shape as ZDT1.

    g(x) = 1 + 10(n - 1) + sum_{i=2}^{n} (x_i^2 - 10 cos(4 pi x_i))
    """

    n_var = 10
    xl = np.array([0.0] + [-5.0] * 9)
    xu = np.array([1.0] + [5.0] * 9)

    def _g(self, x: np.ndarray) -> float:
        return 1.0 + 10.0 * (self.n_var - 1) + np.sum(
            x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:])
        )

    def _h(self, f1: float | np.ndarray, g: float) -> float | np.ndarray:
        return 1.0 - np.sqrt(f1 / g)


class ZDT6(ZDT):
    """ZDT6: n=10, x_i in [0, 1]. Nonconvex, nonuniformly spaced front.

    f1 = 1 - exp(-4 x1) sin^6(6 pi x1)   (note: f1 is not x1 here)
    g(x) = 1 + 9 [(sum_{i=2}^{n} x_i) / (n - 1)]^0.25
    f2 = g(x) [1 - (f1 / g(x))^2]
    The nondominated front has f1 in [0.280775, 1] (smaller f1 values are
    unreachable on the g=1 curve).
    """

    n_var = 10
    xl = np.zeros(10)
    xu = np.ones(10)


    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6.0 * np.pi * x[0]) ** 6
        g = self._g(x)
        return np.array([f1, g * self._h(f1, g)])

    def _g(self, x: np.ndarray) -> float:
        return 1.0 + 9.0 * (np.sum(x[1:]) / (self.n_var - 1)) ** 0.25

    def _h(self, f1: float | np.ndarray, g: float) -> float | np.ndarray:
        return 1.0 - (f1 / g) ** 2

    def pareto_front(self, n_points: int) -> np.ndarray:
        f1 = np.linspace(0.280775, 1.0, n_points)
        return np.column_stack([f1, self._h(f1, 1.0)])
