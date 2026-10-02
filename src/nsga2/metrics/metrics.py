"""Performance metrics of Deb et al. (2002), Section IV-B.

gamma measures convergence (closeness to the true Pareto-optimal front),
delta measures spread (uniformity of the obtained nondominated set).
Both consume plain arrays so they are testable without the algorithm.
"""

import numpy as np


def gamma(F: np.ndarray, reference: np.ndarray) -> float:
    """Convergence metric: mean distance to the closest true-front point.

    Args:
        F: obtained objective values, shape (n, n_obj).
        reference: uniformly spaced points on the true front,
            shape (n_ref, n_obj) -- e.g. ``problem.pareto_front(500)``.
    """
    # (n, n_ref) pairwise Euclidean distances; fine at this scale
    d = np.linalg.norm(F[:, None, :] - reference[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def delta(F: np.ndarray, reference: np.ndarray) -> float:
    """Diversity metric: 0 for a perfectly uniform front with endpoints hit.

    d_f / d_l are the distances from the true front's extreme points to the
    obtained set; d_i are consecutive distances after sorting by f_1.
    """
    if len(F) < 2:
        return float("nan")

    F = F[np.argsort(F[:, 0])]
    d = np.linalg.norm(np.diff(F, axis=0), axis=1)     # d_i, i = 1..N-1
    d_bar = d.mean()

    # extreme solutions = the two endpoints of the true front
    # (pareto_front() generates f_1 ascending, so they are reference[0]/[-1]);
    # nearest obtained solution plays the "boundary solution" role
    d_f = np.linalg.norm(F - reference[0], axis=1).min()
    d_l = np.linalg.norm(F - reference[-1], axis=1).min()

    n = len(F)
    return float((d_f + d_l + np.sum(np.abs(d - d_bar)))
                 / (d_f + d_l + (n - 1) * d_bar))
