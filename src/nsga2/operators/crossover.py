import numpy as np
from nsga2.core.individual import Individual


def single_point_crossover():
    pass


def sbx(eta_c: float, xl: np.ndarray, xu: np.ndarray, pc: float):
    """Bounded SBX (Deb & Agrawal, 1995), per-variable."""
    exponent = 1.0 / (eta_c + 1.0)

    def crossover(p1: Individual, p2: Individual,
                  rng: np.random.Generator) -> tuple[Individual, Individual]:
        if rng.random() > pc:
            return Individual(x=p1.x.copy()), Individual(x=p2.x.copy())
        
        x1 = np.minimum(p1.x, p2.x)
        x2 = np.maximum(p1.x, p2.x)
        diff = x2 - x1

        # 父代重合的维度：β_max 无定义，但此时子代恒等于父代（diff=0），只需避开除零——用 where 兜底成 1
        beta_max = np.where(
            diff > 1e-14,
            1.0 + 2.0 * np.minimum(x1 - xl, xu - x2) / diff,
            1.0,
        )
        alpha = 2.0 - beta_max ** (-(eta_c + 1.0))

        u = rng.random(x1.shape)
        beta_q = np.where(
            u <= 1.0 / alpha,
            (u * alpha) ** exponent,
            (1.0 / (2.0 - u * alpha)) ** exponent,
        )

        o1 = Individual(x=np.clip(0.5 * ((1 + beta_q) * x1 + (1 - beta_q) * x2), xl, xu))
        o2 = Individual(x=np.clip(0.5 * ((1 - beta_q) * x1 + (1 + beta_q) * x2), xl, xu))

        return o1, o2

    return crossover