import json
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import trange

from nsga2.algorithms.nsga2 import NSGA2
from nsga2.operators.crossover import sbx
from nsga2.operators.mutation import polynomial_mutation
from nsga2.metrics.metrics import delta, gamma
from nsga2.problems.base import Problem
from nsga2.problems.unconstrained import (
    FON, KUR, POL, SCH, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6,
)

# matplotlib 使用自定义绘图配置
mpl.rc_file(Path(__file__).parent / "configs" / "matplotlibrc")

# 实验参数
pop_size = 100
n_gen = 250
eta_c = 20
eta_m = 20

n_ref = 500
n_runs = 10
problems: list[Problem] = [SCH, FON, POL, KUR, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]

# 实验结果保存位置
results_dir = Path(__file__).parent.parent / "results"

# 当seed为多少时保存当次实验的可视化结果
figure_seed = 0

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[
            logging.FileHandler(results_dir / "benchmark.log"),
            logging.StreamHandler(),
        ],
    )


def visualize(F0: np.ndarray, problem: Problem, save_dir: Path | None = None):
    name = type(problem).__name__

    fig, ax = plt.subplots()
    if hasattr(problem, "pareto_front"):
        pf = problem.pareto_front(n_points=2000)
        ax.plot(pf[:, 0], pf[:, 1], 'r-', lw=1.5, 
                label="Pareto-optimal front", zorder=0)
        
    ax.scatter(F0[:, 0], F0[:, 1], 
               s=36, c="#2a78d6", linewidths=0, 
               label="Nondominated solutions")
    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$", labelpad=10, rotation=0)
    ax.set_title(f"NSGA-II on {name}")   
    ax.legend()  
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / f"{name}.png")


def main():
    results_dir.mkdir(exist_ok=True)
    setup_logging()
    logging.info("protocol: pop=%d gen=%d runs=%d ref_points=%d",
                 pop_size, n_gen, n_runs, n_ref)

    summary: dict[str, dict] = {}

    for problem_cls in problems:
        problem = problem_cls()
        name = problem_cls.__name__
        has_front = hasattr(problem, "pareto_front")
        reference = problem.pareto_front(n_ref) if has_front else None
        if not has_front:
            logging.warning("%s: no closed-form front, metrics skipped", name)

        rows = []
        for seed in trange(n_runs, desc=f"Problem: {name}"):
            t0 = time.perf_counter()

            nsga2 = NSGA2(problem=problem, pop_size=pop_size, n_gen=n_gen, seed=seed,
                          crossover=sbx(eta_c=eta_c, xl=problem.xl, xu=problem.xu),
                          mutation=polynomial_mutation(eta_m=eta_m, xl=problem.xl, xu=problem.xu))
            pop = nsga2.run()

            X = np.array([ind.x for ind in pop])
            F = np.array([ind.f for ind in pop])
            F0 = np.array([ind.f for ind in pop if ind.rank == 0])

            out = results_dir / name
            out.mkdir(exist_ok=True)
            np.save(out / f"solutions_seed{seed}.npy", X)
            np.save(out / f"function_values_seed{seed}.npy", F)

            if seed == figure_seed:
                figures_dir = results_dir / "figures"
                figures_dir.mkdir(exist_ok=True)
                visualize(F0, problem, save_dir=figures_dir)


            if has_front:                
                g, dl = gamma(F0, reference), delta(F0, reference)
                rows.append({"seed": seed, "gamma": g, "delta": dl})
                logging.info("%s seed=%d gamma=%.6f delta=%.6f (%.1fs)",
                             name, seed, g, dl, time.perf_counter() - t0)
            else:
                logging.info("%s seed=%d saved, metrics skipped (%.1fs)",
                             name, seed, time.perf_counter() - t0)

        if rows:
            g_all = np.array([r["gamma"] for r in rows])
            d_all = np.array([r["delta"] for r in rows])
            summary[name] = {
                "gamma_mean": float(g_all.mean()), "gamma_var": float(g_all.var()),
                "delta_mean": float(d_all.mean()), "delta_var": float(d_all.var()),
                "n_runs": len(rows),
            }
            logging.info("%s summary: gamma %.6f±%.6f, delta %.6f±%.6f",
                         name, g_all.mean(), g_all.std(),
                         d_all.mean(), d_all.std())

        # NOTE
        break

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("wrote %s", results_dir / "summary.json")



if __name__ == "__main__":
    main()