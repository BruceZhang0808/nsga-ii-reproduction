import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
from tqdm import trange, tqdm

from nsga2.algorithms.nsga2 import NSGA2
from nsga2.operators.crossover import sbx
from nsga2.operators.mutation import polynomial_mutation
from nsga2.metrics.metrics import delta, gamma
from nsga2.problems.base import Problem
from nsga2.problems.unconstrained import (
    FON, KUR, POL, SCH, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6,
)


# 实验的结果存放目录，全局使用
RESULTS_DIR = Path(__file__).parents[3] / "results"


@dataclass
class NSGA2_Config:
    pop_size: int
    n_gen: int
    eta_c: float
    eta_m: float
    pc: float
    n_ref: int
    n_runs: int    
    figure_seed: int

class TqdmLoggingHandler(logging.Handler):
    """通过 tqdm.write 输出日志，避免打断进度条刷新"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)

def get_config() -> NSGA2_Config:
    cfg_path = Path(__file__).parent / "configs" / "benchmark.yaml"
    with open(cfg_path, "r") as file:
        configs = yaml.safe_load(file)
    nsga2_cfg = configs["nsga2_real"]
    return NSGA2_Config(**nsga2_cfg)

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[
            logging.FileHandler(RESULTS_DIR / "benchmark.log", mode='w'),   # NOTE mode: 'a' for 'append' / 'w' for 'write'
            TqdmLoggingHandler(),
        ],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


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


def run_zdt4(n_gen: int, eta_m: int):
    RESULTS_DIR.mkdir(exist_ok=True)
    mpl.rc_file(Path(__file__).parent / "configs" / "matplotlibrc")

    problem = ZDT4()
    name = type(problem).__name__
    cfg = get_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[
            logging.FileHandler(RESULTS_DIR / "benchmark.log", mode='a'),   # NOTE mode: 'a' for 'append' / 'w' for 'write'
            TqdmLoggingHandler(),
        ]
    )

    rows = []
    reference = problem.pareto_front(cfg.n_ref)
    pbar = trange(cfg.n_runs, desc=f"Problem: {name}")
    for seed in pbar:
        pbar.set_postfix_str(f"seed={seed}")
        t0 = time.perf_counter()

        nsga2 = NSGA2(problem=problem, pop_size=cfg.pop_size, n_gen=n_gen, seed=seed,
                            crossover=sbx(eta_c=cfg.eta_c, xl=problem.xl, xu=problem.xu, pc=cfg.pc),
                            mutation=polynomial_mutation(eta_m=eta_m, xl=problem.xl, xu=problem.xu))
        pop = nsga2.run()

        F0 = np.array([ind.f for ind in pop if ind.rank == 0])
        if seed == cfg.figure_seed:
            figures_dir = RESULTS_DIR / "figures"
            figures_dir.mkdir(exist_ok=True)

            fig, ax = plt.subplots()
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
            plt.savefig(figures_dir / f"{name}(n_gen={n_gen} eta_m={eta_m}).png")
         
        g, dl = gamma(F0, reference), delta(F0, reference)
        rows.append({"seed": seed, "gamma": g, "delta": dl})
        logging.info("%s(n_gen=%d eta_m=%d) seed=%d gamma=%.6f delta=%.6f (%.1fs)",
                        name, n_gen, eta_m, seed, g, dl, time.perf_counter() - t0)

    g_all = np.array([r["gamma"] for r in rows])
    d_all = np.array([r["delta"] for r in rows])
    logging.info("%s(n_gen=%d eta_m=%d) summary: gamma %.6f±%.6f, delta %.6f±%.6f",
                    name, n_gen, eta_m, g_all.mean(), g_all.std(),
                    d_all.mean(), d_all.std())


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # matplotlib 使用自定义绘图配置
    mpl.rc_file(Path(__file__).parent / "configs" / "matplotlibrc")

    problems: list[Problem] = [SCH, FON, POL, KUR, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]

    cfg = get_config()
    setup_logging(RESULTS_DIR)
    logging.info("operating system: %s", sys.platform)
    logging.info("protocol: pop=%d gen=%d runs=%d",
                 cfg.pop_size, cfg.n_gen, cfg.n_runs)

    summary = {"Parameters": asdict(cfg)}
    for problem_cls in problems:
        problem = problem_cls()
        name = problem_cls.__name__
        has_front = hasattr(problem, "pareto_front")
        reference = problem.pareto_front(cfg.n_ref) if has_front else None
        if not has_front:
            logging.warning("%s: no closed-form front, metrics skipped", name)

        rows = []
        pbar = trange(cfg.n_runs, desc=f"Problem: {name}")
        for seed in pbar:
            pbar.set_postfix_str(f"seed={seed}")
            t0 = time.perf_counter()

            nsga2 = NSGA2(problem=problem, pop_size=cfg.pop_size, n_gen=cfg.n_gen, seed=seed,
                          crossover=sbx(eta_c=cfg.eta_c, xl=problem.xl, xu=problem.xu, pc=cfg.pc),
                          mutation=polynomial_mutation(eta_m=cfg.eta_m, xl=problem.xl, xu=problem.xu))
            pop = nsga2.run()

            X = np.array([ind.x for ind in pop])
            F = np.array([ind.f for ind in pop])
            F0 = np.array([ind.f for ind in pop if ind.rank == 0])

            out = RESULTS_DIR / name
            out.mkdir(exist_ok=True)
            np.save(out / f"solutions_seed{seed}.npy", X)
            np.save(out / f"function_values_seed{seed}.npy", F)

            # 只保存 figure_seed 这一次实验的可视化结果
            if seed == cfg.figure_seed:
                figures_dir = RESULTS_DIR / "figures"
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

            # summary_path = RESULTS_DIR / "summary.json"
            # summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            summary[name] = {
                "gamma_mean": float(g_all.mean()), "gamma_var": float(g_all.var()),
                "delta_mean": float(d_all.mean()), "delta_var": float(d_all.var())
            }
            # summary_path.write_text(json.dumps(summary, indent=2))
            logging.info("%s summary: gamma %.6f±%.6f, delta %.6f±%.6f",
                         name, g_all.mean(), g_all.std(),
                         d_all.mean(), d_all.std())

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    # run_zdt4(500, 10)
