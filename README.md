---
Date: 2026-10-2
Author: Bruce Zhang
Email: zhangyixiang0808@gmail.com
---
# NSGA-II 复现实验报告

本项目为我对 Deb et al. 于 2002 年提出的 NSGA-II 算法论文 [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](docs/Deb_NSGAII.pdf) 进行复现，使用 NumPy 从零实现 NSGA-II Real-coded 与无约束测试问题，并在论文的实验参数下验证 CONVERGENCE metric $\Upsilon$ 和 Diversity metric $\Delta$，并与论文结果进行了对比与分析。

## 一、复现范围

已复现的论文内容：

- **NSGA-II Real-coded core**（Section III）：Fast Nondominated Sorting Approach, Crowding distance
- **GA operators**：Binary tournament selection, Simulated binary crossover, Polynomial mutation
- **Test problems**（Table I）：SCH, FON, POL, KUR, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6 共 9 个无约束问题
- **Metrics**（Section IV-B）：CONVERGENCE metric $\Upsilon$ and Diversity metric $\Delta$
- **Experiment**（Table II, III）：完成了实验并得到了 NSGA-II Real-coded 在九个实验上的指标结果
- **算法在 ZDT4 上的改进分析**（Section IV-D）：针对常规实验参数下，算法在 ZDT4 上的不收敛问题，按照论文分析进行了参数的改进

## 二、算法流程

整体流程如下：

![NSGA-II 算法流程图](docs/flowchart.png)


## 三、Benchmark 实验设置


<div align="center">

| 参数 | 值 |
|:---:|:---:|
| 种群规模 | 100 |
| 代数 | 250 |
| 每个问题独立重复实验次数 | 10 |
| 交叉概率 $p_c$ | 0.9 |
| SBX 分布指数 $\eta_c$ | 20 |
| 变异概率 $p_m$ | $\frac{1}{n}$|
| 多项式变异分布指数 $\eta_m$ | 20 |

</div>

> 注：变异概率中 n 为变量个数

## 四、实验结果

### 4.1 主实验：指标 $\Upsilon$ 和 $\Delta$ 实验结果与论文对照

<div align="center">

| 问题 | 论文 $\overline{\Upsilon}$| 复现 $\overline{\Upsilon}$  | 论文 $\sigma_{\Upsilon}^2$ | 复现 $\sigma_{\Upsilon}^2$ |
|:----:|:-----------------:|:-----------------:|:----------------:|:--------------------------------:|
| **SCH** | 0.003391  | **0.003485**  | 0.000000                        | **0.000000**            |
| **FON** | 0.001931  | **0.002180**  | 0.000000                  | **0.000000**            |
| **ZDT1** | 0.033482  | **0.002822**  | 0.004750                  | **0.000000**            |
| **ZDT2** | 0.072391  | **0.002006**  | 0.031689                  | **0.000000**            |
| **ZDT3** | 0.114500  | **0.004405**  | 0.007940                  | **0.000000**            |
| **ZDT4** | 0.513053  | **2.120603**  | 0.118460                  | **1.982629**            |
| **ZDT6** | 0.296564  | **0.020963**  | 0.013135                  | **0.000175**            |

</div>

<div align="center">

| 问题 | 论文 $\overline{\Delta}$| 复现 $\overline{\Delta}$  | 论文 $\sigma_{\Delta}^2$ | 复现 $\sigma_{\Delta}^2$ |
|:----:|:-----------------:|:-----------------:|:----------------:|:--------------------------------:|
| **SCH** | 0.477899         | **0.279187**                | 0.003471 | **0.000801**     |
| **FON** | 0.378065         | **0.356351**                | 0.000639|  **0.000808**     |
| **ZDT1** | 0.390307         | **0.382973**                | 0.001876|  **0.000944**     |
| **ZDT2** | 0.430776         | **0.388099**                | 0.004721|  **0.000878**     |
| **ZDT3** | 0.738540         | **0.658315**                | 0.019706|  **0.002004**     |
| **ZDT4** | 0.702612         | **1.262784**                | 0.064648|  **0.015270**    |
| **ZDT6** | 0.668025         | **0.684165**                | 0.009923|  **0.070410**    |

</div>

> 注：
> 1. 由于论文没有给出 POL 和 KUR 问题的精确最优解，故本次复现没有在这两个问题上计算指标，收敛曲线见 4.2 小节；
> 2. 所有小数精确到小数点后 6 位；
> 3. 原始结果保存在 `results/benchmark.log` 和 `results/summary.json`。

**结论：**

结果非常的 amazing 啊！除 ZDT4 外，本次复现在两个指标上的结果均持平或优于论文结果（两个指标都是越小越好），方差部分更是显著优于论文1~2个数量级，说明本次复现结果的稳定性更强。针对 ZDT4 问题的分析请看 4.3 小节。

### 4.2 收敛曲线图

每个问题保存第一次独立实验（对应的随机数种子 seed = 0）得到的的非支配解分布图，均保存在 `results/figures`。ZDT4 问题放在下一小节详细分析，其他问题的结果图如下：

<p align="center">
  <img src="results/figures/SCH.png" width="24%" alt="SCH"/>
  <img src="results/figures/FON.png" width="24%" alt="FON"/>
  <img src="results/figures/POL.png" width="24%" alt="POL"/>
  <img src="results/figures/KUR.png" width="24%" alt="KUR"/>
  <br/>
  <img src="results/figures/ZDT1.png" width="24%" alt="ZDT1"/>
  <img src="results/figures/ZDT2.png" width="24%" alt="ZDT2"/>
  <img src="results/figures/ZDT3.png" width="24%" alt="ZDT3"/>
  <img src="results/figures/ZDT6.png" width="24%" alt="ZDT6"/>
</p>



### 4.3 ZDT4 专题分析

#### 4.3.1 初始分析

在初始参数设定下（见第三节，$\eta_m$=20, 250 代），本次复现与论文收敛曲线图对比如下：

<p align="center">
  <img src="results/figures/ZDT4.png" width="45%" alt="ZDT4"/>
  <img src="docs/zdt4_raw.png" width="45%" alt="paper ZDT4"/>
</p>

结果非常差，随后进行改进。

#### 4.3.2 改进分析

论文的 Section IV-C 有原话如下：

> The problem ZDT4 has $21^9$ or 7.94($10^{11}$ ) different local Pareto-optimal fronts in the search space, of which only one corresponds to the global Pareto-optimal front. The Euclidean distance in the decision space between solutions of two consecutive local Pareto-optimal sets is 0.25. Fig. 10 shows that both real-coded NSGA-II and PAES get stuck at different local Pareto-optimal sets, but the convergence and ability to find a diverse set of solutions are definitely better with NSGA-II. 

接下来论文在 Section IV-D 部分尝试改变参数，的确取得了更好的结果。本次复现与论文对应，设置不同的参数得到的结果和收敛图如下：

<div align="center">

| 配置 | $\overline{\Upsilon}$ | $\overline{\Delta}$ |
|:---:|:---:|:---:|
| $\eta_m$=20, 250 代（初始） | 2.120603 | 1.262784 |
| $\eta_m$=10, 250 代 | 1.929798 | 1.357815|
| $\eta_m$=10, 500 代 | 1.206907 | 1.231298 |

</div>

<p align="center">
<table width="100%">
  <tr align="center">
    <td width="33%">
      <figure>
        <img src="results/figures/ZDT4(eta_m=10).png" width="100%" alt="ZDT4(eta_m=10)"/>
        <figcaption>复现：ηₘ=10，250 代</figcaption>
      </figure>
    </td>
    <td width="33%">
      <figure>
        <img src="results/figures/ZDT4(n_gen=500 eta_m=10).png" width="100%" alt="ZDT4(n_gen=500 eta_m=10)"/>
        <figcaption>复现：ηₘ=10，500 代</figcaption>
      </figure>
    </td>
    <td width="33%">
      <figure>
        <img src="docs/zdt4_better.png" width="100%" alt="zdt4_better"/>
        <figcaption>论文 Fig. 12：ηₘ=10，250 代</figcaption>
      </figure>
    </td>
  </tr>
</table>
</p>

**分析：**

本次复现结果仍然没有论文效果好，可能和随机数有关，因为我发现我 10 次实验中，永远是 seed = 0 那一次实验的结果最好。（希望这个借口能安抚我受伤的心灵）



## 五、待完善部分


1. **算法对比 (Table II, III)**：实现 PAES、SPEA 与 NSGA-II Binary-coded，并完成 Benchmark 实验；
2. **参数实验（Table IV）**：POL, KUR, ZDT3, ZDT6 的 500 代版本；
3. **旋转问题（Section V）**：Rotated problems 及其求解；
4. **约束处理（Section VI）**：Constrained NSGA-II，及 Table V 的 CONSTR, SRN, TNK, WATER 四个约束问题。


## 六、复现说明

1. 首先克隆此仓库

```bash
git clone git@github.com:BruceZhang0808/nsga-ii-reproduction.git
```

2. 创建虚拟环境

```
cd nsga-ii-reproduction
uv sync
source .venv/bin/activate
```

3. 在根目录运行实验

```bash
python3 -m nsga2.experiments.run_benchmark
```
实验配置在 `src/nsga2/experiments/configs/benchmark.yaml`，结果输出至 `results/`。


## 七、代码仓库详细说明

本项目代码编写遵从现代 Python "src layout" 结构，项目结构如下

```
nsga-ii-reproduction/
├── docs/                          # 复现所依据的论文与报告插图
├── results/                       # 实验输出：
│   │                              #   各问题的解的 .npy 文件, summary.json, benchmark.log
│   └── figures/                   #   各问题的非支配解分布图
├── src/nsga2/                     # 核心源码
│   ├── algorithms/                # NSGA-II 主流程
│   ├── core/                      # Individual, Fast Nondominated Sorting Approach, crowding distance
│   ├── operators/                 # GA operators: Binary tournament selection, SBX, polynomial mutation
│   ├── problems/                  # 论文 Table I 的 9 个测试问题
│   ├── metrics/                   # 两个指标
│   └── experiments/               # 实验脚本与配置
│       ├── configs/               #   实验及画图配置文件
│       └── run_benchmark.py       #   benchmark 实验入口
├── README.md
├── pyproject.toml
└── uv.lock
```

## 八、项目开发随想

就这么一个小小的大作业，都花了我总共估计 15h+ 才完成。如果只是为了尽快复现论文以完成作业，那我 2h 不到就可以完成，甚至让 ai 5分钟做出来，但这样做后果无疑是灾难：所有代码放在 1~2 个文件当中，想增加新功能得重新改动大部分代码，所有参数全部设置为全局变量……我很清楚这一点，毕竟我大一就是这样写代码的；我也很清楚如果不使用更复杂的提示词的话 ai 也会这样干，因为我打数学建模比赛时 ai 就是给我返回的这样的代码（竟然还混到了省二）。在现在这个 ai 盛行、人心浮躁的时代，我希望能静下心来，好好的琢磨一下如何能真正提升自己的代码能力。我不希望 ai 支配我，而应该是相反，由我来掌控这个工具。

因此在做这次作业时，我给自己立下了几条死规矩：
1. 核心代码必须自己手敲，如果遇到问题可以让 ai 指导，但一定不能直接让他写好
2. 努力培养自己的工程师思维，将设计分成多个模块，让代码便于维护
3. 代码必须简洁易读，并且符合 Python 开发的社区规范

完成情况十分良好：
1. 只有 `src/nsga2/problems/unconstrained.py` 和 `src/nsga2/metrics/metrics.py` 两个文件由 ai 生成，其他所有代码均为自己手敲
2. 在设计时很大程度上参考了 [pymoo: Multi-objective Optimization in Python](https://pymoo.org/) 这个优秀的项目，模块的设计思路使其十分利于维护，方便我增加新功能、完成更完整的复现
3. 好吧反正我认为我写的代码可读性还行，但是我估计其他人应该还是会认为这是屎山（毕竟程序员最讨厌读别人的代码），如果有更好的建议欢迎 email me

我认为收获最大的地方在于实验代码的部分，毕竟我以后还会跑更多更大的深度学习相关的实验，所以这部分的设计很重要，我也学到了许多：
1. 使用 `logging` 模块记录实验日志，我读了一篇很好的 [How-To Guide](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial)
2. 使用 `pathlib` 管理文件路径，这部分太重要了，毕竟我也经历过太多 `FileNotFoundError` 了😭
3. 实验配置以及 `matplotlib` 的配置单独放在一个 `config` 目录下，在脚本中使用 `@dataclass` 创建 `Config` 类；总之不要硬编码，或设置全局常量

写代码很痛苦了，代码出 bug 了更痛苦，但是只要静下心来、保持专注，我发现我也能享受这个过程。不要急于求成，一步一个脚印，去做真正有意义的事情。