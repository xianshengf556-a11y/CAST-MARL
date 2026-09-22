# CAST-MARL — 可复现实验材料

[English](README.md) | **中文**

论文的代码、配置与逐种子实验记录

> **CAST-MARL: An Engineering Framework for Safety-Constrained Cooperative
> Multi-UAV Coverage Planning**
> 冯素慧、李琳、张奇志
> *IEEE Access*

本仓库用于支持论文的独立核验，包含完整实验源码、每一次报告实验所用的确切配置、
每张表格背后的逐种子与逐回合记录，以及一条命令即可从零复现全部运行的驱动脚本。

---

## 目录结构

```
experimental_runs/
  reproducibility/     实验源码与基准模块
  *.pth                两个存档检查点
scripts/               产出论文各表的驱动脚本
  run_terrain_controlled.py         受控基准与组件消融
  collect_controlled_results.py     汇总为论文表格
  exp_fallback_threshold.py         空可行集统计与阈值扫描
  run_runtime_stats.py              滤波器决策耗时分布
  exp_low_coverage.py               低覆盖场景检验
  exp_correction_scale.py           修正半径扫描
  exp_crossmap_comparison.py        未知地图上规划器与安全层对比
  exp_reward_sensitivity.py         奖励系数敏感性
  run_control_single_terrain_160.py 等训练预算对照
archived_configs/      各地形的原始配置（逐字沿用）
dataset/               任务数据集（0.5 MB）
results/               论文每个数字对应的逐种子记录与汇总表（见 results/README.md）；
                       训练日志不随仓库发布，由 run_all.sh 重新生成
run_all.sh             一键复现
```

`exp_*.py` 与预算对照属于单点分析脚本，各自独立运行并输出一个 CSV，
是论文正文与回复信中对应结论的证据来源。

## 运行环境

Python 3.8 及以上，依赖 `numpy`、`scipy`、`matplotlib`、`torch`。

**不需要 GPU。** 模型仅 44.4 万参数，峰值显存约 99 MB，计算瓶颈在仿真（CPU），
因此**核数远比显卡重要**。论文中的全部结果均在纯 CPU 容器上产出。

```bash
pip install -r requirements.txt
python scripts/bundle_paths.py     # 自检：打印全部解析后的路径
```

## 复现方式

```bash
bash run_all.sh                    # 全部阶段，并行数 = 核数
STAGES=0 bash run_all.sh           # 只跑协议自检（约 2 分钟）
WORKERS=16 bash run_all.sh         # 指定并行数
```

| 阶段 | 内容 |
|---|---|
| 0 | 环境检查 + 对已发表数字的协议自检 |
| 1 | 地形基准与组件消融（5 个种子） |
| 2 | 空可行集统计与阈值扫描 |
| 3 | 滤波器计时分布（须在空闲机器上运行） |
| 4 | 汇总为论文表格 |

阶段 1 是主要开销：3 地形 × 5 种子 × 9 个模型 = **135 个训练作业**。每个作业在独立
进程中训练单个模型，因此所有模型从相同的种子状态出发，同时能充分使用多核机器。

## 驱动脚本所用的实验协议

受控驱动 `scripts/run_terrain_controlled.py` 采用如下协议，并会把协议写入每次运行的
输出 JSON：

* 训练与评估使用**同一个地形标签**；
* 评估种子**固定并记录**；
* 每个条件使用 **5 个训练种子**；
* 每个模型在**独立进程**中训练，因此所有模型从相同的种子状态出发；
* 冲突统计以**逐回合样本**导出，据此可还原整数对事件数。

## 校验

`scripts/_validate_protocol.py` 会重跑论文中 *Raw A\*（无滤波器）* 参考行所用的协议，
并把覆盖率、冲突率、路径长度与已发表数值比较，可复现到 5 位有效数字：

```
coverage  99.50529 +- 1.16123         对比已发表  99.505 +- 1.161      偏差 0.000%
conflict  0.01025 +- 0.01928          对比已发表  0.01025 +- 0.01928   偏差 0.035%
path      20884.66659 +- 4264.67166   对比已发表  20884.7 +- 4264.7    偏差 0.000%
轨迹重算自校验：max |sim - recomputed| = 0.000e+00   通过
```

**若此项不通过，后续阶段的结果不可信。**

## 消融变体

| 变体 | 主干网络 | 参数量 |
|---|---|---|
| 完整模型（CAST-MARL） | 交互感知 Transformer | 444,118 |
| 无注意力 | 保留 Transformer 主干，多头注意力替换为对 agent 轴的无参均值聚合 | 444,118 |
| 无 Transformer | 共享两层 MLP | 54,279 |

## 数据与指标

论文中所有数值均来自未改动的仿真器步进函数。覆盖率为至少被观测一次的任务单元比例；
冲突率统计间距低于报告阈值的无序飞行器对，并按"对-步"机会数归一化；路径长度累加
欧氏位移。冲突统计以**整数对事件数配精确 Poisson 置信区间**报告，因为在该量级下
单个事件就会让报告值变动约 2.4e-4。

## 许可

MIT（见 `LICENSE`）。若需复用地形场景素材，请先与通讯作者联系。
