# SDRE 最优性分析：TPBVP 真最优 vs SDRE 近似

## 核心问题

SDRE（State-Dependent Riccati Equation）用代数 Riccati 方程（ARE）的稳态解 $P_\infty$ 近似有限时间水平问题的时变 Riccati 解 $P(t)$。这个近似到底损失了多少最优性？

## 方法论路线

### 阶段一：CasADi + IPOPT 直接配点法
- **文件**: `aerospace/control/optimal_control_solver.py`
- 用 CasADi opti 接口 + 梯形配点法求解时间-能量最优控制
- 归一化时间 $\tau = t/T \in [0,1]$，消除自由终端时间的非线性
- 5 个 $\gamma$ 值 (1e5~1e7) 全部收敛，终端误差 ~1mm
- 每个求解 ~1s，200-350 次 IPOPT 迭代

### 阶段二/三：逆最优 R(x) 拟合
- **文件**: `aerospace/control/inverse_optimal_fit.py`
- 从最优轨迹反演 R(x)：在各状态点搜索使 SDRE 控制匹配 $u^*$ 的标量 r
- Sigmoid 拟合：$R(r) = 1.24e13 + (1.37e12 - 1.24e13) \cdot \sigma((r - 569) / 1194)$
- 远处 R≈1.37e12（小→激进）、近处 R≈1.24e13（大→保守），符合物理直觉
- **但用于闭环仿真时，相对固定 R 的改善仅 3.6%**（7.71h vs 7.99h）

### 阶段四：TPBVP — 真正的最优解
- **文件**: `experiment_tpbvp_compare.py`
- 用 Pontryagin 最小值原理将微分博弈鞍点问题转化为 10 维 TPBVP
- `scipy.integrate.solve_bvp` + continuation（从 T/8 逐步延至 T）求解
- 单次求解 ~5.7s，收敛到 600-700 节点自适应网格

### 阶段五：时变增益因子
- **文件**: `experiment_gain_factor.py`
- 从 TPBVP 提取 $f(t_{go}) = ||u^*|| / ||u_{SDRE}||$，拟合 $f = 1 - e^{-\alpha t_{go}}$

## 关键发现

### 1. 博弈鞍点 TPBVP 不稳定

用 $\gamma=\sqrt{2}$ 的微分博弈求解 TPBVP 时，不同随机初始化给出差异 40 倍的代价函数值。鞍点问题的 BVP 求解器无法可靠收敛到全局鞍点。

**解决方案**：切换到纯追捕（$\gamma \to \infty$），代价变为凸函数，解唯一且稳定。

### 2. SDRE 存在 ~149% 的次优性

| 方法 | 代价 J (纯追捕) | 与最优差距 |
|------|----------------|-----------|
| TPBVP 真最优 | 2.70e+07 | — |
| SDRE (稳态 P) | 6.72e+07 | +149% |
| SDRE + f(t_go) | 6.72e+07 | +149% |

### 3. 增益缩放完全无效

时变增益因子 $f(t_{go})$ 对 SDRE 控制进行缩放，改善 **0%**。

**根本原因**：最优控制与 SDRE 控制的差异不在**大小**，而在**方向**。ARE 的稳态增益矩阵 $K_\infty$ 和真实最优的时变 $K(t)$ 有本质的结构差异。单纯缩放不改变控制方向。

代数 Riccati 方程 vs 微分 Riccati 方程的区别：
- ARE (SDRE): $A^T P + P A - P B R^{-1} B^T P + Q = 0 \rightarrow P_\infty$（常数增益）
- DRE (TPBVP): $-\dot{P} = A^T P + P A - P B R^{-1} B^T P + Q,\ P(T)=0 \rightarrow P(t)$（时变增益）

终端附近 $P(t) \to 0$，SDRE 全程用 $P_\infty$，在终端阶段"用力过猛"。

## 改善方向讨论

### 标量 R(x) 的局限

无论 R 如何依赖状态，SDRE 控制始终是 $u = -R(x)^{-1} B^T P x$，其中 P 来自 ARE（稳态）。ARE 无法编码 DRE 中从 $\lambda(T)=0$ 反向积分的时间演化信息。

### 可选方案对比

| 方案 | 学习目标 | 自由度 | 保留 Riccati 结构 | 捕获时变性 |
|------|---------|--------|------------------|-----------|
| R(x) 标量 | 标量 | 1 | ✓ | ✗ |
| R(x) 2×2 矩阵 | 对称矩阵 | 3 | ✓ | 部分 |
| P(x) 直接 | 对称矩阵 | 10 | ✗ | ✓ |
| u(x) 直接 | 向量 | 2 | ✗ | ✓ (数据来自 TPBVP) |

### smart_uq 项目评估

smart_uq 的 JT（Jacobian-Taylor）方法将 SDRE 的 P(x) 展开为 11 变量 5 阶泰勒多项式，可免在线 ARE 求解。但该方法仍然是 SDRE 近似（非真最优），其轨迹数据与 Python SDRE 仿真本质相同，**不能作为 TPBVP 最优控制的训练集**。方法论（多项式逼近→NN 逼近）可复用。

## 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `aerospace/control/optimal_control_solver.py` | CasADi+IPOPT 最优控制求解 | ✓ 完成 |
| `aerospace/control/inverse_optimal_fit.py` | 逆最优 R(x) 拟合 | ✓ 完成 |
| `experiment_trajectory_compare.py` | 最优 vs 固定R vs R(r) 轨迹对比 | ✓ 完成 |
| `experiment_tpbvp_compare.py` | TPBVP vs SDRE 鞍点最优对比 | ✓ 完成 |
| `experiment_gain_factor.py` | 时变增益因子提取与验证 | ✓ 完成 |
| `experiment_inverse_optimal.py` | 网格搜索 R(r) 参数 | 上游提供 |

## 运行命令

```bash
# 最优控制求解
uv run python -m aerospace.control.optimal_control_solver

# 逆最优拟合
uv run python -m aerospace.control.inverse_optimal_fit

# TPBVP 对比
uv run python experiment_tpbvp_compare.py

# 增益因子分析
uv run python experiment_gain_factor.py

# 三轨迹对比
uv run python experiment_trajectory_compare.py
```

## 输出目录

- `outputs/optimal_control/` — 最优控制解 (.npz)
- `outputs/inverse_optimal/` — 逆最优拟合结果
- `outputs/tpbvp/` — TPBVP 对比图、增益因子图
- `outputs/figures/trajectory_compare/` — 三轨迹对比图
