# SDC + Forward Euler 预测精度诊断

**日期**: 2026-05-30
**实验**: Phase 1 单步预测误差诊断
**脚本**: `aerospace/experiments/prediction_error_diagnostic.py`

## 研究问题

EKF 预测步使用 Forward Euler 离散化 (`F = I + A_SDC·dt`) 传播状态。如果改用 RK4 全非线性积分，单步预测精度能否显著提升？

## 方法

在同一仿真主循环中，每步并行计算三种预测：

| 预测方法 | 公式 | 精度 |
|----------|------|------|
| Forward Euler (当前) | `x + dt·(A_SDC·x + B·du)` | O(dt²) 局部误差 |
| RK4 | 4 阶 Runge-Kutta 积分子步 + 子步中重算 A_SDC | O(dt⁵) 局部误差 |
| 真值 | 13D RK45 (rtol=1e-8) → 做差 | 参考基准 |

参数：e=0.5, dt=10s, σ_ang=0.008°, 基线场景 5480 步。

## 结果

### 总预测误差（包含 EKF 估计误差 + 离散化误差）

| 指标 | Median | Mean | P95 | P99 |
|------|--------|------|-----|-----|
| FE 位置误差 | 4589 m | 5872 m | 15208 m | 23667 m |
| RK4 位置误差 | 4584 m | 5866 m | 15198 m | 23665 m |

> 总预测误差的主体是 EKF 估计误差（数 km），而非离散化误差。

### FE vs RK4 差异（纯离散化误差）

| 指标 | Median | Mean | P95 | P99 |
|------|--------|------|-----|-----|
| \|FE − RK4\| 位置 | **1.15 m** | 6.92 m | 32.81 m | 59.84 m |
| \|FE − RK4\| 速度 | ~0 | ~0 | ~0 | ~0 |

### FE 系统偏置（各分量均值）

| 分量 | 偏置 |
|------|------|
| dx (径向) | −1071 m |
| dy (沿迹向) | −608 m |
| dz (轨道面法向) | −252 m |
| dvx | −30 mm/s |
| dvy | +416 mm/s |
| dvz | +93 mm/s |

### 与噪声尺度的比较

- 仅测角测量噪声 (典型距离 ~800 km): **~140 m**
- EKF 估计误差 RMSE: **~7-8 km**
- FE-RK4 差异 (median): **1.15 m**
- FE-RK4 / 测量噪声: **0.008** (典型距离)

## 结论

1. **SDC 的精确因子化特性 (`A(x)·x ≡ f(x)`) 补偿了 Forward Euler 的低阶离散化**。FE 和 RK4 的单步预测差异仅 ~1 m，比测量噪声 (~140 m) 和 EKF 估计误差 (~7 km) 小两个数量级。

2. **切换到 RK4 全非线性预测没有工程意义**。单步改善 ~1 m 在闭环中会被测量更新完全稀释。速度项几乎无差异（因为速度动力学是线性的）。

3. **FE 存在方向性系统偏置**（径向/沿迹向 ~km 量级累积），但这来自 EKF 估计偏置而非离散化方法。

4. **在极近距（< 10 km）且高精度传感器场景下，FE 的尾部误差（P99 ~60 m）可能成为瓶颈**，但这超出了本文讨论的仅测角传感器精度范围。

## 对论文的意义

此诊断验证了统一 SDC 框架的核心设计选择：用 SDC 的精确性补偿低阶离散化的粗糙性。论文 §3.2 中可以将此作为"SDC 离散化精度论证"的技术论据：

> *The SDC parameterization satisfies A(x)·x ≡ f(x) exactly — it reconstructs the full nonlinear vector field at the linearization point without truncation. Consequently, the only source of prediction error is the O(dt²) Euler discretization, not model mismatch. A numerical comparison with RK4 integration of the full 6-DOF relative dynamics confirms that the median per-step difference is 1.15 m — two orders of magnitude below the angles-only measurement uncertainty (~140 m at typical ranges).*
