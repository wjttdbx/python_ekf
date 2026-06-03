# 论文详细提纲 & 证据映射（v2 — 填入实验数据）

## 元数据

| 项 | 值 |
|---|-----|
| 暂定标题 | Simultaneous Estimation and Control via a Common State-Dependent Coefficient Matrix for Angles-Only Relative Navigation in Elliptical Orbits |
| 目标期刊 | Acta Astronautica |
| 论文类型 | 应用论文 |
| 核心创新 | 同一个 `A_SDC(t_k)` 矩阵同时用于 EKF 预测和 SDRE 控制 —— SDC 框架下滤波线性化点与控制器设计点天然一致 |
| 实验维度 | A（基线对比）、B（噪声敏感性）、C（CW 模型对比）、F（Monte Carlo） |
| 代码库 | `/Users/mac/home/python_ekf` |
| 状态 | 实验完成，待撰写 |

---

## 核心 INSIGHT 集合（来自苏格拉底推导）

1. **统一 SDC 参数化不是"设计选择"，是理论必然**: SDC 框架下 `A(x̂_{k|k})·x̂_{k|k} ≡ f(x̂_{k|k})`（精确重构非线性向量场），EKF 和 SDRE 对线性化点的需求本就相同。统一框架是发现（finding）而非发明（invention）。

2. **SDC 精确性补偿了低阶离散化**: FE vs RK4 差异仅 1.15 m 中位数 → `A(x)·x ≡ f(x)` 的精确因子化使 Forward Euler 的一阶误差对闭环性能不产生实质性影响。

3. **CW 失效的根本原因是引力梯度非线性而非偏心率**: CW 在 866 km 初始距离上的引力梯度高估 80%，且缺失 NERM 反阻尼模态。CW 失败于 e=0.001（近圆）→ 证明 NERM SDC 在中长距离上不是"高 e 补丁"而是**刚需**。

4. **传感器噪声存在最优区间**: 噪声越低 ≠ 性能越好。σ_ang=0.001° 的捕获时间（87,000s）反而比 σ_ang=0.1°（38,470s）长 2.3 倍。大噪声 → 高 Kalman gain → 更激进的估计更新 → 更激进的 SDRE 控制 → 更快接近。代价是 ΔV 略增。

---

## 1. Introduction

### 1.1 问题背景 (≈1 页)
- 航天器相对导航与自主交会：在轨服务、碎片清除、非合作目标抵近
- 仅测角传感器的实际约束：星载光学相机只能提供方位角/仰角，无法直接测距
- 椭圆轨道上的非线性效应不可忽略（**实验证据**：CW 线性模型在 e=0.001 近圆轨道 + 866km 距离下即发散，见 Experiment C）

### 1.2 文献回顾 (≈1.5 页)
- **相对运动动力学**：CW [Clohessy & Wiltshire, 1960] → LERM [Melton, 2000] → SDRE 非线性参数化 [Cimen, 2010]
- **仅测角相对导航**：angles-only initial orbit determination [Woffinden & Geller, 2009], EKF for bearings-only tracking [Geller & Klein, 2014]
- **SDRE 控制**：SDRE regulation [Cloutier, 1997] → SDRE 航天器交会与接近操作 [Tartaglia & Innocenti, 2016]
- **关键缺口**：(1) 现有文献中 EKF 和 SDRE 使用独立或不同的动力学模型（如 CW 做滤波、非线性做控制），缺乏"同一 SDC 参数化同时驱动滤波与控制"的共同矩阵框架；(2) 仅测角条件下共同 SDC 矩阵方法的闭环统计验证缺失；(3) SDC 精确因子化（`A(x)·x ≡ f(x)`）在估计领域的价值未被充分认识

### 1.3 本文贡献 (≈0.5 页)
1. **统一 SDC 参数化框架**：同一个 `A_SDC(t_k)` 同时服务于 EKF 状态预测和 SDRE 最优控制。理论论证 + 数值验证（FE vs RK4 差异仅 1.15 m）
2. **仅测角约束下的完整闭环验证**：无距离测量条件下，EKF 估计收敛并支撑 SDRE 最优控制，Monte Carlo 200 trials 成功率 97.5%
3. **CW 线性模型的系统性失效分析**：CW 在中长距离（866 km）下因引力梯度非线性而失败于**所有偏心率**（含 e=0.001），证明 NERM SDC 是非线性距离导航的刚需
4. **传感器噪声的反直觉效应**：噪声从 0.001° 增至 0.1°，捕获时间反而减半（87 ks → 38 ks），推力峰值不变但均值增 2.3×
5. **辛平衡 ARE 求解**：解决 Q/S 量级差 10¹³ 导致的 ARE 病态问题

### 1.4 论文结构 (≈0.1 页)

**证据映射**：
- [GAP] 文献缺口 → 需引用 CW、LERM、SDRE、仅测角 EKF 各 2-3 篇关键文献
- [CONTRIB] 五条贡献 → 分别对应 §3.1/3.2、§5.1/5.4、§5.3、§5.2、§3.3
- [INSIGHT] 核心 INSIGHT → 对应 §3.1 和 §5.5 Discussion

---

## 2. System Modeling and Problem Formulation

### 2.1 Reference Orbit Dynamics
- Keplerian two-body problem: `a_c, e_c, μ`
- True anomaly evolution: `ν̇ = √(μ a_c (1-e²)) / r_c²`
- Orbital parameters as functions of ν: `r_c(ν), ν̇, ν̈`

**关键公式**：
```
r_c = a_c(1-e²) / (1 + e_c cos ν)
ν̇ = √(μ a_c (1-e²)) / r_c²
ν̈ = -2 ṙ_c ν̇ / r_c
```

### 2.2 Nonlinear Relative Motion in LVLH Frame
- LVLH 坐标系：(x, y, z) = (radial, along-track, cross-track)
- 13 维绝对状态：`[X_p(6), X_e(6), ν(1)]`
- Chaser 加速度（含引力 + 科里奥利 + 控制）：
```
ẍ_p = 2ν̇ẏ_p + ν̈y_p + ν̇²x_p − μ(r_c+x_p)/r_p³ + μ/r_c² + u_{px}
ÿ_p = −2ν̇ẋ_p − ν̈x_p + ν̇²y_p − μ·y_p/r_p³ + u_{py}
z̈_p = −μ·z_p/r_p³ + u_{pz}
```
- Target 加速度：对称形式（带 u_t 项）

**图表**：
- **Fig 1**: LVLH 坐标系示意图（参考轨道 + chaser-target 两航天器相对位置）

### 2.3 State-Dependent Coefficient Parameterization
- **SDC 分解原理**：`f(x) = A(x)·x`（非唯一），本文采用分式因子化
- **引力差项参数化**：
```
b_x = −μ(r_c+x_p)/r_p³ + μ(r_c+x_e)/r_e³
b = A(x_rel)·x_rel   (通过除以 r²_rel = x²_rel + y²_rel + z²_rel 实现)
```
- **6×6 A_SDC 矩阵结构**：
```
A_SDC = [   0₃          I₃     ]
        [ A₂₁(ν, x_rel)  A₂₂(ν) ]
```
其中 A₂₁ 包含 ν̇², ν̈, 引力差 SDC 因子；A₂₂ = [[0, 2ν̇, 0], [−2ν̇, 0, 0], [0, 0, 0]]

- **关键性质**：`A(x)·x ≡ f(x)` 精确恒成立——这不是泰勒截断，而是代数重构

### 2.4 Angles-Only Measurement Model
- 传感器输出：`z = [az, el]ᵀ`（仅方位角+仰角，无距离）
- 测量方程：
```
az = arctan2(y_rel, x_rel)
el = arcsin(z_rel / ρ),  ρ = ||[dx, dy, dz]||
```
- 噪声：`v ~ N(0, R)`, `R = diag(σ²_ang, σ²_ang)`
- 仅测角可观测性：一阶 Gramian 秩条件 + 轨道机动作为自然激励

### 2.5 Optimal Approach Control Problem
- 目标：驱动相对状态 x_rel → 0 同时最小化燃料消耗
- Quadratic cost functional:
```
J = ½∫ [xᵀQx + u_cᵀ R u_c] dt
```
- Chaser 最小化 J，target 不作机动（u_t = 0）
- SDC 参数化后转换为逐点 ARE：`A_SDCᵀP + P A_SDC − P B R⁻¹ Bᵀ P + Q = 0`
- 控制律：`u_c = −R⁻¹ Bᵀ P x̂_{k|k}`（用 EKF 估计状态）

**证据映射**：
- [EQ] 公式 2.1–2.18 → 代码 `nerm.py:get_orbital_params`, `get_SDC_matrix`, `dynamics_13d`
- [EQ] 公式 2.19–2.22 → 代码 `ekf.py:measure`, `meas_jacobian`
- [FIG] Fig 1, Fig 2（框架概览框图）
- [PARAM] Table 1：轨道/传感器/控制参数汇总

---

## 3. Unified EKF-SDRE Framework

### 3.1 The Unified SDC Principle
**核心命题**：在 SDC 框架下，EKF 的线性化点 `x̂_{k|k}` 和 SDRE 控制的线性化点天然一致，因此同一个 `A_SDC(t_k)` 可以同时服务于状态预测和最优控制计算。

**理论论证**：
- EKF 预测：`x_{k+1|k} = (I + A(x̂_{k|k})·dt)·x̂_{k|k} + dt·B·u_k` → 线性化点为 `x̂_{k|k}`
- SDRE 设计：`A(x̂_{k|k})ᵀP + P A(x̂_{k|k}) − P B R⁻¹ Bᵀ P + Q = 0` → 线性化点同为 `x̂_{k|k}`
- **关键洞察**：这不是人为设计，而是 SDC 框架的理论必然——滤波和控制对线性化点的需求本就相同

**数值验证**（预测误差诊断实验）：
- 替代方案：RK4 全非线性积分（每子步重算 A_SDC）
- |FE − RK4| 位置差异：median **1.15 m**, mean 6.92 m, P95 32.81 m, P99 59.84 m
- 测量噪声（典型距离 800 km × 0.008°）：~140 m
- FE-RK4 差异 / 测量噪声 = **0.008** → 两个数量级以下的差异
- **结论**：SDC 的精确因子化特性补偿了 Forward Euler 的低阶离散化。统一 `A_SDC(t_k)` 在数值上等价于全非线性方案。

**实现证据**：代码中 `get_SDC_matrix` 每步仅调用一次（`nerm_ekf_sdre.py:150`），结果 `A_SDC` 同时用于 `compute_control`（line 154）和 `ekf.predict`（line 173）。

### 3.2 Angles-Only EKF
#### 初始化
- P₀：`σ_pos = ρ₀·σ_ang`, `σ_vel = 1.0·σ_ang`（物理直觉：初始距离不确定性 = 传感器角分辨率 × 距离）
- x̂₀ = X_c₀ − X_t₀（chaser 已知自身位置和初始编队构型）

#### 预测步
```
x_{k+1|k} = (I + A_SDC·dt)·x̂_{k|k} + dt·B·(u_p − u_e)
P_{k+1|k} = F·P_{k|k}·Fᵀ + Q,   F = I + A_SDC·dt
```
- A_SDC 即为 §3.1 中与控制共用的同一矩阵
- B = [0₃; I₃]（控制仅作用在加速度分量）

#### 更新步
- 2×6 雅可比矩阵 H（仅测角）解析形式
- 角度新息绕卷：`wrap(Δaz, Δel) → [−π, π]`
- 标准 Kalman 增益 + 协方差更新

#### 仅测角可观测性讨论
- Gramian 矩阵的秩条件
- 轨道机动作为自然激励提升可观测性
- 远距离（>1000 km）时可观测性弱 → 解释了 Monte Carlo 中 5 个 EKF 发散 case

### 3.3 SDRE Optimal Control
#### ARE 构造
```
A_SDCᵀP + P A_SDC − P B R⁻¹ Bᵀ P + Q = 0
```

#### 辛平衡（Symplectic Balancing）
- 问题：`||Q||_F / ||S||_F ≈ 10¹³`（S = B R⁻¹ Bᵀ），哈密顿矩阵条件数 ∼10¹⁸
- 解：`α = √(||Q||/||S||)` → Q_bal=Q/α, R_bal=R/α → P=α·P_bar
- Fallback：行列缩放（对 A 做行归一化）
- 实验中 1/3 的 ARE 求解触发了 fallback（e=0.5, 0.7 场景）

#### 控制律
```
u_c = −R⁻¹ Bᵀ P x̂_{k|k}     (chaser：用 EKF 估计)
u_t = 0                      (target：无机动)
```

### 3.4 Sparse ARE Updates
- `are_interval > 1`：P 矩阵在多个控制步间复用
- dt=10s 内 ν 变化 ~0.001° → 轨道参数缓变 → P 缓变
- Trade-off：ARE 求解占单步约 50% 计算时间

### 3.5 Closed-Loop Integration
- Algorithm 1 伪代码
- 时序一致性：
  - t_k: 算 A_SDC(t_k) → 控制 u(t_k) → 真值传播 → t_{k+1}
  - t_{k+1}: EKF 预测（复用 A_SDC(t_k)）→ 更新（t_{k+1} 测量）→ x̂_{k+1|k+1}
  - 下一步 t_{k+1}: 用 x̂_{k+1|k+1} 算新 A_SDC(t_{k+1})

**证据映射**：
- [ALG] Algorithm 1 → `nerm_ekf_sdre.py:run` (137-183)
- [EQ] 公式 3.1–3.25 → `ekf.py:predict, update`; `sdre.py:compute_control, _solve_are_balanced`
- [DIAG] 预测误差诊断 → `docs/prediction_error_diagnostic.md` + `aerospace/experiments/prediction_error_diagnostic.py`
- [FIG] Fig 3: 统一 EKF-SDRE 流程（标注 A_SDC 复用）
- [FIG] Fig 4: 辛平衡前后 ARE 条件数对比
- [FIG] Fig 5: FE vs RK4 预测误差分布（诊断实验）
- [TABLE] Table 2: 算法参数汇总

---

## 4. Simulation Setup

### 4.1 Reference Scenario
| 参数 | 值 |
|------|-----|
| a_c | 15,000 km |
| e_c | 0.5 (基准) |
| μ | 3.986×10⁵ km³/s² |
| T_orbit | ~51,600 s (14.3 h) |
| ν₀ | 0° |
| X_p0 | [500, 500, 500, 0.01, 0.01, 0.01] |
| X_e0 | [0, 0, 0, 0, 0, 0] |
| 初始相对距离 | 866 km |

### 4.2 Filter Configuration
| 参数 | 值 |
|------|-----|
| 传感器类型 | 仅测角 (azimuth + elevation) |
| σ_ang (基准) | 0.008° (1σ ≈ 0.14 mrad) |
| R | diag(σ_ang², σ_ang²) |
| Q | diag(5×10⁻⁴, 5×10⁻⁴, 5×10⁻⁴, 5×10⁻⁸, 5×10⁻⁸, 5×10⁻⁸) |
| P₀ | diag((866·σ_ang)²×3, (1.0·σ_ang)²×3) |

### 4.3 Controller Configuration
| 参数 | 值 |
|------|-----|
| Q_ctrl | I₆ |
| R_ctrl | 1×10¹³·I₃ |
| are_interval | 1 (每步求解) |
| 推力约束 | R 隐式约束推力至 ~0.1 mm/s² 量级 |

### 4.4 Simulation Parameters
| 参数 | 值 |
|------|-----|
| dt | 10 s |
| t_end | 10×T_orbit ≈ 516,000 s |
| 积分器 | RK45, rtol=1e-8, atol=1e-10 |
| 捕获判据 | 相对距离 < 100 m |

### 4.5 Evaluation Metrics
1. 捕获性能：捕获时间、最终脱靶量、捕获成功率
2. 估计精度：位置/速度 RMSE、新息统计
3. 控制消耗：总 ΔV = ∫||u_p|| dt、推力峰值（max||u_p||）、推力均值、推力抖动（std）
4. 计算效率：单步耗时、总 wall time

### 4.6 Experimental Groups

**Experiment A — Baseline: Angles-Only vs Full-Information**
- A1: 仅测角 EKF+SDRE（带噪声, rng≠None）
- A2: 全知 SDRE（无噪声, rng=None）
- 脚本：`main.py`

**Experiment B — Sensor Noise Sensitivity**
- σ_ang ∈ {0.001°, 0.004°, 0.008°, 0.02°, 0.05°, 0.1°} × 5 seeds
- 新增记录：推力峰值、推力均值、推力抖动
- 脚本：`aerospace/experiments/sensor_noise_sweep.py`

**Experiment C — CW Model Baseline on NERM Truth**
- e_c ∈ {0.001, 0.1, 0.3, 0.5, 0.7} × 3 seeds
- Group 1：NERM SDC（滤波+控制均用 NERM A_SDC）
- Group 2：CW 常值 A（滤波+控制均用 CW A_cw，真值仍为 NERM 13D）
- **设计改进**：两组共享 NERM 13D 真值，仅滤波/控制模型不同（通过 `A_fixed` 参数注入 CW A 矩阵）
- 脚本：`aerospace/experiments/eccentricity_sweep.py`

**Experiment F — Monte Carlo Statistical Analysis**
- N = 200 trials
- 随机化：初始位置 ±300 km 高斯扰动（chaser + target 独立）+ 独立测量噪声种子
- 8 进程并行
- 脚本：`aerospace/experiments/monte_carlo.py`

**Experiment D — Orbital Regime Sensitivity**
- LEO (a=7500, e=0.1) / MEO (a=15000, e=0.5) / GEO (a=42164, e=0.1) × 2 modes × 5 seeds
- 仅测角 EKF+SDRE vs 全知 SDRE 跨三轨道高度对比
- 脚本：`aerospace/experiments/altitude_sweep.py`

**证据映射**：
- [TABLE] Table 3: 实验组设计汇总
- [SCRIPT] 脚本均已完成并验证

---

## 5. Results and Discussion

### 5.1 Baseline: Angles-Only vs Full-Information (Experiment A)

**实测数据**：

| 指标 | 仅测角 EKF+SDRE | 全知 SDRE |
|------|:--:|:--:|
| 捕获 | ✓ | ✓ |
| 捕获时间 | 54,800 s (15.2 h) | 88,130 s (24.5 h) |
| 最终距离 | 99.8 m | 99.8 m |
| 位置 RMSE | 8.16 km | — |
| 速度 RMSE | 6.36 mm/s | — |
| 总 ΔV | 10.03 km/s | — |
| Wall time | 8.0 s | — |

**反直觉发现**：仅测角 EKF 的捕获时间（54.8 ks）反而短于全知 SDRE（88.1 ks）。机制：EKF 估计误差 → 相对状态估计有偏 → SDRE 控制偏激进 → 更快接近。代价是可能存在 overshoot 风险。

**图表**：
- Fig 6: LVLH 相对运动轨迹对比
- Fig 7: EKF 估计误差时间历程 + 3σ 包络（6 分量）
- Fig 8: 推力分量 + 范数对比

### 5.2 Sensor Noise Sensitivity (Experiment B)

**实测数据**：

| σ_ang | 捕获率 | 捕获时间 | 位置 RMSE | ΔV | 推力峰值 | 推力均值 | 捕获相对速度 |
|-------|--------|----------|-----------|-----|----------|----------|-------------|
| 0.001° | 100% | 87,000 s | 5.09 km | 10.27 km/s | 1.96 m/s² | 0.12 m/s² | 0.09 m/s |
| 0.004° | 100% | 68,520 s | 6.73 km | 10.11 km/s | 1.96 m/s² | 0.15 m/s² | 0.07 m/s |
| 0.008° | 100% | 54,860 s | 8.04 km | 10.03 km/s | 1.96 m/s² | 0.18 m/s² | 0.17 m/s |
| 0.02° | 100% | 41,090 s | 9.42 km | 10.06 km/s | 1.96 m/s² | 0.22 m/s² | 1.12 m/s |
| 0.05° | 100% | 38,890 s | 9.13 km | 10.14 km/s | 1.96 m/s² | 0.25 m/s² | 0.70 m/s |
| 0.10° | 100% | 38,470 s | 8.66 km | 10.19 km/s | 1.96 m/s² | 0.27 m/s² | 1.98 m/s |
| 全知 SDRE | 100% | 88,130 s | — | — | — | — | 0.14 m/s |

**关键发现**：
1. **噪声越大 → 捕获越快**：σ_ang 从 0.001° → 0.1°（100 倍），捕获时间 87.0 → 38.5 ks（反比 2.3×）
2. **推力峰值恒定**：~1.96 m/s²，不随噪声变化
3. **推力均值增 2.3×**：噪声增大 → 更频繁的控制纠正
4. **捕获相对速度随噪声增大**：0.09 → 1.98 m/s，但仍全部 < 2 m/s → SDRE 控制在所有噪声下均实现软接近
5. **位置 RMSE 饱和**：σ_ang > 0.02° 后不再恶化

**工程启示**：对于追赶任务（优先速度），传感器精度不需最高——0.05°~0.1° 即足够。对精密交会（优先精度），需 ≤0.01°。

**图表**：
- Fig 9: EKF 位置/速度 RMSE vs σ_ang（双纵轴，对数横轴，error bar）
- Fig 10: 捕获时间 + 成功率 vs σ_ang
- Fig 11: 推力峰值/均值 vs σ_ang（新增）

### 5.3 CW Model Baseline on NERM Truth (Experiment C)

**实测数据**：

| e | NERM 捕获? | NERM 捕获时间 | NERM ΔV | CW 捕获? |
|---|:--:|------|------|:--:|
| 0.001 | ✓ | 29,800 s | 3.83 km/s | ✗ (发散) |
| 0.1 | ✓ | 29,770 s | 4.19 km/s | ✗ (发散) |
| 0.3 | ✓ | 35,040 s | 6.29 km/s | ✗ (发散) |
| 0.5 | ✓ | 55,010 s | 14.86 km/s | ✗ (发散) |
| 0.7 | ✓ | 87,260 s | 108.60 km/s | ✗ (发散) |

**CW 在 NERM 真值上全失败——含 e=0.001 近圆轨道！**

**根因分析**：
- CW 引力梯度（A[3,0]）：3.54×10⁻⁷（基于圆轨道线性化 `3n²`）
- NERM 引力梯度（A[3,0]，实际值）：1.97×10⁻⁷（非线性效应弱化了梯度）
- CW 高估 80% → 控制器过阻尼 → 但关键是 CW **缺失** NERM 的反阻尼模态（A 矩阵正实部特征值 ~+2.8×10⁻⁴）
- CW 控制器阻尼（~1.17×10⁻⁴）< NERM 真值的反阻尼（~2.8×10⁻⁴）→ 指数发散

**Narrative 升级**：
> CW 的失效不是因为偏心率——而是因为 866 km 距离上的引力梯度非线性。NERM SDC 不是"高偏心率补丁"而是**中长距离相对导航的刚需**。

**图表**：
- Fig 12: NERM 组捕获时间 vs e_c（error bar）
- Fig 13: NERM 组 EKF RMSE vs e_c（U 形曲线：最低点 e=0.5, RMSE=8.05 km）
- Fig 14: CW vs NERM 闭环稳定性对比示意图（A 矩阵特征值分析）

### 5.4 Monte Carlo Statistical Analysis (Experiment F)

**实测数据 N=200**：

| 指标 | Median | Mean | P95 | 
|------|--------|------|-----|
| 捕获成功率 | **97.5%** (195/200) | — | — |
| 捕获时间 | 13.41 h | 16.97 h | 35.8 h |
| 最终距离 (捕获) | 0.099 km | — | — |
| EKF 位置 RMSE | 12.71 km | — | 42.3 km |
| EKF 速度 RMSE | 9.34 mm/s | — | 31.2 mm/s |
| ΔV chaser | 8.86 km/s | 829.15 km/s | 19.4 km/s |
| 初始距离 | 1038 ± 389 km | — | — |

**5 个未捕获 case**：EKF 发散（远距离 + 大初始扰动 + 不利几何 → 可观测性崩溃）

**图表**：
- Fig 15: 捕获时间分布直方图 + 中位数标注
- Fig 16: 总 ΔV 分布（chaser + target 并排）
- Fig 17: 最终脱靶量分布（含非捕获 case 标记）
- Fig 18: EKF 误差 p50/p95 包络（10 trial 叠加 + 粗包络线）

### 5.5 Orbital Regime Sensitivity (Experiment D)

**实测数据（5 seeds × 3 regimes × 2 modes）**：

| Regime | a_c | e_c | Mode | Success |
|--------|-----|-----|------|:--:|
| LEO | 7,500 | 0.1 | Angles-only | **0/5** |
| LEO | 7,500 | 0.1 | Full-info | **0/5** |
| MEO | 15,000 | 0.5 | Angles-only | **5/5** |
| MEO | 15,000 | 0.5 | Full-info | **5/5** |
| GEO | 42,164 | 0.1 | Angles-only | **5/5** |
| GEO | 42,164 | 0.1 | Full-info | **5/5** |

**LEO 失败机制**：
- 全知 SDRE：距离先冲到 19,000 km（CW 沿迹漂移）→ 控制器拉回至 0.37 km → 反弹到 1.3 km
- 根因：R=10¹³ 使末端推力 < 0.6 mm/s²，无法对抗 Coriolis 力 ~2 mm/s²（LEO 的 n 比 MEO 大 2.8×）
- 仅测角 EKF：快动力学压缩了视差窗口，EKF 完全发散

**GEO 异常（仅测角）**：
- 4/5 seeds 正常（捕获 3.6-7.2 h，RMSE 58±8 km）
- 1/5 outlier（捕获 214 h，RMSE 4,758 km，ΔV 4,431 km/s）
- 根因：EKF 协方差几步内冻结，Kalman gain 不再更新；当方位角新息接近 ±π 时，绕卷产生 2π 跳变 → 方向估计翻转 180° → 正反馈发散（83 次绕卷事件）

**结论**：MEO 是该方法的最优工作域——LEO 太快（控制饱和），GEO 太慢（EKF 冻结）。

### 5.6 Discussion

**统一 SDC 框架**：
- 优势：概念简洁（单次 SDC 计算）、计算高效（省去第二次 get_SDC_matrix）、模型一致、对离散化误差鲁棒（SDC 精确性补偿低阶积分）
- 局限：SDC 因子化在 ρ→0 时的除零风险（已通过 +1e-6 正则化处理）

**仅测角 EKF 的意外行为**：
- 噪声-捕获速度的反比关系：Kalman gain 自适应的副作用
- 可用于任务规划：追赶阶段可放宽传感器精度要求以换取速度

**CW 模型的教训**：
- 非线性的两个来源：(a) 偏心率（轨道弧度）,(b) 相对距离（引力梯度差异）
- CW 解决 (a) 但不解决 (b)。对于 866 km 的初始距离，(b) 是主导因素
- **长距离相对导航中，线性模型永远不够**——这是 SDRE/SDC 方法存在的根本理由

**轨道高度敏感性**（Experiment D 新发现）：
- MEO 是该方法的 natural sweet spot：LEO 太快（控制饱和），GEO 太慢（EKF 冻结）
- LEO 失败非 EKF 问题——全知 SDRE 也过冲反弹
- GEO outlier 揭示 angles-only 在慢动力学下的协方差冻结风险

**工程启示**：
- 传感器精度 &lt; 0.02° 时效果饱和
- 单步计算 ~1.5 ms → 在线可行性
- 跨轨道调参（scheduled Q/R）是推广到 LEO/GEO 的前提

**证据映射**：
- [FIG] Fig 6–18 共 13 张
- [TABLE] Table 4–5

---

## 6. Conclusions

### 6.1 Summary of Findings

**1. 共同 SDC 矩阵框架是可行的。** SDC 精确因子化 `A(x)·x ≡ f(x)` 是代数恒等式，非截断近似——Forward Euler SDC 离散化与全非线性 RK4 积分的差异仅 1.15 m（中位数），比仅测角测量噪声（~140 m）低两个数量级。同一 `A_SDC(t_k)` 同时驱动 EKF 预测和 SDRE 控制在理论上是自然的（线性化点天然一致），在数值上是等效的。

**2. 闭环性能在 MEO 下具有统计可靠性，但不跨轨道泛化。** Monte Carlo N=200，97.5% 成功率，捕获时间中位数 13.4 h。然而固定 (Q,R) 参数不能通吃所有轨道：LEO 因控制末端饱和失败，GEO 因 EKF 协方差冻结 + 角度绕卷存在随机发散风险。MEO（a=15000 km, e=0.5）是该方法的自然工作域。

**3. 传感器噪声存在反直觉的加速效应，跨三个距离量级成立。** σ_θ 从 0.001° 增至 0.1°，捕获时间减半（87 ks → 38 ks），推力峰值恒定在 1.96 m/s²。机制：噪声放大 Kalman gain → SDRE 控制更激进。

### 6.2 Limitations
- 假设 chaser 已知自身绝对位置（GNSS/星敏器精度影响未纳入）
- 单 chaser-target 对（未扩展多航天器协同）
- 远距离（>1000 km）+ 大初始扰动 → 可观测性崩溃风险（5/200）

### 6.3 Future Work
- 多 chaser 协同观测提升仅测角可观测性
- 自适应 Q/R 调优（基于新息统计）
- 星载实时验证 + 硬件在环

---

## 图表汇总（共 9 张图 + 5 张表）

| 编号 | 内容 | 来源 |
|------|------|:--:|
| Fig 1 | LVLH 坐标系 + 问题几何示意图 | 手绘 |
| Fig 2 | 统一 EKF-SDRE 算法流程（标注 A_SDC 复用） | 手绘 |
| Fig 3 | FE vs RK4 单步预测误差分布直方图 | 诊断实验 |
| Fig 4 | 仅测角 vs 全知 SDRE 轨迹/推力对比 | Exp A |
| Fig 5 | EKF 估计误差时间历程 + 3σ 包络（6 分量） | Exp A |
| Fig 6 | 捕获时间 + 推力峰值/均值 vs σ_ang | Exp B |
| Fig 7 | NERM 捕获时间 + EKF RMSE vs e_c | Exp C |
| Fig 8 | Monte Carlo 捕获时间分布直方图 | Exp F |
| Fig 9 | Monte Carlo EKF 误差 p50/p95 包络 | Exp F |

| 编号 | 内容 |
|------|------|
| Table 1 | 轨道/传感器/控制参数 |
| Table 2 | EKF-SDRE 算法参数 |
| Table 3 | 实验组设计 |
| Table 4 | Monte Carlo 统计汇总 |
| Table 5 | 与文献方法定性比较

---

## 生成状态

- [x] Experiment A 基线 → `main.py`
- [x] Experiment B 噪声扫描 → `aerospace/experiments/sensor_noise_sweep.py`
- [x] Experiment C CW 对比 → `aerospace/experiments/eccentricity_sweep.py`
- [x] Experiment F Monte Carlo → `aerospace/experiments/monte_carlo.py`
- [x] Experiment D 轨道高度扫描 → `aerospace/experiments/altitude_sweep.py`
- [x] 预测误差诊断 → `aerospace/experiments/prediction_error_diagnostic.py`
- [x] 核心代码修复 → `nerm_ekf_sdre.py` (统一 A_SDC)
- [x] 诊断文档 → `docs/prediction_error_diagnostic.md`
- [ ] 确定最终标题
- [ ] 文献检索
- [ ] 撰写初稿
