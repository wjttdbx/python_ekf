# 共享SDC矩阵的滤波-控制SDRE方法

## 一、方法总览

本文档阐述一种面向**航天器追逃博弈**的闭环估计-控制统一框架。其核心思想是：

> **扩展卡尔曼滤波（EKF）和状态依赖黎卡提方程（SDRE）控制器共享同一个 SDC（State-Dependent Coefficient）参数化矩阵。** EKF 用它对非线性系统做局部线性化预测，SDRE 控制器用它求解代数黎卡提方程（ARE）产生最优推力——二者在同一个线性化点上协同工作，形成"先估计，后控制，同一点，同一模型"的紧耦合架构。

整个闭环回路如下：

```
真实状态 ──→ 传感器测量（含噪声） ──→ EKF 估计相对状态 x̂
                                          │
                          ┌───────────────┘
                          ▼
                   构建 A_SDC(x̂)  ←── 共享 SDC 矩阵
                     ╱           ╲
                    ╱             ╲
            EKF 预测步         SDRE 控制
         (F = I + A_SDC·dt)   (求解 ARE → u_p, u_e)
                    ╲             ╱
                     ╲           ╱
                      ▼         ▼
                  真实动力学推进（RK45）
                          │
                          ▼
                    下一时间步 ──→ 循环
```

---

## 二、数学基础：NERM 动力学与 SDC 参数化

### 2.1 坐标系统与状态定义

- **参考坐标系**：LVLH（Local Vertical Local Horizontal）
  - $x$：径向（地心指向航天器）
  - $y$：沿迹方向（运动方向）
  - $z$：法向（轨道面法向，右手定则）

- **绝对状态**（每颗星 6 维）：
  $$X = [x,\ y,\ z,\ v_x,\ v_y,\ v_z]^\top \in \mathbb{R}^6$$
  追踪星（Pursuer）记为 $X_p$，逃逸星（Evader）记为 $X_e$。

- **相对状态**（6 维）：
  $$x_{\text{rel}} = X_p - X_e = [dx,\ dy,\ dz,\ dv_x,\ dv_y,\ dv_z]^\top$$

- **扩展状态**（13 维闭环系统）：
  $$\mathbf{X} = [X_p(6),\ X_e(6),\ \nu(1)]^\top$$
  其中 $\nu$ 是真近点角，它的引入使参考轨道随时间演化，驱动时变动力学。

### 2.2 参考轨道参数

给定真近点角 $\nu$，参考轨道由三个标量完全刻画：

| 参数 | 符号 | 公式 | 物理意义 |
|------|------|------|----------|
| 地心距离 | $r_c$ | $\displaystyle r_c = \frac{a_c(1-e_c^2)}{1+e_c\cos\nu}$ | 虚拟参考点到地心的距离 |
| 真近点角速度 | $\dot\nu$ | $\displaystyle \dot\nu = \frac{\sqrt{\mu a_c(1-e_c^2)}}{r_c^2}$ | 参考轨道角速度 |
| 真近点角加速度 | $\ddot\nu$ | $\displaystyle \ddot\nu = -\frac{2\dot r_c \dot\nu}{r_c}$ | 椭圆轨道的角加速度（圆轨道时为零） |

其中 $a_c$ 是半长轴，$e_c$ 是偏心率，$\mu$ 是引力常数。

### 2.3 非线性相对运动方程

在 LVLH 系下，航天器的加速度由三项构成：科里奥利力、离心力/时变项、
二体引力差（相对于参考轨道）：

$$\begin{aligned}
\ddot x &= 2\dot\nu \dot y + \ddot\nu y + \dot\nu^2 x - \frac{\mu(r_c+x)}{r_p^3} + \frac{\mu}{r_c^2} + u_x \\[4pt]
\ddot y &= -2\dot\nu \dot x - \ddot\nu x + \dot\nu^2 y - \frac{\mu y}{r_p^3} + u_y \\[4pt]
\ddot z &= -\frac{\mu z}{r_p^3} + u_z
\end{aligned}$$

式中 $r_p = \sqrt{(r_c+x)^2 + y^2 + z^2}$，$u_x, u_y, u_z$ 是推力加速度。

这套方程对追踪星和逃逸星分别适用，且**对所有偏心率 $e_c \in [0,1)$ 精确成立**——不是小偏心率近似，是全非线性模型（NERM, Nonlinear Elliptical Relative Motion）。

---

## 三、SDC 矩阵的构造（核心）

### 3.1 SDC 参数化的原理

SDRE 方法的关键是将非线性动力学写成**伪线性形式**：

$$\dot x_{\text{rel}} = A(x) \cdot x_{\text{rel}} + B \cdot (u_p - u_e)$$

其中 $A(x)$ 不是常数矩阵，而是**依赖于当前状态的系数矩阵**——这就是"状态依赖系数"（SDC）的含义。

### 3.2 非线性引力差值的 SDC 分解

这是整个方法中最精妙的一步。两颗星的引力差向量为：

$$b = \begin{bmatrix}
-\frac{\mu(r_c+x_p)}{r_p^3} + \frac{\mu(r_c+x_e)}{r_e^3} \\[4pt]
-\frac{\mu y_p}{r_p^3} + \frac{\mu y_e}{r_e^3} \\[4pt]
-\frac{\mu z_p}{r_p^3} + \frac{\mu z_e}{r_e^3}
\end{bmatrix}$$

这一步的诀窍是**将引力差向量分解为状态依赖矩阵乘以相对位置**：

$$b = \begin{bmatrix}
\frac{b_x \cdot dx}{r_{\text{rel}}^2} & \frac{b_x \cdot dy}{r_{\text{rel}}^2} & \frac{b_x \cdot dz}{r_{\text{rel}}^2} \\[4pt]
\frac{b_y \cdot dx}{r_{\text{rel}}^2} & \frac{b_y \cdot dy}{r_{\text{rel}}^2} & \frac{b_y \cdot dz}{r_{\text{rel}}^2} \\[4pt]
\frac{b_z \cdot dx}{r_{\text{rel}}^2} & \frac{b_z \cdot dy}{r_{\text{rel}}^2} & \frac{b_z \cdot dz}{r_{\text{rel}}^2}
\end{bmatrix} \cdot \begin{bmatrix} dx \\ dy \\ dz \end{bmatrix}$$

其中 $r_{\text{rel}}^2 = dx^2 + dy^2 + dz^2 + \varepsilon$（$\varepsilon = 10^{-6}$ 防止除零）。

这一分解在数学上是**精确的**：矩阵乘以 $[dx,dy,dz]^\top$ 正好恢复原引力差向量。它不是近似，而是重新排列。

### 3.3 $A_{\text{SDC}}$ 的完整结构

$$A_{\text{SDC}} = \begin{bmatrix}
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\[4pt]
\dot\nu^2 + \frac{b_x dx}{r_{\text{rel}}^2} & \ddot\nu + \frac{b_x dy}{r_{\text{rel}}^2} & \frac{b_x dz}{r_{\text{rel}}^2} & 0 & 2\dot\nu & 0 \\[4pt]
-\ddot\nu + \frac{b_y dx}{r_{\text{rel}}^2} & \dot\nu^2 + \frac{b_y dy}{r_{\text{rel}}^2} & \frac{b_y dz}{r_{\text{rel}}^2} & -2\dot\nu & 0 & 0 \\[4pt]
\frac{b_z dx}{r_{\text{rel}}^2} & \frac{b_z dy}{r_{\text{rel}}^2} & \frac{b_z dz}{r_{\text{rel}}^2} & 0 & 0 & 0
\end{bmatrix}$$

分块解释：

| 子块 | 位置 | 含义 |
|------|------|------|
| $A_{12} = I_3$ | 右上 (0:3, 3:6) | 运动学恒等式 $\dot r = v$ |
| $A_{21}$ | 左下 (3:6, 0:3) | 引力差 + 离心力 + 时变项 |
| $A_{22}$ | 右下 (3:6, 3:6) | 科里奥利力 $2\dot\nu$ 耦合 |

控制输入矩阵：
$$B = \begin{bmatrix} 0_{3\times3} \\ I_3 \end{bmatrix} \in \mathbb{R}^{6\times 3}$$

相对动力学因此写为：
$$\dot x_{\text{rel}} = A_{\text{SDC}} \cdot x_{\text{rel}} + B \cdot (u_p - u_e)$$

---

## 四、EKF：用 SDC 矩阵做状态预测

### 4.1 EKF 状态与测量模型

EKF 估计的是 6 维**相对状态** $x_{\text{rel}} = [dx, dy, dz, dv_x, dv_y, dv_z]^\top$。

**测量模型**（角度-only 模式，也是主要研究对象）：

$$z = h(x_{\text{rel}}) = \begin{bmatrix}
\text{az} \\ \text{el}
\end{bmatrix} = \begin{bmatrix}
\arctan2(dy, dx) \\[4pt]
\arcsin\left(\frac{dz}{\rho}\right)
\end{bmatrix}$$

其中 $\rho = \sqrt{dx^2 + dy^2 + dz^2}$ 是相对距离。

### 4.2 预测步：SDC 矩阵驱动状态转移

EKF 的预测步**直接利用 SDC 矩阵做一阶泰勒离散化**：

$$F = I_{6\times 6} + A_{\text{SDC}} \cdot \Delta t$$

$$\hat x_{k|k-1} = F \cdot \hat x_{k-1|k-1} + \Delta t \cdot B \cdot (u_p - u_e)$$

$$P_{k|k-1} = F \cdot P_{k-1|k-1} \cdot F^\top + Q$$

这相当于一步**前向欧拉积分**，将 SDC 矩阵在估计状态处冻结为常数矩阵。

### 4.3 更新步：非线性测量更新

测量雅可比矩阵 $H$ 在当前先验估计处解析计算：

$$H = \left.\frac{\partial h}{\partial x_{\text{rel}}}\right|_{\hat x_{k|k-1}}$$

随后执行标准 EKF 更新：

$$\begin{aligned}
S_k &= H P_{k|k-1} H^\top + R \\
K_k &= P_{k|k-1} H^\top S_k^{-1} \\
\hat x_{k|k} &= \hat x_{k|k-1} + K_k (z_k - h(\hat x_{k|k-1})) \\
P_{k|k} &= (I - K_k H) P_{k|k-1}
\end{aligned}$$

角度分量需做 $\pm\pi$ 归一化处理（wrap-aware innovation）。

---

## 五、SDRE 控制：用 SDC 矩阵求解 ARE

### 5.1 零和追逃博弈建模

SDRE 控制将追逃问题建模为**无限时域非线性零和微分博弈**：

**追踪星**（Pursuer，追方）最小化：
$$J_p = \frac{1}{2} \int_0^\infty \left( x_{\text{rel}}^\top Q x_{\text{rel}} + u_p^\top R u_p - \gamma^2 u_e^\top R u_e \right) dt$$

**逃逸星**（Evader，逃方）最大化 $J_p$。参数 $\gamma > 1$ 是博弈调节因子（默认 $\gamma = \sqrt{2}$）。

### 5.2 代数黎卡提方程（ARE）

在 $A = A_{\text{SDC}}$ 处冻结系统矩阵后，解如下 ARE：

$$A^\top P + P A - P B R_{\text{eff}}^{-1} B^\top P + Q = 0$$

其中 **有效控制权重**（博弈修正后）为：

$$R_{\text{eff}} = \frac{R}{1 - \gamma^{-2}}$$

这个修正将二人零和博弈转化为等价的最优控制问题，保证 ARE 有对称正定解。

### 5.3 控制律

$$\begin{aligned}
u_p &= -R_{\text{eff}}^{-1} B^\top P \cdot \hat x_{\text{rel}} \quad &\text{(追方：使用 EKF 估计)} \\[4pt]
u_e &= +\gamma^{-2} R_{\text{eff}}^{-1} B^\top P \cdot x_{\text{rel}}^{\text{true}} \quad &\text{(逃方：假设全知，用于博弈平衡)}
\end{aligned}$$

> **关键差异**：追方输入 EKF 估计值 $\hat x_{\text{rel}}$（现实约束），逃方使用真实相对状态（博弈理论需要）。这是混合信息结构博弈的体现。

### 5.4 数值求解策略

项目使用 `scipy.linalg.solve_continuous_are`，但面临一个问题：$Q$ 和 $B R_{\text{eff}}^{-1} B^\top$ 的量级差距可达 $10^{13}$ 倍，导致哈密顿矩阵病态。因此引入了**辛平衡（Symplectic Balancing）**策略：

$$\alpha = \sqrt{\frac{\|Q\|_F}{\|B R_{\text{eff}}^{-1} B^\top\|_F}}, \quad Q' = \frac{Q}{\alpha}, \quad R'_{\text{eff}} = \frac{R_{\text{eff}}}{\alpha}$$

解平衡后 ARE 得 $P'$，恢复 $P = \alpha \cdot P'$。

此外支持**稀疏 ARE 更新**：间隔若干步才重解 ARE，中间步复用缓存的 $P$ 矩阵。大幅降低计算开销，因为相邻步的 $A_{\text{SDC}}$ 变化缓慢。

---

## 六、闭环仿真流程（时间步内详解）

每个仿真步长 $\Delta t$ 内的执行顺序：

```
┌─────────────────────────────────────────────────────────┐
│ Step k → k+1                                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 获取轨道参数                                         │
│     r_c, ν̇, ν̈ ← get_orbital_params(ν_k)                 │
│                                                         │
│  2. 推算逃方近似位置（仅用于构建 A_SDC）                   │
│     X̃_e = X_p_true - x̂_rel                               │
│                                                         │
│  3. 构建共享 SDC 矩阵（关键！位置唯一）                     │
│     A_SDC = get_SDC_matrix(X_p_true, X̃_e, r_c, ν̇, ν̈)     │
│                                                         │
│  4. SDRE 控制计算                                        │
│     if (k % are_interval == 0):                          │
│         求解 ARE → P                                     │
│     u_p = -R_eff⁻¹ Bᵀ P x̂_rel    (追方，用估计值)         │
│     u_e = +γ⁻² R_eff⁻¹ Bᵀ P x_true (逃方，用真值)        │
│                                                         │
│  5. 真实状态推进（RK45 积分非线性 ODE）                     │
│     X_k+1 ← dynamics_13d(t, X_k, u_p, u_e)               │
│                                                         │
│  6. EKF 预测 + 更新（同一 A_SDC！）                        │
│     F = I + A_SDC·Δt                                    │
│     x̂⁻ = F·x̂ + Δt·B·(u_p - u_e)                         │
│     P⁻ = F·P·Fᵀ + Q                                     │
│     z_meas = measure(X_p, X_e) + noise                   │
│     x̂⁺, P⁺ ← EKF update(x̂⁻, P⁻, z_meas)                  │
│                                                         │
│  7. 检查终止条件                                          │
│     if ‖dx‖ < capture_dist → 捕获！                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 理由 |
|------|------|
| 同一步内 EKF 和 SDRE 使用同一个 $A_{\text{SDC}}$ | 滤波和控制在此步的线性化点一致——都是 $(\hat x_{k|k}, X_{p,\text{true}})$ 处；避免了模型不一致带来的额外误差 |
| 控制计算在状态推进**之前** | $u_k$ 是零阶保持输入，在整个 $[t_k, t_{k+1}]$ 内恒定，符合实际推力器特性 |
| EKF 预测在状态推进**之后** | $A_{\text{SDC}}$ 是 $t_k$ 时刻的线性化，预测到 $t_{k+1}$；测量是 $t_{k+1}$ 时刻采样的 |
| 逃方使用真实相对状态 | 逃方"全知"是博弈理论的 Nash 平衡假设——计算的是博弈理论中的逃方最优策略，而非实际的逃方行为 |
| 无噪声时跳过 EKF | 当 `rng=None`，直接将 EKF 状态设为真实相对状态，等效于全状态反馈 SDRE |

---

## 七、方法优势与适用边界

### 7.1 为什么共享 SDC 矩阵是好的设计

1. **模型一致性**：EKF 的线性化预测与 SDRE 的线性化控制源自**同一数学模型**的同一线性化点。不存在"滤波器用一个近似模型，控制器用另一个近似模型"的失配。

2. **计算复用**：SDC 矩阵每步只构造一次，一个 $A_{\text{SDC}}$ 供 EKF 预测和 SDRE 控制两个模块使用——避免重复计算引力差和非线性项。

3. **理论优雅**：这是"分离原理在非线性系统中的自然推广"——虽然非线性系统不严格满足分离原理，但 SDC 框架下滤波与控制共享同一 $A(x)$ 参数化，使估计误差和控制误差向同一方向近似。

4. **适用于任意偏心率**：NERM 模型对 $e_c \in [0,1)$ 精确成立，不像 CW 方程那样局限于近圆轨道。这使得本方法可用于大椭圆轨道（$e_c = 0.5$）甚至更高偏心率的场景。

### 7.2 需要警惕的地方

1. **SDC 分解不唯一**：非线性系统 $\dot x = f(x)$ 的 SDC 分解 $f(x) = A(x)x$ 有无穷多种（$A(x)x = (A(x)+E(x))x$ 若 $E(x)x=0$ 则等效）。目前采用的"引力差按比例分配"分解是合理的，但不一定是最优的。

2. **仅一步欧拉积分的精度**：EKF 预测使用 $F = I + A\Delta t$（一阶），长时间或大步长下可能引入系统性的预测偏差。RK45 只用于真实状态传播，EKF 内仍是一阶。

3. **ARE 求解失败的回退**：大相对距离下 $A_{\text{SDC}}$ 的特征值分散可能导致 ARE 无解或数值崩溃——代码中采用"复用上一时刻 $P$ 矩阵"的策略，但这在快速机动时可能不够准确。

4. **不保证全局稳定性**：SDRE 方法的闭环稳定性只在局部有理论保证（需要逐点可控性条件）。实际应用中通过仿真验证。

### 7.3 适用场景

- ✅ 大椭圆参考轨道（$e_c$ 接近 0.8）
- ✅ 仅测角传感器（主被动相对导航）
- ✅ 追逃博弈（二人零和非线性微分博弈）
- ✅ 非合作目标（逃方不配合，仅通过测量获取信息）
- ⚠️ 极高精度要求（可能需要 TPBVP 直接优化而非 SDRE）

---

## 八、代码实现索引

| 文件 | 角色 |
|------|------|
| `aerospace/dynamics/nerm.py` | SDC 矩阵构造（`get_SDC_matrix`），13D ODE 右端（`dynamics_13d`），轨道参数计算（`get_orbital_params`） |
| `aerospace/dynamics/nerm_2d.py` | 面内 2D 版本，同上结构，4×4 SDC 矩阵 |
| `aerospace/estimation/ekf.py` | EKF 实现：`predict(A, B, u_p, u_e, dt)` 使用 SDC 矩阵做一阶离散化，`update()` 执行角度测量更新 |
| `aerospace/control/sdre.py` | SDRE 控制器：`compute_control(A_SDC, x_rel, ...)` 用 SDC 矩阵求解 ARE 并输出 $u_p, u_e$，含辛平衡策略 |
| `aerospace/simulation/nerm_ekf_sdre.py` | 闭环仿真引擎：串联动力学、EKF、SDRE，每个时间步按上述流程执行 |
| `main.py` | 主入口：配置仿真参数（轨道、噪声、权重），跑有噪声+无噪声两遍，画对比图 |

---

## 九、2D 简化版本

项目同时提供了面内 2D 版本（`nerm_2d.py`），将状态降为 4 维 $[x, y, v_x, v_y]$（设 $z=0, v_z=0$）。其 SDC 矩阵为 4×4：

$$A_{\text{SDC}}^{\text{2D}} = \begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\dot\nu^2 + \frac{b_x dx}{r_{\text{rel}}^2} & \ddot\nu + \frac{b_x dy}{r_{\text{rel}}^2} & 0 & 2\dot\nu \\
-\ddot\nu + \frac{b_y dx}{r_{\text{rel}}^2} & \dot\nu^2 + \frac{b_y dy}{r_{\text{rel}}^2} & -2\dot\nu & 0
\end{bmatrix}$$

EKF 测量退化为单个方位角 `[az]`（$R$ 为 1×1），控制量退化为 2D 推力 $[u_x, u_y]$。其余逻辑与 3D 版本完全一致。

---

*文档版本：2026-06-09，基于代码仓库 `python_ekf` 当前 master 分支。*
