# 多项式混沌 SDRE（PC-SDRE）：理论与实现探索

## 目录

1. [问题定位](#1-问题定位)
2. [数学基础：广义多项式混沌 (gPC)](#2-数学基础广义多项式混沌-gpc)
3. [随机 SDRE 问题的 gPC 重构](#3-随机-sdre-问题的-gpc-重构)
4. [三种计算路径](#4-三种计算路径)
5. [与 HNEKF 的统一框架](#5-与-hnekf-的统一框架)
6. [原型实现路线图](#6-原型实现路线图)
7. [文献定位与创新空间](#7-文献定位与创新空间)
8. [风险与缓解](#8-风险与缓解)

---

## 1. 问题定位

### 1.1 当前状态

本项目的 EKF+SDRE 框架存在一个结构性割裂：

```
 ┌─────────────────────┐        ┌─────────────────────┐
 │   EKF 估计侧        │        │   SDRE 控制侧       │
 │                     │        │                     │
 │  输出: x̂ (均值)     │  ───→  │  输入: A(x̂) (单点)  │
 │  输出: P (协方差)   │  ✗     │  丢弃了 P 的信息     │
 └─────────────────────┘        └─────────────────────┘
```

**核心浪费**：EKF 花了一半的计算量来维护协方差矩阵 P（描述估计的不确定性），但 SDRE 控制器完全忽略了这些不确定信息，只在均值点 `x̂` 处评估动力学矩阵。

### 1.2 方案 C 的目标

将 SDRE 控制律从**确定性点映射**升级为**随机映射**，使得控制量 `u` 不仅是状态均值 `x̂` 的函数，也显式依赖于状态协方差 `P`：

```
u = κ(x̂, P)   而不是   u = κ(x̂)
```

当 P 大（估计不可靠）时，控制器自动变得更"谨慎"；当 P 小（估计精确）时，收敛到经典 SDRE。

### 1.3 核心数学问题

给定：
- 随机状态 `x(ξ)`，其中 `ξ ~ N(0, I_6)` 是标准正态随机向量
- 状态分布由均值 `x̄` 和协方差 `P` 表征（来自 HNEKF/EKF）
- SDC 参数化 `A(x(ξ))`，满足 `f(x) = A(x) x`

求解：
- 期望二次型代价 `J = E_ξ[∫ (x^T Q x + u^T R u) dt]` 的最优控制
- 或更一般的风险敏感代价

---

## 2. 数学基础：广义多项式混沌 (gPC)

### 2.1 Wiener-Askey 多项式混沌

对于标准正态随机向量 `ξ ∈ R^d`，状态和控制量展开在一组正交多项式基上：

```
x(t, ξ) = Σ_{α∈J} x̂_α(t) · ψ_α(ξ)     (状态展开)
u(t, ξ) = Σ_{α∈J} û_α(t) · ψ_α(ξ)     (控制展开)
```

其中：
- `α = (α_1, ..., α_d)` 是多重索引，`|α| = α_1 + ... + α_d`
- `ψ_α(ξ)` 是 d 维 Hermite 多项式（因为 ξ 是 Gaussian）
- `J = {α : |α| ≤ p}` 是截断到 p 阶的多重指标集
- 展开维度：`P_dim = (d+p)! / (d! p!)`

对 d=6 维状态、p=2 阶展开：`P_dim = (8)!/(6! 2!) = 28` 个基函数

### 2.2 正交性与内积

基函数的正交性（对 Hermite 多项式）：

```
E[ψ_α(ξ) · ψ_β(ξ)] = α! · δ_{αβ}
```

其中 `α! = α_1! · α_2! · ... · α_d!`。

**Gram 矩阵** W，尺寸 `P_dim × P_dim`：

```
W_{ij} = E[ψ_i(ξ) · ψ_j(ξ)]    ← 对角矩阵（正交性）
```

### 2.3 三重积张量（关键结构）

当两个 gPC 展开相乘再投影时，出现三重积：

```
ψ_i(ξ) · ψ_j(ξ) = Σ_k c_{ijk} · ψ_k(ξ)
```

其中 `c_{ijk} = E[ψ_i ψ_j ψ_k] / E[ψ_k²]` 是**Galerkin 三重积系数**。

这个三重积张量 `C = [c_{ijk}]` 是整个 gPC 方法的核心——它编码了"基函数相乘后如何投影回基函数空间"。对于 Hermite 多项式，`c_{ijk}` 有解析闭式表达式，不需要数值积分。

---

## 3. 随机 SDRE 问题的 gPC 重构

### 3.1 随机动力学

相对运动动力学（6 维相对状态）：

```
ẋ(t, ξ) = A(x(t, ξ)) · x(t, ξ) + B · u(t, ξ)
```

其中 `A(x)` 来自 `get_SDC_matrix()`，`B = [0₃; I₃]`。

将 gPC 展开代入：

```
Σ_α ẋ̂_α ψ_α = A(Σ_β x̂_β ψ_β) · Σ_γ x̂_γ ψ_γ + B · Σ_α û_α ψ_α
```

左端对 `ψ_δ` 做 Galerkin 投影（取 `E[· ψ_δ]`），得到确定性的系数演化方程：

```
ẋ̂_δ = Σ_β,γ A_{βγδ} · x̂_γ + B · û_δ       (δ ∈ J)
```

其中 `A_{βγδ}` 是一个三阶张量，编码了 SDC 矩阵的 gPC 展开与 Galerkin 投影。

#### 3.1.1 关键简化：SDC 矩阵的 gPC 展开

SDC 矩阵 `A(x(ξ))` 本身也是随机变量的函数，可以展开：

```
A(ξ) = Σ_{β} Â_β · ψ_β(ξ)
```

系数 `Â_β`（每个是 6×6 矩阵）可通过**非侵入式投影**计算：

```
Â_β = E[A(ξ) · ψ_β(ξ)] / E[ψ_β²]
    = ∫ A(x(ξ)) · ψ_β(ξ) · ρ(ξ) dξ / (β!)
```

这个积分用 Gauss-Hermite 张量积求积法数值计算。对 p=2 阶，每维 3 个节点，6 维空间需 3^6 = 729 个求积点（或用稀疏网格 Smolyak 算法降到约 200 个点）。

#### 3.1.2 投影动力学的紧凑形式

定义展开系数向量（将所有 PC 系数堆叠）：

```
X = [x̂_1; x̂_2; ...; x̂_{P_dim}]  ∈ R^{6·P_dim}
U = [û_1; û_2; ...; û_{P_dim}]  ∈ R^{3·P_dim}
```

则投影动力学为：

```
Ẋ = A_gPC · X + B_gPC · U
```

其中：
- `A_gPC` 是 `(6·P_dim) × (6·P_dim)` 的**分块矩阵**，每个 `6×6` 块为 `Σ_{β} c_{βγδ} · Â_β`
- `B_gPC = I_{P_dim} ⊗ B` 是 `(6·P_dim) × (3·P_dim)` 的块对角矩阵

### 3.2 期望代价函数的 gPC 表示

无限时域期望二次型代价：

```
J = E_ξ [∫_0^∞ (x^T Q x + u^T R u) dt]
```

利用 gPC 展开的正交性：

```
E[x^T Q x] = E[(Σ_α x̂_α ψ_α)^T Q (Σ_β x̂_β ψ_β)]
           = Σ_{α,β} x̂_α^T Q x̂_β · E[ψ_α ψ_β]
           = Σ_α x̂_α^T Q x̂_α · α!        (因为正交性)
           = X^T · (Q ⊗ W) · X
```

其中 `⊗` 是 Kronecker 积，`W = diag(α!)` 是 Hermite 多项式的 Gram 矩阵。

同理：`E[u^T R u] = U^T · (R ⊗ W) · U`

因此**随机最优控制问题完全转化为确定性问题**：

```
min_U  ∫_0^∞ [X^T Q̄ X + U^T R̄ U] dt
s.t.   Ẋ = A_gPC · X + B_gPC · U
```

其中 `Q̄ = Q ⊗ W`，`R̄ = R ⊗ W`。

### 3.3 确定性 SDARE

上述确定性 LQR 问题的解由**代数 Riccati 方程**给出：

```
A_gPC^T · Π + Π · A_gPC - Π · B_gPC · R̄^{-1} · B_gPC^T · Π + Q̄ = 0
```

其中 `Π` 是 `(6·P_dim) × (6·P_dim)` 的对称正定矩阵。

这是方案 C 的**核心方程**。一旦求解出 `Π`，最优反馈控制律为：

```
U = -R̄^{-1} · B_gPC^T · Π · X
```

#### 3.3.1 控制律的还原

由于 `X` 包含了所有 gPC 系数，需要还原到物理空间。展开反馈增益：

```
U = [û_1; ...; û_{P_dim}] = -R̄^{-1} · B_gPC^T · Π · [x̂_1; ...; x̂_{P_dim}]
```

零阶项（`û_1`，对应 `ψ_1(ξ) = 1`）给出**均值控制**：

```
ū = û_1 = -R^{-1} · B^T · (某个 6×6 有效增益矩阵) · x̂_1
```

高阶项（`û_β, β≠1`）给出**控制的不确定性依赖**：

```
u(ξ) = ū + Σ_{β≠1} û_β · ψ_β(ξ)
```

这告诉我们：控制的**置信区间**以及控制量如何随状态不确定性调整。

---

## 4. 三种计算路径

### 路径 1：侵入式 Galerkin（完全展开）

**步骤**：

1. **SDC 矩阵的 gPC 展开**（离线/在线一次）
   - 选择 Gauss-Hermite 求积节点 `{ξ_q, w_q}_{q=1}^{N_q}`
   - 对每个求积点，用当前均值 `x̄` 和协方差 `P` 构造状态样本：
     ```
     x(ξ_q) = x̄ + L · ξ_q     其中 P = L · L^T
     ```
   - 计算 `A_q = get_SDC_matrix(x(ξ_q))`
   - gPC 系数投影：`Â_β = Σ_q w_q · A_q · ψ_β(ξ_q) / (β!)`

2. **组装 Galerkin 系统**
   - 计算 `A_gPC` 的每个 `(6×6)` 分块：`[A_gPC]_{γδ} = Σ_β c_{βγδ} · Â_β`
   - 构造 `B_gPC = I_{P_dim} ⊗ B`
   - 构造 `Q̄ = Q ⊗ W`, `R̄ = R ⊗ W`

3. **求解确定性 ARE**
   ```
   Π = solve_continuous_are(A_gPC, B_gPC, Q̄, R̄)     ← 尺寸 (6·P_dim)×(6·P_dim)
   ```

4. **提取控制律**
   ```
   K_gPC = R̄^{-1} · B_gPC^T · Π     ← (3·P_dim) × (6·P_dim)
   U = -K_gPC · X
   u_physical = û_1 = U 的前 3 个分量（均值控制）
   ```

**优点**：理论严格，一次 ARE 解出完整的随机反馈律
**缺点**：`A_gPC` 维度随 PC 阶数指数增长（p=2, d=6 → Π 为 168×168）

### 路径 2：摄动法（低阶近似，推荐优先实现）

不求解完整的 Galerkin 系统，而是做**均值 + 一阶摄动**展开。

**假设**：协方差 P 较小（状态估计的不确定性"小"），可以在均值处做摄动展开。

**步骤**：

1. **均值 SDRE**（标准，已经在做）
   ```
   Ā = A(x̄)       ← SDC 矩阵在均值处
   P̄ = solve_continuous_are(Ā, B, Q, R_eff)     ← 标准 ARE (6×6)
   ū = -R^{-1} · B^T · P̄ · x̄                      ← 标准控制律
   ```

2. **敏感性方程**：对状态变量的每个方向求 A 的导数
   ```
   ∂A/∂x_i|_{x=x̄}      i = 1,...,6       ← 6 个 (6×6) 矩阵
   ```
   这可通过有限差分或解析微分（利用 SDC 矩阵的代数结构）计算。

3. **ARE 解的摄动**：令 `Π ≈ P̄ ⊗ e₁e₁^T + Σ_i P̃_i ⊗ (·)`，
   一阶摄动给出关于 `P̃_i` 的 Lyapunov 方程：
   ```
   (Ā - S P̄)^T · P̃_i + P̃_i · (Ā - S P̄) = -(∂A/∂x_i)^T · P̄ - P̄ · (∂A/∂x_i)
   ```
   对每个 i=1,...,6 解 6 个 Lyapunov 方程（每个 6×6，计算代价可忽略）。

4. **协方差依赖的控制修正**
   ```
   u = ū - R^{-1} B^T · Σ_i P̃_i · x̄ · σ_i + R^{-1} B^T · Σ_i P̃_i · L_i · ξ
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               均值修正项（O(P)）              线性随机反馈项
   ```
   其中 `σ_i = √P_{ii}`，`L_i` 是 P 的 Cholesky 因子的第 i 列。

**优点**：
- 计算代价仅比标准 SDRE 多 6 个 Lyapunov 方程和 6 个敏感性矩阵
- 与现有 SDRE 代码几乎无侵入
- 物理意义清晰：协方差的每个方向独立修正控制

**缺点**：
- 仅对小不确定性（P 较小）有效
- 丢失高阶（O(P²)）效应

### 路径 3：随机搭配法（Stochastic Collocation，最实用）

**核心思想**：用少量精心选择的采样点代替全 Galerkin 展开。

**步骤**：

1. **生成配置点**：在 6 维标准正态空间中选择 M 个配置点 `{ξ_m}`
   - 稀疏网格 Smolyak 算法：p=2 时约需 M ≈ 85 个点（vs. 729 个全张量积点）
   - 或 Sigma 点方法（无迹变换）：M = 13 个点（2×6+1）

2. **并行求解 M 个确定性 SDRE**：
   ```
   对每个配置点 m:
       x_m = x̄ + L · ξ_m        ← 配置点对应的状态
       A_m = A(x_m)              ← SDC 矩阵
       P_m = solve_are(A_m, B, Q, R_eff)     ← 确定性 ARE
       u_m = -R^{-1} · B^T · P_m · x_m       ← 配置点控制
   ```

3. **回归/插值得到期望控制**：
   ```
   ū = Σ_m w_m · u_m            ← 加权平均（求积权重）
   或
   û_α = Σ_m w_m · u_m · ψ_α(ξ_m)    ← gPC 系数（多项式插值）
   ```
   然后只用 `û_0`（均值控制）或带上前几个高阶系数。

**优点**：
- **非侵入式**——不改动现有 SDRE 求解器
- M 个独立的 ARE 可完全并行化（GPU 或多线程友好）
- 可处理任意大小不确定性
- Sigma 点版本（M=13）计算代价极低

**缺点**：
- 对配置点数敏感（过高则计算量大，过低则精度差）
- 需要从 M 个控制量中"融合"出一个实际执行的控制量

---

## 5. 与 HNEKF 的统一框架

### 5.1 概念统一

```
                    状态不确定性表示
                          │
          ┌───────────────┼───────────────┐
          │               │               │
     Taylor 多项式     gPC (Hermite)    无迹变换 (UT)
     (HNEKF 用)       (PC-SDRE 用)      (工程桥梁)
          │               │               │
          ▼               ▼               ▼
     RK78 流映射     Galerkin 投影     Sigma 点传播
     (状态预测)      (代价/动力学)     (配置点 ARE)
          │               │               │
          ▼               ▼               ▼
    矩匹配 E[x],E[xx^T]  确定性 SDARE   M 个确定性 ARE
```

### 5.2 统一的多项式表示

HNEKF 的 Taylor 多项式和 PC-SDRE 的 Hermite 多项式描述的是**同一个底层随机量**——只是在不同基函数系下的展开：

| 属性 | HNEKF | PC-SDRE |
|------|-------|---------|
| 基函数 | `{(x_i - x̄_i)^α}` (Taylor) | `{He_α(ξ)}` (Hermite) |
| 变量 | 物理状态 `x` | 标准正态 `ξ` |
| 正交性 | 非正交（依赖 P） | 正交（与 P 无关） |
| 传播 | RK78 流映射 | Galerkin / 配置点 |
| 输出 | `(x̄, P)` (前两阶矩) | `(x̄, P)` + 高阶矩 |

#### 基变换公式

从 Taylor 多项式到 Hermite 多项式的变换：

```
x = x̄ + L · ξ              （仿射变换，L 是 P 的 Cholesky 因子）
(x - x̄)^α = L^α · ξ^α = L^α · Σ_β d_{αβ} · He_β(ξ)
```

其中 `d_{αβ}` 是单项式 `ξ^α` 在 Hermite 基下的展开系数（可从 Hermite 多项式的显式表达式计算）。

**这意味着**：给定 HNEKF 的 Taylor 多项式表示，可以**无损转换**为 Hermite gPC 表示，然后在 Hermite 空间里做 PC-SDRE。反之亦然。

### 5.3 循环耦合：从 PC-SDRE 回到 HNEKF

完整的"统一框架"不是单向的：

```
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  HNEKF: x(ξ) 的 Taylor 表示 → (x̄, P)                 │
  │       │                                              │
  │       ▼ 基变换                                        │
  │  gPC:  x(ξ) 的 Hermite 表示                           │
  │       │                                              │
  │       ▼ Galerkin / 配置点                             │
  │  PC-SDRE: A_gPC, B_gPC → ARE → u(ξ) 的 gPC 表示      │
  │       │                                              │
  │       ▼                                              │
  │  u(ξ) = ū + Σ_{β≠1} û_β He_β(ξ)                     │
  │       │                                              │
  │       ▼ 物理空间控制量                                 │
  │  执行 ū（均值）或 ū + δu（采样）                       │
  │       │                                              │
  │       ▼                                              │
  │  HNEKF 预测步: 传播 x(ξ) 的 Taylor 多项式              │
  │       │                                              │
  │       ▼                                              │
  │  新的 (x̄, P) → 循环                                  │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

---

## 6. 原型实现路线图

### 阶段 1：SC-SDRE（随机搭配 SDRE）—— 2-3 天

**目标**：验证"协方差纳入控制"是否有效

**实现**（路径 3 的简化版）：

```python
# aerospace/control/sc_sdre.py — Stochastic Collocation SDRE

class SCSDREController:
    """随机搭配 SDRE：用无迹变换将协方差纳入 ARE 求解"""
    
    def __init__(self, Q, R, gamma, n_sigma=3):
        self.Q = Q
        self.R = R
        self.gamma = gamma
        self.n_sigma = n_sigma  # 每维 sigma 点数
    
    def compute_control(self, x_mean, P, get_SDC_matrix, t):
        """输入均值和协方差，输出鲁棒控制量"""
        # 步骤 1：生成 Sigma 点 (2n+1 = 13 个)
        L = np.linalg.cholesky(P)
        sigma_points = [x_mean]
        weights = [κ/(6+κ)]  # 中心权重
        for i in range(6):
            sigma_points.append(x_mean + np.sqrt(6+κ) * L[:, i])
            sigma_points.append(x_mean - np.sqrt(6+κ) * L[:, i])
            weights.extend([0.5/(6+κ), 0.5/(6+κ)])
        
        # 步骤 2：对每个 Sigma 点求解 ARE
        P_matrices = []  # ARE 解
        u_samples = []   # 控制量
        for chi in sigma_points:
            A = get_SDC_matrix(chi)
            P_i = solve_continuous_are(A, B, Q, R_eff)
            u_i = -R_inv @ B.T @ P_i @ chi[:6]
            P_matrices.append(P_i)
            u_samples.append(u_i)
        
        # 步骤 3：加权融合
        u_mean = np.average(u_samples, weights=weights, axis=0)
        return u_mean
```

**实验**：修改 `nerm_ekf_sdre.py`，在 EKF 协方差较大时切换为 SC-SDRE，比较追踪精度。

**预期**：EKF 初期（P 大时）SC-SDRE 比标准 SDRE 更保守但更稳定；后期（P 小时）两者等价。

### 阶段 2：摄动 PC-SDRE —— 1 周

**目标**：实现低阶摄动的解析修正

**新增模块**：

```python
# aerospace/control/pc_sdre_perturbation.py

def compute_A_sensitivity(x_mean, dynamics_params):
    """计算 ∂A/∂x_i|_{x=x_mean} (6 个 6×6 矩阵)"""
    # 用解析微分或复步微分

def solve_perturbation_P(A_mean, P_mean, A_sensitivities, S):
    """求解摄动 ARE 修正项"""
    A_cl = A_mean - S @ P_mean     # 闭环矩阵 (6×6)
    P_pert = []
    for dA_i in A_sensitivities:
        rhs = -dA_i.T @ P_mean - P_mean @ dA_i
        P_i = solve_lyapunov(A_cl.T, rhs)
        P_pert.append(P_i)
    return P_pert

def compute_covariance_aware_control(x_mean, P, P_mean, P_pert, B, R_inv):
    """计算带有协方差修正的控制量"""
    u_nominal = -R_inv @ B.T @ P_mean @ x_mean
    u_correction = 0
    for i in range(6):
        sigma_i = np.sqrt(P[i,i])
        u_correction -= R_inv @ B.T @ P_pert[i] @ x_mean * sigma_i
    return u_nominal + u_correction
```

### 阶段 3：全 Galerkin PC-SDRE —— 2-3 周

**目标**：实现完整的侵入式 Galerkin 方案（路径 1）

**关键子模块**：

1. **Hermite 基函数系统** (`aerospace/pc/hermite_basis.py`)
   - 多维 Hermite 多项式的评估：`He_α(ξ)`
   - 三重积张量 `c_{αβγ}` 的预计算和存储
   - 支持稀疏存储（绝大多数 `c_{αβγ} = 0`）

2. **SDC 矩阵的 gPC 展开** (`aerospace/pc/sdc_gpc.py`)
   - 稀疏 Gauss-Hermite 求积（Smolyak 算法）
   - SDC 矩阵在求积节点上的批量求值
   - gPC 系数的投影计算

3. **Galerkin 系统组装** (`aerospace/pc/galerkin_system.py`)
   ```python
   def assemble_A_gpc(A_hat, triple_prod_tensor):
       """组装 (6·P_dim)×(6·P_dim) 的 Galerkin 矩阵"""
       for gamma in range(P_dim):
           for delta in range(P_dim):
               block = np.zeros((6, 6))
               for beta in range(P_dim):
                   c = triple_prod_tensor[beta, gamma, delta]
                   if abs(c) > 1e-12:
                       block += c * A_hat[beta]
               A_gpc[6*gamma:6*(gamma+1), 6*delta:6*(delta+1)] = block
       return A_gpc
   ```

4. **大维度 ARE 求解器** (`aerospace/pc/block_are_solver.py`)
   - 直接 `solve_continuous_are`（如果 dim ≤ 200）
   - 或利用分块结构的迭代法（Newton-Kleinman）
   - 或降阶模型（POD/平衡截断）

### 阶段 4：HNEKF + PC-SDRE 闭环仿真 —— 1-2 周

将 PC-SDRE 嵌入 `nerm_ekf_sdre.py` 的仿真循环中，与标准 SDRE 对比：

- 追踪精度（最终距离）
- 控制能量（总 ΔV）
- 鲁棒性（对初始误差的敏感度）
- 蒙特卡洛统计分析

---

## 7. 文献定位与创新空间

### 7.1 已有工作

| 工作 | 核心贡献 | 与本项目的差异 |
|------|----------|---------------|
| Fisher & Bhattacharya (2008, 2009) | 随机 LQR + gPC Galerkin | **线性**系统，非 SDC |
| Bhusal & Subbarao (ACC 2022) | gPC + SDRE + 参数不确定性 | 仅**参数**不确定性（惯性矩阵），非**状态**不确定性 |
| Nakka & Chung (2021) | gPC-SCP 轨迹优化 | 轨迹优化（开环），非闭环反馈 SDRE |
| Wan, Harinath & Braatz (CDC 2017) | 分段 PC 随机 LQR | 线性系统，保证代价公式 |

### 7.2 本项目的创新空间

#### 创新点 1（中等创新）：**SDC 矩阵的状态不确定性传播到 ARE 解**

> 现有 gPC-SDRE 文献（Bhusal 2022）只处理被控对象的**参数**不确定性（如惯性矩阵未知但固定）。本项目将 gPC 应用于**状态估计**的不确定性——协方差 P 随时间演化（来自 EKF/HNEKF），导致 SDC 矩阵 A(x) 在每个时刻有**时变的随机特性**。
>
> 这是一个新的问题设定，因为参数不确定性的 gPC 展开是**一次性**的（参数固定），而状态不确定性的 gPC 展开需要**每步重新计算**。

#### 创新点 2（较高创新）：**Taylor → Hermite 基变换桥接 HNEKF 与 SDRE**

> HNEKF 在 Taylor 多项式基下工作，SDRE 需要状态分布的信息。我们提出通过基变换矩阵 `d_{αβ}`（单项式 → Hermite）实现两个框架的无缝对接。这使得 HNEKF 的 Taylor 多项式可以在不重新采样的条件下直接投影到 Hermite 空间进行 PC-SDRE 计算。
>
> 据我所知，**Taylor 多项式（HNEKF）与 Hermite gPC（SDRE）之间的基变换桥尚未在文献中出现**。

#### 创新点 3（高创新）：**闭环联合随机框架**

> 将状态估计（HNEKF）和最优控制（SDRE）统一在同一个多项式混沌框架下。两者共享：
> - 同一个随机变量 `ξ` 的参数化
> - 同一套多项式基函数（通过基变换连接）
> - 同一个不确定性的传播机制
>
> 这超越了现有的"先估计、后控制"串行结构，形成一个**估计-控制随机耦合系统**。理论上，这个框架允许分析"估计误差如何通过控制律反馈影响闭环稳定性"——这是一个公认难题（dual control / 随机分离原理的破坏）。

### 7.3 建议的发表策略

| 阶段 | 产出 | 目标期刊/会议 |
|------|------|--------------|
| 阶段 1+2 | "Covariance-Aware SDRE via Unscented Transform" | ACC 或 CDC（短文） |
| 阶段 3 | "Polynomial Chaos-Based Stochastic SDRE for State-Estimation-Aware Optimal Control" | AIAA JGCD 或 Automatica |
| 阶段 4 | "Unified gPC Framework for Joint Estimation and Control of Nonlinear Stochastic Systems" | IEEE TAC 或 IJRNC |

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 全 Galerkin ARE 维度爆炸 | 高 | 高 | 优先走阶段 1/2（低维）；阶段 3 用稀疏网格 + 降阶 |
| P 较小时 PC 修正不明显 | 中 | 中 | 设计大不确定性场景（初始误差大、弱可观测轨道） |
| 控制"过于保守" | 中 | 中 | 引入风险调节参数（类似 LEQG） |
| SDC 非唯一性在随机版本中更严重 | 低 | 中 | 约束 SDC 参数化（当前 SDC 已经是自然的） |
| 三重积张量内存爆炸 | 高 | 低 | 稀疏存储（95%+ 的 `c_{αβγ} = 0`） |

---

## 参考文献

1. Bhusal, R. & Subbarao, K. (2022). "A State-Dependent Riccati Equation-Based Robust Control Approach for Nonlinear Systems with Parametric Uncertainties." ACC 2022.
2. Bhusal, R. (2021). "Uncertainty Propagation, Control, and Estimation of Stochastic Dynamic Systems Using Generalized Polynomial Chaos Expansion." Ph.D. Dissertation, UT Arlington.
3. Fisher, J. & Bhattacharya, R. (2009). "Linear quadratic regulation of systems with stochastic parameter uncertainties." Automatica, 45(12).
4. Nakka, Y. & Chung, S.-J. (2021). "Trajectory Optimization of Chance-Constrained Nonlinear Stochastic Systems for Motion Planning Under Uncertainty." arXiv:2106.02801.
5. Wan, Harinath & Braatz (2017). "A piecewise polynomial chaos approach to stochastic linear quadratic regulation." CDC 2017.
6. Xiu, D. & Karniadakis, G.E. (2002). "The Wiener-Askey polynomial chaos for stochastic differential equations." SIAM J. Sci. Comput., 24(2).
