# NERM + EKF + SDRE 航天器追逃仿真报告

## 问题描述

椭圆轨道上的航天器追逃博弈：追踪星通过测量相对距离-方位角-俯仰角，经 EKF 估计相对状态，再由 SDRE 博弈控制器计算推力，实现对逃逸星的追捕。逃逸星同样由 SDRE 计算最优逃逸策略。

---

## 系统模型

### 轨道动力学（NERM）

采用非线性椭圆相对运动方程（NERM），状态向量为 13 维：

$$\mathbf{s} = [\mathbf{X}_p^\top,\ \mathbf{X}_e^\top,\ \nu]^\top \in \mathbb{R}^{13}$$

其中 $\mathbf{X}_p, \mathbf{X}_e \in \mathbb{R}^6$ 分别为追踪星和逃逸星在 LVLH 坐标系下的绝对位置-速度，$\nu$ 为参考轨道真近点角。

参考轨道参数：

| 参数 | 值 |
|------|-----|
| 引力常数 $\mu$ | 3.986×10⁵ km³/s² |
| 半长轴 $a_c$ | 15000 km |
| 离心率 $e_c$ | 0.5 |
| 轨道周期 $T$ | 5.08 h |

### 观测模型

追踪星仅配备测角传感器（精度 $0.008°$），对逃逸星的相对测量为方位角-俯仰角（AE），无距离测量：

$$\mathbf{z} = \begin{bmatrix} \alpha \\ \varepsilon \end{bmatrix} = \begin{bmatrix} \arctan2(\Delta y,\ \Delta x) \\ \arcsin(\Delta z / \rho) \end{bmatrix}$$

其中 $\Delta\mathbf{r} = \mathbf{r}_p - \mathbf{r}_e$ 为追踪星坐标系下逃逸星的相对位移。

测量噪声标准差：$\sigma_\alpha = \sigma_\varepsilon = 0.008° = 1.396 \times 10^{-4}\ \text{rad}$

EKF 实现中保留 $[\rho, \alpha, \varepsilon]$ 三维测量向量，通过设置 $R_{\rho\rho} = 10^{10}$ 使距离测量权重趋近于零，等效为纯测角模型。

### EKF 状态估计

状态：相对位置-速度 $\mathbf{x} = \mathbf{X}_p - \mathbf{X}_e \in \mathbb{R}^6$

**预测步**（对齐 C++ 离散化结构）：

$$\mathbf{F} = \mathbf{I} + \mathbf{A}_\text{SDC} \cdot \Delta t$$

$$\hat{\mathbf{x}}^- = \mathbf{F}\hat{\mathbf{x}} + \Delta t \cdot \mathbf{B}(\mathbf{u}_p - \mathbf{u}_e)$$

$$\mathbf{P}^- = \mathbf{F}\mathbf{P}\mathbf{F}^\top + \mathbf{Q}$$

**更新步**（在预测状态处线性化）：

$$\mathbf{H} = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}^-}, \quad \mathbf{K} = \mathbf{P}^- \mathbf{H}^\top (\mathbf{H}\mathbf{P}^-\mathbf{H}^\top + \mathbf{R})^{-1}$$

$$\hat{\mathbf{x}} = \hat{\mathbf{x}}^- + \mathbf{K}(\mathbf{z} - h(\hat{\mathbf{x}}^-)), \quad \mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}^-$$

过程噪声：$\mathbf{Q} = \text{diag}(5\times10^{-4},\ 5\times10^{-4},\ 5\times10^{-4},\ 5\times10^{-8},\ 5\times10^{-8},\ 5\times10^{-8})$

### SDRE 博弈控制器

相对状态方程的 SDC 分解：$\dot{\mathbf{x}} = \mathbf{A}_\text{SDC}(\mathbf{x})\mathbf{x} + \mathbf{B}\mathbf{u}_p - \mathbf{B}\mathbf{u}_e$

求解连续时间代数 Riccati 方程（ARE）：

$$\mathbf{A}^\top \mathbf{P} + \mathbf{P}\mathbf{A} + \mathbf{Q} - \mathbf{P}\mathbf{B}\left(\mathbf{R}^{-1} - \frac{1}{\gamma^2}\mathbf{R}^{-1}\right)\mathbf{B}^\top\mathbf{P} = 0$$

控制律：

$$\mathbf{u}_p = -\mathbf{R}^{-1}\mathbf{B}^\top\mathbf{P}\hat{\mathbf{x}}, \quad \mathbf{u}_e = \frac{1}{\gamma^2}\mathbf{R}^{-1}\mathbf{B}^\top\mathbf{P}\hat{\mathbf{x}}$$

控制参数：$\mathbf{Q}_c = \mathbf{I}_6$，$\mathbf{R}_c = 10^{13}\mathbf{I}_3$，$\gamma = \sqrt{2}$

---

## 仿真设置

| 参数 | 值 |
|------|-----|
| 追踪星初始位置 | [500, 500, 500] km |
| 追踪星初始速度 | [0.01, 0.01, 0.01] km/s |
| 逃逸星初始状态 | 原点静止 |
| 初始相对距离 | 866.0 km |
| 控制步长 $\Delta t$ | 10 s |
| 捕获阈值 | 100 m |
| 随机种子 | 42 |

---

## 多场景仿真结果

针对五种典型的追逃几何场景进行了仿真验证。每种场景均运行了“有噪声（EKF+SDRE）”和“理想测量（Ideal SDRE）”两个版本。

| 场景 | 初始距离 (km) | 捕获时间 (Noisy) | 捕获时间 (Ideal) |
| :--- | :--- | :--- | :--- |
| **场景1：同平面同轨道** | 100.4 | 14130 s (3.93 h) | 18580 s (5.16 h) |
| **场景2：不同平面同轨道** | 100.3 | 20560 s (5.71 h) | 25700 s (7.14 h) |
| **场景3：同平面不同轨道(低)** | 100.5 | 16560 s (4.60 h) | 20730 s (5.76 h) |
| **场景4：同平面不同轨道(高)** | 100.2 | 16380 s (4.55 h) | 22950 s (6.38 h) |
| **场景5：不同平面不同轨道(高)** | 100.1 | 23610 s (6.56 h) | 25600 s (7.11 h) |


---

## 输出图表

每次运行 `uv run python main.py` 生成以下图表：

| 文件 | 内容 |
|------|------|
| `*_rel.png` | LVLH 相对运动轨迹（3D + 投影）、位置/速度各分量时间历程、相对距离 |
| `*_ctrl.png` | 追踪星/逃逸星推力三分量及范数时间历程 |
| `*_ekf.png` | EKF 位置/速度误差 + 3σ 包络、新息序列（含捕获时刻标注） |
| `nerm_comparison.png` | 有/无噪声相对距离与估计误差对比 |

---

## 代码结构

```
aerospace/
├── dynamics/nerm.py        OrbitalDynamics — NERM 动力学、SDC 矩阵
├── control/sdre.py         SDREGameController — ARE 求解、博弈控制律
├── estimation/ekf.py       RelativeStateEKF — RAE 观测模型、EKF 预测/更新
├── simulation/
│   └── nerm_ekf_sdre.py    EKFSDRESimulation — 闭环仿真引擎
└── visualization/
    └── ekf_plots.py        plot_single_simulation / plot_comparison
main.py                     主入口
```
