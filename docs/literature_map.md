# 文献检索与定位分析

**日期**: 2026-05-30
**用途**: Acta Astronautica 论文 "Angles-Only Relative Navigation and Game-Theoretic Control via Unified SDC Parameterization"

---

## 文献地图

### 1. 相对运动动力学

| 文献 | 贡献 | 与本论文关系 |
|------|------|:--:|
| **Clohessy & Wiltshire (1960)** — J. Aerospace Sciences | CW 方程：圆轨道线性相对运动 | 线性基线，§5.3 对比 |
| **Tschauner & Hempel (1965)** — Astronautica Acta | TH 方程：椭圆轨道线性时变相对运动 | NERM 的理论前身 |
| **Yamanaka & Ankersen (2002)** — JGCD | 椭圆轨道非奇异状态转移矩阵 | 椭圆轨道建模参考 |
| **Dwidar & Owis (2013)** — IJARAI | TH+J₂+SDRE 椭圆轨道编队控制 | TH-SDRE 结合的先例 |

### 2. 仅测角相对导航

| 文献 | 贡献 | 与本论文关系 |
|------|------|:--:|
| **Woffinden & Geller (2009)** — IEEE TAES | 仅测角可观测性判据：CW 线性系统下距离不完全可观 | 理论基础，§2.4 |
| **Woffinden & Geller (2009)** — JGCD | 最优可观测性机动设计 | 对比参考 |
| **Geller & Klein (2014)** — JGCD | 相机偏置实现零-Δv 仅测角可观性 | 仅测角方案对比 |
| **Gaias, D'Amico & Ardaens (2014)** — JGCD | PRISMA/AVANTI 飞行实验：相对轨道要素 + 仅测角 | 实验验证参考 |
| **Sullivan & D'Amico (2017)** — JGCD | 非线性 Kalman + 相对轨道要素 仅测角 | EKF 对比方法 |

### 3. SDRE 控制

| 文献 | 贡献 | 与本论文关系 |
|------|------|:--:|
| **Cloutier (1997)** — 开创性论文 | SDRE 方法的建立：SDC 参数化 + 逐点 ARE | 理论奠基 |
| **Çimen (2010)** — Annual Reviews in Control | SDRE 综述 (第一部分) | 方法总览 |
| **Çimen (2012)** — JGCD | SDRE 综述 (全面版)，~321+ 引用 | **核心引用**，§3.3 |
| **Stansbery & Cloutier (2000)** | SDRE 追逃博弈 | SDRE 博弈理论前身 |
| **Tartaglia & Innocenti (2016)** — AIAA GNC | 航天器交会的 SDRE 非零和博弈 | SDRE 交会参考 |
| **Li, Huang, Wang & Cui (2016)** — 系统工程与电子技术 | 终端碰撞角约束 SDRE 微分对策制导律 | SDRE 博弈对比 |

### 4. SDREF（SDRE 滤波）

| 文献 | 贡献 | 与本论文关系 |
|------|------|:--:|
| **Mracek, Cloutier & D'Souza (1996)** | SDREF 的首次提出 | 理论奠基 |
| **Park & Kim (2016)** — Advances in Space Research | **SDREF 用于编队飞行相对导航**（J₂ 扰动 + ECEF 坐标系）| **最接近的估计方法** |
| **Simhamed & Ykhlef (2021)** — Measurement | SDREF vs EKF：频踪，SDREF 精度和鲁棒性均优于 EKF | SDREF 对比数据 |
| **Tahirovic & Redzovic (2025)** — arXiv | SDRE-KF vs EKF vs 粒子滤波：SDRE-KF 在强非线性下优于 EKF | 最新对比 |

### 5. 估计 + 控制结合

| 文献 | 贡献 | 与本论文关系 |
|------|------|:--:|
| **Jagat (2015)** — Auburn 博士论文 | SDRE 追逃博弈 + 仅测角导航（LQG 双控制 + 信息加权 LQG）| **最接近的问题设定** |
| **Vepa (2018)** — Journal of Navigation | 非线性 TH 方程 + 估计 + 控制（LQNG 方法）| **估计+控制结合的先例** |
| **Okasha & Newman (2011)** | TH+EKF+滑翔制导用于自主交会 | 组合方案对比 |
| **Choukroun & Tekinalp (2013)** — EUCASS | **SDRE 同时用于航天器姿态估计与控制**：单/双环 SDRE 控制器 + 伪线性四元数 SDRE 滤波器 | **"同方法做估计+控制"的最接近先例**（姿态域） |
| **Lee, Cochran & No (2012)** — J. Aerospace Eng. | SDRE 控制器 + EKF 估计器用于编队飞行位置+姿态控制 | SDRE+EKF 组合，但**模型不同** |
| **Muralidhar & Kumar (2021)** — IAC | SDRE 控制器 + UKF 估计器，TH 方程，椭圆轨道交会对接 | SDRE 控制 + UKF 估计，TH 非 SDC |
| **Wang, Butcher & Lovell (2018)** — Acta Astronautica | 仅测距 EKF 中模糊轨道的非线性可观测性分析 | 椭圆轨道相对导航 EKF 模糊性 |

---

## 文献缺口确认

### 没有人做过：统一 SDC 参数化同时驱动 EKF 预测和 SDRE 控制（相对位置导航）

| 文献 | 滤波用 SDC? | 控制用 SDC? | 同一矩阵? | 域 |
|------|:--:|:--:|:--:|:--:|
| Park & Kim (2016) SDREF | ✓ SDC | ✗ (无控制) | N/A | 相对位置 |
| Jagat (2015) | ✗ (独立 LQG/EKF) | ✓ SDRE | N/A | 相对位置 |
| Vepa (2018) | ✓ TH 方程 | ✓ TH 方程 | TH≠SDC | 相对位置 |
| Choukroun & Tekinalp (2013) | ✓ SDRE | ✓ SDRE | **同方法** | **姿态** |
| Lee, Cochran & No (2012) | ✗ (EKF) | ✓ SDRE | ✗ | 相对位置 |
| Muralidhar & Kumar (2021) | ✗ (UKF) | ✓ SDRE | ✗ | 相对位置 |
| **本论文** | ✓ SDC | ✓ SDC | **同一个 `A_SDC(t_k)`** | **相对位置** |

### 关键差异化

1. **Choukroun & Tekinalp (2013)** 用 SDRE 同时做姿态估计和控制——"概念上最接近"。但：(a) 域不同（姿态 vs 相对位置），(b) 未明确论证"同一 SDC 矩阵"的复用，(c) 无双星相对导航与仅测角约束。

2. **Lee et al. (2012)** 用 SDRE+EKF 但模型不同——SDRE 用非线性模型，EKF 用独立线性化模型。这正是本文要消除的"模型失配"。

3. **Muralidhar & Kumar (2021)** 最接近的问题设定（椭圆轨道交会），但用 UKF 而非 EKF，TH 而非 SDC，不存在"同一矩阵复用"。

4. **没有任何工作**同时满足：(a) 相对位置导航，(b) SDC 参数化，(c) 同一 A 矩阵用于 EKF 和 SDRE，(d) 仅测角，(e) 系统实验验证。

---

## 建议引用列表（约 25-30 篇）

### 必引（8-10 篇）

1. Clohessy & Wiltshire (1960) — CW 方程
2. Cloutier (1997) — SDRE 奠基
3. Çimen (2012) — SDRE 综述 (§3.3 方法基础)
4. Park & Kim (2016) — SDREF 相对导航 (§1.2 最接近的估计方法)
5. Jagat (2015) — SDRE 追逃博弈 + 仅测角 (§1.2 最接近的问题设定)
6. Woffinden & Geller (2009) — 仅测角可观测性 (§2.4)
7. Geller & Klein (2014) — 仅测角接近操作 (§5.5 对比)
8. Vepa (2018) — 非线性 TH + 估计 + 控制 (§1.2 先例 + §5.5 对比)

### 补充引用（12-15 篇）

9. Tschauner & Hempel (1965) — TH 方程
10. Yamanaka & Ankersen (2002) — 椭圆轨道 STM
11. Dwidar & Owis (2013) — TH+SDRE 编队
12. Stansbery & Cloutier (2000) — SDRE 追逃
13. Tartaglia & Innocenti (2016) — SDRE 航天器交会
14. Simhamed & Ykhlef (2021) — SDREF vs EKF 对比
15. Okasha & Newman (2011) — TH+EKF 交会
16. Sullivan & D'Amico (2017) — 非线性 KF 仅测角
17. Gaias, D'Amico & Ardaens (2014) — PRISMA 飞行实验
18. Mracek, Cloutier & D'Souza (1996) — SDREF 首次提出
19. Li et al. (2016) — SDRE 微分对策
20. Tahirovic & Redzovic (2025) — SDRE-KF 最新对比

### 可选（3-5 篇，如需扩展章节）

21. Woffinden & Geller (2009) JGCD — 最优可观测性机动
22. Carter (1998) — TH 方程闭式解
23. Schweighart & Sedwick (2002) — J₂ 编队模型
24. Melton (2000) — LERM 椭圆轨道线性时变模型
25. Roa & Peláez (2017) — 异步相对运动非线性修正

---

## 定位陈述（论文 §1.2 缺口论据的核心句）

> While the SDRE technique has been extensively surveyed by Çimen (2012) and applied to spacecraft formation control (Dwidar & Owis, 2013), and the SDREF has been validated for relative state estimation (Park & Kim, 2016), **no prior work has exploited the natural coincidence of linearization points in the SDC framework to unify estimation and control within a single SDC parameterization.** The closest efforts either separate the estimation and control models (Jagat, 2015) or use alternative linearizations that forfeit the exact reconstruction property `A(x)·x ≡ f(x)` of the SDC factorization (Vepa, 2018).

---

## 状态

- [x] 文献检索完成
- [ ] 下载/获取全文（关键 8 篇）
- [ ] 撰写 §1.2 文献回顾
- [ ] 撰写 §5.5 比较讨论
