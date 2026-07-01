# 3D SDRE vs CasADi 最优控制对比分析

## 核心问题

2D 中 SDRE 存在 ~149% 次优性，3D 情况如何？用不同轨道场景系统对比。

## 方法论

### CasADi 3D 直接配点求解器
- **文件**: `aerospace/control/optimal_control_solver_3d.py`
- 2D→3D 扩展：状态 5D→7D (增加 dz, dvz)，控制 2D→3D (增加 uz)
- z 方向无科氏力耦合，纯引力恢复：dz/dt = dvz, dvz/dt = -mu * dz / r_p^3 + uz
- 5 个 gamma (1e5~1e7) 全部收敛，终端误差 ~1mm，求解时间 1.2~4.6s

### 公平对比脚本
- **文件**: `experiment_sdre_vs_optimal_3d.py`
- 双方使用相同代价函数：J = int(x^T Q x + u^T R u) dt（纯追捕）
- Q=diag(1,1,1,10,10,10), R=1e13 * I_3 (SDRE 默认参数)
- CasADi 固定 T = T_sdre，从 SDRE 轨迹 warm-start
- 大椭圆轨道 (a=15000, e=0.5) 初始距离 510km

### 多场景批量对比
- **文件**: `experiment_batch_sdre_vs_optimal.py`
- 导入 run_scenarios.py 的 5 个场景，ECI→LVLH 转换
- 近圆轨道 (a=6871, e=0.1)，初始距离 100~200km

## 关键发现

### 1. 大椭圆轨道：SDRE 准最优

| 方法 | J | 次优 gap | peak ||u|| |
|------|---|---------|---------------|
| SDRE 3D | 1.357e+09 | — | 0.21 mm/s^2 |
| CasADi 3D | 1.205e+09 | **12.6%** | 0.22 mm/s^2 |

R=1e13 的强惩罚使推力接近 0，轨迹基本是弹道漂移。SDRE 的恒定增益近似在弱控制区域表现得很好。

### 2. 近圆轨道：SDRE 差异巨大（65%~268%）

| 场景 | T_sdre (h) | SDRE gap |
|------|-----------|---------|
| 场景1：同平面同轨道 | 5.11 | **65.1%** |
| 场景2：不同平面同轨道 | 5.82 | **115.8%** |
| 场景3：同平面不同轨道（低） | 4.59 | **146.6%** |
| 场景4：同平面不同轨道（高） | 5.38 | **268.1%** |
| 场景5：不同平面不同轨道（高） | 5.83 | **245.1%** |

### 3. 跨轨道面 (z) 分量放大次优性

- 场景 1 (纯平面内，z=0) → 65%，次优性最小
- 场景 2/5 (有 z 偏移) → 116%~245%，SDRE 在跨轨道面控制上更次优
- 场景 4 (高轨+大初始距离) → 268%，最差

### 4. 3D CasADi 收敛问题

公平对比中 CasADi 全部 hit 1000 iter limit 未收敛到最优。debug 解给出可行下界，真正的 gap 只会更大。可能原因：
- 固定 T 约束过紧
- 强 R 惩罚使目标曲率大，IPOPT 收敛慢
- 需要更好的 warm-start 或 continuation 策略

## 与 2D 对比

| | 2D (a=15000,e=0.5) | 3D (a=15000,e=0.5) | 3D (a=6871,e=0.1, 5场景) |
|---|---|---|---|
| SDRE 次优性 | 149% | 12.6% | 65%~268% |
| 推力水平 | 饱和 (0.01 km/s^2) | ~0.2 mm/s^2 | ~mm/s^2 |
| 主要因素 | R 值 + 控制方向 | z 分量分散推力 | z 偏移 + 初始距离 |

## 改进方向

1. **SDRE 在 z 方向的修正**：z 方向无科氏力，增益与 xy 面不同，可对角化 R 为 [r_xy, r_z]
2. **按场景分类**：纯平面场景用 SDRE，有 z 偏移场景需要修正
3. **学习 z 方向修正因子**：小自由度 (1 个参数)，比全状态学习简单

## 新增文件

| 文件 | 作用 |
|------|------|
| `aerospace/control/optimal_control_solver_3d.py` | CasADi+IPOPT 3D 直接配点求解器 |
| `experiment_sdre_vs_optimal_3d.py` | 3D SDRE vs CasADi 公平对比（同 Q,R） |
| `experiment_batch_sdre_vs_optimal.py` | 5 场景批量对比 |

## 运行命令

```bash
# 3D 最优控制求解 (5个gamma值)
uv run python -m aerospace.control.optimal_control_solver_3d

# 单场景公平对比
uv run python experiment_sdre_vs_optimal_3d.py

# 5 场景批量对比
uv run python experiment_batch_sdre_vs_optimal.py
```
