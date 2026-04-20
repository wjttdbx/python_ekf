"""
对比 SDRE 和 MPC 在追逃博弈中的性能

实验设计：
1. 相同初始条件
2. SDRE: 瞬时最优反馈控制
3. MPC: 有限时域轨迹优化（预测时域 N=10, 20, 50）
4. 比较指标：
   - 终端距离
   - 总燃料消耗
   - Cost-to-go: J = ∫[x^T Q x + u_p^T R u_p - γ^-2 u_e^T R u_e] dt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from aerospace.dynamics.nerm_2d import OrbitalDynamics2D
from aerospace.control.sdre_2d import SDRE2DController
from aerospace.paths import FIGURES_DIR
import warnings
warnings.filterwarnings('ignore')


class SimpleMPC2D:
    """
    简化的 2D MPC 控制器（用于概念验证）
    使用梯度下降优化，不是 DA-MPC（先验证思路）
    """

    def __init__(self, dynamics, nu, horizon=10, dt=1.0, max_iter=50):
        self.dynamics = dynamics
        self.nu = nu  # 当前真近点角
        self.horizon = horizon
        self.dt = dt
        self.max_iter = max_iter

        # 权重矩阵（与 SDRE 相同）
        self.Q = np.eye(4)
        self.R = 1e13 * np.eye(2)
        self.gamma = np.sqrt(2)

        # 控制序列初始化
        self.U_p = np.zeros((horizon, 2))
        self.U_e = np.zeros((horizon, 2))

    def predict_trajectory(self, x_rel, U_p, U_e):
        """前向预测轨迹（相对状态）"""
        X = [x_rel]
        x = x_rel.copy()

        # 需要绝对位置来计算动力学，这里简化假设两者初始位置
        X_p = np.array([x[0]/2, x[1]/2, x[2]/2, x[3]/2])
        X_e = np.array([-x[0]/2, -x[1]/2, -x[2]/2, -x[3]/2])
        nu = self.nu

        for k in range(self.horizon):
            u_p = U_p[k] if k < len(U_p) else np.zeros(2)
            u_e = U_e[k] if k < len(U_e) else np.zeros(2)

            # 构建 9D 状态
            state_9d = np.concatenate([X_p, X_e, [nu]])

            # 计算导数
            dstate = self.dynamics.dynamics_9d(0, state_9d, u_p, u_e)

            # 欧拉积分
            X_p = X_p + dstate[0:4] * self.dt
            X_e = X_e + dstate[4:8] * self.dt
            nu = nu + dstate[8] * self.dt

            # 相对状态
            x = X_p - X_e
            X.append(x)

        return np.array(X)

    def compute_cost(self, X, U_p, U_e):
        """计算轨迹代价"""
        J = 0.0

        for k in range(self.horizon):
            x = X[k]
            u_p = U_p[k] if k < len(U_p) else np.zeros(2)
            u_e = U_e[k] if k < len(U_e) else np.zeros(2)

            # 阶段代价
            J += x @ self.Q @ x
            J += u_p @ self.R @ u_p
            J -= (1 / self.gamma**2) * (u_e @ self.R @ u_e)

        # 终端代价
        x_N = X[-1]
        J += 10 * (x_N @ self.Q @ x_N)

        return J

    def compute_gradient_fd(self, x0, U_p, U_e):
        """有限差分计算梯度（简化版）"""
        eps = 1e-6

        # 当前代价
        X0 = self.predict_trajectory(x0, U_p, U_e)
        J0 = self.compute_cost(X0, U_p, U_e)

        grad_U_p = np.zeros_like(U_p)
        grad_U_e = np.zeros_like(U_e)

        # 对追击者控制求梯度
        for k in range(len(U_p)):
            for j in range(2):
                U_p_pert = U_p.copy()
                U_p_pert[k, j] += eps
                X_pert = self.predict_trajectory(x0, U_p_pert, U_e)
                J_pert = self.compute_cost(X_pert, U_p_pert, U_e)
                grad_U_p[k, j] = (J_pert - J0) / eps

        # 对逃逸者控制求梯度（注意符号：逃逸者最大化）
        for k in range(len(U_e)):
            for j in range(2):
                U_e_pert = U_e.copy()
                U_e_pert[k, j] += eps
                X_pert = self.predict_trajectory(x0, U_p, U_e_pert)
                J_pert = self.compute_cost(X_pert, U_p, U_e_pert)
                grad_U_e[k, j] = -(J_pert - J0) / eps  # 负号：最大化

        return grad_U_p, grad_U_e

    def solve(self, x0):
        """求解 MPC（鞍点迭代）"""
        U_p = self.U_p.copy()
        U_e = self.U_e.copy()

        alpha = 0.01  # 步长

        for iter in range(self.max_iter):
            grad_U_p, grad_U_e = self.compute_gradient_fd(x0, U_p, U_e)

            # 梯度下降/上升
            U_p_new = U_p - alpha * grad_U_p
            U_e_new = U_e - alpha * grad_U_e

            # 控制约束投影
            u_max = 0.1  # m/s^2
            U_p_new = np.clip(U_p_new, -u_max, u_max)
            U_e_new = np.clip(U_e_new, -u_max, u_max)

            # 检查收敛
            if np.linalg.norm(U_p_new - U_p) < 1e-4 and \
               np.linalg.norm(U_e_new - U_e) < 1e-4:
                break

            U_p = U_p_new
            U_e = U_e_new

        # Warm start：保存优化结果
        self.U_p = np.vstack([U_p[1:], np.zeros((1, 2))])
        self.U_e = np.vstack([U_e[1:], np.zeros((1, 2))])

        return U_p[0], U_e[0]


def run_comparison():
    """运行对比实验"""

    # 轨道参数
    mu = 3.986004418e14
    a = 6378137.0 + 400000.0
    n = np.sqrt(mu / a**3)

    # 初始条件（2D 面内）
    x0 = np.array([100.0, 50.0, -0.1, 0.05])  # [x, y, vx, vy]

    # 仿真参数
    dt = 1.0
    t_final = 200.0
    steps = int(t_final / dt)

    # 创建动力学
    dynamics = OrbitalDynamics2D(
        mu=mu, a=a, e=0.0,
        m_p=100.0, m_e=100.0,
        Q=np.eye(4), R=1e13 * np.eye(2),
        gamma=np.sqrt(2)
    )

    print("=" * 60)
    print("追逃博弈性能对比：SDRE vs MPC")
    print("=" * 60)
    print(f"初始状态: x={x0[0]:.1f}m, y={x0[1]:.1f}m")
    print(f"初始距离: {np.linalg.norm(x0[:2]):.1f}m")
    print()

    # ========== 方法 1: SDRE ==========
    print("运行 SDRE 控制器...")
    sdre = SDRE2DController(dynamics)

    x_sdre = x0.copy()
    traj_sdre = [x_sdre.copy()]
    cost_sdre = 0.0
    fuel_sdre = 0.0

    for step in range(steps):
        u_p, u_e = sdre.compute_control(x_sdre)

        # 计算代价
        cost_sdre += x_sdre @ dynamics.Q @ x_sdre * dt
        cost_sdre += u_p @ dynamics.R @ u_p * dt
        cost_sdre -= (1 / dynamics.gamma**2) * (u_e @ dynamics.R @ u_e) * dt

        fuel_sdre += np.linalg.norm(u_p) * dt

        # 积分
        dx = dynamics._dynamics(0, x_sdre, u_p, u_e)
        x_sdre = x_sdre + dx * dt
        traj_sdre.append(x_sdre.copy())

        if np.linalg.norm(x_sdre[:2]) < 1.0:
            print(f"  SDRE: 捕获成功 @ t={step*dt:.1f}s")
            break

    traj_sdre = np.array(traj_sdre)
    dist_sdre = np.linalg.norm(traj_sdre[-1, :2])

    print(f"  终端距离: {dist_sdre:.2f}m")
    print(f"  总代价: {cost_sdre:.2e}")
    print(f"  总燃料: {fuel_sdre:.2f} m/s")
    print()

    # ========== 方法 2: MPC (不同时域) ==========
    results = {}

    for horizon in [5, 10, 20]:
        print(f"运行 MPC 控制器 (horizon={horizon})...")
        mpc = SimpleMPC2D(dynamics, horizon=horizon, dt=dt, max_iter=20)

        x_mpc = x0.copy()
        traj_mpc = [x_mpc.copy()]
        cost_mpc = 0.0
        fuel_mpc = 0.0

        for step in range(steps):
            u_p, u_e = mpc.solve(x_mpc)

            # 计算代价
            cost_mpc += x_mpc @ dynamics.Q @ x_mpc * dt
            cost_mpc += u_p @ dynamics.R @ u_p * dt
            cost_mpc -= (1 / dynamics.gamma**2) * (u_e @ dynamics.R @ u_e) * dt

            fuel_mpc += np.linalg.norm(u_p) * dt

            # 积分
            dx = dynamics._dynamics(0, x_mpc, u_p, u_e)
            x_mpc = x_mpc + dx * dt
            traj_mpc.append(x_mpc.copy())

            if np.linalg.norm(x_mpc[:2]) < 1.0:
                print(f"  MPC-{horizon}: 捕获成功 @ t={step*dt:.1f}s")
                break

            if step % 50 == 0:
                print(f"    步骤 {step}: 距离={np.linalg.norm(x_mpc[:2]):.1f}m")

        traj_mpc = np.array(traj_mpc)
        dist_mpc = np.linalg.norm(traj_mpc[-1, :2])

        print(f"  终端距离: {dist_mpc:.2f}m")
        print(f"  总代价: {cost_mpc:.2e}")
        print(f"  总燃料: {fuel_mpc:.2f} m/s")
        print(f"  相对 SDRE 改进: {(cost_sdre - cost_mpc) / cost_sdre * 100:.1f}%")
        print()

        results[f'MPC-{horizon}'] = {
            'traj': traj_mpc,
            'cost': cost_mpc,
            'fuel': fuel_mpc,
            'dist': dist_mpc
        }

    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 轨迹对比
    ax = axes[0, 0]
    ax.plot(traj_sdre[:, 0], traj_sdre[:, 1], 'b-', label='SDRE', linewidth=2)
    for horizon, data in results.items():
        ax.plot(data['traj'][:, 0], data['traj'][:, 1], '--', label=horizon, alpha=0.7)
    ax.plot(0, 0, 'r*', markersize=15, label='Target')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('轨迹对比')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # 距离曲线
    ax = axes[0, 1]
    t_sdre = np.arange(len(traj_sdre)) * dt
    ax.plot(t_sdre, np.linalg.norm(traj_sdre[:, :2], axis=1), 'b-', label='SDRE', linewidth=2)
    for horizon, data in results.items():
        t_mpc = np.arange(len(data['traj'])) * dt
        ax.plot(t_mpc, np.linalg.norm(data['traj'][:, :2], axis=1), '--', label=horizon, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('距离演化')
    ax.legend()
    ax.grid(True)

    # 性能指标对比
    ax = axes[1, 0]
    methods = ['SDRE'] + list(results.keys())
    costs = [cost_sdre] + [data['cost'] for data in results.values()]
    ax.bar(methods, costs)
    ax.set_ylabel('Total Cost')
    ax.set_title('总代价对比')
    ax.grid(True, axis='y')

    # 燃料消耗对比
    ax = axes[1, 1]
    fuels = [fuel_sdre] + [data['fuel'] for data in results.values()]
    ax.bar(methods, fuels)
    ax.set_ylabel('Total Fuel (m/s)')
    ax.set_title('燃料消耗对比')
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sdre_vs_mpc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {FIGURES_DIR / 'sdre_vs_mpc_comparison.png'}")

    # ========== 结论 ==========
    print("\n" + "=" * 60)
    print("实验结论：")
    print("=" * 60)

    best_mpc = min(results.items(), key=lambda x: x[1]['cost'])
    improvement = (cost_sdre - best_mpc[1]['cost']) / cost_sdre * 100

    if improvement > 5:
        print(f"✅ MPC 显著优于 SDRE！")
        print(f"   最佳 MPC ({best_mpc[0]}) 相对 SDRE 改进: {improvement:.1f}%")
        print(f"\n💡 建议：用 DA-MPC 生成训练数据，训练 PINN 学习更优策略")
    elif improvement > 0:
        print(f"⚠️  MPC 略优于 SDRE（改进 {improvement:.1f}%）")
        print(f"   考虑计算成本，可能不值得")
    else:
        print(f"❌ MPC 未能超越 SDRE")
        print(f"   可能原因：")
        print(f"   1. MPC 优化未收敛（迭代次数不够）")
        print(f"   2. 预测时域太短")
        print(f"   3. SDRE 本身已经很优（对于这个问题）")


if __name__ == '__main__':
    run_comparison()
