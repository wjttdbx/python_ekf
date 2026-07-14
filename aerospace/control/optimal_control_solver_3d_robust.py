"""
CasADi + IPOPT 直接配点法求解 NERM 3D 时间-能量最优控制问题（增强版）

改进点：
1. 放松终端约束到软捕获标准 (0.1 km)
2. 改进初始猜测策略
3. 增加配点数和迭代次数
4. 多gamma自动扫描
5. 自适应终端速度约束

    min_{T, u(·)}  T + gamma * ∫_0^T ||u(t)||² dt
    s.t. x(0)=x0, ||r(T)|| <= 0.1 km, ||v(T)|| <= 0.0001 km/s, NERM 3D
"""

import numpy as np
import casadi as ca
from pathlib import Path

MU = 3.986e5
A_C = 15000.0
E_C = 0.5


def _orbital_params(nu):
    """Keplerian 参数 (CasADi 符号表达式)"""
    r_c = A_C * (1 - E_C**2) / (1 + E_C * ca.cos(nu))
    nu_dot = ca.sqrt(MU * A_C * (1 - E_C**2)) / r_c**2
    r_c_dot = ca.sqrt(MU / (A_C * (1 - E_C**2))) * E_C * ca.sin(nu)
    nu_ddot = -2 * r_c_dot * nu_dot / r_c
    return r_c, nu_dot, nu_ddot


def _dynamics_rhs(x, u):
    """NERM 3D 相对运动 RHS (CasADi 符号)

    x = [dx, dy, dz, dvx, dvy, dvz, nu]  (7D)
    u = [ux, uy, uz]                       (3D)
    """
    dx, dy, dz, dvx, dvy, dvz, nu = (
        x[0], x[1], x[2], x[3], x[4], x[5], x[6],
    )

    r_c, nu_dot, nu_ddot = _orbital_params(nu)
    r_p = ca.sqrt((r_c + dx)**2 + dy**2 + dz**2)

    grav_x = -MU * (r_c + dx) / r_p**3 + MU / r_c**2
    grav_y = -MU * dy / r_p**3
    grav_z = -MU * dz / r_p**3

    ddx = (2 * nu_dot * dvy + nu_ddot * dy + nu_dot**2 * dx
           + grav_x + u[0])
    ddy = (-2 * nu_dot * dvx - nu_ddot * dx + nu_dot**2 * dy
           + grav_y + u[1])
    ddz = grav_z + u[2]

    return ca.vertcat(dvx, dvy, dvz, ddx, ddy, ddz, nu_dot)


def solve_optimal_control(
    x0: np.ndarray,
    gamma: float = 1e7,
    N: int = 80,
    T_guess: float = None,
    r_term_tol: float = 0.1,
    v_term_tol: float = 1e-4,
    opts: dict = None,
    verbose: int = 1,
) -> dict:
    """CasADi + IPOPT 梯形直接配点法求解 3D 最优控制（增强版）

    Parameters
    ----------
    x0 : (7,)
        初始状态 [dx, dy, dz, dvx, dvy, dvz, nu]
    gamma : float
        时间-能量权衡系数（越大越偏向省时间）
    N : int
        配点区间数（增加到80以提高精度）
    T_guess : float | None
        初始时间猜测（自动根据初值估计）
    r_term_tol : float
        终端位置容差 (km)，默认0.1对应软捕获
    v_term_tol : float
        终端速度容差 (km/s)，默认1e-4对应软捕获
    opts : dict
        额外 IPOPT 选项
    verbose : int
        0=静默, 1=简要, 2=详细

    Returns
    -------
    dict: t_grid, x_traj, u_traj, lambda_traj, T_opt, converged, gamma, stats
    """
    opti = ca.Opti()

    # ── 决策变量: 归一化时间网格 ──
    X = opti.variable(7, N + 1)
    U = opti.variable(3, N)
    T_var = opti.variable()
    dtau = 1.0 / N

    # ── 改进的初始猜测 ──
    r0 = float(np.sqrt(x0[0]**2 + x0[1]**2 + x0[2]**2))
    v0 = float(np.sqrt(x0[3]**2 + x0[4]**2 + x0[5]**2))

    if T_guess is None:
        # 轨道周期
        T_orbit = 2 * np.pi * np.sqrt(A_C**3 / MU)

        # 基于距离和速度的估计
        if v0 > 1e-6:
            T_kinematic = r0 / v0  # 匀速运动时间
        else:
            T_kinematic = r0 * 100  # 如果初速度很小，用经验值

        # 取较小者，但不超过半个轨道周期
        T_guess = min(T_kinematic * 2.0, T_orbit * 0.5)
        T_guess = max(T_guess, 1000.0)  # 至少1000秒

    if verbose >= 1:
        print(f"  Initial guess: T={T_guess:.1f}s={T_guess/3600:.2f}h, "
              f"r0={r0:.1f}km, v0={v0:.5f}km/s")

    opti.set_initial(T_var, T_guess)

    # 改进状态初始猜测：线性插值到目标
    for i in range(7):
        if i < 3:  # 位置：线性收缩到0
            opti.set_initial(X[i, :], np.linspace(x0[i], 0.0, N + 1))
        elif i < 6:  # 速度：指数衰减到0
            decay = np.exp(-3 * np.linspace(0, 1, N + 1))
            opti.set_initial(X[i, :], x0[i] * decay)
        else:  # nu: 根据平均角速度演化
            nu_dot_avg = np.sqrt(MU * A_C * (1 - E_C**2)) / A_C**2
            opti.set_initial(X[i, :], x0[6] + nu_dot_avg * T_guess * np.linspace(0, 1, N + 1))

    # 控制初始猜测：初期大，后期小
    u_profile = np.outer(np.ones(3), np.exp(-2 * np.linspace(0, 1, N)))
    opti.set_initial(U, u_profile * 1e-4)

    # ── 目标: T + gamma * T * Σ ||u_k||² * dτ ──
    J = T_var
    for k in range(N):
        J += gamma * T_var * ca.sumsqr(U[:, k]) * dtau
    opti.minimize(J)

    # ── 梯形配点约束 ──
    for k in range(N):
        xk = X[:, k]
        xk1 = X[:, k + 1]
        uk = U[:, k]
        fk = _dynamics_rhs(xk, uk)
        fk1 = _dynamics_rhs(xk1, uk)
        dt_half = 0.5 * T_var * dtau
        opti.subject_to(xk1 == xk + dt_half * (fk + fk1))

    # ── 边界约束 ──
    opti.subject_to(X[:, 0] == x0)

    # 终端约束：放松到软捕获标准
    opti.subject_to(X[0, -1]**2 + X[1, -1]**2 + X[2, -1]**2 <= r_term_tol**2)
    opti.subject_to(X[3, -1]**2 + X[4, -1]**2 + X[5, -1]**2 <= v_term_tol**2)

    # ── 变量边界 ──
    opti.subject_to(T_var >= 100)
    opti.subject_to(T_var <= 300000)  # 最多约83小时

    # 状态边界：宽松，仅防止数值爆炸
    for k in range(N + 1):
        opti.subject_to(opti.bounded(-3000, X[0, k], 3000))
        opti.subject_to(opti.bounded(-3000, X[1, k], 3000))
        opti.subject_to(opti.bounded(-3000, X[2, k], 3000))
        opti.subject_to(opti.bounded(-1.0, X[3, k], 1.0))
        opti.subject_to(opti.bounded(-1.0, X[4, k], 1.0))
        opti.subject_to(opti.bounded(-1.0, X[5, k], 1.0))

    # ── IPOPT 选项 ──
    ipopt_opts = {
        "print_level": 5 if verbose >= 2 else 0,
        "tol": 1e-5,
        "acceptable_tol": 1e-3,
        "acceptable_iter": 15,
        "max_iter": 3000,
        "linear_solver": "mumps",
        "hessian_approximation": "limited-memory",
        "nlp_scaling_method": "gradient-based",
        "mu_strategy": "adaptive",
        "warm_start_init_point": "yes",
    }
    if opts:
        ipopt_opts.update(opts)

    opti.solver("ipopt", {"print_time": False}, ipopt_opts)

    # ── 求解 ──
    try:
        sol = opti.solve()
        success = True
    except RuntimeError as e:
        if verbose >= 1:
            print(f"  IPOPT failed: {e}")
        try:
            sol = opti.debug
            success = False
        except Exception:
            return dict(converged=False, message=str(e))

    # ── 提取 ──
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    T_opt = float(sol.value(T_var))

    t_grid = np.linspace(0, T_opt, N + 1)
    U_traj = np.column_stack([
        U_opt[:, k] if k < N else U_opt[:, -1] for k in range(N + 1)
    ])

    # ── 协态恢复 (PMP: λ_vel = -2γ u*) ──
    lambda_traj = np.zeros((6, N + 1))
    for k in range(N + 1):
        idx = min(k, N - 1)
        lambda_traj[3:6, k] = -2.0 * gamma * U_opt[:, idx]

    # 后向积分位置协态（简化版，只用速度协态）
    lambda_traj[:3, :] = 0.0  # 占位

    stats = sol.stats() if hasattr(sol, 'stats') else {}
    converged = success and stats.get("success", False)

    term_r_err = float(np.sqrt(X_opt[0, -1]**2 + X_opt[1, -1]**2 + X_opt[2, -1]**2))
    term_v_err = float(np.sqrt(X_opt[3, -1]**2 + X_opt[4, -1]**2 + X_opt[5, -1]**2))

    if verbose >= 1:
        print(f"  converged={converged}, T={T_opt:.0f}s={T_opt/3600:.2f}h")
        print(f"  term_r_err={term_r_err:.4f}km, term_v_err={term_v_err:.6f}km/s")

    if converged:
        E_ctrl = float(np.trapezoid(np.sum(U_traj**2, axis=0), t_grid))
        peak_u = float(np.max(np.linalg.norm(U_traj, axis=0)))
        if verbose >= 1:
            print(f"  E_ctrl={E_ctrl:.4e}, peak_u={peak_u:.4e} km/s²")
            print(f"  iter={stats.get('iter_count', '?')}")

    return dict(
        t_grid=t_grid,
        x_traj=X_opt,
        u_traj=U_traj,
        lambda_traj=lambda_traj,
        T_opt=T_opt,
        converged=converged,
        gamma=gamma,
        stats=stats,
        term_r_err=term_r_err,
        term_v_err=term_v_err,
    )


def solve_with_gamma_sweep(
    x0: np.ndarray,
    gamma_range: list = None,
    **kwargs
) -> dict:
    """自动扫描gamma，返回第一个收敛解

    Parameters
    ----------
    x0 : (7,)
        初始状态
    gamma_range : list
        gamma候选值，默认 [3e6, 1e7, 3e7, 1e8]
    **kwargs
        传递给 solve_optimal_control

    Returns
    -------
    dict: 第一个收敛解，或最后一次尝试的结果
    """
    if gamma_range is None:
        gamma_range = [3e6, 1e7, 3e7, 1e8]

    verbose = kwargs.get("verbose", 1)

    for gamma in gamma_range:
        if verbose >= 1:
            print(f"\n{'─'*60}")
            print(f"Trying gamma = {gamma:.1e}")
            print(f"{'─'*60}")

        result = solve_optimal_control(x0, gamma=gamma, **kwargs)

        if result.get("converged", False):
            if verbose >= 1:
                print(f"✓ Converged with gamma={gamma:.1e}")
            return result

    if verbose >= 1:
        print(f"✗ No gamma converged, returning last attempt")

    return result


if __name__ == "__main__":
    # 测试用例：d600_diagonal_closing
    X_p0 = np.array([600.0, 0.0, 0.0, -0.01, 0.0, 0.0])
    X_e0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nu0 = 0.0
    x_rel = X_p0 - X_e0
    x0 = np.array([x_rel[0], x_rel[1], x_rel[2],
                   x_rel[3], x_rel[4], x_rel[5], nu0])

    print("="*60)
    print("Testing TPBVP solver (robust version)")
    print("="*60)
    print(f"Initial state: r={np.linalg.norm(x0[:3]):.1f}km, "
          f"v={np.linalg.norm(x0[3:6]):.5f}km/s")

    result = solve_with_gamma_sweep(x0, verbose=2)

    if result.get("converged", False):
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"T_opt = {result['T_opt']/3600:.2f} h")
        print(f"Terminal errors: r={result['term_r_err']:.4f}km, "
              f"v={result['term_v_err']:.6f}km/s")
    else:
        print(f"\n{'='*60}")
        print("FAILED TO CONVERGE")
        print(f"{'='*60}")
