"""
CasADi + IPOPT 直接配点法 TPBVP 求解器

用 Hermite-Simpson 直接配点法将 NERM 追逃博弈的两点边值问题（TPBVP）
转化为 NLP，通过 CasADi 符号框架 + IPOPT 求解。

状态：x_aug = [x, y, z, vx, vy, vz, nu]（7D，相对状态 + 真近点角）
控制：u = [ux, uy, uz]（3D，追踪星推力加速度，km/s²）
代价：J = x_f^T S x_f + ∫(x_rel^T Q x_rel + u^T R u) dt
"""

from __future__ import annotations

import time
import zhplot
from dataclasses import dataclass, field

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class TPBVPResult:
    t: np.ndarray          # (N+1,) 时间节点（秒）
    x_traj: np.ndarray     # (7, N+1) 状态轨迹（含 nu）
    u_traj: np.ndarray     # (3, N+1) 控制序列
    J: float               # 最优代价
    solved: bool           # 是否收敛
    solver_stats: dict     # IPOPT 统计信息
    cpu_time: float        # 求解耗时（秒）


class TPBVPCollocationSolver:
    """Hermite-Simpson 直接配点法 TPBVP 求解器。

    Parameters
    ----------
    dynamics : OrbitalDynamics
        NERM 动力学对象，提供 mu, a_c, e_c 参数
    Q : (6,6) ndarray
        状态代价矩阵（作用于 6D 相对状态）
    R : (3,3) ndarray
        控制代价矩阵
    S : (6,6) ndarray
        终端代价矩阵
    N : int
        配点段数（时间节点数为 N+1）
    T : float or None
        仿真时长（秒）；None 时使用 1 个轨道周期
    u_max : float or None
        推力上限（km/s²）；None 时不施加推力约束
    nu0 : float
        初始真近点角（rad）
    ipopt_opts : dict or None
        额外的 IPOPT 求解器选项
    """

    def __init__(
        self,
        dynamics,
        Q: np.ndarray,
        R: np.ndarray,
        S: np.ndarray,
        N: int = 100,
        T: float | None = None,
        u_max: float | None = None,
        nu0: float = 0.0,
        ipopt_opts: dict | None = None,
    ):
        self.dyn = dynamics
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.S = np.asarray(S, dtype=float)
        self.N = N
        self.T = T if T is not None else dynamics.T_orbit
        self.u_max = u_max
        self.nu0 = nu0
        self.h = self.T / N  # 每段步长

        # IPOPT 默认选项
        self._ipopt_opts: dict = {
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.print_level": 3,
            "ipopt.linear_solver": "mumps",
            "print_time": 0,
        }
        if ipopt_opts:
            self._ipopt_opts.update(ipopt_opts)

        self._f_cas = self._build_casadi_dynamics()

    # ------------------------------------------------------------------
    # 符号动力学
    # ------------------------------------------------------------------

    def _build_casadi_dynamics(self) -> ca.Function:
        """构建 CasADi 符号 ODE：f(x_aug, u) -> dx_aug/dt。"""
        mu = self.dyn.mu
        a_c = self.dyn.a_c
        e_c = self.dyn.e_c
        p = a_c * (1.0 - e_c**2)          # 半通径

        x_aug = ca.MX.sym("x_aug", 7)     # [x,y,z,vx,vy,vz,nu]
        u = ca.MX.sym("u", 3)             # [ux,uy,uz]

        x, y, z = x_aug[0], x_aug[1], x_aug[2]
        vx, vy, vz = x_aug[3], x_aug[4], x_aug[5]
        nu = x_aug[6]

        # 参考轨道参数（与 nerm.py get_orbital_params 完全一致）
        r_c = p / (1.0 + e_c * ca.cos(nu))
        h_orb = ca.sqrt(mu * p)
        nu_dot = h_orb / r_c**2
        r_c_dot = ca.sqrt(mu / p) * e_c * ca.sin(nu)
        nu_ddot = -2.0 * r_c_dot * nu_dot / r_c

        # 追踪星地心距离（追踪星在 LVLH 中坐标为 (x,y,z)）
        r_p = ca.sqrt((r_c + x)**2 + y**2 + z**2)

        # 加速度（与 nerm.py dynamics_13d 第3行一致，注意 -nu_ddot*x）
        ax = (2.0 * nu_dot * vy + nu_ddot * y + nu_dot**2 * x
              - mu * (r_c + x) / r_p**3 + mu / r_c**2 + u[0])
        ay = (-2.0 * nu_dot * vx - nu_ddot * x + nu_dot**2 * y
              - mu * y / r_p**3 + u[1])
        az = -mu * z / r_p**3 + u[2]

        dx_aug = ca.vertcat(vx, vy, vz, ax, ay, az, nu_dot)

        return ca.Function("f_nerm", [x_aug, u], [dx_aug],
                           ["x_aug", "u"], ["dx_aug"])

    # ------------------------------------------------------------------
    # 热启动：无控制轨迹前向积分
    # ------------------------------------------------------------------

    def _warm_start(self, x0: np.ndarray, x_f: np.ndarray | None) -> np.ndarray:
        """生成初始猜测。

        Returns
        -------
        w0 : (10*(N+1),) ndarray
            决策变量初始猜测，布局 [x0,u0, x1,u1, ..., xN,uN]
        """
        N, h, nu0 = self.N, self.h, self.nu0
        t_span = (0.0, self.T)
        t_eval = np.linspace(0.0, self.T, N + 1)

        # 无控制前向积分，仅用于获取 nu(t) 轨迹
        def ode_free(t, s):
            nu = s[6]
            r_c, nu_dot, _ = self.dyn.get_orbital_params(nu)
            # 追踪星无控制（u=0）
            dx = np.zeros(7)
            dx[0:3] = s[3:6]
            xp, yp, zp = s[0], s[1], s[2]
            vxp, vyp, vzp = s[3], s[4], s[5]
            r_p = np.sqrt((r_c + xp)**2 + yp**2 + zp**2)
            dx[3] = (2*nu_dot*vyp + 0*yp + nu_dot**2*xp
                     - self.dyn.mu*(r_c+xp)/r_p**3 + self.dyn.mu/r_c**2)
            dx[4] = (-2*nu_dot*vxp + nu_dot**2*yp
                     - self.dyn.mu*yp/r_p**3)
            dx[5] = -self.dyn.mu*zp/r_p**3
            dx[6] = nu_dot
            return dx

        s0_aug = np.append(x0, nu0)
        sol = solve_ivp(ode_free, t_span, s0_aug, t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-9)

        x_guess = np.zeros((7, N + 1))
        x_guess[6, :] = sol.y[6, :]      # nu(t) 来自无控制积分

        # 位置/速度线性插值 x0 -> x_f（或 0）
        target = x_f if x_f is not None else np.zeros(6)
        for i in range(N + 1):
            alpha = i / N
            x_guess[:6, i] = (1 - alpha) * x0 + alpha * target

        # 拼装决策变量 [xi(7), ui(3)] * (N+1)
        w0 = np.zeros(10 * (N + 1))
        for i in range(N + 1):
            w0[10*i:10*i+7] = x_guess[:, i]
            # 控制猜测：零
            w0[10*i+7:10*i+10] = 0.0

        return w0

    # ------------------------------------------------------------------
    # NLP 组装
    # ------------------------------------------------------------------

    def _build_nlp(self, x0: np.ndarray, x_f: np.ndarray | None):
        """组装 Hermite-Simpson NLP。

        Returns
        -------
        nlp : dict
            CasADi NLP 字典 {"x": w, "f": J, "g": g}
        lbw, ubw : list
            决策变量上下界
        lbg, ubg : list
            约束上下界
        """
        N, h = self.N, self.h
        f = self._f_cas
        Q, R, S = self.Q, self.R, self.S

        # 决策变量列表（交错布局）
        w = []       # 符号变量
        lbw, ubw = [], []   # 变量界
        g = []       # 约束
        lbg, ubg = [], []

        J = ca.MX(0)         # 目标函数

        # 边界（无穷大）
        x_lb = [-ca.inf] * 7
        x_ub = [ca.inf] * 7
        u_lb = [-self.u_max]*3 if self.u_max else [-ca.inf]*3
        u_ub = [self.u_max]*3  if self.u_max else [ca.inf]*3

        # 存储各节点状态/控制符号，用于 Hermite-Simpson
        X_nodes = []
        U_nodes = []

        for k in range(N + 1):
            xk = ca.MX.sym(f"x_{k}", 7)
            uk = ca.MX.sym(f"u_{k}", 3)
            w += [xk, uk]
            lbw += x_lb + u_lb
            ubw += x_ub + u_ub
            X_nodes.append(xk)
            U_nodes.append(uk)

            # 初始条件约束
            if k == 0:
                g.append(xk[:6] - x0)           # 位置/速度固定
                lbg += [0.0] * 6
                ubg += [0.0] * 6
                g.append(xk[6] - self.nu0)       # nu0 固定
                lbg += [0.0]
                ubg += [0.0]

            # 积分代价（Simpson 积分在段末处理，此处仅记录节点）
            # 终端代价在循环后处理

        # Hermite-Simpson 连续性约束 + 积分代价
        for k in range(N):
            xk  = X_nodes[k]
            xk1 = X_nodes[k + 1]
            uk  = U_nodes[k]
            uk1 = U_nodes[k + 1]

            fk  = f(xk,  uk)
            fk1 = f(xk1, uk1)

            # 中点插值
            xm = 0.5 * (xk + xk1) + (h / 8.0) * (fk - fk1)
            um = 0.5 * (uk + uk1)
            fm = f(xm, um)

            # 连续性约束（7 个方程）
            defect = xk1 - xk - (h / 6.0) * (fk + 4.0 * fm + fk1)
            g.append(defect)
            lbg += [0.0] * 7
            ubg += [0.0] * 7

            # Simpson 积分代价（仅 6D 状态参与 Q，u 参与 R）
            x_rel_k  = xk[:6]
            x_rel_m  = xm[:6]
            x_rel_k1 = xk1[:6]

            Lk  = x_rel_k.T  @ Q @ x_rel_k  + uk.T  @ R @ uk
            Lm  = x_rel_m.T  @ Q @ x_rel_m  + um.T  @ R @ um
            Lk1 = x_rel_k1.T @ Q @ x_rel_k1 + uk1.T @ R @ uk1

            J += (h / 6.0) * (Lk + 4.0 * Lm + Lk1)

        # 终端代价
        xN = X_nodes[N]
        J += xN[:6].T @ S @ xN[:6]

        # 终端约束（位置硬约束，速度软化进 S）
        if x_f is not None:
            g.append(xN[:3] - x_f[:3])     # 终端位置
            lbg += [0.0] * 3
            ubg += [0.0] * 3

        # 拼装 w
        w_cat = ca.vertcat(*w)

        return (
            {"x": w_cat, "f": J, "g": ca.vertcat(*g)},
            lbw, ubw, lbg, ubg,
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def solve(self, x0: np.ndarray, x_f: np.ndarray | None = None) -> TPBVPResult:
        """求解 TPBVP。

        Parameters
        ----------
        x0 : (6,) ndarray
            初始 6D 相对状态 [x,y,z,vx,vy,vz]（km, km/s）
        x_f : (6,) ndarray or None
            终端目标状态；None 时只用终端代价 S 软约束

        Returns
        -------
        TPBVPResult
        """
        x0 = np.asarray(x0, dtype=float)
        if x_f is not None:
            x_f = np.asarray(x_f, dtype=float)

        # 组装 NLP
        nlp, lbw, ubw, lbg, ubg = self._build_nlp(x0, x_f)

        # 初始猜测
        w0 = self._warm_start(x0, x_f)

        # 构建 IPOPT 求解器
        solver = ca.nlpsol("tpbvp_solver", "ipopt", nlp, self._ipopt_opts)

        t0 = time.perf_counter()
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        cpu_time = time.perf_counter() - t0

        # 判断收敛
        stats = solver.stats()
        solved = stats.get("success", False)

        # 提取结果
        w_opt = np.array(sol["x"]).flatten()
        N = self.N
        t_nodes = np.linspace(0.0, self.T, N + 1)
        x_traj = np.zeros((7, N + 1))
        u_traj = np.zeros((3, N + 1))

        for k in range(N + 1):
            x_traj[:, k] = w_opt[10*k : 10*k+7]
            u_traj[:, k] = w_opt[10*k+7 : 10*k+10]

        J_opt = float(sol["f"])

        return TPBVPResult(
            t=t_nodes,
            x_traj=x_traj,
            u_traj=u_traj,
            J=J_opt,
            solved=solved,
            solver_stats=stats,
            cpu_time=cpu_time,
        )
