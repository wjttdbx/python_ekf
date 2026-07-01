"""
多场景批量: 3D SDRE vs CasADi 最优控制对比

从 run_scenarios.py 导入所有场景, ECI→LVLH, 统一用 Q=dia(1,1,1,10,10,10), R=1e13·I₃.
"""
import numpy as np
import casadi as ca
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

OUT_DIR = Path("outputs/sdre_vs_optimal_3d")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 场景适配的参数化动力学 ──
def _orbital_params(nu, mu, a_c, e_c):
    r_c = a_c * (1 - e_c**2) / (1 + e_c * ca.cos(nu))
    nu_dot = ca.sqrt(mu * a_c * (1 - e_c**2)) / r_c**2
    r_c_dot = ca.sqrt(mu / (a_c * (1 - e_c**2))) * e_c * ca.sin(nu)
    nu_ddot = -2 * r_c_dot * nu_dot / r_c
    return r_c, nu_dot, nu_ddot


def _dynamics_rhs(x, u, mu, a_c, e_c):
    dx, dy, dz, dvx, dvy, dvz, nu = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    r_c, nu_dot, nu_ddot = _orbital_params(nu, mu, a_c, e_c)
    r_p = ca.sqrt((r_c + dx)**2 + dy**2 + dz**2)
    grav_x = -mu * (r_c + dx) / r_p**3 + mu / r_c**2
    grav_y = -mu * dy / r_p**3
    grav_z = -mu * dz / r_p**3
    ddx = 2*nu_dot*dvy + nu_ddot*dy + nu_dot**2*dx + grav_x + u[0]
    ddy = -2*nu_dot*dvx - nu_ddot*dx + nu_dot**2*dy + grav_y + u[1]
    ddz = grav_z + u[2]
    return ca.vertcat(dvx, dvy, dvz, ddx, ddy, ddz, nu_dot)


def run_sdre_3d(x_rel0, nu0, Q, R, mu, a_c, e_c, dt=20.0, t_max_h=24):
    """3D SDRE 闭环仿真, 参数化轨道."""
    T_end = t_max_h * 3600; N_steps = int(T_end / dt)
    X_p = np.array([*x_rel0[:3], *x_rel0[3:6]]); X_e = np.zeros(6)
    nu = nu0; t_now = 0.0
    traj_rel, u_hist = [x_rel0[:6].copy()], [np.zeros(3)]
    B = np.zeros((6, 3)); B[3:, :] = np.eye(3)

    class OrbWrapper:
        def __init__(s): s.mu, s.a_c, s.e_c = mu, a_c, e_c
        def get_orbital_params(s, n):
            rc = s.a_c*(1-s.e_c**2)/(1+s.e_c*np.cos(n))
            nd = np.sqrt(s.mu*s.a_c*(1-s.e_c**2))/rc**2
            rcd = np.sqrt(s.mu/(s.a_c*(1-s.e_c**2)))*s.e_c*np.sin(n)
            ndd = -2*rcd*nd/rc
            return rc, nd, ndd
        def get_SDC_matrix(s, X_p, X_e, rc, nd, ndd):
            xr = X_p - X_e; r = np.sqrt(xr[0]**2+xr[1]**2+xr[2]**2)+1e-12
            rp = np.sqrt((rc+X_p[0])**2+X_p[1]**2+X_p[2]**2)
            re = np.sqrt((rc+X_e[0])**2+X_e[1]**2+X_e[2]**2)
            bx = -s.mu*(rc+X_p[0])/rp**3 + s.mu*(rc+X_e[0])/re**3
            by = -s.mu*X_p[1]/rp**3 + s.mu*X_e[1]/re**3
            bz = -s.mu*X_p[2]/rp**3 + s.mu*X_e[2]/re**3
            A = np.zeros((6,6))
            A[0,3]=A[1,4]=A[2,5]=1
            A[3,0]=nd**2+bx*xr[0]/r**2; A[3,1]=ndd+bx*xr[1]/r**2
            A[3,2]=bx*xr[2]/r**2; A[3,4]=2*nd
            A[4,0]=-ndd+by*xr[0]/r**2; A[4,1]=nd**2+by*xr[1]/r**2
            A[4,2]=by*xr[2]/r**2; A[4,3]=-2*nd
            A[5,0]=bz*xr[0]/r**2; A[5,1]=bz*xr[1]/r**2; A[5,2]=bz*xr[2]/r**2
            return A
        def dynamics_13d(s, t, state, u_p, u_e):
            Xp=state[0:6]; Xe=state[6:12]; n=state[12]
            rc,nd,ndd=s.get_orbital_params(n)
            rp=np.sqrt((rc+Xp[0])**2+Xp[1]**2+Xp[2]**2)
            re=np.sqrt((rc+Xe[0])**2+Xe[1]**2+Xe[2]**2)
            d=np.zeros(13); d[0:3]=Xp[3:6]; d[6:9]=Xe[3:6]
            d[3]=2*nd*Xp[4]+ndd*Xp[1]+nd**2*Xp[0]-s.mu*(rc+Xp[0])/rp**3+s.mu/rc**2+u_p[0]
            d[4]=-2*nd*Xp[3]-ndd*Xp[0]+nd**2*Xp[1]-s.mu*Xp[1]/rp**3+u_p[1]
            d[5]=-s.mu*Xp[2]/rp**3+u_p[2]
            d[9]=2*nd*Xe[4]+ndd*Xe[1]+nd**2*Xe[0]-s.mu*(rc+Xe[0])/re**3+s.mu/rc**2+u_e[0]
            d[10]=-2*nd*Xe[3]-ndd*Xe[0]+nd**2*Xe[1]-s.mu*Xe[1]/re**3+u_e[1]
            d[11]=-s.mu*Xe[2]/re**3+u_e[2]; d[12]=nd
            return d

    orb = OrbWrapper()
    for k in range(N_steps):
        x_rel = X_p - X_e
        rc, nd, ndd = orb.get_orbital_params(nu)
        A = orb.get_SDC_matrix(X_p, X_e, rc, nd, ndd)
        try: P = solve_continuous_are(A, B, Q, R)
        except Exception: P = np.zeros((6,6))
        u_p = -np.linalg.inv(R) @ B.T @ P @ x_rel
        state = np.concatenate([X_p, X_e, [nu]])
        sol = solve_ivp(orb.dynamics_13d, [t_now, t_now+dt], state,
                        args=(u_p, np.zeros(3)), method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:,-1]; X_p=state[0:6]; X_e=state[6:12]; nu=state[12]; t_now+=dt
        traj_rel.append((X_p - X_e).copy()); u_hist.append(u_p.copy())
        if np.linalg.norm(X_p[:3]-X_e[:3]) < 0.1: break

    t_arr = np.arange(len(traj_rel))*dt; x_arr=np.array(traj_rel).T; u_arr=np.array(u_hist).T
    d_arr = np.linalg.norm(x_arr[:3], axis=0)
    ci = np.argmax(d_arr<0.1) if np.any(d_arr<0.1) else len(d_arr)-1
    return dict(t=t_arr, x=x_arr, u=u_arr, T_cap=t_arr[ci], cap_idx=ci)


def solve_casadi(x0, T_fixed, Q_diag, R_diag, mu, a_c, e_c, N=60,
                 warm_x=None, warm_u=None):
    opti = ca.Opti(); X=opti.variable(7,N+1); U=opti.variable(3,N); dt=1.0/N
    if warm_x is not None and warm_u is not None:
        tw=np.linspace(0,T_fixed,warm_x.shape[1]); tn=np.linspace(0,T_fixed,N+1)
        Xg=np.zeros((7,N+1))
        for i in range(7): Xg[i,:]=np.interp(tn,tw,warm_x[i])
        Ug=np.zeros((3,N)); tu=np.linspace(0,T_fixed,N)
        for i in range(3): Ug[i,:]=np.interp(tu,tw,warm_u[i])
    else:
        Xg=np.zeros((7,N+1))
        for i in range(7):
            yf=0.0 if i<6 else(x0[6]+T_fixed*1e-4)
            Xg[i,:]=np.linspace(x0[i],yf,N+1)
        Ug=np.zeros((3,N))
    opti.set_initial(X,Xg); opti.set_initial(U,Ug)
    Qm=np.diag(Q_diag); Rm=np.diag(R_diag); J=0
    for k in range(N):
        xk=X[:6,k]; uk=U[:,k]
        J+=(xk.T@Qm@xk+uk.T@Rm@uk)*T_fixed*dt
    opti.minimize(J)
    for k in range(N):
        xk=X[:,k]; xk1=X[:,k+1]; uk=U[:,k]
        fk=_dynamics_rhs(xk,uk,mu,a_c,e_c); fk1=_dynamics_rhs(xk1,uk,mu,a_c,e_c)
        opti.subject_to(xk1==xk+0.5*T_fixed*dt*(fk+fk1))
    opti.subject_to(X[:,0]==x0)
    opti.subject_to(X[0,-1]**2+X[1,-1]**2+X[2,-1]**2<=1e-4)
    # u_max 约束已移除，允许 CasADi 自由选择推力
    for k in range(N+1):
        opti.subject_to(opti.bounded(-5000,X[0,k],5000))
        opti.subject_to(opti.bounded(-5000,X[1,k],5000))
        opti.subject_to(opti.bounded(-5000,X[2,k],5000))
    opti.solver("ipopt",{},{"print_level":0,"tol":1e-6,"max_iter":5000,
                "linear_solver":"mumps","hessian_approximation":"limited-memory",
                "acceptable_tol":1e-3,"acceptable_iter":15})
    try:
        sol=opti.solve(); conv=True
    except RuntimeError as e:
        try: sol=opti.debug
        except: return dict(converged=False,message=str(e))
        conv=False
    Xo=sol.value(X); Uo=sol.value(U); tg=np.linspace(0,T_fixed,N+1)
    Ut=np.column_stack([Uo[:,k] if k<N else Uo[:,-1] for k in range(N+1)])
    te=float(np.sqrt(Xo[0,-1]**2+Xo[1,-1]**2+Xo[2,-1]**2))
    return dict(converged=conv or te<1e-2, t=tg, x=Xo, u=Ut, T=T_fixed, term_err=te)


def compute_cost(t, x, u, Q, R):
    J=0.0
    for i in range(1,len(t)):
        J+=(x[:6,i]@Q@x[:6,i]+u[:,i]@R@u[:,i])*(t[i]-t[i-1])
    return float(J)


if __name__ == "__main__":
    from run_scenarios import SCENARIOS, eci_to_lvlh_scenario, MU

    Q = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    R = np.eye(3) * 1e13

    print("=" * 80)
    print(f"{'Scenario':<28} {'T(h)':>7} {'J_sdre':>13} {'J_casadi':>13} {'gap':>8}")
    print("-" * 80)

    for key, cfg in SCENARIOS.items():
        name = cfg["name"]; orb = cfg["chief_orbit"]
        a_c, e_c = orb["a"], orb["e"]
        X_p0, X_e0, nu0 = eci_to_lvlh_scenario(cfg)
        x_rel0 = X_p0 - X_e0
        x0 = np.array([*x_rel0, nu0])

        # SDRE
        try:
            sdre = run_sdre_3d(x_rel0, nu0, Q, R, MU, a_c, e_c)
        except Exception as e:
            print(f"  {name:<26} SDRE FAIL: {e}")
            continue

        T_s = sdre["T_cap"]; J_s = compute_cost(sdre["t"], sdre["x"], sdre["u"], Q, R)

        # CasADi
        try:
            ns = sdre["x"].shape[1]
            nh = np.linspace(nu0, nu0+T_s*1e-4, ns)
            wx = np.vstack([sdre["x"][:6], nh])
            opt = solve_casadi(x0, T_s, np.diag(Q), np.diag(R), MU, a_c, e_c,
                               warm_x=wx, warm_u=sdre["u"])
        except Exception as e:
            print(f"  {name:<26} CasADi FAIL: {e}")
            continue

        if opt["converged"] or ('u' in opt and len(opt['u'])>0):
            J_o = compute_cost(opt["t"], opt["x"], opt["u"], Q, R)
            gap = (J_s/J_o-1)*100
        else:
            J_o, gap = float("nan"), float("nan")

        gs = f"{gap:.1f}%" if not np.isnan(gap) else "FAIL"
        print(f"  {name:<26} {T_s/3600:>7.2f} {J_s:>13.4e} {J_o:>13.4e} {gs:>8}")

    print("=" * 80)
