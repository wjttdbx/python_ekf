"""
Run SDRE sensitivity tests varying R and U_MAX and simulation duration.
Generates a summary printed to stdout and saved to scripts/sdre_sensitivity_results.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController

OUT = Path("scripts") / "sdre_sensitivity_results.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

X0 = np.array([500.0, 500.0, 200.0, 0.05, -0.02, 0.01])
N_base = 120


def run_sdre(orb, x0, R_val, U_max, T_mult=1, N_mult=1):
    controller = SDREGameController(Q=None, R=np.eye(3) * R_val)
    T = orb.T_orbit * T_mult
    N = int(N_base * N_mult * T_mult)
    dt = T / N

    state = np.zeros(13)
    state[0:6] = x0
    state[6:12] = 0.0
    state[12] = 0.0

    t_cur = 0.0
    captured = False
    for i in range(N):
        X_p = state[0:6]
        X_e = state[6:12]
        x_rel = X_p - X_e
        if np.linalg.norm(x_rel[:3]) < 1.0:
            captured = True
            break
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(state[12])
        A_sdc = orb.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)
        u_p, _ = controller.compute_control(A_sdc, x_rel, t_cur)
        u_norm = np.linalg.norm(u_p)
        if u_norm > U_max:
            u_p = u_p / u_norm * U_max
        u_e = np.zeros(3)

        sol = solve_ivp(lambda t,s: orb.dynamics_13d(t, s, u_p, u_e), [t_cur, t_cur+dt], state,
                        method='RK45', rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]
        t_cur += dt

    return dict(captured=captured, t_end=t_cur, final_dist=np.linalg.norm(state[0:3]-state[6:9]))


def main():
    orb = OrbitalDynamics()
    R_list = [1e4, 1e2, 1e6]
    U_list = [1e-4, 1e-3, 1e-2]
    T_mult_list = [1, 3]

    lines = []
    header = f"SDRE sensitivity test - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(header)
    lines.append(header)
    for Rv in R_list:
        for Uv in U_list:
            for Tm in T_mult_list:
                res = run_sdre(orb, X0, Rv, Uv, T_mult=Tm)
                s = f"R={Rv:.1e}, U_max={Uv:.1e} km/s^2, T_mult={Tm} -> captured={res['captured']}, t_end={res['t_end']:.1f}s, final_dist={res['final_dist']:.3f} km"
                print(s)
                lines.append(s)

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines))
    print('\nResults saved to', OUT)


if __name__ == '__main__':
    main()
