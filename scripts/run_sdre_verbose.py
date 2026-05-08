"""
Short SDRE run with verbose controller to print P and control diagnostics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.integrate import solve_ivp
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController

X0 = np.array([500.0, 500.0, 200.0, 0.05, -0.02, 0.01])

orb = OrbitalDynamics()
controller = SDREGameController(Q=np.diag([1.0]*6), R=np.eye(3)*1e4, verbose=True, verbose_interval=1)

T = orb.T_orbit
N = 10
dt = T / N
state = np.zeros(13)
state[0:6] = X0
state[6:12] = 0.0
state[12] = 0.0

a = 0
for k in range(N):
    X_p = state[0:6]
    X_e = state[6:12]
    x_rel = X_p - X_e
    r_c, nu_dot, nu_ddot = orb.get_orbital_params(state[12])
    A_sdc = orb.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)
    u_p, u_e = controller.compute_control(A_sdc, x_rel, t=k*dt, solve_are=True)
    print(f"step {k}: u_p norm={np.linalg.norm(u_p):.6e} km/s^2, u_e norm={np.linalg.norm(u_e):.6e}")
    sol = solve_ivp(lambda t,s: orb.dynamics_13d(t,s,u_p,np.zeros(3)), [0, dt], state, method='RK45')
    state = sol.y[:, -1]
    print(f"  new dist = {np.linalg.norm(state[0:3]-state[6:9]):.6f} km")

print('Done')
