"""Paired J2 robustness pilot for the current main.py configuration.

Run: uv run python -m experiments.j2_capture_validation
J2 acts on truth only; the reference frame remains the nominal Kepler frame.
RAAN=argument of periapsis=0. Both spacecraft receive their own J2 acceleration;
do not subtract reference J2, since the reference itself is unperturbed.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.integrate import solve_ivp

from main import create_ekf
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation

J2 = 1.08262668e-3
RE = 6378.1363


class J2Truth(OrbitalDynamics):
    def __init__(self, inclination_deg=45.0, scale=1.0):
        super().__init__()
        self.inc = np.deg2rad(inclination_deg)
        self.scale = scale

    def rotation(self, nu):
        c, s = np.cos(nu), np.sin(nu)
        ci, si = np.cos(self.inc), np.sin(self.inc)
        return np.array([[c, -s, 0], [ci*s, ci*c, -si], [si*s, si*c, ci]])

    def acceleration(self, r):
        radius = np.linalg.norm(r)
        z2 = (r[2] / radius)**2
        return (1.5*self.scale*J2*self.mu*RE**2/radius**5
                * r * np.array([5*z2-1, 5*z2-1, 5*z2-3]))

    def dynamics_13d(self, t, state, u_p, u_e):
        derivative = super().dynamics_13d(t, state, u_p, u_e)
        if self.scale:
            rc = self.get_orbital_params(state[12])[0]
            rot = self.rotation(state[12])
            for start in (0, 6):
                r = rot @ (state[start:start+3] + np.array([rc, 0., 0.]))
                derivative[start+3:start+6] += rot.T @ self.acceleration(r)
        return derivative


class CaptureSimulation(EKFSDRESimulation):
    """Record continuous entry without changing the engine's sampled stop rule."""
    entry = None

    def _propagate_true_state(self, state, u_p, u_e, t, dt):
        def entry_event(t, state, *args):
            return np.linalg.norm(state[:3]-state[6:9])-self.capture_dist
        entry_event.direction = -1
        sol = solve_ivp(self.dynamics.dynamics_13d, (t, t+dt), state,
                        args=(u_p, u_e), rtol=1e-8, atol=1e-10, events=entry_event)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError(sol.message)
        if self.entry is None and len(sol.t_events[0]):
            st = sol.y_events[0][0]
            self.entry = dict(time_s=float(sol.t_events[0][0]),
                              distance_m=float(1000*np.linalg.norm(st[:3]-st[6:9])),
                              speed_m_s=float(1000*np.linalg.norm(st[3:6]-st[9:12])))
        return sol.y[:, -1]


def validate_physics():
    orb = J2Truth()
    r = np.array([7300., 1400., 2200.])
    def potential(r):
        radius = np.linalg.norm(r)
        return orb.mu*J2*RE**2/(2*radius**3)*(3*(r[2]/radius)**2-1)
    h = 0.01
    fd = np.array([-(potential(r+h*e)-potential(r-h*e))/(2*h) for e in np.eye(3)])
    np.testing.assert_allclose(orb.acceleration(r), fd, rtol=1e-8, atol=1e-14)
    st = np.r_[500., 500., 500., .01, .01, .01, np.zeros(6), 0.]
    zero = np.zeros(3)
    np.testing.assert_array_equal(J2Truth(scale=0).dynamics_13d(0, st, zero, zero),
                                  OrbitalDynamics().dynamics_13d(0, st, zero, zero))
    # Independent inertial propagation checks the rotating-frame force injection.
    def to_eci(st, start):
        rc, w, _ = orb.get_orbital_params(st[12])
        rot = orb.rotation(st[12])
        rdot = np.sqrt(orb.mu/(orb.a_c*(1-orb.e_c**2)))*orb.e_c*np.sin(st[12])
        pos = st[start:start+3]+np.array([rc, 0., 0.])
        vel = st[start+3:start+6]+np.array([rdot, 0., 0.])+np.cross([0., 0., w], pos)
        return np.r_[rot@pos, rot@vel]
    def inertial(t, x):
        return np.r_[x[3:], -orb.mu*x[:3]/np.linalg.norm(x[:3])**3+orb.acceleration(x[:3])]
    rel = solve_ivp(orb.dynamics_13d, (0, 1000), st, args=(zero, zero), rtol=1e-11, atol=1e-12).y[:, -1]
    errors = []
    for start in (0, 6):
        truth = solve_ivp(inertial, (0, 1000), to_eci(st, start), rtol=1e-12, atol=1e-13).y[:, -1]
        errors.append(float(np.linalg.norm(to_eci(rel, start)[:3]-truth[:3])*1000))
    assert max(errors) < 0.01, errors
    return dict(potential_gradient_check="PASS", zero_j2_check="PASS", eci_position_errors_m=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-m', type=float, default=100.)
    parser.add_argument('--output', default='outputs/j2_capture_validation')
    args = parser.parse_args()
    if not np.isfinite(args.capture_m) or args.capture_m <= 0:
        parser.error('--capture-m must be finite and positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(seeds=[42, 43, 44], inclinations_deg=[0, 45, 90],
                  dt_s=10, horizon_orbits=10, capture_km=args.capture_m/1000, truth_only_j2=True,
                  passive_evader=[True, False], J2=J2, earth_radius_km=RE,
                  a_km=15000, e=.5, raan_deg=0, argument_periapsis_deg=0,
                  controller_Q="I6", controller_R="1e13 I3", gamma=float(np.sqrt(2)),
                  navigation="main.create_ekf noisy=True; Euler SDC EKF; exact initial estimate",
                  initial_state=[500,500,500,.01,.01,.01,0,0,0,0,0,0,0])
    (out/'manifest.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    checks = validate_physics()
    print('PHYSICS', checks, flush=True)
    rows = []
    for passive in config['passive_evader']:
        for seed in config['seeds']:
            for inc in [None]+config['inclinations_deg']:
                label = f"{'passive' if passive else 'active'}_seed{seed}_" + ('nominal' if inc is None else f'j2_i{inc}')
                print('START', label, flush=True)
                orb = J2Truth(inclination_deg=inc or 0, scale=0 if inc is None else 1)
                xp = np.array([500.,500.,500.,.01,.01,.01])
                ctrl = SDREGameController(Q=np.eye(6), R=1e13*np.eye(3), gamma=np.sqrt(2))
                sim = CaptureSimulation(dynamics=orb, controller=ctrl,
                    ekf=create_ekf(True, xp.copy(), float(np.linalg.norm(xp[:3]))),
                    X_p0=xp, X_e0=np.zeros(6), rng=np.random.default_rng(seed),
                    passive_evader=passive, early_stop=True, capture_source='true', dt=10,
                    capture_dist=args.capture_m/1000)
                start = time.perf_counter()
                result = sim.run(10*orb.T_orbit)
                assert np.isfinite(result.states).all() and np.isfinite(result.x_est_history).all()
                row = dict(label=label, seed=seed, passive=passive, inclination_deg=inc,
                    captured=sim.entry is not None, entry=sim.entry,
                    sampled_capture=bool(result.captured), final_time_s=float(result.t[-1]),
                    min_sampled_distance_m=float(1000*result.dist_history.min()),
                    final_estimation_error_m=float(1000*np.linalg.norm(result.ekf_err_history[:3,-1])),
                    peak_pursuer_accel_m_s2=float(1000*np.linalg.norm(result.u_p_history,axis=0).max()),
                    are_fallbacks=ctrl._are_fallback_count, wall_s=time.perf_counter()-start)
                np.savez_compressed(out/f'{label}.npz', t=result.t, states=result.states,
                                    estimates=result.x_est_history, up=result.u_p_history, ue=result.u_e_history)
                rows.append(row)
                (out/'summary.json').write_text(json.dumps(dict(physics_checks=checks, runs=rows), indent=2), encoding='utf-8')
                print('RESULT', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
