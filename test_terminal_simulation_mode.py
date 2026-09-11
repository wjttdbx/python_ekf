"""Integration tests for terminal-study simulation behavior."""

import numpy as np

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.estimation.navigators import JacobianEKFNavigator
from aerospace.simulation.metrics import compute_metrics
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation


class _ControllerWithEvaderCommand:
    """Returns a nonzero evader command so passive mode must override it."""

    def compute_control(self, A_SDC, x_rel, **kwargs):
        return np.zeros(3), np.array([1.0, -2.0, 3.0])


class _FixedNavigator:
    """Keeps a prescribed estimate so the stopping source is observable."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        self.P = np.eye(6)
        self.R = np.eye(2)
        self.angles_only = True

    def predict(self, *args, **kwargs):
        return self.x.copy(), self.P.copy()

    def update(self, x_priori, P_priori, z_meas):
        return np.zeros(2)


def test_estimated_capture_source_stops_on_estimate_not_true_distance():
    dynamics = OrbitalDynamics()
    x_p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    navigator = _FixedNavigator(
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    simulation = EKFSDRESimulation(
        dynamics=dynamics,
        controller=_ControllerWithEvaderCommand(),
        navigator=navigator,
        X_p0=x_p0,
        X_e0=np.zeros(6),
        dt=1.0,
        capture_dist=0.75,
        rng=np.random.default_rng(0),
        passive_evader=True,
        early_stop=True,
        capture_source="estimated",
    )

    result = simulation.run(t_end=2.0)

    assert result.captured is True
    assert result.t.tolist() == [0.0, 1.0]
    assert np.linalg.norm(result.x_est_history[:3, -1]) < 0.75
    assert result.dist_history[-1] > 0.75


def test_passive_terminal_study_runs_full_horizon_after_initial_entry():
    dynamics = OrbitalDynamics()
    x_p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_e0 = np.zeros(6)
    ekf = RelativeStateEKF(
        x0=x_p0 - x_e0,
        P0=np.eye(6),
        Q=np.zeros((6, 6)),
        R=np.eye(2),
        angles_only=True,
    )
    simulation = EKFSDRESimulation(
        dynamics=dynamics,
        controller=_ControllerWithEvaderCommand(),
        ekf=ekf,
        X_p0=x_p0,
        X_e0=x_e0,
        dt=1.0,
        capture_dist=2.0,
        passive_evader=True,
        early_stop=False,
    )

    result = simulation.run(t_end=2.0)

    assert result.t.tolist() == [0.0, 1.0, 2.0]
    assert np.allclose(result.u_e_history, 0.0)
    assert result.captured is False
    metrics = compute_metrics(
        result, dt=1.0, capture_dist=2.0,
        v_thresh=0.1, sustain_dur=0.0,
    )
    assert metrics.T_FP == 0.0


def test_flow_prediction_map_matches_one_step_truth_propagation():
    dynamics = OrbitalDynamics()
    x_p0 = np.array([1.0, 0.2, -0.1, 0.0, 0.001, -0.0002])
    x_e0 = np.zeros(6)
    navigator = JacobianEKFNavigator(
        x0=x_p0-x_e0,
        P0=np.eye(6)*1e-4,
        Q=np.zeros((6, 6)),
        R=np.eye(2)*1e-8,
        angles_only=True,
        state_scales=np.array([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]),
    )
    simulation = EKFSDRESimulation(
        dynamics=dynamics,
        controller=_ControllerWithEvaderCommand(),
        navigator=navigator,
        X_p0=x_p0,
        X_e0=x_e0,
        dt=1.0,
        passive_evader=True,
        prediction_map="flow",
    )
    state = simulation.state0.copy()
    r_c, nu_dot, nu_ddot = simulation._get_orbital_context(state[12])
    context = simulation._build_navigator_ctx(
        state, r_c, nu_dot, nu_ddot, np.zeros(3), np.zeros(3),
    )
    predicted_relative = context["f_discrete"](x_p0-x_e0)
    propagated = simulation._propagate_true_state(
        state, np.zeros(3), np.zeros(3), 0.0, 1.0,
    )
    true_relative = propagated[:6]-propagated[6:12]

    np.testing.assert_allclose(predicted_relative, true_relative, atol=1e-10, rtol=0.0)
