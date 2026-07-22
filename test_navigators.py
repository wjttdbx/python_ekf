"""Numerical regression tests for injectable navigators."""

import numpy as np
import pytest

from aerospace.estimation.navigators import SquareRootUKFNavigator


class _LinearMeasurementUKF(SquareRootUKFNavigator):
    def __init__(self, measurement_matrix, *args, **kwargs):
        self._measurement_matrix = measurement_matrix
        super().__init__(*args, **kwargs)

    def _measure_from_rel_state(self, x_rel):
        return self._measurement_matrix @ x_rel


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.3, 1e-3])
def test_square_root_ukf_matches_linear_kalman_update(alpha):
    x = np.array([1.0, 0.2, -0.1, 0.01, -0.02, 0.03])
    P = np.array([
        [2.0, 0.3, 0.1, 0.0, 0.0, 0.0],
        [0.3, 1.5, 0.2, 0.0, 0.0, 0.0],
        [0.1, 0.2, 1.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.05, 0.0],
        [0.0, 0.0, 0.0, 0.05, 0.3, 0.02],
        [0.0, 0.0, 0.0, 0.0, 0.02, 0.2],
    ]) * 1e-3
    H = np.array([
        [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.2, 0.0, 0.0],
    ])
    R = np.diag([2e-4, 3e-4, 4e-4])
    z = H @ x + np.array([1e-3, -2e-3, 1.5e-3])

    navigator = _LinearMeasurementUKF(
        H, x0=x, P0=P, Q=np.zeros((6, 6)), R=R,
        angles_only=False, alpha=alpha, beta=2.0, kappa=0.0,
    )
    x_prior, P_prior = navigator.predict(
        np.eye(6), np.zeros((6, 3)), np.zeros(3), np.zeros(3), 1.0,
        f_discrete=lambda state: state,
    )
    navigator.update(x_prior, P_prior, z)

    innovation_cov = H @ P @ H.T + R
    gain = P @ H.T @ np.linalg.inv(innovation_cov)
    x_expected = x + gain @ (z - H @ x)
    P_expected = P - gain @ innovation_cov @ gain.T

    tolerance = 1e-8 if alpha == 1e-3 else 1e-11
    np.testing.assert_allclose(x_prior, x, atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(P_prior, P, atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(navigator._last_S, innovation_cov, atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(navigator.x, x_expected, atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(navigator.P, P_expected, atol=tolerance, rtol=0.0)
    assert np.all(np.linalg.eigvalsh(navigator.P) > 0.0)

def test_e0_paired_noise_is_added_to_propagated_measurement():
    from scipy.integrate import solve_ivp

    from aerospace.dynamics.nerm import OrbitalDynamics
    from aerospace.estimation.ekf import RelativeStateEKF
    from aerospace.estimation.navigators import SDCEKFNavigator
    from experiments.e0_filter_validation import _eval_one_step

    class CapturingNavigator(SDCEKFNavigator):
        def update(self, x_priori, P_priori, z_meas):
            self.received_measurement = z_meas.copy()
            return super().update(x_priori, P_priori, z_meas)

    dynamics = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    X_p = np.array([30.0, 2.0, -1.0, 0.0, 0.01, -0.002])
    X_e = np.zeros(6)
    nu = 0.4
    u_p = np.array([1e-5, -2e-5, 0.5e-5])
    u_e = np.zeros(3)
    dt = 10.0
    noise = np.array([2e-5, -3e-5])
    R = np.diag([1e-8, 1e-8])
    navigator = CapturingNavigator(
        x0=X_p-X_e, P0=np.eye(6)*1e-3,
        Q=np.zeros((6, 6)), R=R, angles_only=True,
    )

    _eval_one_step(
        dynamics, X_p, X_e, nu, u_p, u_e, dt,
        navigator, np.random.default_rng(1), z_noise=noise,
    )

    state0 = np.concatenate([X_p, X_e, [nu]])
    propagated = solve_ivp(
        dynamics.dynamics_13d, [0.0, dt], state0,
        args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10,
    ).y[:, -1]
    expected = RelativeStateEKF.measure(
        propagated[:6], propagated[6:12], angle_only=True,
    ) + noise
    np.testing.assert_allclose(navigator.received_measurement, expected)