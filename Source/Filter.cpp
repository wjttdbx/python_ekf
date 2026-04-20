#include "../Header/Filter.h"

/* KalmanFilter	  KF */
Filter_KF::Filter_KF(MotionModel::CV_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	if(model_observation.Observation_Mode() != 0) throw std::invalid_argument("Filter -> Filter_KF -> error:Kalman filter is limited to using linear observation models");
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = X_priori + K * (Z - H * X_priori);
	P_Optimal = (I - K * H) * P_priori;
}

Filter_KF::Filter_KF(MotionModel::CA_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation){
	if (model_observation.Observation_Mode() != 0) throw std::invalid_argument("Filter -> Filter_KF -> error:Kalman filter is limited to using linear observation models");
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = X_priori + K * (Z - H * X_priori);
	P_Optimal = (I - K * H) * P_priori;
}

Filter_KF::Filter_KF(MotionModel::Singer_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	if (model_observation.Observation_Mode() != 0) throw std::invalid_argument("Filter -> Filter_KF -> error:Kalman filter is limited to using linear observation models");
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = X_priori + K * (Z - H * X_priori);
	P_Optimal = (I - K * H) * P_priori;
}

Filter_KF::Filter_KF(MotionModel::CS_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	if (model_observation.Observation_Mode() != 0) throw std::invalid_argument("Filter -> Filter_KF -> error:Kalman filter is limited to using linear observation models");
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	B = model.B_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;
	Vector Average_acceleration = MotionModel::Extract_acceleration(model.Dim, X_Optimal);

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal + B * Average_acceleration;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = X_priori + K * (Z - H * X_priori);
	P_Optimal = (I - K * H) * P_priori;
}

Filter_KF::Filter_KF(MotionModel::Jerk_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	if (model_observation.Observation_Mode() != 0) throw std::invalid_argument("Filter -> Filter_KF -> error:Kalman filter is limited to using linear observation models");
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = X_priori + K * (Z - H * X_priori);
	P_Optimal = (I - K * H) * P_priori;
}


/* Expanded Kalman Filter	EKF */
Filter_EKF::Filter_EKF(MotionModel::CV_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = (model_observation.Observation_Mode() == 0) ? X_priori + K * (Z - H * X_priori) :
		X_priori + K * (Z - model_observation.measurements(X_priori, model.Dim, observation));
	P_Optimal = (I - K * H) * P_priori;
}

Filter_EKF::Filter_EKF(MotionModel::CA_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	Vector middle = H * X_priori;
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = (model_observation.Observation_Mode() == 0) ? X_priori + K * (Z - H * X_priori) :
		X_priori + K * (Z - model_observation.measurements(X_priori, model.Dim, observation));
	P_Optimal = (I - K * H) * P_priori;
}

Filter_EKF::Filter_EKF(MotionModel::Singer_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = (model_observation.Observation_Mode() == 0) ? X_priori + K * (Z - H * X_priori) :
		X_priori + K * (Z - model_observation.measurements(X_priori, model.Dim, observation));
	P_Optimal = (I - K * H) * P_priori;
}

Filter_EKF::Filter_EKF(MotionModel::CS_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	B = model.B_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;
	Vector Average_acceleration = MotionModel::Extract_acceleration(model.Dim, X_Optimal);

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal + B * Average_acceleration;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = (model_observation.Observation_Mode() == 0) ? X_priori + K * (Z - H * X_priori) :
		X_priori + K * (Z - model_observation.measurements(X_priori, model.Dim, observation));
	P_Optimal = (I - K * H) * P_priori;
}

Filter_EKF::Filter_EKF(MotionModel::CS_Improvement_model& model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, size_t observe_num) {
	F = model.F_Calculation();
	Q = model.Q_Calculation(observe_num);
	B = model.B_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;
	Vector Average_acceleration = MotionModel::Extract_acceleration(model.Dim, X_Optimal);

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal + B * Average_acceleration;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();
		
	// Step 3: Update
	Matrix Pzz = H * P_priori * H.transpose() + R;
	Vector Z_pre = model_observation.measurements(X_priori, model.Dim, observation);
	model.alpha_adaption = model.alpha * maxDiagRatio(vector_mulTranspose(Z - Z_pre), Pzz);
	Vector X_old = X_Optimal;
	for (size_t i = 0; i < MotionModel_dim; ++i) model.acc[i] = (X_Optimal[MotionModel_dim * i + 2] - X_old[MotionModel_dim * i + 2]) * model.alpha_adaption * std::exp(-model.alpha_adaption * model.T) / (1. - std::exp(-model.alpha_adaption * model.T));

	X_Optimal = (model_observation.Observation_Mode() == 0) ? X_priori + K * (Z - H * X_priori) :
		X_priori + K * (Z - Z_pre);
	P_Optimal = (I - K * H) * P_priori;
}

Filter_EKF::Filter_EKF(MotionModel::Jerk_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Step 1: Prediction
	Vector X_priori = F * X_Optimal;
	Matrix P_priori = F * P_Optimal * F.transpose() + Q;

	// Step 2: Calculate the Kalman gain - K
	H = model_observation.H_Calculation(X_priori, model.Dim, observation);
	R = model_observation.R_Calculation();
	Matrix K = P_priori * H.transpose() * (H * P_priori * H.transpose() + R).inverse();

	// Step 3: Update
	X_Optimal = (model_observation.Observation_Mode() == 0) ? X_priori + K * (Z - H * X_priori) :
		X_priori + K * (Z - model_observation.measurements(X_priori, model.Dim, observation));
	P_Optimal = (I - K * H) * P_priori;
}


/* Unscented Kalman Filter */
Filter_UKF::Filter_UKF(MotionModel::CV_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	R = model_observation.R_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Calculation UKF parameter
	size_t n = X_Optimal.getDimension();
	size_t num_Sigma = 2 * n + 1;
	double kappa = (std::abs(UKF_parameter.Kappa) < tor_Discrimination) ? 3 - static_cast<int>(n) : UKF_parameter.Kappa;
	double alpha = UKF_parameter.Alpha;		double beta = UKF_parameter.Beta;
	double lambda = alpha * alpha * (n + kappa) - n;

	// Step 1: Calculate the sampling points
	Matrix cho = ((n + lambda) * P_Optimal).cholesky();
	Matrix Chi1_1(n, n);		Matrix Chi1_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi1_1[j][i] = X_Optimal[j] + cho[j][i];
			Chi1_2[j][i] = X_Optimal[j] - cho[j][i];
		}
	}
	Matrix Chi(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi[j][i + 1] = Chi1_1[j][i];
			Chi[j][i + n + 1] = Chi1_2[j][i];
		}
		Chi[i][0] = X_Optimal[i];
	}
	Vector W_m(num_Sigma);		Vector W_c(num_Sigma);
	W_m[0] = lambda / (n + lambda);		W_c[0] = W_m[0] + 1. - std::pow(alpha, 2) + beta;
	for (size_t i = 1; i < num_Sigma; ++i) {
		W_m[i] = 1. / (2 * (n + lambda));
		W_c[i] = 1. / (2 * (n + lambda));
	}

	// Step 2: Further prediction is made on the sigma point set
	Matrix Chi_pre = F * Chi;

	// Step 3: Calculate the mean and covariance
	Vector X_pre(n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = W_m[i] * Chi_pre[j][i];
		X_pre = X_pre + middle;
	}
	Matrix P_pre(n, n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = Chi_pre[j][i] - X_pre[j];
		P_pre = P_pre + W_m[i] * vector_mulTranspose(middle);
	}
	P_pre = P_pre + Q;

	// Step 4: According to the predicted values, the UT transformation is carried out successively to obtain the new sigma point set
	Matrix cho2 = ((n + lambda) * P_pre).cholesky();
	Matrix Chi2_1(n, n);		Matrix Chi2_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi2_1[j][i] = X_pre[j] + cho2[j][i];
			Chi2_2[j][i] = X_pre[j] - cho2[j][i];
		}
	}
	Matrix Chi2(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		Chi2[i][0] = X_pre[i];
		for (size_t j = 0; j < n; ++j) {
			Chi2[j][i + 1] = Chi2_1[j][i];
			Chi2[j][i + n + 1] = Chi2_2[j][i];
		}
	}

	// Step 5: Observation and prediction
	Matrix Z_i(Z.getDimension(), num_Sigma);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Chi2_i(n);
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Vector Z_pre_i = model_observation.measurements(Chi2_i, model.Dim, observation);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_i[j][i] = Z_pre_i[j];
	}

	// Step 6: Calculate the mean and covariance of the predicted observed values
	Vector Z_pre(Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Z_pre = Z_pre + W_m[i] * Z_pre_i;
	}
	Matrix Pzz(Z.getDimension(), Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Pzz = Pzz + W_c[i] * vector_mulTranspose(Z_pre_i - Z_pre);
	}
	Pzz = Pzz + R;
	Matrix Pxz(n, Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		Vector Chi2_i(n);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Pxz = Pxz + W_c[i] * vector_mulTranspose1(Chi2_i - X_pre, Z_pre_i - Z_pre);
	}

	// Step 7: Calculate the Kalman gain - K
	Matrix K = Pxz * (Pzz.inverse());

	// Step 8: Update Status/Variance 
	X_Optimal = X_pre + K * (Z - Z_pre);
	P_Optimal = P_pre - (K * Pzz) * K.transpose();
}

Filter_UKF::Filter_UKF(MotionModel::CA_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	R = model_observation.R_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Calculation UKF parameter
	size_t n = X_Optimal.getDimension();
	size_t num_Sigma = 2 * n + 1;
	double kappa = (std::abs(UKF_parameter.Kappa) < tor_Discrimination) ? 3 - static_cast<int>(n) : UKF_parameter.Kappa;
	double alpha = UKF_parameter.Alpha;		double beta = UKF_parameter.Beta;
	double lambda = alpha * alpha * (n + kappa) - n;

	// Step 1: Calculate the sampling points
	Matrix cho = ((n + lambda) * P_Optimal).cholesky();
	Matrix Chi1_1(n, n);		Matrix Chi1_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi1_1[j][i] = X_Optimal[j] + cho[j][i];
			Chi1_2[j][i] = X_Optimal[j] - cho[j][i];
		}
	}
	Matrix Chi(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi[j][i + 1] = Chi1_1[j][i];
			Chi[j][i + n + 1] = Chi1_2[j][i];
		}
		Chi[i][0] = X_Optimal[i];
	}
	Vector W_m(num_Sigma);		Vector W_c(num_Sigma);
	W_m[0] = lambda / (n + lambda);		W_c[0] = W_m[0] + 1. - std::pow(alpha, 2) + beta;
	for (size_t i = 1; i < num_Sigma; ++i) {
		W_m[i] = 1. / (2 * (n + lambda));
		W_c[i] = 1. / (2 * (n + lambda));
	}

	// Step 2: Further prediction is made on the sigma point set
	Matrix Chi_pre = F * Chi;

	// Step 3: Calculate the mean and covariance
	Vector X_pre(n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = W_m[i] * Chi_pre[j][i];
		X_pre = X_pre + middle;
	}
	Matrix P_pre(n, n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = Chi_pre[j][i] - X_pre[j];
		P_pre = P_pre + W_m[i] * vector_mulTranspose(middle);
	}
	P_pre = P_pre + Q;

	// Step 4: According to the predicted values, the UT transformation is carried out successively to obtain the new sigma point set
	Matrix cho2 = ((n + lambda) * P_pre).cholesky();
	Matrix Chi2_1(n, n);		Matrix Chi2_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi2_1[j][i] = X_pre[j] + cho2[j][i];
			Chi2_2[j][i] = X_pre[j] - cho2[j][i];
		}
	}
	Matrix Chi2(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		Chi2[i][0] = X_pre[i];
		for (size_t j = 0; j < n; ++j) {
			Chi2[j][i + 1] = Chi2_1[j][i];
			Chi2[j][i + n + 1] = Chi2_2[j][i];
		}
	}

	// Step 5: Observation and prediction
	Matrix Z_i(Z.getDimension(), num_Sigma);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Chi2_i(n);
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Vector Z_pre_i = model_observation.measurements(Chi2_i, model.Dim, observation);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_i[j][i] = Z_pre_i[j];
	}

	// Step 6: Calculate the mean and covariance of the predicted observed values
	Vector Z_pre(Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Z_pre = Z_pre + W_m[i] * Z_pre_i;
	}
	Matrix Pzz(Z.getDimension(), Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Pzz = Pzz + W_c[i] * vector_mulTranspose(Z_pre_i - Z_pre);
	}
	Pzz = Pzz + R;
	Matrix Pxz(n, Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		Vector Chi2_i(n);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Pxz = Pxz + W_c[i] * vector_mulTranspose1(Chi2_i - X_pre, Z_pre_i - Z_pre);
	}

	// Step 7: Calculate the Kalman gain - K
	Matrix K = Pxz * (Pzz.inverse());

	// Step 8: Update Status/Variance 
	X_Optimal = X_pre + K * (Z - Z_pre);
	P_Optimal = P_pre - (K * Pzz) * K.transpose();
}

Filter_UKF::Filter_UKF(MotionModel::Singer_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	R = model_observation.R_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Calculation UKF parameter
	size_t n = X_Optimal.getDimension();
	size_t num_Sigma = 2 * n + 1;
	double kappa = (std::abs(UKF_parameter.Kappa) < tor_Discrimination) ? 3 - static_cast<int>(n) : UKF_parameter.Kappa;
	double alpha = UKF_parameter.Alpha;		double beta = UKF_parameter.Beta;
	double lambda = alpha * alpha * (n + kappa) - n;

	// Step 1: Calculate the sampling points
	Matrix cho = ((n + lambda) * P_Optimal).cholesky();
	Matrix Chi1_1(n, n);		Matrix Chi1_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi1_1[j][i] = X_Optimal[j] + cho[j][i];
			Chi1_2[j][i] = X_Optimal[j] - cho[j][i];
		}
	}
	Matrix Chi(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi[j][i + 1] = Chi1_1[j][i];
			Chi[j][i + n + 1] = Chi1_2[j][i];
		}
		Chi[i][0] = X_Optimal[i];
	}
	Vector W_m(num_Sigma);		Vector W_c(num_Sigma);
	W_m[0] = lambda / (n + lambda);		W_c[0] = W_m[0] + 1. - std::pow(alpha, 2) + beta;
	for (size_t i = 1; i < num_Sigma; ++i) {
		W_m[i] = 1. / (2 * (n + lambda));
		W_c[i] = 1. / (2 * (n + lambda));
	}

	// Step 2: Further prediction is made on the sigma point set
	Matrix Chi_pre = F * Chi;

	// Step 3: Calculate the mean and covariance
	Vector X_pre(n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = W_m[i] * Chi_pre[j][i];
		X_pre = X_pre + middle;
	}
	Matrix P_pre(n, n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = Chi_pre[j][i] - X_pre[j];
		P_pre = P_pre + W_m[i] * vector_mulTranspose(middle);
	}
	P_pre = P_pre + Q;

	// Step 4: According to the predicted values, the UT transformation is carried out successively to obtain the new sigma point set
	Matrix cho2 = ((n + lambda) * P_pre).cholesky();
	Matrix Chi2_1(n, n);		Matrix Chi2_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi2_1[j][i] = X_pre[j] + cho2[j][i];
			Chi2_2[j][i] = X_pre[j] - cho2[j][i];
		}
	}
	Matrix Chi2(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		Chi2[i][0] = X_pre[i];
		for (size_t j = 0; j < n; ++j) {
			Chi2[j][i + 1] = Chi2_1[j][i];
			Chi2[j][i + n + 1] = Chi2_2[j][i];
		}
	}

	// Step 5: Observation and prediction
	Matrix Z_i(Z.getDimension(), num_Sigma);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Chi2_i(n);
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Vector Z_pre_i = model_observation.measurements(Chi2_i, model.Dim, observation);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_i[j][i] = Z_pre_i[j];
	}

	// Step 6: Calculate the mean and covariance of the predicted observed values
	Vector Z_pre(Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Z_pre = Z_pre + W_m[i] * Z_pre_i;
	}
	Matrix Pzz(Z.getDimension(), Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Pzz = Pzz + W_c[i] * vector_mulTranspose(Z_pre_i - Z_pre);
	}
	Pzz = Pzz + R;
	Matrix Pxz(n, Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		Vector Chi2_i(n);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Pxz = Pxz + W_c[i] * vector_mulTranspose1(Chi2_i - X_pre, Z_pre_i - Z_pre);
	}

	// Step 7: Calculate the Kalman gain - K
	Matrix K = Pxz * (Pzz.inverse());

	// Step 8: Update Status/Variance 
	X_Optimal = X_pre + K * (Z - Z_pre);
	P_Optimal = P_pre - (K * Pzz) * K.transpose();
}

Filter_UKF::Filter_UKF(MotionModel::CS_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	B = model.B_Calculation();
	R = model_observation.R_Calculation();
	Vector xk_acc = MotionModel::Extract_acceleration(model.Dim, X_Optimal);
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Calculation UKF parameter
	size_t n = X_Optimal.getDimension();
	size_t num_Sigma = 2 * n + 1;
	double kappa = (std::abs(UKF_parameter.Kappa) < tor_Discrimination) ? 3 - static_cast<int>(n) : UKF_parameter.Kappa;
	double alpha = UKF_parameter.Alpha;		double beta = UKF_parameter.Beta;
	double lambda = alpha * alpha * (n + kappa) - n;

	// Step 1: Calculate the sampling points
	Matrix cho = ((n + lambda) * P_Optimal).cholesky();
	Matrix Chi1_1(n, n);		Matrix Chi1_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi1_1[j][i] = X_Optimal[j] + cho[j][i];
			Chi1_2[j][i] = X_Optimal[j] - cho[j][i];
		}
	}
	Matrix Chi(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi[j][i + 1] = Chi1_1[j][i];
			Chi[j][i + n + 1] = Chi1_2[j][i];
		}
		Chi[i][0] = X_Optimal[i];
	}
	Vector W_m(num_Sigma);		Vector W_c(num_Sigma);
	W_m[0] = lambda / (n + lambda);		W_c[0] = W_m[0] + 1. - std::pow(alpha, 2) + beta;
	for (size_t i = 1; i < num_Sigma; ++i) {
		W_m[i] = 1. / (2 * (n + lambda));
		W_c[i] = 1. / (2 * (n + lambda));
	}
	
	// Step 2: Further prediction is made on the sigma point set
	Matrix Chi_pre = F * Chi;
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle = B * xk_acc;
		for (size_t j = 0; j < n; ++j) Chi_pre[j][i] = Chi_pre[j][i] + middle[j];
	}

	// Step 3: Calculate the mean and covariance
	Vector X_pre(n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = W_m[i] * Chi_pre[j][i];
		X_pre = X_pre + middle;
	}
	Matrix P_pre(n, n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = Chi_pre[j][i] - X_pre[j];
		P_pre = P_pre + W_m[i] * vector_mulTranspose(middle);
	}
	P_pre = P_pre + Q;
	Vector lzz(n);
	for (size_t i = 0; i < n; ++i) lzz[i] = P_pre[i][i];

	// Step 4: According to the predicted values, the UT transformation is carried out successively to obtain the new sigma point set
	Matrix cho2 = ((n + lambda) * P_pre).cholesky();
	Matrix Chi2_1(n, n);		Matrix Chi2_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi2_1[j][i] = X_pre[j] + cho2[j][i];
			Chi2_2[j][i] = X_pre[j] - cho2[j][i];
		}
	}
	Matrix Chi2(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		Chi2[i][0] = X_pre[i];
		for (size_t j = 0; j < n; ++j) {
			Chi2[j][i + 1] = Chi2_1[j][i];
			Chi2[j][i + n + 1] = Chi2_2[j][i];
		}
	}

	// Step 5: Observation and prediction
	Matrix Z_i(Z.getDimension(), num_Sigma);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Chi2_i(n);
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Vector Z_pre_i = model_observation.measurements(Chi2_i, model.Dim, observation);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_i[j][i] = Z_pre_i[j];
	}

	// Step 6: Calculate the mean and covariance of the predicted observed values
	Vector Z_pre(Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Z_pre = Z_pre + W_m[i] * Z_pre_i;
	}
	Matrix Pzz(Z.getDimension(), Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Pzz = Pzz + W_c[i] * vector_mulTranspose(Z_pre_i - Z_pre);
	}
	Pzz = Pzz + R;
	Matrix Pxz(n, Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		Vector Chi2_i(n);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Pxz = Pxz + W_c[i] * vector_mulTranspose1(Chi2_i - X_pre, Z_pre_i - Z_pre);
	}

	// Step 7: Calculate the Kalman gain - K
	Matrix K = Pxz * (Pzz.inverse());

	// Step 8: Update Status/Variance 
	X_Optimal = X_pre + K * (Z - Z_pre);
	P_Optimal = P_pre - (K * Pzz) * K.transpose();
}


Filter_UKF::Filter_UKF(MotionModel::CS_Improvement_model& model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter, size_t observe_num) {
	F = model.F_Calculation();
	Q = model.Q_Calculation(observe_num);
	B = model.B_Calculation();
	R = model_observation.R_Calculation();
	Vector xk_acc = MotionModel::Extract_acceleration(model.Dim, X_Optimal);
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Calculation UKF parameter
	size_t n = X_Optimal.getDimension();
	size_t num_Sigma = 2 * n + 1;
	double kappa = (std::abs(UKF_parameter.Kappa) < tor_Discrimination) ? 3 - static_cast<int>(n) : UKF_parameter.Kappa;
	double alpha = UKF_parameter.Alpha;		double beta = UKF_parameter.Beta;
	double lambda = alpha * alpha * (n + kappa) - n;

	// Step 1: Calculate the sampling points
	Matrix cho = ((n + lambda) * P_Optimal).cholesky();
	Matrix Chi1_1(n, n);		Matrix Chi1_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi1_1[j][i] = X_Optimal[j] + cho[j][i];
			Chi1_2[j][i] = X_Optimal[j] - cho[j][i];
		}
	}
	Matrix Chi(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi[j][i + 1] = Chi1_1[j][i];
			Chi[j][i + n + 1] = Chi1_2[j][i];
		}
		Chi[i][0] = X_Optimal[i];
	}
	Vector W_m(num_Sigma);		Vector W_c(num_Sigma);
	W_m[0] = lambda / (n + lambda);		W_c[0] = W_m[0] + 1. - std::pow(alpha, 2) + beta;
	for (size_t i = 1; i < num_Sigma; ++i) {
		W_m[i] = 1. / (2 * (n + lambda));
		W_c[i] = 1. / (2 * (n + lambda));
	}

	// Step 2: Further prediction is made on the sigma point set
	Matrix Chi_pre = F * Chi;
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle = B * xk_acc;
		for (size_t j = 0; j < n; ++j) Chi_pre[j][i] = Chi_pre[j][i] + middle[j];
	}

	// Step 3: Calculate the mean and covariance
	Vector X_pre(n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = W_m[i] * Chi_pre[j][i];
		X_pre = X_pre + middle;
	}
	Matrix P_pre(n, n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = Chi_pre[j][i] - X_pre[j];
		P_pre = P_pre + W_m[i] * vector_mulTranspose(middle);
	}
	P_pre = P_pre + Q;
	Vector lzz(n);
	for (size_t i = 0; i < n; ++i) lzz[i] = P_pre[i][i];

	// Step 4: According to the predicted values, the UT transformation is carried out successively to obtain the new sigma point set
	Matrix cho2 = ((n + lambda) * P_pre).cholesky();
	Matrix Chi2_1(n, n);		Matrix Chi2_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi2_1[j][i] = X_pre[j] + cho2[j][i];
			Chi2_2[j][i] = X_pre[j] - cho2[j][i];
		}
	}
	Matrix Chi2(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		Chi2[i][0] = X_pre[i];
		for (size_t j = 0; j < n; ++j) {
			Chi2[j][i + 1] = Chi2_1[j][i];
			Chi2[j][i + n + 1] = Chi2_2[j][i];
		}
	}

	// Step 5: Observation and prediction
	Matrix Z_i(Z.getDimension(), num_Sigma);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Chi2_i(n);
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Vector Z_pre_i = model_observation.measurements(Chi2_i, model.Dim, observation);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_i[j][i] = Z_pre_i[j];
	}

	// Step 6: Calculate the mean and covariance of the predicted observed values
	Vector Z_pre(Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Z_pre = Z_pre + W_m[i] * Z_pre_i;
	}
	Matrix Pzz(Z.getDimension(), Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Pzz = Pzz + W_c[i] * vector_mulTranspose(Z_pre_i - Z_pre);
	}
	Pzz = Pzz + R;
	Matrix Pxz(n, Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		Vector Chi2_i(n);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Pxz = Pxz + W_c[i] * vector_mulTranspose1(Chi2_i - X_pre, Z_pre_i - Z_pre);
	}

	// Step 7: Calculate the Kalman gain - K
	Matrix K = Pxz * (Pzz.inverse());
	model.alpha_adaption = model.alpha * maxDiagRatio(vector_mulTranspose(Z - Z_pre), Pzz);
	

	// Step 8: Update Status/Variance 
	Vector X_old = X_Optimal;
	X_Optimal = X_pre + K * (Z - Z_pre);
	P_Optimal = P_pre - (K * Pzz) * K.transpose();
	for (size_t i = 0; i < MotionModel_dim; ++i) model.acc[i] = (X_Optimal[MotionModel_dim * i + 2] - X_old[MotionModel_dim * i + 2]) * model.alpha_adaption * std::exp(-model.alpha_adaption * model.T) / (1. - std::exp(-model.alpha_adaption * model.T));
}

Filter_UKF::Filter_UKF(MotionModel::Jerk_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter) {
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	R = model_observation.R_Calculation();
	Matrix I(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < F.getRows(); ++i) I[i][i] = 1;

	// Calculation UKF parameter
	size_t n = X_Optimal.getDimension();
	size_t num_Sigma = 2 * n + 1;
	double kappa = (std::abs(UKF_parameter.Kappa) < tor_Discrimination) ? 3 - static_cast<int>(n) : UKF_parameter.Kappa;
	double alpha = UKF_parameter.Alpha;		double beta = UKF_parameter.Beta;
	double lambda = alpha * alpha * (n + kappa) - n;

	// Step 1: Calculate the sampling points
	Matrix cho = ((n + lambda) * P_Optimal).cholesky();
	Matrix Chi1_1(n, n);		Matrix Chi1_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi1_1[j][i] = X_Optimal[j] + cho[j][i];
			Chi1_2[j][i] = X_Optimal[j] - cho[j][i];
		}
	}
	Matrix Chi(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi[j][i + 1] = Chi1_1[j][i];
			Chi[j][i + n + 1] = Chi1_2[j][i];
		}
		Chi[i][0] = X_Optimal[i];
	}
	Vector W_m(num_Sigma);		Vector W_c(num_Sigma);
	W_m[0] = lambda / (n + lambda);		W_c[0] = W_m[0] + 1. - std::pow(alpha, 2) + beta;
	for (size_t i = 1; i < num_Sigma; ++i) {
		W_m[i] = 1. / (2 * (n + lambda));
		W_c[i] = 1. / (2 * (n + lambda));
	}

	// Step 2: Further prediction is made on the sigma point set
	Matrix Chi_pre = F * Chi;

	// Step 3: Calculate the mean and covariance
	Vector X_pre(n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = W_m[i] * Chi_pre[j][i];
		X_pre = X_pre + middle;
	}
	Matrix P_pre(n, n);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector middle(n);
		for (size_t j = 0; j < n; ++j) middle[j] = Chi_pre[j][i] - X_pre[j];
		P_pre = P_pre + W_m[i] * vector_mulTranspose(middle);
	}
	P_pre = P_pre + Q;

	// Step 4: According to the predicted values, the UT transformation is carried out successively to obtain the new sigma point set
	Matrix cho2 = ((n + lambda) * P_pre).cholesky();
	Matrix Chi2_1(n, n);		Matrix Chi2_2(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			Chi2_1[j][i] = X_pre[j] + cho2[j][i];
			Chi2_2[j][i] = X_pre[j] - cho2[j][i];
		}
	}
	Matrix Chi2(n, num_Sigma);
	for (size_t i = 0; i < n; ++i) {
		Chi2[i][0] = X_pre[i];
		for (size_t j = 0; j < n; ++j) {
			Chi2[j][i + 1] = Chi2_1[j][i];
			Chi2[j][i + n + 1] = Chi2_2[j][i];
		}
	}

	// Step 5: Observation and prediction
	Matrix Z_i(Z.getDimension(), num_Sigma);
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Chi2_i(n);
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Vector Z_pre_i = model_observation.measurements(Chi2_i, model.Dim, observation);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_i[j][i] = Z_pre_i[j];
	}

	// Step 6: Calculate the mean and covariance of the predicted observed values
	Vector Z_pre(Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Z_pre = Z_pre + W_m[i] * Z_pre_i;
	}
	Matrix Pzz(Z.getDimension(), Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		Pzz = Pzz + W_c[i] * vector_mulTranspose(Z_pre_i - Z_pre);
	}
	Pzz = Pzz + R;
	Matrix Pxz(n, Z.getDimension());
	for (size_t i = 0; i < num_Sigma; ++i) {
		Vector Z_pre_i(Z.getDimension());
		Vector Chi2_i(n);
		for (size_t j = 0; j < Z.getDimension(); ++j) Z_pre_i[j] = Z_i[j][i];
		for (size_t j = 0; j < n; ++j) Chi2_i[j] = Chi2[j][i];
		Pxz = Pxz + W_c[i] * vector_mulTranspose1(Chi2_i - X_pre, Z_pre_i - Z_pre);
	}

	// Step 7: Calculate the Kalman gain - K
	Matrix K = Pxz * (Pzz.inverse());

	// Step 8: Update Status/Variance 
	X_Optimal = X_pre + K * (Z - Z_pre);
	P_Optimal = P_pre - (K * Pzz) * K.transpose();
}


/* Particle Filter */
void Filter_PF::initialize(std::vector<Particle>& particles, Vector X_Optimal) {
	size_t num_particles = particles[0].Num_Particle;
	size_t dim = X_Optimal.getDimension();

	// Generate random numbers
	std::random_device random;
	std::mt19937 generator(random());
	std::normal_distribution<double> distribution(0., 1.);

	// Initialize
	for (size_t i = 0; i < num_particles; ++i) {
		particles[i].X = X_Optimal;
		for(size_t j = 0; j < dim; ++j) particles[i].X[j] += distribution(generator);
		particles[i].Weight = 1. / num_particles;
	}
}

void Filter_PF::resampling(std::vector<Particle>& particles) {
	size_t num_particles = particles[0].Num_Particle;

	// Calculate the cumulative weight
	double sum_weights = 0.;
	for (size_t i = 0; i < num_particles; ++i) sum_weights += particles[i].Weight;

	// Normalize weights
	for (size_t i = 0; i < num_particles; ++i) particles[i].Weight /= sum_weights;

	// Create a cumulative weight array
	Vector cumulative_weight(num_particles);
	cumulative_weight[0] = particles[0].Weight;
	for (size_t i = 1; i < num_particles; ++i) cumulative_weight[i] = cumulative_weight[i - 1] + particles[i].Weight;

	// Create a new set of particles
	std::vector<Particle> new_particles(num_particles);
	std::random_device random;
	std::mt19937 generator(random());
	std::uniform_real_distribution<double> dist(0., 1. / num_particles);
	double start = dist(generator);
	for (size_t i = 0; i < num_particles; ++i) {
		double target = start + i * (1. / num_particles);
		size_t index = 0;
		while (target > cumulative_weight[index] && index < num_particles - 1) index++;
		new_particles[i] = particles[index];
		new_particles[i].Weight = 1.0 / num_particles;
	}

	// Complete resampling
	for(size_t i = 0; i < num_particles; ++i) particles[i] = new_particles[i];
}

Filter_PF::Filter_PF(MotionModel::CV_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, std::vector<Particle>& particles) {
	size_t num_particles = particles[0].Num_Particle;
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	R = model_observation.R_Calculation();

	// Step 1: Prediction
	for (size_t i = 0; i < num_particles; ++i) particles[i].X = F * particles[i].X;
	
	// Step 2: Calculate
	double total_weight = 0.;
	for (size_t i = 0; i < num_particles; ++i) {
		Vector z_priori = model_observation.measurements(particles[i].X, model.Dim, observation);
		Vector exponent = (Z - z_priori).transpose() * R.inverse() * (Z - z_priori);
		particles[i].Weight *= std::exp(-0.5 * exponent[0]);
		total_weight += particles[i].Weight;
	}
	for (size_t i = 0; i < num_particles; ++i) particles[i].Weight /= total_weight;
	resampling(particles);
	Vector X_pre(X_Optimal.getDimension());
	Matrix P_pre(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < num_particles; ++i) X_pre = X_pre + particles[i].Weight * particles[i].X;
	for (size_t i = 0; i < num_particles; ++i) {
		Vector difference = particles[i].X - X_pre;
		P_pre = P_pre + particles[i].Weight * vector_mulTranspose(difference);
	}
	P_pre = P_pre + Q;

	// Step 3: Update
	X_Optimal = X_pre;
	P_Optimal = P_pre;
}

Filter_PF::Filter_PF(MotionModel::CA_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, std::vector<Particle>& particles) {
	size_t num_particles = particles[0].Num_Particle;
	F = model.F_Calculation();
	Q = model.Q_Calculation();
	R = model_observation.R_Calculation();

	// Step 1: Prediction
	Filter_PF::initialize(particles, X_Optimal);
	for (size_t i = 0; i < num_particles; ++i) particles[i].X = F * particles[i].X;

	// Step 2: Calculate
	double total_weight = 0.;
	for (size_t i = 0; i < num_particles; ++i) {
		Vector z_priori = model_observation.measurements(particles[i].X, model.Dim, observation);
		Vector exponent = (Z - z_priori).transpose() * R.inverse() * (Z - z_priori);
		particles[i].Weight *= std::exp(-0.5 * exponent[0]);
		total_weight += particles[i].Weight;
	}
	for (size_t i = 0; i < num_particles; ++i) particles[i].Weight /= total_weight;
	resampling(particles);
	Vector X_pre(X_Optimal.getDimension());
	Matrix P_pre(X_Optimal.getDimension(), X_Optimal.getDimension());
	for (size_t i = 0; i < num_particles; ++i) X_pre = X_pre + particles[i].Weight * particles[i].X;
	for (size_t i = 0; i < num_particles; ++i) {
		Vector difference = particles[i].X - X_pre;
		P_pre = P_pre + particles[i].Weight * vector_mulTranspose(difference);
	}
	P_pre = P_pre + Q;

	// Step 3: Update
	X_Optimal = X_pre;
	P_Optimal = P_pre;
}