#pragma once
/// ukf_7state.h —— 7 状态联合 UKF (四元数 + 角速度)
/// 对齐 MATLAB main2.m + UKF.m
///
/// 特色：双轮 sigma 点采样
///   第一轮: 2n+1 点 → 动力学传播 → 预测均值和 PXX
///   第二轮: +2n 点 (从 Q 生成) → 总共 4n+1 点 → 测量映射 → PXZ, PZZ → K 增益

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "math_utils.h"
#include "quaternion.h"
#include "dynamics.h"

namespace jt_ekf {

// ── UKF 参数 (对齐 MATLAB 全局变量) ────────────────────────────────────────────
struct UKFParams {
    int n;
    double alpha;
    double beta;
    double lambda;
    double gamma;

    UKFParams(int dim, double a = 1.0, double b = 2.0)
        : n(dim), alpha(a), beta(b) {
        lambda = 3.0 * alpha * alpha - static_cast<double>(n);
        gamma = std::sqrt(static_cast<double>(n) + lambda);
    }
};

// ── JointUKF: 7 状态联合 UKF ─────────────────────────────────────────────────
class JointUKF {
public:
    static constexpr int N = 7;

    Eigen::Matrix<double, N, 1> X;
    Eigen::Matrix<double, N, N> P;
    UKFParams ukf_param;
    Eigen::Matrix<double, N, N> Q;
    Eigen::Matrix<double, N, N> R_mea;

    JointUKF(const Eigen::Matrix<double, N, 1>& X0,
             const Eigen::Matrix<double, N, N>& P0,
             const Eigen::Matrix<double, N, N>& Q_proc,
             const Eigen::Matrix<double, N, N>& R_meas)
        : X(X0), P(P0), ukf_param(N, 1.0, 2.0), Q(Q_proc), R_mea(R_meas) {
        X.head<4>() = quat_normalize(X.head<4>());
    }

    // ── 生成 sigma 点 (第一轮: 2N+1) ──
    void generate_sigma_points(Eigen::Matrix<double, N, 2*N+1>& XSam) const {
        Eigen::Matrix<double, N, N> P_reg = P + 1e-18 * Eigen::Matrix<double, N, N>::Identity();
        Eigen::LLT<Eigen::Matrix<double, N, N>> llt(P_reg);
        Eigen::Matrix<double, N, N> choP = ukf_param.gamma * llt.matrixL().toDenseMatrix().transpose();

        XSam.col(0) = X;
        for (int j = 0; j < N; ++j) {
            XSam.col(1 + j)     = X + choP.col(j);
            XSam.col(1 + N + j) = X - choP.col(j);
        }
    }

    // ── 第二轮额外采样: Q 生成 2N 点 ──
    void generate_Q_sigma_points(
            const Eigen::Matrix<double, N, 1>& x_center,
            Eigen::Matrix<double, N, 2*N>& XSam) const {
        Eigen::Matrix<double, N, N> Q_reg = Q + 1e-18 * Eigen::Matrix<double, N, N>::Identity();
        Eigen::LLT<Eigen::Matrix<double, N, N>> llt(Q_reg);
        Eigen::Matrix<double, N, N> choQ = ukf_param.gamma * llt.matrixL().toDenseMatrix().transpose();

        for (int j = 0; j < N; ++j) {
            XSam.col(j)     = x_center - choQ.col(j);
            XSam.col(N + j) = x_center + choQ.col(j);
        }
    }

    // ── 完整一步 ──
    void step(const Eigen::Vector3d& T_torque,
              const Eigen::Matrix<double, N, 1>& Z_meas,
              const RigidBodyParams& params) {
        // ===== 第一轮: 状态传播 =====
        Eigen::Matrix<double, N, 2*N+1> XSam;
        generate_sigma_points(XSam);

        Eigen::Matrix<double, N, 2*N+1> XPred;
        for (int j = 0; j < 2*N + 1; ++j) {
            XPred.col(j) = xStateFunction(XSam.col(j), T_torque, params);
        }

        // 权重 (第一轮)
        double lam = ukf_param.lambda;
        Eigen::VectorXd Wm1(2*N+1), Wc1(2*N+1);
        Wm1.setConstant(1.0/(2.0*(N + lam)));
        Wc1.setConstant(1.0/(2.0*(N + lam)));
        Wm1(0) = lam/(N + lam);
        Wc1(0) = lam/(N + lam) + 1.0 - ukf_param.alpha*ukf_param.alpha + ukf_param.beta;

        Eigen::Matrix<double, N, 1> XPredAve = XPred * Wm1;
        XPredAve.head<4>() = quat_normalize(XPredAve.head<4>());

        Eigen::Matrix<double, N, N> PXX = Q;
        for (int j = 0; j < 2*N + 1; ++j) {
            Eigen::Matrix<double, N, 1> diff = XPred.col(j) - XPredAve;
            PXX.noalias() += Wc1(j) * (diff * diff.transpose());
        }

        // ===== 第二轮: 测量更新 =====
        Eigen::Matrix<double, N, 2*N> XSamQ;
        generate_Q_sigma_points(XPred.col(0), XSamQ);

        static constexpr int TOTAL = 4*N + 1;
        Eigen::Matrix<double, N, TOTAL> XSamSEC;
        XSamSEC.template leftCols<2*N+1>() = XPred;
        XSamSEC.template rightCols<2*N>() = XSamQ;

        // 测量映射 (obs = identity)
        Eigen::Matrix<double, N, TOTAL> ZPred;
        for (int j = 0; j < TOTAL; ++j) {
            ZPred.col(j) = XSamSEC.col(j);
        }

        // 第二轮权重
        Eigen::VectorXd Wm2(TOTAL), Wc2(TOTAL);
        Wm2.setConstant(1.0/(2.0*(2.0*N + lam)));
        Wc2.setConstant(1.0/(2.0*(2.0*N + lam)));
        Wm2(0) = lam/(2.0*N + lam);
        Wc2(0) = lam/(2.0*N + lam) + 1.0 - ukf_param.alpha*ukf_param.alpha + ukf_param.beta;

        Eigen::Matrix<double, N, 1> ZPredAve = ZPred * Wm2;

        Eigen::Matrix<double, N, N> PXZ = Eigen::Matrix<double, N, N>::Zero();
        Eigen::Matrix<double, N, N> PZZ = R_mea;
        for (int j = 0; j < TOTAL; ++j) {
            Eigen::Matrix<double, N, 1> diff_x = XSamSEC.col(j) - XPredAve;
            Eigen::Matrix<double, N, 1> diff_z = ZPred.col(j) - ZPredAve;
            PXZ.noalias() += Wc2(j) * (diff_x * diff_z.transpose());
            PZZ.noalias() += Wc2(j) * (diff_z * diff_z.transpose());
        }

        // Kalman 更新
        Eigen::Matrix<double, N, N> K = PXZ * PZZ.inverse();
        X = XPredAve + K * (Z_meas - ZPredAve);
        X.head<4>() = quat_normalize(X.head<4>());
        P = PXX - K * PZZ * K.transpose();
    }
};

} // namespace jt_ekf
