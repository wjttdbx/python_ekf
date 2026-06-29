#pragma once
/// cascaded_ukf.h —— 级联双 UKF (先 w 后 q)
/// 对齐 MATLAB main3.m + UKF1.m + UKF2.m

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "math_utils.h"
#include "quaternion.h"
#include "dynamics.h"

namespace jt_ekf {

// ── UKF2: 3 状态角速度 UKF ────────────────────────────────────────────────────
class UKF2_AngularVelocity {
public:
    static constexpr int N = 3;

    Eigen::Vector3d X;
    Eigen::Matrix3d P;
    UKFParams param;
    Eigen::Matrix3d Q2;
    Eigen::Matrix3d R_mea2;

    UKF2_AngularVelocity(const Eigen::Vector3d& X0,
                          const Eigen::Matrix3d& P0,
                          const Eigen::Matrix3d& Q,
                          const Eigen::Matrix3d& R_meas,
                          double alpha = 1.0, double beta = 2.0)
        : X(X0), P(P0), param(N, alpha, beta), Q2(Q), R_mea2(R_meas) {}

    // 生成 2N+1 sigma 点 (静态大小)
    Eigen::Matrix<double, N, 2*N+1> make_sigma() const {
        Eigen::Matrix3d P_reg = P + 1e-18 * Eigen::Matrix3d::Identity();
        Eigen::LLT<Eigen::Matrix3d> llt(P_reg);
        Eigen::Matrix3d choP = param.gamma * llt.matrixL().toDenseMatrix().transpose();

        Eigen::Matrix<double, N, 2*N+1> XSam;
        XSam.col(0) = X;
        for (int j = 0; j < N; ++j) {
            XSam.col(1 + j)     = X + choP.col(j);
            XSam.col(1 + N + j) = X - choP.col(j);
        }
        return XSam;
    }

    void step(const Eigen::Vector3d& Z_w,
              const Eigen::Vector3d& T_torque,
              const RigidBodyParams& params) {
        double lam = param.lambda;

        // 第一轮权重
        Eigen::VectorXd Wm(2*N+1), Wc(2*N+1);
        Wm.setConstant(1.0/(2.0*(N + lam)));
        Wc.setConstant(1.0/(2.0*(N + lam)));
        Wm(0) = lam/(N + lam);
        Wc(0) = lam/(N + lam) + 1.0 - param.alpha*param.alpha + param.beta;

        // Sigma 点
        Eigen::Matrix<double, N, 2*N+1> XSam = make_sigma();

        // 传播
        Eigen::Matrix<double, N, 2*N+1> XPred;
        for (int j = 0; j < 2*N + 1; ++j) {
            XPred.col(j) = xStateFunction2(XSam.col(j), T_torque, params);
        }

        Eigen::Vector3d XPredAve = XPred * Wm;

        Eigen::Matrix3d PXX = Q2;
        for (int j = 0; j < 2*N + 1; ++j) {
            Eigen::Vector3d diff = XPred.col(j) - XPredAve;
            PXX.noalias() += Wc(j) * (diff * diff.transpose());
        }

        // 第二轮: Q 采样
        Eigen::Matrix3d Q_reg = Q2 + 1e-18 * Eigen::Matrix3d::Identity();
        Eigen::LLT<Eigen::Matrix3d> lltQ(Q_reg);
        Eigen::Matrix3d choQ = param.gamma * lltQ.matrixL().toDenseMatrix().transpose();

        Eigen::Matrix<double, N, 2*N> XTempQ;
        for (int j = 0; j < N; ++j) {
            XTempQ.col(j)     = XPred.col(0) - choQ.col(j);
            XTempQ.col(N + j) = XPred.col(0) + choQ.col(j);
        }

        static constexpr int TOTAL = 4*N + 1;
        Eigen::Matrix<double, N, TOTAL> XSamSEC;
        XSamSEC.template leftCols<2*N+1>() = XPred;
        XSamSEC.template rightCols<2*N>() = XTempQ;

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
        Wc2(0) = lam/(2.0*N + lam) + 1.0 - param.alpha*param.alpha + param.beta;

        Eigen::Vector3d ZPredAve = ZPred * Wm2;

        Eigen::Matrix3d PXZ = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d PZZ = R_mea2;
        for (int j = 0; j < TOTAL; ++j) {
            Eigen::Vector3d diff_x = XSamSEC.col(j) - XPredAve;
            Eigen::Vector3d diff_z = ZPred.col(j) - ZPredAve;
            PXZ.noalias() += Wc2(j) * (diff_x * diff_z.transpose());
            PZZ.noalias() += Wc2(j) * (diff_z * diff_z.transpose());
        }

        Eigen::Matrix3d K = PXZ * PZZ.inverse();
        X = XPredAve + K * (Z_w - ZPredAve);
        P = PXX - K * PZZ * K.transpose();
    }
};

// ── UKF1: 4 状态四元数 UKF ────────────────────────────────────────────────────
class UKF1_Quaternion {
public:
    static constexpr int N = 4;

    Eigen::Vector4d X;
    Eigen::Matrix4d P;
    UKFParams param;
    Eigen::Matrix4d Q1;
    Eigen::Matrix4d R_mea1;

    UKF1_Quaternion(const Eigen::Vector4d& X0,
                    const Eigen::Matrix4d& P0,
                    const Eigen::Matrix4d& Q,
                    const Eigen::Matrix4d& R_meas,
                    double alpha = 1.0, double beta = 2.0)
        : X(X0), P(P0), param(N, alpha, beta), Q1(Q), R_mea1(R_meas) {
        X = quat_normalize(X);
    }

    Eigen::Matrix<double, N, 2*N+1> make_sigma() const {
        Eigen::Matrix4d P_reg = P + 1e-10 * Eigen::Matrix4d::Identity();
        Eigen::LLT<Eigen::Matrix4d> llt(P_reg);
        Eigen::Matrix4d choP = param.gamma * llt.matrixL().toDenseMatrix().transpose();

        Eigen::Matrix<double, N, 2*N+1> XSam;
        XSam.col(0) = X;
        for (int j = 0; j < N; ++j) {
            XSam.col(1 + j)     = X + choP.col(j);
            XSam.col(1 + N + j) = X - choP.col(j);
        }
        return XSam;
    }

    void step(const Eigen::Vector4d& Z_q,
              const Eigen::Vector3d& w_est,
              double dt) {
        double lam = param.lambda;

        Eigen::VectorXd Wm(2*N+1), Wc(2*N+1);
        Wm.setConstant(1.0/(2.0*(N + lam)));
        Wc.setConstant(1.0/(2.0*(N + lam)));
        Wm(0) = lam/(N + lam);
        Wc(0) = lam/(N + lam) + 1.0 - param.alpha*param.alpha + param.beta;

        Eigen::Matrix<double, N, 2*N+1> XSam = make_sigma();

        Eigen::Matrix<double, N, 2*N+1> XPred;
        for (int j = 0; j < 2*N + 1; ++j) {
            XPred.col(j) = xStateFunction1(XSam.col(j), w_est, dt);
        }

        Eigen::Vector4d XPredAve = XPred * Wm;
        XPredAve = quat_normalize(XPredAve);

        Eigen::Matrix4d PXX = Q1;
        for (int j = 0; j < 2*N + 1; ++j) {
            Eigen::Vector4d diff = XPred.col(j) - XPredAve;
            PXX.noalias() += Wc(j) * (diff * diff.transpose());
        }

        // 第二轮: Q 采样
        Eigen::Matrix4d Q_reg = Q1 + 1e-18 * Eigen::Matrix4d::Identity();
        Eigen::LLT<Eigen::Matrix4d> lltQ(Q_reg);
        Eigen::Matrix4d choQ = param.gamma * lltQ.matrixL().toDenseMatrix().transpose();

        Eigen::Matrix<double, N, 2*N> XTempQ;
        for (int j = 0; j < N; ++j) {
            XTempQ.col(j)     = XPred.col(0) - choQ.col(j);
            XTempQ.col(N + j) = XPred.col(0) + choQ.col(j);
        }

        static constexpr int TOTAL = 4*N + 1;
        Eigen::Matrix<double, N, TOTAL> XSamSEC;
        XSamSEC.template leftCols<2*N+1>() = XPred;
        XSamSEC.template rightCols<2*N>() = XTempQ;

        Eigen::Matrix<double, N, TOTAL> ZPred;
        for (int j = 0; j < TOTAL; ++j) {
            ZPred.col(j) = XSamSEC.col(j);
        }

        Eigen::VectorXd Wm2(TOTAL), Wc2(TOTAL);
        Wm2.setConstant(1.0/(2.0*(2.0*N + lam)));
        Wc2.setConstant(1.0/(2.0*(2.0*N + lam)));
        Wm2(0) = lam/(2.0*N + lam);
        Wc2(0) = lam/(2.0*N + lam) + 1.0 - param.alpha*param.alpha + param.beta;

        Eigen::Vector4d ZPredAve = ZPred * Wm2;

        Eigen::Matrix4d PXZ = Eigen::Matrix4d::Zero();
        Eigen::Matrix4d PZZ = R_mea1;
        for (int j = 0; j < TOTAL; ++j) {
            Eigen::Vector4d diff_x = XSamSEC.col(j) - XPredAve;
            Eigen::Vector4d diff_z = ZPred.col(j) - ZPredAve;
            PXZ.noalias() += Wc2(j) * (diff_x * diff_z.transpose());
            PZZ.noalias() += Wc2(j) * (diff_z * diff_z.transpose());
        }

        Eigen::Matrix4d K = PXZ * PZZ.inverse();
        X = XPredAve + K * (Z_q - ZPredAve);
        P = PXX - K * PZZ * K.transpose();
    }
};

// ── CascadedUKF: 级联双 UKF ──────────────────────────────────────────────────
class CascadedUKF {
public:
    UKF2_AngularVelocity ukf_w;
    UKF1_Quaternion ukf_q;

    CascadedUKF(const Eigen::Vector3d& w0, const Eigen::Matrix3d& Pw0,
                const Eigen::Vector4d& q0, const Eigen::Matrix4d& Pq0,
                const Eigen::Matrix3d& Q_w, const Eigen::Matrix3d& R_w,
                const Eigen::Matrix4d& Q_q, const Eigen::Matrix4d& R_q)
        : ukf_w(w0, Pw0, Q_w, R_w)
        , ukf_q(q0, Pq0, Q_q, R_q) {}

    void step(const Eigen::Vector3d& Z_w,
              const Eigen::Vector4d& Z_q,
              const Eigen::Vector3d& T_torque,
              const RigidBodyParams& params) {
        ukf_w.step(Z_w, T_torque, params);
        ukf_q.step(Z_q, ukf_w.X, params.dt);
    }

    Eigen::Vector3d get_w() const { return ukf_w.X; }
    Eigen::Vector4d get_q() const { return ukf_q.X; }
};

} // namespace jt_ekf
