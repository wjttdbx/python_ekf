#pragma once
/// ekf_6state.h —— 6 状态 EKF (MRP姿态 + 角速度)
/// 对齐 MATLAB Main.m (EKF 部分)

#include <Eigen/Dense>
#include "math_utils.h"
#include "quaternion.h"
#include "dynamics.h"

namespace jt_ekf {

// ── RecursionFunction: 状态转移矩阵 phi 和系统矩阵 F ──────────────────────────
// 对齐 MATLAB RecursionFunction.m
// 输入 xForm = [q(1:3); w(1:3)]  (6D, MRP参数化)
inline std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 6>>
RecursionFunction(const Eigen::Matrix<double, 6, 1>& xForm,
                  const RigidBodyParams& params) {
    Eigen::Vector3d qForm = xForm.head<3>();
    Eigen::Vector3d wForm = xForm.tail<3>();

    // 惯量比参数
    double Ixx = params.inertia(0, 0);
    double Iyy = params.inertia(1, 1);
    double Izz = params.inertia(2, 2);
    double p1 = (Iyy - Izz) / Ixx;
    double p2 = (Izz - Ixx) / Iyy;
    double p3 = (Ixx - Iyy) / Izz;

    // 系统矩阵 F (6x6)
    Eigen::Matrix<double, 6, 6> FX = Eigen::Matrix<double, 6, 6>::Zero();

    // FX(1:3, 1:3) = -tilde(w)  ——  姿态对姿态的偏导
    FX.block<3, 3>(0, 0) = -tilde(wForm);

    // FX(1:3, 4:6) = 0.5 * I_3  ——  角速度对姿态的贡献
    FX.block<3, 3>(0, 3) = 0.5 * Eigen::Matrix3d::Identity();

    // FX(4:6, 4:6) = [0  -p1  p1; p2  0  -p2; -p3  p3  0] * tilde(w)
    //  —— 角速度对角速度的偏导 (Euler方程线性化)
    Eigen::Matrix3d M;
    M << 0.0,  -p1,   p1,
          p2,  0.0,  -p2,
         -p3,   p3,  0.0;
    FX.block<3, 3>(3, 3) = M * tilde(wForm);

    // 状态转移矩阵: phi = I + dt * F (一阶泰勒离散化)
    Eigen::Matrix<double, 6, 6> phiX = Eigen::Matrix<double, 6, 6>::Identity()
                                        + params.dt * FX;

    return {phiX, FX};
}

// ── calc_Q: 过程噪声协方差矩阵 ─────────────────────────────────────────────────
// 对齐 MATLAB calc_Q.m
inline Eigen::Matrix<double, 6, 6> calc_Q(const Eigen::Matrix<double, 6, 1>& x,
                                           const RigidBodyParams& params,
                                           double var_proc_scale) {
    // B = diag(1./I)  —— 控制输入矩阵
    Eigen::Vector3d inv_diag = params.inv_inertia.diagonal();
    Eigen::Matrix3d B = inv_diag.asDiagonal();

    auto [phiX, FX] = RecursionFunction(x, params);
    Eigen::Matrix3d M_block = FX.block<3, 3>(3, 3);

    double dt = params.dt;
    Eigen::Matrix<double, 6, 6> QQ = Eigen::Matrix<double, 6, 6>::Zero();

    Eigen::Matrix3d BB = B * B;  // B * B^T (对角)

    // QQ(1:3, 1:3) = B*B^T * dt^3 / 12
    QQ.block<3, 3>(0, 0) = BB * (dt * dt * dt) / 12.0;

    // QQ(1:3, 4:6) = B*B^T * M^T / 6 * dt^3 + B*B^T / 4 * dt^2
    QQ.block<3, 3>(0, 3) = BB * M_block.transpose() / 6.0 * (dt * dt * dt)
                           + BB / 4.0 * (dt * dt);

    // QQ(4:6, 1:3) = QQ(1:3, 4:6)^T
    QQ.block<3, 3>(3, 0) = QQ.block<3, 3>(0, 3).transpose();

    // QQ(4:6, 4:6) = M*BB*M^T * dt^3/3 + dt^2/2*(BB*M^T + M*BB) + BB*dt
    QQ.block<3, 3>(3, 3) = M_block * BB * M_block.transpose() * (dt * dt * dt) / 3.0
                           + (dt * dt) / 2.0 * (BB * M_block.transpose() + M_block * BB)
                           + BB * dt;

    return 2.0 * var_proc_scale * QQ;
}

// ── Deriv: 状态导数计算 ───────────────────────────────────────────────────────
// 对齐 MATLAB Deriv.m (用于 EKF 预测)
inline std::pair<Eigen::Matrix<double, 6, 1>, double>
Deriv(const Eigen::Matrix<double, 6, 1>& xGuess,
      const Eigen::Vector4d& qGuess,
      const Eigen::Vector3d& Toe,
      const RigidBodyParams& params) {
    Eigen::Vector3d wGuess = xGuess.tail<3>();
    Eigen::Matrix<double, 6, 1> xDot;

    // 四元数导数: dq/dt = 0.5 * Omega(w) * q
    Eigen::Vector4d qDot = 0.5 * Omega(wGuess) * qGuess;

    // xDot(1:4) 存储四元数导数
    xDot.head<4>() = qDot;
    double q0chang = qDot(3);  // qw 的导数

    // 角加速度: J * dw/dt = Toe - cross(w, J*w)
    Eigen::Vector3d wd = params.inv_inertia * (Toe - cross_vec(wGuess, params.inertia * wGuess));
    xDot.tail<3>() = wd;

    // 注意：EKF 只用前3个MRP分量，但这里返回完整4D四元数导数
    // 实际使用 xDot(1:3) 和 xDot(4:6)=wd
    return {xDot, q0chang};
}

// ── EKF6State: 6状态 EKF 滤波器 ────────────────────────────────────────────────
// 对齐 MATLAB Main.m 中的 EKF 循环
class EKF6State {
public:
    // 状态: x = [q_mrp(3); w(3)]  — 6D
    Eigen::Matrix<double, 6, 1> x;      // 状态估计
    Eigen::Matrix<double, 6, 6> P;      // 协方差矩阵
    Eigen::Vector4d q;                   // 完整四元数 (用于传播)

    EKF6State(const Eigen::Matrix<double, 6, 1>& x0,
              const Eigen::Matrix<double, 6, 6>& P0,
              const Eigen::Vector4d& q0)
        : x(x0), P(P0), q(q0) {}

    // ── 更新步 (每 interval 步执行一次) ──
    void update(const Eigen::Matrix<double, 6, 1>& z,
                const Eigen::Matrix<double, 6, 6>& R_meas,
                const Eigen::Matrix<double, 6, 6>& H) {
        // K = P * H^T / (H * P * H^T + R)
        Eigen::Matrix<double, 6, 6> S = H * P * H.transpose() + R_meas;
        Eigen::Matrix<double, 6, 6> K = P * H.transpose() * S.inverse();

        Eigen::Matrix<double, 6, 1> hPred = H * x;
        Eigen::Matrix<double, 6, 1> xDelta = K * (z - hPred);

        // 角速度更新：直接加
        x.tail<3>() += xDelta.tail<3>();

        // 姿态更新：乘性 (delta_q ⊗ q_pred)
        Eigen::Vector3d d_mrp = xDelta.head<3>();
        Eigen::Vector4d dq = delta_quat_from_mrp(d_mrp);
        q = quat_multiply(dq, q);  // 乘性误差四元数更新
        q = quat_normalize(q);
        x.head<3>() = q.head<3>(); // 更新 MRP 表示

        // 协方差更新
        P = (Eigen::Matrix<double, 6, 6>::Identity() - K * H) * P;
    }

    // ── 预测步 ──
    void predict(const Eigen::Vector3d& Toe,
                 const RigidBodyParams& params,
                 double var_proc_scale) {
        // 1. 计算状态转移矩阵和过程噪声
        auto [phiX, FX] = RecursionFunction(x, params);
        Eigen::Matrix<double, 6, 6> Q_proc = calc_Q(x, params, var_proc_scale);

        // 2. 状态导数
        auto [xDot, q0Dot] = Deriv(x, q, Toe, params);

        // 3. 预测
        // xPred = x + xDot * dt (MRP + w)
        x.head<3>() += xDot.head<3>() * params.dt;
        x.tail<3>() += params.inv_inertia * (Toe - cross_vec(x.tail<3>(),
                                             params.inertia * x.tail<3>())) * params.dt;

        // 四元数预测 (完整四元数)
        Eigen::Vector4d qPred_raw;
        qPred_raw.head<3>() = x.head<3>();
        qPred_raw(3) = q(3) + q0Dot * params.dt;
        q = quat_normalize(qPred_raw);
        x.head<3>() = q.head<3>();

        // 协方差预测
        P = phiX * P * phiX.transpose() + Q_proc;
    }

    // ── 完整一步 (预测 + 可选更新) ──
    void step(const Eigen::Vector3d& Toe,
              const Eigen::Matrix<double, 6, 1>& z,
              const Eigen::Matrix<double, 6, 6>& R_meas,
              int step_idx, int update_interval,
              const RigidBodyParams& params,
              double var_proc_scale) {
        // 先预测
        auto [phiX, FX] = RecursionFunction(x, params);
        Eigen::Matrix<double, 6, 6> Q_proc = calc_Q(x, params, var_proc_scale);
        auto [xDot, q0Dot] = Deriv(x, q, Toe, params);

        Eigen::Matrix<double, 6, 1> xPred = x;
        xPred.head<3>() += xDot.head<3>() * params.dt;
        xPred.tail<3>() += params.inv_inertia * (Toe - cross_vec(x.tail<3>(),
                                                  params.inertia * x.tail<3>())) * params.dt;

        Eigen::Vector4d qPred_raw;
        qPred_raw.head<3>() = xPred.head<3>();
        qPred_raw(3) = q(3) + q0Dot * params.dt;
        Eigen::Vector4d qPred = quat_normalize(qPred_raw);
        xPred.head<3>() = qPred.head<3>();

        Eigen::Matrix<double, 6, 6> PPred = phiX * P * phiX.transpose() + Q_proc;

        // 条件更新
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
        if (step_idx % update_interval == 0) {
            Eigen::Matrix<double, 6, 6> S = H * PPred * H.transpose() + R_meas;
            Eigen::Matrix<double, 6, 6> K = PPred * H.transpose() * S.inverse();

            Eigen::Matrix<double, 6, 1> hPred = H * xPred;
            Eigen::Matrix<double, 6, 1> xDelta = K * (z - hPred);

            x = xPred;
            x.tail<3>() += xDelta.tail<3>();

            Eigen::Vector3d d_mrp = xDelta.head<3>();
            Eigen::Vector4d dq = delta_quat_from_mrp(d_mrp);
            q = quat_multiply(dq, qPred);
            q = quat_normalize(q);
            x.head<3>() = q.head<3>();

            P = (Eigen::Matrix<double, 6, 6>::Identity() - K * H) * PPred;
        } else {
            x = xPred;
            q = qPred;
            P = PPred;
        }
    }
};

} // namespace jt_ekf
