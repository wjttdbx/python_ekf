#pragma once
/// dynamics.h —— 刚体动力学 + 四元数运动学 (RK4 积分)
/// 对齐 MATLAB f_dyn_U.m, f_dyn_U2.m, xStateFunction.m, xStateFunction1.m, xStateFunction2.m

#include <Eigen/Dense>
#include "math_utils.h"
#include "quaternion.h"

namespace jt_ekf {

// ── 刚体状态结构体 ────────────────────────────────────────────────────────────
struct RigidBodyState {
    Eigen::Vector3d R;       // 位置 (km)
    Eigen::Matrix3d A;       // 姿态矩阵 (DCM)
    Eigen::Vector3d v;       // 速度 (km/s)
    Eigen::Vector3d w;       // 角速度 (rad/s)
    Eigen::Vector4d q;       // 姿态四元数 [qx, qy, qz, qw]
};

// ── 刚体动力学参数 ────────────────────────────────────────────────────────────
struct RigidBodyParams {
    double mass;                    // 质量 (kg)
    Eigen::Matrix3d inertia;        // 转动惯量矩阵
    Eigen::Matrix3d inv_inertia;    // 转动惯量逆矩阵
    double dt;                      // 仿真步长

    RigidBodyParams(double m = 10.0,
                    const Eigen::Matrix3d& J = Eigen::Matrix3d::Identity(),
                    double time_step = 0.01)
        : mass(m), inertia(J), inv_inertia(J.inverse()), dt(time_step) {}
};

// ── f_dyn_U: 计算力和力矩下的加速度 ────────────────────────────────────────────
// 对齐 MATLAB f_dyn_U.m
// 返回: (vd, wd) 速度和角速度的时间导数
inline std::pair<Eigen::Vector3d, Eigen::Vector3d>
f_dyn_U(const Eigen::Vector3d& R, const Eigen::Vector3d& R_touch_e,
        const Eigen::Vector3d& v, const Eigen::Vector3d& w,
        const Eigen::Vector3d& F, const Eigen::Vector3d& T,
        const RigidBodyParams& params) {
    // 线加速度: a = F / m
    Eigen::Vector3d vd = F / params.mass;

    // 外力矩 + 力对质心的附加力矩: T_ex = T + cross(R_touch_e - R, F)
    Eigen::Vector3d T_ex = T + cross_vec(R_touch_e - R, F);

    // 角加速度: J * dw/dt = T_ex - cross(w, J*w)
    // MATLAB: wd = inertia \ (T_ex - cross(w, inertia*w))
    Eigen::Vector3d wd = params.inv_inertia * (T_ex - cross_vec(w, params.inertia * w));

    return {vd, wd};
}

// ── f_dyn_U2: RK4 积分 (完整状态传播) ─────────────────────────────────────────
// 对齐 MATLAB f_dyn_U2.m
inline RigidBodyState f_dyn_U2(const RigidBodyState& state,
                                const Eigen::Vector3d& R_touch_e,
                                const Eigen::Vector3d& F,
                                const Eigen::Vector3d& T,
                                const RigidBodyParams& params) {
    double dt = params.dt;
    RigidBodyState s = state;
    auto compute_derivs = [&](const RigidBodyState& ss) {
        return f_dyn_U(ss.R, R_touch_e, ss.v, ss.w, F, T, params);
    };

    // ── Step 1 ──
    auto [vd1, wd1] = compute_derivs(s);
    Eigen::Vector3d k1_R  = dt * s.v;
    Eigen::Matrix3d kk1_A = aw(s.w, dt) - Eigen::Matrix3d::Identity();
    // 注意：aw 返回的是 exp(tilde(w)*dt)，而 k_A = A * tilde(w) * dt ≈ A*(aw - I)
    // MATLAB 中: k_A = A * tilde(w) * dt, kk_A = aw(w) - I  (近似)
    Eigen::Vector3d k1_v  = dt * vd1;
    Eigen::Vector3d k1_w  = dt * wd1;
    Eigen::Vector4d k1_q  = 0.5 * dt * Omega(s.w) * s.q;
    Eigen::Matrix3d k1_A  = dt * s.A * tilde(s.w);

    // ── Step 2 ──
    RigidBodyState s2 = s;
    s2.R = s.R + 0.5 * k1_R;
    s2.A = s.A + 0.5 * k1_A;  // 注意：这里用 k1_A 而非 kk1_A（MATLAB原意）
    s2.v = s.v + 0.5 * k1_v;
    s2.w = s.w + 0.5 * k1_w;
    auto [vd2, wd2] = compute_derivs(s2);
    Eigen::Vector3d k2_R  = dt * s2.v;  // MATLAB 用 vU + k1_v/2，等价
    Eigen::Matrix3d kk2_A = aw(s2.w, dt) - Eigen::Matrix3d::Identity();
    Eigen::Vector3d k2_v  = dt * vd2;
    Eigen::Vector3d k2_w  = dt * wd2;
    Eigen::Vector4d k2_q  = 0.5 * dt * Omega(s2.w) * s.q;
    Eigen::Matrix3d k2_A  = dt * s.A * tilde(s2.w);

    // ── Step 3 ──
    RigidBodyState s3 = s;
    s3.R = s.R + 0.5 * k2_R;
    s3.A = s.A + 0.5 * k2_A;
    s3.v = s.v + 0.5 * k2_v;
    s3.w = s.w + 0.5 * k2_w;
    auto [vd3, wd3] = compute_derivs(s3);
    Eigen::Vector3d k3_R  = dt * s3.v;
    Eigen::Matrix3d kk3_A = aw(s3.w, dt) - Eigen::Matrix3d::Identity();
    Eigen::Vector3d k3_v  = dt * vd3;
    Eigen::Vector3d k3_w  = dt * wd3;
    Eigen::Vector4d k3_q  = 0.5 * dt * Omega(s3.w) * s.q;
    Eigen::Matrix3d k3_A  = dt * s.A * tilde(s3.w);

    // ── Step 4 ──
    RigidBodyState s4 = s;
    s4.R = s.R + k3_R;
    s4.A = s.A + k3_A;
    s4.v = s.v + k3_v;
    s4.w = s.w + k3_w;
    auto [vd4, wd4] = compute_derivs(s4);
    Eigen::Vector3d k4_R  = dt * s4.v;
    Eigen::Matrix3d kk4_A = aw(s4.w, dt) - Eigen::Matrix3d::Identity();
    Eigen::Vector3d k4_v  = dt * vd4;
    Eigen::Vector3d k4_w  = dt * wd4;
    Eigen::Vector4d k4_q  = 0.5 * dt * Omega(s4.w) * s.q;
    Eigen::Matrix3d k4_A  = dt * s.A * tilde(s4.w);

    // ── RK4 加权平均 ──
    RigidBodyState next;
    next.R = s.R + (k1_R + 2.0*k2_R + 2.0*k3_R + k4_R) / 6.0;
    next.v = s.v + (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v) / 6.0;
    next.w = s.w + (k1_w + 2.0*k2_w + 2.0*k3_w + k4_w) / 6.0;
    next.A = s.A + (kk1_A + 2.0*kk2_A + 2.0*kk3_A + kk4_A) / 6.0;
    // 四元数更新后归一化
    Eigen::Vector4d q_next_raw = s.q + (k1_q + 2.0*k2_q + 2.0*k3_q + k4_q) / 6.0;
    next.q = quat_normalize(q_next_raw);

    return next;
}

// ── xStateFunction: 7D 状态传播 (q + w) 用于 UKF ───────────────────────────────
// 对齐 MATLAB xStateFunction.m
inline Eigen::Matrix<double, 7, 1> xStateFunction(const Eigen::Matrix<double, 7, 1>& X,
                                                    const Eigen::Vector3d& T,
                                                    const RigidBodyParams& params) {
    Eigen::Vector4d q = X.head<4>();
    Eigen::Vector3d w = X.tail<3>();

    // 角加速度: J * dw/dt = T - cross(w, J*w)
    Eigen::Vector3d wd = params.inv_inertia * (T - cross_vec(w, params.inertia * w));
    Eigen::Vector3d w_new = w + params.dt * wd;

    // 四元数传播: dq/dt = 0.5 * Omega(w) * q
    Eigen::Vector4d q_new_raw = q + 0.5 * params.dt * Omega(w) * q;
    Eigen::Vector4d q_new = quat_normalize(q_new_raw);

    Eigen::Matrix<double, 7, 1> XNew;
    XNew << q_new, w_new;
    return XNew;
}

// ── xStateFunction1: 4D 四元数传播 (给定 w) ─────────────────────────────────────
// 对齐 MATLAB xStateFunction1.m
inline Eigen::Vector4d xStateFunction1(const Eigen::Vector4d& q,
                                        const Eigen::Vector3d& w,
                                        double dt) {
    Eigen::Vector4d q_new_raw = q + 0.5 * dt * Omega(w) * q;
    return quat_normalize(q_new_raw);
}

// ── xStateFunction2: 3D 角速度传播 ─────────────────────────────────────────────
// 对齐 MATLAB xStateFunction2.m
inline Eigen::Vector3d xStateFunction2(const Eigen::Vector3d& w,
                                        const Eigen::Vector3d& T,
                                        const RigidBodyParams& params) {
    Eigen::Vector3d wd = params.inv_inertia * (T - cross_vec(w, params.inertia * w));
    return w + params.dt * wd;
}

// ── 外部力矩生成 (用于仿真) ─────────────────────────────────────────────────────
// 对齐 MATLAB Main.m 中的 T = Toe + ProcNoise
inline Eigen::Vector3d external_torque(double time) {
    double t = time;
    double scale = 0.1 * (0.5 * std::sin(M_PI/5.0 * t) +
                          0.2 * std::cos(M_PI/2.0 * t) +
                          0.1 * std::sin(M_PI * t));
    return Eigen::Vector3d::Constant(scale);
}

} // namespace jt_ekf
