#pragma once
/// quaternion.h —— 四元数基本运算
/// 对齐 MATLAB QuanMatrix.m (Omega矩阵), 四元数归一化, 四元数乘法

#include <Eigen/Dense>
#include <cmath>
#include "math_utils.h"

namespace jt_ekf {

// ── 四元数归一化 ─────────────────────────────────────────────────────────────
inline Eigen::Vector4d quat_normalize(const Eigen::Vector4d& q) {
    return q.normalized();
}

// ── Omega(w) 矩阵 (4x4) : QuanMatrix.m ────────────────────────────────────────
// Omega(w) = [ -tilde(w)  w ]
//            [   -w^T     0 ]
// 使得 dq/dt = 0.5 * Omega(w) * q
inline Eigen::Matrix4d Omega(const Eigen::Vector3d& w) {
    Eigen::Matrix4d O = Eigen::Matrix4d::Zero();
    O.block<3,3>(0,0) = -tilde(w);
    O.block<3,1>(0,3) = w;
    O.block<1,3>(3,0) = -w.transpose();
    return O;
}

// ── 四元数乘法 q_out = QuanMatrix(q2) * q1 ────────────────────────────────────
// 注意：QuanMatrix 在 MATLAB 中根据输入维度不同有两种行为：
//   输入 3D → 用 [w;0] 构造；输入 4D → 直接用
// 这里拆成两个重载。
inline Eigen::Matrix4d QuanMatrix_from_w(const Eigen::Vector3d& w) {
    return Omega(w);  // 与 MATLAB 一致：n = w4*I + [-tilde(w) w; -w' 0]
    // 但 MATLAB 里 w4=0，所以 n = [-tilde(w) w; -w' 0]，等价于 Omega(w)
}

inline Eigen::Matrix4d QuanMatrix_from_q(const Eigen::Vector4d& q) {
    double qw = q.w();
    Eigen::Vector3d qv = q.head<3>();
    Eigen::Matrix4d n;
    n.block<3,3>(0,0) = qw * Eigen::Matrix3d::Identity() - tilde(qv);
    n.block<3,1>(0,3) = qv;
    n.block<1,3>(3,0) = -qv.transpose();
    n(3,3) = qw;
    return n;
}

// ── 四元数乘法 (实用版): r = p ⊗ q ────────────────────────────────────────────
inline Eigen::Vector4d quat_multiply(const Eigen::Vector4d& p, const Eigen::Vector4d& q) {
    return QuanMatrix_from_q(p) * q;
}

// ── MRP (修正罗德里格斯参数) ↔ 四元数 ─────────────────────────────────────────
// 对齐 MATLAB Main.m 中使用的参数化：x = q(1:3), q(4) = sqrt(1 - x'*x)
inline Eigen::Vector3d quat_to_mrp(const Eigen::Vector4d& q) {
    return q.head<3>();
}

inline Eigen::Vector4d mrp_to_quat(const Eigen::Vector3d& mrp) {
    double sq = mrp.squaredNorm();
    double qw = std::sqrt(std::max(1.0 - sq, 0.0));
    Eigen::Vector4d q;
    q << mrp, qw;
    return q;
}

// ── 误差四元数 (delta_q): 用于 EKF 中乘性姿态更新 ──────────────────────────────
// 对齐 Main.m 中的 delta q 计算
inline Eigen::Vector4d delta_quat_from_mrp(const Eigen::Vector3d& delta_x) {
    double sq = delta_x.squaredNorm();
    double qw = std::sqrt(std::max(1.0 - sq, 0.0));
    Eigen::Vector4d dq;
    dq << delta_x, qw;
    return dq;
}

} // namespace jt_ekf
