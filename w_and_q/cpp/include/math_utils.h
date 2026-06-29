#pragma once
/// math_utils.h —— 反对称矩阵 / 叉乘 / Rodrigues 旋转 / 四元数→DCM
/// 完全对齐 MATLAB w_and_q/ 目录中的 tilde.m, cross.m, aw.m, quat2cosmatrix.m

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

namespace jt_ekf {

// ── tilde(A) : 3x1 → 3x3 反对称矩阵 ──────────────────────────────────────────
// 对齐 MATLAB tilde.m
inline Eigen::Matrix3d tilde(const Eigen::Vector3d& a) {
    Eigen::Matrix3d B;
    B <<  0.0,  -a.z(),  a.y(),
          a.z(),  0.0,   -a.x(),
         -a.y(),  a.x(),  0.0;
    return B;
}

// ── cross(u, v) : 3D 叉乘 ────────────────────────────────────────────────────
// 对齐 MATLAB cross.m
inline Eigen::Vector3d cross_vec(const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
    return u.cross(v);
}

// ── aw(w0) : 角速度 → 旋转矩阵 (Rodrigues formula) ────────────────────────────
// 对齐 MATLAB aw.m
// E0 = exp( tilde(w0) * dt )   ≈  I + tilde(w)*sin(th) + tilde(w)^2*(1-cos(th))
inline Eigen::Matrix3d aw(const Eigen::Vector3d& w0, double dt) {
    double norm_w = w0.norm();
    if (norm_w < 1e-30) {
        return Eigen::Matrix3d::Identity();
    }
    double th = norm_w * dt;
    Eigen::Vector3d w = w0 / norm_w;
    double c = std::cos(th);
    double s = std::sin(th);

    Eigen::Matrix3d E0;
    E0(0,0) = c + w.x()*w.x()*(1.0 - c);
    E0(0,1) = w.x()*w.y()*(1.0 - c) - w.z()*s;
    E0(0,2) = w.z()*w.x()*(1.0 - c) + w.y()*s;

    E0(1,0) = w.x()*w.y()*(1.0 - c) + w.z()*s;
    E0(1,1) = c + w.y()*w.y()*(1.0 - c);
    E0(1,2) = w.z()*w.y()*(1.0 - c) - w.x()*s;

    E0(2,0) = w.z()*w.x()*(1.0 - c) - w.y()*s;
    E0(2,1) = w.z()*w.y()*(1.0 - c) + w.x()*s;
    E0(2,2) = c + w.z()*w.z()*(1.0 - c);
    return E0;
}

// ── quat2cosmatrix(q) : 四元数 → 方向余弦矩阵 ──────────────────────────────────
// 对齐 MATLAB quat2cosmatrix.m
inline Eigen::Matrix3d quat2cosmatrix(const Eigen::Vector4d& q) {
    Eigen::Vector3d qv = q.head<3>();
    double qw = q.w();
    Eigen::Matrix<double, 3, 4> R1, R2;
    R1.leftCols<3>()  = (-qv).asDiagonal();
    R1.rightCols<1>() = tilde(qv) + qw * Eigen::Matrix3d::Identity();
    R1.col(3) = qv;
    R2.leftCols<3>()  = (-qv).asDiagonal();
    R2.rightCols<1>() = -tilde(qv) + qw * Eigen::Matrix3d::Identity();
    R2.col(3) = qv;
    return R1 * R2.transpose();
}

} // namespace jt_ekf
