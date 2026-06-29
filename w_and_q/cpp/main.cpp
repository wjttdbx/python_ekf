/// main.cpp —— JT-EKF/UKF 姿态估计仿真程序
/// 对齐 MATLAB w_and_q/ 目录下的 Main.m, main2.m, main3.m
///
/// 三种滤波器对比:
///   1. EKF (6-state MRP)        — 对齐 Main.m
///   2. Joint UKF (7-state)      — 对齐 main2.m + UKF.m
///   3. Cascaded UKF (UKF2→UKF1) — 对齐 main3.m + UKF1.m + UKF2.m
///
/// 真实动力学: RK4 积分刚体转动 + 四元数运动学

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

// 包含所有滤波器实现
#include "math_utils.h"
#include "quaternion.h"
#include "dynamics.h"
#include "ekf_6state.h"
#include "ukf_7state.h"
#include "cascaded_ukf.h"

using namespace jt_ekf;

// ── 仿真配置 ──────────────────────────────────────────────────────────────────
struct SimConfig {
    double dt         = 0.1;       // 时间步长 (s)
    double total_time = 50.0;      // 总仿真时间 (s)
    double mass       = 10.0;      // 质量 (kg)
    int    update_interval = 1;    // EKF 更新间隔
};

// ── 输出 CSV 文件 ─────────────────────────────────────────────────────────────
void write_csv(const std::string& filename,
               const std::vector<double>& time,
               const std::vector<Eigen::Vector4d>& q_true,
               const std::vector<Eigen::Vector3d>& w_true,
               const std::vector<Eigen::Vector4d>& q_est,
               const std::vector<Eigen::Vector3d>& w_est,
               const std::vector<Eigen::Vector4d>& q_meas,
               const std::vector<Eigen::Vector3d>& w_meas) {
    std::ofstream f(filename);
    f << "time,q_true_x,q_true_y,q_true_z,q_true_w,"
      << "w_true_x,w_true_y,w_true_z,"
      << "q_est_x,q_est_y,q_est_z,q_est_w,"
      << "w_est_x,w_est_y,w_est_z,"
      << "q_meas_x,q_meas_y,q_meas_z,q_meas_w,"
      << "w_meas_x,w_meas_y,w_meas_z\n";
    f << std::scientific << std::setprecision(8);
    for (size_t i = 0; i < time.size(); ++i) {
        f << time[i] << ","
          << q_true[i].x() << "," << q_true[i].y() << "," << q_true[i].z() << "," << q_true[i].w() << ","
          << w_true[i].x() << "," << w_true[i].y() << "," << w_true[i].z() << ","
          << q_est[i].x()  << "," << q_est[i].y()  << "," << q_est[i].z()  << "," << q_est[i].w()  << ","
          << w_est[i].x()  << "," << w_est[i].y()  << "," << w_est[i].z()  << ","
          << q_meas[i].x() << "," << q_meas[i].y() << "," << q_meas[i].z() << "," << q_meas[i].w() << ","
          << w_meas[i].x() << "," << w_meas[i].y() << "," << w_meas[i].z() << "\n";
    }
    std::cout << "  输出: " << filename << " (" << time.size() << " 行)" << std::endl;
}

// ── 打印 RMSE 统计 ────────────────────────────────────────────────────────────
void print_rmse(const std::string& label,
                const std::vector<Eigen::Vector4d>& q_true,
                const std::vector<Eigen::Vector3d>& w_true,
                const std::vector<Eigen::Vector4d>& q_est,
                const std::vector<Eigen::Vector3d>& w_est) {
    double q_err_sum = 0.0, w_err_sum = 0.0;
    size_t N = std::min({q_true.size(), q_est.size(), w_true.size(), w_est.size()});
    for (size_t i = 0; i < N; ++i) {
        // 四元数误差: 2*acos(|dot(q_true, q_est)|)  (弧度)
        double dot = std::abs(q_true[i].dot(q_est[i]));
        dot = std::min(dot, 1.0);
        q_err_sum += 2.0 * std::acos(dot) * 180.0 / M_PI;  // 转换为度

        w_err_sum += (w_true[i] - w_est[i]).norm();
    }
    std::cout << "  [" << label << "] 姿态 RMSE: " << q_err_sum/N << " deg, "
              << "角速度 RMSE: " << w_err_sum/N << " rad/s" << std::endl;
}

// ── 运行级联 UKF 仿真 (对齐 main3.m) ──────────────────────────────────────────
void run_cascaded_ukf(const SimConfig& cfg) {
    std::cout << "\n========== Cascaded UKF (UKF2→UKF1, 对齐 main3.m) ==========\n";

    // ── 参数 ──
    RigidBodyParams params(cfg.mass, Eigen::Matrix3d::Identity(), cfg.dt);

    // 测量噪声方差
    Eigen::Vector3d VarOfMear1_vec(1e-2 * 1.0, 1e-2 * 1.5, 1e-2 * 2.0);
    double         VarOfMear1_qw   = 1e-2 * 1.0;
    Eigen::Vector3d VarOfMear2_vec(1e-4 * 1.3, 1e-4 * 3.0, 1e-4 * 4.0);

    Eigen::Vector3d VarOfProc2_vec(1e-6 * 1.0, 1e-6 * 2.0, 1e-6 * 1.0);

    // 测量噪声协方差
    Eigen::Matrix4d RMea1 = Eigen::Matrix4d::Zero();
    RMea1.diagonal() << VarOfMear1_vec, VarOfMear1_qw;

    Eigen::Matrix3d RMea2 = VarOfMear2_vec.asDiagonal();

    // 过程噪声协方差
    Eigen::Matrix4d Q1 = Eigen::Matrix4d::Zero();  // q 过程噪声 = 0

    Eigen::Matrix3d Q2 = params.inv_inertia * VarOfProc2_vec.asDiagonal() * params.inv_inertia.transpose();

    // ── 初始真实状态 ──
    RigidBodyState true_state;
    true_state.R  = Eigen::Vector3d::Zero();
    true_state.A  = Eigen::Matrix3d::Identity();
    true_state.v  = Eigen::Vector3d::Zero();
    true_state.w  = Eigen::Vector3d(1.0, -0.2, -0.05);
    true_state.q  = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);

    // ── 滤波器初始化 ──
    Eigen::Vector3d w_init(0.1, 0.02, 0.3);     // 带偏差的初始估计
    Eigen::Vector4d q_init(0.2, 0.3, 0.1, std::sqrt(0.86));
    q_init = quat_normalize(q_init);

    Eigen::Matrix3d Pw0 = Eigen::Matrix3d::Identity();
    Eigen::Matrix4d Pq0 = Eigen::Matrix4d::Identity();

    CascadedUKF cascaded_ukf(w_init, Pw0, q_init, Pq0, Q2, RMea2, Q1, RMea1);

    // ── 随机数生成器 ──
    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // ── 存储容器 ──
    std::vector<double> time_vec;
    std::vector<Eigen::Vector4d> q_true_vec, q_est_vec, q_meas_vec;
    std::vector<Eigen::Vector3d> w_true_vec, w_est_vec, w_meas_vec;

    int N_steps = static_cast<int>(cfg.total_time / cfg.dt);
    time_vec.reserve(N_steps + 1);
    q_true_vec.reserve(N_steps + 1);
    w_true_vec.reserve(N_steps + 1);
    q_est_vec.reserve(N_steps + 1);
    w_est_vec.reserve(N_steps + 1);
    q_meas_vec.reserve(N_steps + 1);
    w_meas_vec.reserve(N_steps + 1);

    // ── 仿真主循环 ──
    Eigen::Vector3d R_touch_e = Eigen::Vector3d::Zero();
    Eigen::Vector3d F = Eigen::Vector3d::Zero();

    for (int i = 0; i <= N_steps; ++i) {
        double t = i * cfg.dt;

        // 外部力矩
        Eigen::Vector3d T_torque = external_torque(t);

        // 过程噪声 (加在力矩上)
        Eigen::Vector3d proc_noise(
            std::sqrt(VarOfProc2_vec.x()) * gauss(rng),
            std::sqrt(VarOfProc2_vec.y()) * gauss(rng),
            std::sqrt(VarOfProc2_vec.z()) * gauss(rng)
        );
        Eigen::Vector3d T_total = T_torque + proc_noise;

        // ── 真实状态传播 (RK4) ──
        true_state = f_dyn_U2(true_state, R_touch_e, F, T_total, params);

        // ── 测量 ──
        Eigen::Vector4d q_meas = true_state.q;
        q_meas(0) += std::sqrt(VarOfMear1_vec.x()) * gauss(rng);
        q_meas(1) += std::sqrt(VarOfMear1_vec.y()) * gauss(rng);
        q_meas(2) += std::sqrt(VarOfMear1_vec.z()) * gauss(rng);
        q_meas(3) += std::sqrt(VarOfMear1_qw) * gauss(rng);
        q_meas = quat_normalize(q_meas);

        Eigen::Vector3d w_meas = true_state.w;
        w_meas(0) += std::sqrt(VarOfMear2_vec.x()) * gauss(rng);
        w_meas(1) += std::sqrt(VarOfMear2_vec.y()) * gauss(rng);
        w_meas(2) += std::sqrt(VarOfMear2_vec.z()) * gauss(rng);

        // ── UKF 滤波 ──
        cascaded_ukf.step(w_meas, q_meas, T_torque, params);

        // ── 记录 ──
        time_vec.push_back(t);
        q_true_vec.push_back(true_state.q);
        w_true_vec.push_back(true_state.w);
        q_est_vec.push_back(cascaded_ukf.get_q());
        w_est_vec.push_back(cascaded_ukf.get_w());
        q_meas_vec.push_back(q_meas);
        w_meas_vec.push_back(w_meas);
    }

    print_rmse("CascadedUKF", q_true_vec, w_true_vec, q_est_vec, w_est_vec);
    write_csv("output_cascaded_ukf.csv", time_vec, q_true_vec, w_true_vec,
              q_est_vec, w_est_vec, q_meas_vec, w_meas_vec);
}

// ── 运行联合 UKF 仿真 (对齐 main2.m) ──────────────────────────────────────────
void run_joint_ukf(const SimConfig& cfg) {
    std::cout << "\n========== Joint UKF (7-state, 对齐 main2.m) ==========\n";

    RigidBodyParams params(cfg.mass, Eigen::Matrix3d::Identity(), cfg.dt);

    // 噪声 (对齐 main2.m)
    Eigen::Vector3d VarOfMear1_vec(1e-1 * 1.0, 1e-1 * 1.5, 1e-1 * 2.0);
    double VarOfMear1_qw = 1e-1 * 1.0;

    // 注意: main2.m 中 VarOfMear2 = 1e-2 * [1.3; 3; 4]，与 main3.m 不同
    Eigen::Vector3d VarOfMear2_vec(1e-2 * 1.3, 1e-2 * 3.0, 1e-2 * 4.0);
    Eigen::Vector3d VarOfProc2_vec(1e-4 * 1.0, 1e-4 * 2.0, 1e-4 * 1.0);

    // 构建 7x7 协方差矩阵
    Eigen::Matrix<double, 7, 7> R_mea = Eigen::Matrix<double, 7, 7>::Zero();
    R_mea.diagonal() << VarOfMear1_vec, VarOfMear1_qw, VarOfMear2_vec;

    Eigen::Matrix<double, 7, 7> Q_proc = Eigen::Matrix<double, 7, 7>::Zero();
    Eigen::Matrix3d Q_w = params.inv_inertia * VarOfProc2_vec.asDiagonal() * params.inv_inertia.transpose();
    Q_proc.block<3, 3>(4, 4) = Q_w;  // 下标 4:6 (0-indexed)

    // 初始状态
    RigidBodyState true_state;
    true_state.R  = Eigen::Vector3d::Zero();
    true_state.A  = Eigen::Matrix3d::Identity();
    true_state.v  = Eigen::Vector3d::Zero();
    true_state.w  = Eigen::Vector3d(1.0, -0.2, -0.05);
    true_state.q  = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);

    Eigen::Matrix<double, 7, 1> X_init;
    X_init << 0.2, 0.3, 0.1, std::sqrt(0.86), 0.1, 0.02, 0.3;
    X_init.head<4>() = quat_normalize(X_init.head<4>());

    Eigen::Matrix<double, 7, 7> P0 = Eigen::Matrix<double, 7, 7>::Identity();
    JointUKF joint_ukf(X_init, P0, Q_proc, R_mea);

    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<double> time_vec;
    std::vector<Eigen::Vector4d> q_true_vec, q_est_vec, q_meas_vec;
    std::vector<Eigen::Vector3d> w_true_vec, w_est_vec, w_meas_vec;

    int N_steps = static_cast<int>(cfg.total_time / cfg.dt);
    time_vec.reserve(N_steps + 1);

    Eigen::Vector3d R_touch_e = Eigen::Vector3d::Zero();
    Eigen::Vector3d F = Eigen::Vector3d::Zero();

    for (int i = 0; i <= N_steps; ++i) {
        double t = i * params.dt;

        Eigen::Vector3d T_torque = external_torque(t);
        Eigen::Vector3d proc_noise(
            std::sqrt(VarOfProc2_vec.x()) * gauss(rng),
            std::sqrt(VarOfProc2_vec.y()) * gauss(rng),
            std::sqrt(VarOfProc2_vec.z()) * gauss(rng)
        );
        Eigen::Vector3d T_total = T_torque + proc_noise;

        true_state = f_dyn_U2(true_state, R_touch_e, F, T_total, params);

        Eigen::Matrix<double, 7, 1> Z_meas;
        Z_meas.head<4>() = true_state.q;
        Z_meas.head<4>()(0) += std::sqrt(VarOfMear1_vec.x()) * gauss(rng);
        Z_meas.head<4>()(1) += std::sqrt(VarOfMear1_vec.y()) * gauss(rng);
        Z_meas.head<4>()(2) += std::sqrt(VarOfMear1_vec.z()) * gauss(rng);
        Z_meas.head<4>()(3) += std::sqrt(VarOfMear1_qw) * gauss(rng);
        Z_meas.head<4>() = quat_normalize(Z_meas.head<4>());

        Z_meas.tail<3>() = true_state.w;
        Z_meas.tail<3>()(0) += std::sqrt(VarOfMear2_vec.x()) * gauss(rng);
        Z_meas.tail<3>()(1) += std::sqrt(VarOfMear2_vec.y()) * gauss(rng);
        Z_meas.tail<3>()(2) += std::sqrt(VarOfMear2_vec.z()) * gauss(rng);

        joint_ukf.step(T_torque, Z_meas, params);

        time_vec.push_back(t);
        q_true_vec.push_back(true_state.q);
        w_true_vec.push_back(true_state.w);
        q_est_vec.push_back(joint_ukf.X.head<4>());
        w_est_vec.push_back(joint_ukf.X.tail<3>());
        q_meas_vec.push_back(Z_meas.head<4>());
        w_meas_vec.push_back(Z_meas.tail<3>());
    }

    print_rmse("JointUKF   ", q_true_vec, w_true_vec, q_est_vec, w_est_vec);
    write_csv("output_joint_ukf.csv", time_vec, q_true_vec, w_true_vec,
              q_est_vec, w_est_vec, q_meas_vec, w_meas_vec);
}

// ── 运行 EKF 仿真 (对齐 Main.m) ───────────────────────────────────────────────
void run_ekf_6state(const SimConfig& cfg) {
    std::cout << "\n========== EKF (6-state MRP, 对齐 Main.m) ==========\n";

    RigidBodyParams params(cfg.mass, Eigen::Matrix3d::Identity(), 0.01);  // Main.m 用 d_time=0.01

    // 噪声方差 (对齐 Main.m)
    Eigen::Matrix<double, 6, 1> VarOfMear;
    VarOfMear << 0.0001 * 1.0, 0.0001 * 1.5, 0.0001 * 2.0,
                 0.0001 * 1.3, 0.0001 * 3.0, 0.0001 * 4.0;
    Eigen::Matrix<double, 6, 6> R_meas = VarOfMear.asDiagonal();

    double var_proc_scale = 0.0001;  // VarOfProc 的均值量级

    // 初始状态
    RigidBodyState true_state;
    true_state.R  = Eigen::Vector3d::Zero();
    true_state.A  = Eigen::Matrix3d::Identity();
    true_state.v  = Eigen::Vector3d::Zero();
    true_state.w  = Eigen::Vector3d(1.0, -0.2, -0.05);
    true_state.q  = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);

    // EKF 初始化: x0 = [q(1:3); 0; 0; 0]  (MRP + w)
    Eigen::Matrix<double, 6, 1> x0;
    x0 << true_state.q.head<3>(), 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Vector4d q0 = true_state.q;

    EKF6State ekf(x0, P0, q0);

    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<double> time_vec;
    std::vector<Eigen::Vector4d> q_true_vec, q_est_vec, q_meas_vec;
    std::vector<Eigen::Vector3d> w_true_vec, w_est_vec, w_meas_vec;

    int N_steps = static_cast<int>(cfg.total_time / params.dt);
    time_vec.reserve(N_steps + 1);

    Eigen::Vector3d R_touch_e = Eigen::Vector3d::Zero();
    Eigen::Vector3d F = Eigen::Vector3d::Zero();

    for (int i = 0; i <= N_steps; ++i) {
        double t = i * params.dt;

        Eigen::Vector3d T_torque = external_torque(t);
        Eigen::Vector3d proc_noise(
            std::sqrt(var_proc_scale) * gauss(rng),
            std::sqrt(var_proc_scale * 2.0) * gauss(rng),
            std::sqrt(var_proc_scale) * gauss(rng)
        );
        Eigen::Vector3d T_total = T_torque + proc_noise;

        true_state = f_dyn_U2(true_state, R_touch_e, F, T_total, params);

        // 测量: z = [q(1:3); w]
        Eigen::Matrix<double, 6, 1> z;
        z.head<3>() = true_state.q.head<3>();
        z.head<3>()(0) += std::sqrt(VarOfMear(0)) * gauss(rng);
        z.head<3>()(1) += std::sqrt(VarOfMear(1)) * gauss(rng);
        z.head<3>()(2) += std::sqrt(VarOfMear(2)) * gauss(rng);
        z.tail<3>() = true_state.w;
        z.tail<3>()(0) += std::sqrt(VarOfMear(3)) * gauss(rng);
        z.tail<3>()(1) += std::sqrt(VarOfMear(4)) * gauss(rng);
        z.tail<3>()(2) += std::sqrt(VarOfMear(5)) * gauss(rng);

        ekf.step(T_torque, z, R_meas, i, cfg.update_interval, params, var_proc_scale);

        time_vec.push_back(t);
        q_true_vec.push_back(true_state.q);
        w_true_vec.push_back(true_state.w);
        q_est_vec.push_back(ekf.q);
        w_est_vec.push_back(ekf.x.tail<3>());

        Eigen::Vector4d q_meas_full = mrp_to_quat(z.head<3>());
        q_meas_vec.push_back(q_meas_full);
        w_meas_vec.push_back(z.tail<3>());
    }

    print_rmse("EKF(6)    ", q_true_vec, w_true_vec, q_est_vec, w_est_vec);
    write_csv("output_ekf_6state.csv", time_vec, q_true_vec, w_true_vec,
              q_est_vec, w_est_vec, q_meas_vec, w_meas_vec);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  JT-EKF/UKF 姿态估计仿真 — C++ 复现                ║\n";
    std::cout << "║  对齐 MATLAB w_and_q/ 目录                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";

    SimConfig cfg;
    cfg.dt         = 0.1;    // 对齐 main3.m
    cfg.total_time = 50.0;

    // ── 运行三组对比 ──
    std::cout << "\n配置: dt=" << cfg.dt << "s, T=" << cfg.total_time
              << "s, J=I₃, m=" << cfg.mass << "kg\n";

    run_ekf_6state(cfg);
    run_joint_ukf(cfg);
    run_cascaded_ukf(cfg);

    std::cout << "\n完成! 输出文件:\n";
    std::cout << "  output_ekf_6state.csv      — EKF (6-state)\n";
    std::cout << "  output_joint_ukf.csv       — Joint UKF (7-state)\n";
    std::cout << "  output_cascaded_ukf.csv    — Cascaded UKF (UKF2→UKF1)\n";

    return 0;
}
