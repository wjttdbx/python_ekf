#include "../Header/MotionModel.h"

namespace MotionModel {

    /* Observation model */
    Observation_model::Observation_model(size_t Observation_mode, size_t Observation_num, Vector Measurement_error)
        :mode(Observation_mode), num(Observation_num), mea_err(Measurement_error){}

    size_t Observation_model::Observation_Mode() const {
        return mode;
    }

    size_t Observation_model::observation_dim_calculation() const {
        size_t observation_dim = (mode == 0) ? 3 :
            (mode == 1) ? 3 :
            (mode == 2) ? 2 : 0;
        if (observation_dim == 0) throw std::invalid_argument("error:Observation mode error");

        Matrix R(num * observation_dim, num * observation_dim);
        if (R.getRows() != mea_err.getDimension()) throw std::invalid_argument("error:Observation error dimension error - inconsistent with the number of observation stations/measurement mode");

        Vector Z(num * observation_dim);
        if (Z.getDimension() != mea_err.getDimension()) throw std::invalid_argument("error:Observation error dimension error - inconsistent with the number of observation stations/measurement mode");

        return observation_dim;
    }


    Matrix Observation_model::H_Calculation(Vector state, size_t state_dim, Vector Observation_position) const {
        size_t observation_dim = observation_dim_calculation();
        Matrix H(num * observation_dim, MotionModel_dim * state_dim);

        for (size_t i = 0; i < num; ++i) {
            if (mode == 0) 
                for (size_t j = 0; j < observation_dim; ++j)
                    H[observation_dim * i + j][state_dim * j] = 1;
            if (mode == 1) {
                Vector dpos(MotionModel_dim);
                for (size_t j = 0; j < MotionModel_dim; ++j) {
                    dpos[j] = state[state_dim * j] - Observation_position[MotionModel_dim * i + j];
                }
                double r = std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2) + std::pow(dpos[2], 2));
                H[observation_dim * i][state_dim * 0] = dpos[0] / r;
                H[observation_dim * i][state_dim * 1] = dpos[1] / r;
                H[observation_dim * i][state_dim * 2] = dpos[2] / r;
                H[observation_dim * i + 1][state_dim * 0] = - dpos[1] / (std::pow(dpos[0], 2) + std::pow(dpos[1], 2));
                H[observation_dim * i + 1][state_dim * 1] = dpos[0] / (std::pow(dpos[0], 2) + std::pow(dpos[1], 2));
                H[observation_dim * i + 2][state_dim * 0] = - dpos[2] * dpos[0] / (std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2)) * std::pow(r, 2));
                H[observation_dim * i + 2][state_dim * 1] = - dpos[2] * dpos[1] / (std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2)) * std::pow(r, 2));
                H[observation_dim * i + 2][state_dim * 2] = std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2)) / std::pow(r, 2);
            }
            if (mode == 2) {
                Vector dpos(MotionModel_dim);
                for (size_t j = 0; j < MotionModel_dim; ++j) {
                    dpos[j] = state[state_dim * j] - Observation_position[MotionModel_dim * i + j];
                }
                double r = std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2) + std::pow(dpos[2], 2));
                //H[observation_dim * i][state_dim * 0] = - dpos[1] / (std::pow(dpos[0], 2) + std::pow(dpos[1], 2));
                //H[observation_dim * i][state_dim * 1] = dpos[0] / (std::pow(dpos[0], 2) + std::pow(dpos[1], 2));
                //H[observation_dim * i + 1][state_dim * 0] = - dpos[2] * dpos[0] / (std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2)) * r * r);
                //H[observation_dim * i + 1][state_dim * 1] = - dpos[2] * dpos[1] / (std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2)) * r * r);
                //H[observation_dim * i + 1][state_dim * 2] = std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2)) / std::pow(r, 2);
                H[observation_dim * i][state_dim * 0] = dpos[2] / (std::pow(dpos[0], 2) + std::pow(dpos[2], 2));
                H[observation_dim * i][state_dim * 2] = - dpos[0] / (std::pow(dpos[0], 2) + std::pow(dpos[2], 2));
                H[observation_dim * i + 1][state_dim * 0] = -dpos[1] * dpos[0] / (std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[2], 2)) * r * r);
                H[observation_dim * i + 1][state_dim * 1] = std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[2], 2)) / std::pow(r, 2);
                H[observation_dim * i + 1][state_dim * 2] = -dpos[2] * dpos[1] / (std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[2], 2)) * r * r);

            }
        }

        return H;
    }

    Matrix Observation_model::R_Calculation() const {
        size_t observation_dim = observation_dim_calculation();
        Matrix R(num * observation_dim, num * observation_dim);

        for (size_t i = 0; i < R.getRows(); ++i) R[i][i] = mea_err[i] * mea_err[i];

        return R;
    }

    Vector Observation_model::measurements(Vector state, size_t state_dim, Vector Observation_position) const {
        size_t observation_dim = observation_dim_calculation();
        Vector Z(num * observation_dim);

        if (mode == 0) 
            for (size_t i = 0; i < num; ++i)
                for (size_t j = 0; j < MotionModel_dim; ++j)
                    Z[observation_dim * i + j] = state[state_dim * j];
        if (mode == 1)
            for (size_t i = 0; i < num; ++i) {
                Vector dpos(MotionModel_dim);
                for (size_t j = 0; j < MotionModel_dim; ++j) 
                    dpos[j] = state[state_dim * j] - Observation_position[MotionModel_dim * i + j];
                Z[observation_dim * i] = std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2) + std::pow(dpos[2], 2));
                Z[observation_dim * i + 1] = std::atan2(dpos[1], dpos[0]);
                Z[observation_dim * i + 2] = std::atan2(dpos[2], std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2)));
            }
        if (mode == 2)
            for (size_t i = 0; i < num; ++i) {
                Vector dpos(MotionModel_dim);
                for (size_t j = 0; j < MotionModel_dim; ++j)
                    dpos[j] = state[state_dim * j] - Observation_position[MotionModel_dim * i + j];
                //Z[observation_dim * i] = std::atan2(dpos[1], dpos[0]);
                //Z[observation_dim * i + 1] = std::atan2(dpos[2], std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2)));
                Z[observation_dim * i] = std::atan2(-dpos[2], dpos[0]);
                Z[observation_dim * i + 1] = std::atan2(dpos[1], std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[2], 2)));
            }

        return Z;
    }

    Vector Observation_model::measurements_test(Vector state, size_t state_dim, Vector Observation_position) const {
        size_t observation_dim = observation_dim_calculation();
        Vector Z_test(num * observation_dim);
        if (mode == 2)
            for (size_t i = 0; i < num; ++i) {
                Vector dpos(MotionModel_dim);
                for (size_t j = 0; j < MotionModel_dim; ++j)
                    dpos[j] = state[state_dim * j] - Observation_position[MotionModel_dim * i + j];
                Z_test[observation_dim * i] = dpos[1] / dpos[0];
                Z_test[observation_dim * i + 1] =dpos[2] / std::sqrt(std::pow(dpos[0], 2) + std::pow(dpos[1], 2));
            }

        return Z_test;
    }


    /*==================================================================================================================================*/
    /* Constant Velocity model    CV model */
    CV_model::CV_model(double dt, Vector sigma)
        : T(dt), sigma(sigma) {}

    Matrix CV_model::F_Calculation() const {
        Matrix F(Dim * MotionModel_dim, Dim * MotionModel_dim);

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            F[Dim * i][Dim * i] = 1.;           F[Dim * i][Dim * i + 1] = T;
                                                F[Dim * i + 1][Dim * i + 1] = 1.;
        }

        return F;
    }

    Matrix CV_model::Q_Calculation() const {
        Matrix Q(Dim * MotionModel_dim, Dim * MotionModel_dim);
        double q11 = std::pow(T, 5) / 20.;
        double q12 = std::pow(T, 4) / 8.;
        double q22 = std::pow(T, 3) / 3.;

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            Q[Dim * i][Dim * i] = q11 * std::pow(sigma[i], 2);          Q[Dim * i][Dim * i + 1] = q12 * std::pow(sigma[i], 2);
            Q[Dim * i + 1][Dim * i] = q12 * std::pow(sigma[i], 2);      Q[Dim * i + 1][Dim * i + 1] = q22 * std::pow(sigma[i], 2);
        }

        return Q;
    }


    /*==================================================================================================================================*/
    /* Constant accelerated motion   CA model */
    CA_model::CA_model(double dt, Vector sigma)
        : T(dt), sigma(sigma) {}

    Matrix CA_model::F_Calculation() const{
        Matrix F(Dim * MotionModel_dim, Dim * MotionModel_dim);

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            F[Dim * i][Dim * i] = 1.;           F[Dim * i][Dim * i + 1] = T;                F[Dim * i][Dim * i + 2] = .5 * T * T;
                                                F[Dim * i + 1][Dim * i + 1] = 1.;           F[Dim * i + 1][Dim * i + 2] = T;
                                                                                            F[Dim * i + 2][Dim * i + 2] = 1.;
        }

        return F;
    }

    Matrix CA_model::Q_Calculation() const{
        Matrix Q(Dim * MotionModel_dim, Dim * MotionModel_dim);
        double q11 = std::pow(T, 5) / 20.;
        double q12 = std::pow(T, 4) / 8.;
        double q13 = std::pow(T, 3) / 6.;
        double q22 = std::pow(T, 3) / 3.;
        double q23 = std::pow(T, 2) / 2.;
        double q33 = T;

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            Q[Dim * i][Dim * i] = q11 * std::pow(sigma[i], 2);          Q[Dim * i][Dim * i + 1] = q12 * std::pow(sigma[i], 2);              Q[Dim * i][Dim * i + 2] = q13 * std::pow(sigma[i], 2);
            Q[Dim * i + 1][Dim * i] = q12 * std::pow(sigma[i], 2);      Q[Dim * i + 1][Dim * i + 1] = q22 * std::pow(sigma[i], 2);          Q[Dim * i + 1][Dim * i + 2] = q23 * std::pow(sigma[i], 2);
            Q[Dim * i + 2][Dim * i] = q13 * std::pow(sigma[i], 2);      Q[Dim * i + 2][Dim * i + 1] = q23 * std::pow(sigma[i], 2);          Q[Dim * i + 2][Dim * i + 2] = q33 * std::pow(sigma[i], 2);
        }

        return Q;
    }


    /*==================================================================================================================================*/
    /* Singer model */
    Singer_model::Singer_model(double dt, Vector sigma, double alpha)
        : T(dt), sigma(sigma), alpha(alpha) {}

    Matrix Singer_model::F_Calculation() const {
        Matrix F(Dim * MotionModel_dim, Dim * MotionModel_dim);

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            F[Dim * i][Dim * i] = 1.;           F[Dim * i][Dim * i + 1] = T;                F[Dim * i][Dim * i + 2] = (alpha * T - 1. + std::exp(-alpha * T)) / std::pow(alpha, 2);
                                                F[Dim * i + 1][Dim * i + 1] = 1.;           F[Dim * i + 1][Dim * i + 2] = (1. - std::exp(-alpha * T)) / alpha;
                                                                                            F[Dim * i + 2][Dim * i + 2] = std::exp(-alpha * T);
        }

        return F;
    }

    Matrix Singer_model::Q_Calculation() const {
        Matrix Q(Dim * MotionModel_dim, Dim * MotionModel_dim);
        Vector q11(MotionModel_dim);        Vector q12(MotionModel_dim);        Vector q13(MotionModel_dim);
                                            Vector q22(MotionModel_dim);        Vector q23(MotionModel_dim);
                                                                                Vector q33(MotionModel_dim);
        double aT = alpha * T;
        for (size_t i = 0; i < MotionModel_dim; ++i) {
            q11[i] = (1. - std::exp(-2. * aT) + 2. * aT + 2. * std::pow(aT, 3) / 3. - 2. * std::pow(aT, 2) - 4. * aT * std::exp(-aT)) / (2. * std::pow(alpha, 5));
            q12[i] = (std::exp(-2. * aT) + 1 - 2. * std::exp(-aT) + 2. * aT * std::exp(-aT) - 2. * aT + std::pow(alpha, 2) * std::pow(T, 2)) / (2. * std::pow(alpha, 4));
            q13[i] = (1. - std::exp(-2. * aT) - 2. * aT * std::exp(-aT)) / (2. * std::pow(alpha, 3));
            q22[i] = (4. * std::exp(-aT) - 3. - std::exp(-2. * aT) + 2. * aT) / (2. * std::pow(alpha, 3));
            q23[i] = (std::exp(-2. * aT) + 1. - 2. * std::exp(-aT)) / (2. * std::pow(alpha, 2));
            q33[i] = (1. - std::exp(-2. * aT)) / (2. * alpha);
        }
        
        for (size_t i = 0; i < MotionModel_dim; ++i) {
            Q[Dim * i][Dim * i] = 2. * alpha * q11[i] * std::pow(sigma[i], 2);          Q[Dim * i][Dim * i + 1] = 2. * alpha * q12[i] * std::pow(sigma[i], 2);              Q[Dim * i][Dim * i + 2] = 2. * alpha * q13[i] * std::pow(sigma[i], 2);
            Q[Dim * i + 1][Dim * i] = 2. * alpha * q12[i] * std::pow(sigma[i], 2);      Q[Dim * i + 1][Dim * i + 1] = 2. * alpha * q22[i] * std::pow(sigma[i], 2);          Q[Dim * i + 1][Dim * i + 2] = 2. * alpha * q23[i] * std::pow(sigma[i], 2);
            Q[Dim * i + 2][Dim * i] = 2. * alpha * q13[i] * std::pow(sigma[i], 2);      Q[Dim * i + 2][Dim * i + 1] = 2. * alpha * q23[i] * std::pow(sigma[i], 2);          Q[Dim * i + 2][Dim * i + 2] = 2. * alpha * q33[i] * std::pow(sigma[i], 2);
        }

        return Q;
    }


    /*==================================================================================================================================*/
    /* Current Statistical Model   CS model */
    CS_model::CS_model(double dt, Vector a_max, double alpha, Vector ak)
        : T(dt), amax(a_max), alpha(alpha), ak(ak){}

    Matrix CS_model::F_Calculation() const {
        Matrix F(Dim * MotionModel_dim, Dim * MotionModel_dim);

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            F[Dim * i][Dim * i] = 1.;           F[Dim * i][Dim * i + 1] = T;                F[Dim * i][Dim * i + 2] = (alpha * T - 1. + std::exp(-alpha * T)) / std::pow(alpha, 2);
                                                F[Dim * i + 1][Dim * i + 1] = 1.;           F[Dim * i + 1][Dim * i + 2] = (1. - std::exp(-alpha * T)) / alpha;
                                                                                            F[Dim * i + 2][Dim * i + 2] = std::exp(-alpha * T);
        }

        return F;
    }

    Matrix CS_model::Q_Calculation() const {
        Matrix Q(Dim * MotionModel_dim, Dim * MotionModel_dim);
        Vector q11(MotionModel_dim);        Vector q12(MotionModel_dim);        Vector q13(MotionModel_dim);
                                            Vector q22(MotionModel_dim);        Vector q23(MotionModel_dim);
                                                                                Vector q33(MotionModel_dim);
        double aT = alpha * T;
        for (size_t i = 0; i < MotionModel_dim; ++i) {
            q11[i] = (1. - std::exp(-2. * aT) + 2. * aT + 2. * std::pow(aT, 3) / 3. - 2. * std::pow(aT, 2) - 4. * aT * std::exp(-aT)) / (2. * std::pow(alpha, 5));
            q12[i] = (std::exp(-2. * aT) + 1 - 2. * std::exp(-aT) + 2. * aT * std::exp(-aT) - 2. * aT + std::pow(alpha, 2) * std::pow(T, 2)) / (2. * std::pow(alpha, 4));
            q13[i] = (1. - std::exp(-2. * aT) - 2. * aT * std::exp(-aT)) / (2. * std::pow(alpha, 3));
            q22[i] = (4. * std::exp(-aT) - 3. - std::exp(-2. * aT) + 2. * aT) / (2. * std::pow(alpha, 3));
            q23[i] = (std::exp(-2. * aT) + 1. - 2. * std::exp(-aT)) / (2. * std::pow(alpha, 2));
            q33[i] = (1. - std::exp(-2. * aT)) / (2. * alpha);
        }
        Vector a_sigma(MotionModel_dim);        
        for (size_t i = 0; i < MotionModel_dim; ++i) a_sigma[i] = (4. - pi) / pi * std::pow(amax[i] - std::abs(ak[i]), 2);
        for (size_t i = 0; i < MotionModel_dim; ++i) {
            Q[Dim * i][Dim * i] = 2. * alpha * q11[i] * a_sigma[i];          Q[Dim * i][Dim * i + 1] = 2. * alpha * q12[i] * a_sigma[i];              Q[Dim * i][Dim * i + 2] = 2. * alpha * q13[i] * a_sigma[i];
            Q[Dim * i + 1][Dim * i] = 2. * alpha * q12[i] * a_sigma[i];      Q[Dim * i + 1][Dim * i + 1] = 2. * alpha * q22[i] * a_sigma[i];          Q[Dim * i + 1][Dim * i + 2] = 2. * alpha * q23[i] * a_sigma[i];
            Q[Dim * i + 2][Dim * i] = 2. * alpha * q13[i] * a_sigma[i];      Q[Dim * i + 2][Dim * i + 1] = 2. * alpha * q23[i] * a_sigma[i];          Q[Dim * i + 2][Dim * i + 2] = 2. * alpha * q33[i] * a_sigma[i];
        }
        
        return Q;
    }

    Matrix CS_model::B_Calculation() const {
        Matrix B(Dim * MotionModel_dim, Dim);

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            B[Dim * i][i] = std::pow(T, 2) / 2. - (alpha * T - 1 + std::exp(-alpha * T)) / std::pow(alpha, 2);
            B[Dim * i + 1][i] = T - (1 - std::exp(-alpha * T)) / alpha;
            B[Dim * i + 2][i] = 1 - std::exp(-alpha * T);
        }

        return B;
    }

    /*==================================================================================================================================*/
   /* Improvement Current Statistical Model   CS model */
    CS_Improvement_model::CS_Improvement_model(double dt, Vector a_max, double alpha, double alpha_adaption, Vector ak, Vector acc)
        : T(dt), amax(a_max), alpha(alpha), alpha_adaption(alpha_adaption), ak(ak), acc(acc) {
    }

    Matrix CS_Improvement_model::F_Calculation() const {
        Matrix F(Dim * MotionModel_dim, Dim * MotionModel_dim);

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            F[Dim * i][Dim * i] = 1.;           F[Dim * i][Dim * i + 1] = T;                F[Dim * i][Dim * i + 2] = (alpha * T - 1. + std::exp(-alpha * T)) / std::pow(alpha, 2);
                                                F[Dim * i + 1][Dim * i + 1] = 1.;           F[Dim * i + 1][Dim * i + 2] = (1. - std::exp(-alpha * T)) / alpha;
                                                                                            F[Dim * i + 2][Dim * i + 2] = std::exp(-alpha * T);
        }

        return F;
    }

    Matrix CS_Improvement_model::Q_Calculation(size_t num) const {
        static size_t Counter = 0;
        if (Counter == num - 1) Counter = 0;
        Matrix Q(Dim * MotionModel_dim, Dim * MotionModel_dim);
        Vector q11(MotionModel_dim);        Vector q12(MotionModel_dim);        Vector q13(MotionModel_dim);
                                            Vector q22(MotionModel_dim);        Vector q23(MotionModel_dim);
                                                                                Vector q33(MotionModel_dim);
        double aT = alpha_adaption * T;
        for (size_t i = 0; i < MotionModel_dim; ++i) {
            q11[i] = (1. - std::exp(-2. * aT) + 2. * aT + 2. * std::pow(aT, 3) / 3. - 2. * std::pow(aT, 2) - 4. * aT * std::exp(-aT)) / (2. * std::pow(alpha_adaption, 5));
            q12[i] = (std::exp(-2. * aT) + 1 - 2. * std::exp(-aT) + 2. * aT * std::exp(-aT) - 2. * aT + std::pow(alpha_adaption, 2) * std::pow(T, 2)) / (2. * std::pow(alpha_adaption, 4));
            q13[i] = (1. - std::exp(-2. * aT) - 2. * aT * std::exp(-aT)) / (2. * std::pow(alpha_adaption, 3));
            q22[i] = (4. * std::exp(-aT) - 3. - std::exp(-2. * aT) + 2. * aT) / (2. * std::pow(alpha_adaption, 3));
            q23[i] = (std::exp(-2. * aT) + 1. - 2. * std::exp(-aT)) / (2. * std::pow(alpha_adaption, 2));
            q33[i] = (1. - std::exp(-2. * aT)) / (2. * alpha_adaption);
        }
        Vector a_sigma(MotionModel_dim);
        if (Counter == 0) {
            for (size_t i = 0; i < 3; ++i) a_sigma[i] = (4. - pi) / pi * std::pow(amax[i] - std::abs(ak[i]), 2);
        }
        else {
            for (size_t i = 0; i < 3; ++i) a_sigma[i] = (4. - pi) / pi * std::pow(ak[i] + T * acc[i], 2);
        }
        for (size_t i = 0; i < MotionModel_dim; ++i) {
            Q[Dim * i][Dim * i] = 2. * alpha_adaption * q11[i] * a_sigma[i];          Q[Dim * i][Dim * i + 1] = 2. * alpha_adaption * q12[i] * a_sigma[i];              Q[Dim * i][Dim * i + 2] = 2. * alpha_adaption * q13[i] * a_sigma[i];
            Q[Dim * i + 1][Dim * i] = 2. * alpha_adaption * q12[i] * a_sigma[i];      Q[Dim * i + 1][Dim * i + 1] = 2. * alpha_adaption * q22[i] * a_sigma[i];          Q[Dim * i + 1][Dim * i + 2] = 2. * alpha_adaption * q23[i] * a_sigma[i];
            Q[Dim * i + 2][Dim * i] = 2. * alpha_adaption * q13[i] * a_sigma[i];      Q[Dim * i + 2][Dim * i + 1] = 2. * alpha_adaption * q23[i] * a_sigma[i];          Q[Dim * i + 2][Dim * i + 2] = 2. * alpha_adaption * q33[i] * a_sigma[i];
        }
        Counter++;

        return Q;
    }

    Matrix CS_Improvement_model::B_Calculation() const {
        Matrix B(Dim * MotionModel_dim, Dim);

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            B[Dim * i][i] = std::pow(T, 2) / 2. - (alpha * T - 1 + std::exp(-alpha * T)) / std::pow(alpha, 2);
            B[Dim * i + 1][i] = T - (1 - std::exp(-alpha * T)) / alpha;
            B[Dim * i + 2][i] = 1 - std::exp(-alpha * T);
        }

        return B;
    }

    /*==================================================================================================================================*/
    /* Jerk model */
    Jerk_model::Jerk_model(double dt, Vector sigma, double alpha)
        : T(dt), sigma(sigma), alpha(alpha) {}

    Matrix Jerk_model::F_Calculation() const {
        Matrix F(Dim * MotionModel_dim, Dim * MotionModel_dim);

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            F[Dim * i][Dim * i] = 1.;    F[Dim * i][Dim * i + 1] = T;       F[Dim * i][Dim * i + 2] = T * T / 2.;   F[Dim * i][Dim * i + 3] = (1. - alpha * T + .5 * std::pow(alpha * T, 2) - std::exp(-alpha * T)) / std::pow(alpha, 3);
                                         F[Dim * i + 1][Dim * i + 1] = 1.;  F[Dim * i + 1][Dim * i + 2] = T;        F[Dim * i + 1][Dim * i + 3] = (alpha * T - 1. + std::exp(-alpha * T)) / std::pow(alpha, 2);
                                                                            F[Dim * i + 2][Dim * i + 2] = 1.;       F[Dim * i + 2][Dim * i + 3] = (1. - std::exp(-alpha * T)) / alpha;
                                                                                                                    F[Dim * i + 3][Dim * i + 3] = std::exp(-alpha * T);
        }

        return F;
    }

    Matrix Jerk_model::Q_Calculation() const {
        Matrix Q(Dim * MotionModel_dim, Dim * MotionModel_dim);
        Vector q11(MotionModel_dim);        Vector q12(MotionModel_dim);        Vector q13(MotionModel_dim);        Vector q14(MotionModel_dim);
                                            Vector q22(MotionModel_dim);        Vector q23(MotionModel_dim);        Vector q24(MotionModel_dim);
                                                                                Vector q33(MotionModel_dim);        Vector q34(MotionModel_dim);
                                                                                                                    Vector q44(MotionModel_dim);
        double aT = alpha * T;
        for (size_t i = 0; i < MotionModel_dim; ++i) {
            q11[i] = ((std::pow(aT, 5) / 10.) - (std::pow(aT, 4) / 2.) + (4. * std::pow(aT, 3) / 3.) - (2. * std::pow(aT, 2)) + (2. * aT) - 3. + (4. * std::exp(-aT))
                + (2. * std::pow(aT, 2) * std::exp(-aT)) - std::exp(-2. * aT)) / (2. * std::pow(alpha, 7));
            q12[i] = ((std::pow(aT, 4) / 4.) - std::pow(aT, 3) + (2. * std::pow(aT, 2)) - (2. * aT) + 1. - (2. * std::exp(-aT)) + (2. * aT * std::exp(-aT)) 
                - (std::pow(aT, 2) * std::exp(-aT)) + std::exp(-2. * aT)) / (2. * std::pow(alpha, 6));
            q13[i] = ((std::pow(aT, 3) / 3.) - std::pow(aT, 2) + (2. * aT) - 3. + (4. * std::exp(-aT)) + (std::pow(aT, 2) * std::exp(-aT)) - std::exp(-2. * aT) ) / (2. * std::pow(alpha, 5));
            q14[i] = (1. - (2. * std::exp(-aT)) - (std::pow(aT, 2) * std::exp(-aT)) + std::exp(-2. * aT)) / (2. * std::pow(alpha, 4));
            q22[i] = ((2. * std::pow(aT, 3) / 3.) - (2. * std::pow(aT, 2)) + (2. * aT) + 1. - (4. * aT * std::exp(-aT)) - std::exp(-2. * aT)) / (2. * std::pow(alpha, 5));
            q23[i] = (std::pow(aT, 2) - (2. * aT) + 1. - (2. * std::exp(-aT)) + (2. * aT * std::exp(-aT)) + std::exp(-2. * aT) ) / (2. * std::pow(alpha, 4));
            q24[i] = (1. - (2. * aT * std::exp(-aT)) - std::exp(-2. * aT) ) / (2. * std::pow(alpha, 3));
            q33[i] = ((2. * aT) - 3. + (4. * std::exp(-aT)) - std::exp(-2. * aT)) / (2. * std::pow(alpha, 3));
            q34[i] = (1. - (2. * std::exp(-aT)) + std::exp(-2. * aT)) / (2. * std::pow(alpha, 2));
            q34[i] = (1. - std::exp(-2. * aT)) / (2. * alpha);
        }

        for (size_t i = 0; i < MotionModel_dim; ++i) {
            Q[Dim * i][Dim * i] = 2. * alpha * q11[i] * std::pow(sigma[i], 2);          Q[Dim * i][Dim * i + 1] = 2. * alpha * q12[i] * std::pow(sigma[i], 2);              Q[Dim * i][Dim * i + 2] = 2. * alpha * q13[i] * std::pow(sigma[i], 2);
            Q[Dim * i + 1][Dim * i] = 2. * alpha * q12[i] * std::pow(sigma[i], 2);      Q[Dim * i + 1][Dim * i + 1] = 2. * alpha * q22[i] * std::pow(sigma[i], 2);          Q[Dim * i + 1][Dim * i + 2] = 2. * alpha * q23[i] * std::pow(sigma[i], 2);
            Q[Dim * i + 2][Dim * i] = 2. * alpha * q13[i] * std::pow(sigma[i], 2);      Q[Dim * i + 2][Dim * i + 1] = 2. * alpha * q23[i] * std::pow(sigma[i], 2);          Q[Dim * i + 2][Dim * i + 2] = 2. * alpha * q33[i] * std::pow(sigma[i], 2);
        }

        return Q;
    }


    Vector Extract_acceleration(size_t State_dim, Vector X) {
        int dim = X.getDimension() / MotionModel_dim;
        if(dim < 3) throw std::invalid_argument("error:The state vector does not contain the acceleration term");
        Vector Acceleration(MotionModel_dim);
        for (size_t i = 0; i < MotionModel_dim; ++i) Acceleration[i] = X[State_dim * i + 2];
        return Acceleration;
    }
}