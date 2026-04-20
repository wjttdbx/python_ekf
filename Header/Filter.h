#ifndef _FILTER_H_	
#define _FILTER_H_

#include <iostream>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "Matrix.h"
#include "MotionModel.h"


// Kalman Filter
class Filter_KF {
protected:
    Matrix F;       Matrix B;       Matrix Q;       Matrix H;       Matrix R;

public:
    Filter_KF(MotionModel::CV_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
    Filter_KF(MotionModel::CA_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
    Filter_KF(MotionModel::Singer_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
    Filter_KF(MotionModel::CS_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
    Filter_KF(MotionModel::Jerk_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
};

// Expanded Kalman Filter
class Filter_EKF {
protected:
    Matrix F;       Matrix B;       Matrix Q;       Matrix H;       Matrix R;

public:
    Filter_EKF(MotionModel::CV_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
    Filter_EKF(MotionModel::CA_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
    Filter_EKF(MotionModel::Singer_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
    Filter_EKF(MotionModel::CS_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
    Filter_EKF(MotionModel::CS_Improvement_model& model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, size_t observe_num);
    Filter_EKF(MotionModel::Jerk_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation);
};

// Unscented Kalman Filter
struct UKF {
    double Alpha = 1.;
    double Kappa = 0.;
    double Beta  = 2.;
    UKF(){}
    UKF(const double& alpha,
        const double& kappa,
        const double& beta
    ) :Alpha(alpha), Kappa(kappa), Beta(beta) {}
};
class Filter_UKF {
    
protected:
    Matrix F;       Matrix B;       Matrix Q;       Matrix H;       Matrix R;

public:
    Filter_UKF(MotionModel::CV_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter);
    Filter_UKF(MotionModel::CA_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter);
    Filter_UKF(MotionModel::Singer_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter);
    Filter_UKF(MotionModel::CS_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter);
    Filter_UKF(MotionModel::CS_Improvement_model& model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter, size_t observe_num);
    Filter_UKF(MotionModel::Jerk_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, UKF UKF_parameter);
};

//Particle structure
struct Particle{
    Vector X;
    double Weight = 0;
    size_t Num_Particle = 0;
    Particle(){}
    Particle(Vector& x,
    double weight, size_t num_particle):X(x), Weight(weight), Num_Particle(num_particle){}
};
// Particle Filter
class Filter_PF {

protected:
    Matrix F;       Matrix B;       Matrix Q;       Matrix H;       Matrix R;

    // Resampling particle
    void resampling(std::vector<Particle>& particles);

public:
    // Initialize particle
    static void initialize(std::vector<Particle>& particles, Vector X_Optimal);

    Filter_PF(MotionModel::CV_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, std::vector<Particle>& particles);
    Filter_PF(MotionModel::CA_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, std::vector<Particle>& particles);
    //Filter_PF(MotionModel::Singer_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, std::vector<Particle>& particles);
    //Filter_PF(MotionModel::CS_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, std::vector<Particle>& particles);
    //Filter_PF(MotionModel::Jerk_model model, MotionModel::Observation_model model_observation, Vector& X_Optimal, Matrix& P_Optimal, Vector Z, Vector observation, std::vector<Particle>& particles);
};


#endif