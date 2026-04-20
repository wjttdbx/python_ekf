#ifndef _MOTIONMODEL_H_	
#define _MOTIONMODEL_H_

#include <iostream>
#include "Constant.h"
#include "Matrix.h"
#include <cmath>


// Motion model (discrete), state vector format[X, dX/dt, d(dX/dt)/dt]
namespace MotionModel {
   /* ===========  Observation model  =========== */
    class Observation_model {
    public:
        // Measurement model Observation model
        // (Observation_mode -> observation mode, Observation_num -> number of observation stations, Measurement_error -> measurement error)
        // Observation mode : 0-> Return the Cartesian coordinates of the target; 
        //                                  1-> Return the distance/azimuth/pitch Angle of the target; 
        //                                  2-> Return the azimuth/pitch Angle of the target.
        Observation_model(size_t Observation_mode, size_t Observation_num, Vector Measurement_error);

        // Return the observation mode
        size_t Observation_Mode() const;

        // Observation mode discrimination
        size_t observation_dim_calculation() const;

        // Calculate the measurement matrix 
        // (state -> state vector, state_dim -> state dimension required by the motion model, Observation_position -> position of the observation station)
        Matrix H_Calculation(Vector state, size_t state_dim, Vector Observation_position) const;

        // Calculate the measurement error matrix
        Matrix R_Calculation() const;

        // Calculate the measured values 
        // (state -> state vector, state_dim -> state dimension required by the motion model, Observation_position -> position of the observation station)
        Vector measurements(Vector state, size_t state_dim,Vector Observation_position) const;
        Vector measurements_test(Vector state, size_t state_dim, Vector Observation_position) const;
    private:
        size_t mode;       // Observation mode
        size_t num;        // The number of observation stations
        Vector mea_err;    // Measurement error
    };

/*=============================================================================================================================*/
   /* =========== Constant Velocity model   CV model  =========== */
    class CV_model { 
    public:
        // Constant Velocity model -- CV_model (dt -> Time interval, sigma -> standard deviation of process noise)
        CV_model(double dt, Vector sigma);        

        // Calculate the F matrix of the CV motion model
        Matrix F_Calculation() const;

        // Calculate the Q matrix of the CV motion model
        Matrix Q_Calculation() const;          

        static constexpr size_t Dim = 2; // The state dimensions required by the CV model
        double T;       // Time interval
    private:
        Vector sigma;   // The standard deviation of process noise
    };


    /* =========== Constant accelerated motion   CA_model  =========== */
    class CA_model {
    public:
        // Constant accelerated motion -- CA_model (dt -> Time interval, sigma -> standard deviation of process noise)
        CA_model(double dt, Vector sigma);

        // Calculate the F matrix of the CA motion model
        Matrix F_Calculation() const;  

        // Calculate the Q matrix of the CA motion model
        Matrix Q_Calculation() const;      

        static constexpr size_t Dim = 3; //The state dimensions required by the CA model
        double T;       // Time interval
    private:
        Vector sigma;   //The standard deviation of process noise
    };



    /* ===========  Singer_model  =========== */
    class Singer_model {
    public:
        // Singer_model (dt -> Time interval, sigma -> The standard deviation of process noise, alpha -> Maneuvering frequency)
        Singer_model(double dt, Vector sigma, double alpha);

        // Calculate the F-matrix of the Singer motion model
        Matrix F_Calculation() const;

        // Calculate the Q matrix of the Singer motion model
        Matrix Q_Calculation() const;

        static constexpr size_t Dim = 3; //SingerThe state dimensions required by the Singer model
        double alpha;   // Target maneuvering frequency
        double T;       // Time interval
    private:
        Vector sigma;   //The standard deviation of process noise
    };


    /* =========== Current Statistical Model   CS_model  =========== */
    class CS_model {
    public:
        // Current Statistical Model -- CS_model (dt -> Time interval, a_max -> Maximum target acceleration, alpha -> Maneuvering frequency, ak -> Current acceleration)
        CS_model(double dt, Vector a_max, double alpha, Vector ak);

        // Calculate the F matrix of the CS motion model
        Matrix F_Calculation() const;

        // Calculate the Q matrix of the CS motion model
        Matrix Q_Calculation() const;

        // Calculate the B matrix of the CS motion model
        Matrix B_Calculation() const;

        static constexpr size_t Dim = 3; //The state dimensions required by the CS model
        double alpha;   // Maneuvering frequency
        Vector ak;      // Current acceleration
        double T;       // Time interval
    private:
        Vector amax;    // Maximum target acceleration
    };

    /* =========== Improvement Current Statistical Model   CS_Improvement_model  =========== */
    class CS_Improvement_model {
    public:
        // Improvement Current Statistical Model -- CS_model (dt -> Time interval, a_max -> Maximum target acceleration, alpha -> Maneuvering frequency, ak -> Current acceleration)
        CS_Improvement_model(double dt, Vector a_max, double alpha, double alpha_adaption, Vector ak, Vector acc);

        // Calculate the F matrix of the CS_Improvement motion model
        Matrix F_Calculation() const;

        // Calculate the Q matrix of the CS_Improvement motion model
        Matrix Q_Calculation(size_t num) const;

        // Calculate the B matrix of the CS_Improvement motion model
        Matrix B_Calculation() const;

        static constexpr size_t Dim = 3; // The state dimensions required by the CS_Improvement model
        double alpha;   //Maneuvering frequency
        double alpha_adaption;
        Vector ak;      // Current acceleration
        Vector acc;
        double T;       // Time interval
    private:        
        Vector amax;    //Maximum target acceleration
    };

    /* ===========  Jerk_model  =========== */
    class Jerk_model {
    public:
        // Jerk_model (dt -> Time interval, a_max -> Maximum target acceleration, alpha -> Maneuvering frequency)
        Jerk_model(double dt, Vector sigma, double alpha);

        // Calculate the F-matrix of the Jerk motion model
        Matrix F_Calculation() const;

        // Calculate the Q matrix of the Jerk motion model
        Matrix Q_Calculation() const;

        static constexpr size_t Dim = 4; //The state dimensions required by the Jerk model
        double alpha;   // Maneuvering frequency
        double T;       // Time interval
    private:        
        Vector sigma;   // Add the standard deviation of the acceleration
    };


    // Extract the acceleration term from the state vector
    Vector Extract_acceleration(size_t State_dim, Vector X);
}

#endif