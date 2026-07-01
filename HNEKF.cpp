//C++ file
//***********************************************
//      Filename: 标定程序.cpp
//
//        Author: jianlin chen
//         Email: jianlin.chen@upc.edu
//                chenjl@mail.nwpu.edu.cn
//***********************************************
//
//   Description:   
//   cartesian representation is exploited to implement the state and thrust estimation
//  
//	OUTPUT:
//	state
//	parameter, such as area to mass ratio
//***********************************************
#include "/usr/local/smart-uq/include/smartuq.h"
#include "cal2jul.h"
#include "wgs2eci.h"
#include "gmst.h"
#include "gast.h"
#include "eci2ecef.h"
#include "rk78.h"
#include "rk78num.h"
#include "utc2tt.h"
#include "cartesian.h"//dynamics
#include "cartesianNum.h"//dynamics
#include "transformation.h"
#include "inv_mat.h"//calculate the inverse and product of matrix, calculate transpose of the matrix
#include "expectation.h"//expectation of delta x
#include <iostream>
#include <fstream>
using namespace std;
using namespace smartuq;
using namespace inv_space;
using namespace tool_space;

int main(){
//normal distribution
    ifstream filein;
    filein.open("random_num.plt",ios::in);//open file
    vector<vector<double> > random_num;
    for(int i=0;i<3000;i++){
        vector<double> rand_tmp(3,0.);
        for(int j=0;j<3;j++)
            filein >> rand_tmp[j];
        random_num.push_back(rand_tmp);
    }
    filein.close();
//simulation configuration
    int nvp = 9;//number of estimated variables
    int nobs = 3;//number of measurements
    int degree;//the order of taylor expansions
//******* constant ************//
    double utc  = 20151115.+0./24.+0./60./24.+0./24./3600.;//UTC simulation start time
    double CR   = 1.21;
    double areatomass = 0.2/194.;
    int flag_har;//xxxxx: xxxx1 indicates central force is not taken into account
    int flag_srp;//x:1->solar radiation pressure 0->no
    int flag_third;//xx: 11->sun and moon,10->sun,01->moon,00->no
    double period;
    double uncp;
    double uncr;
    double uncv;
    double mea_err;
    filein.open("init.txt",ios::in);//open `file
    filein >> degree;
    filein >> flag_har;
    filein >> flag_srp;
    filein >> flag_third;
    filein >> period;
    filein >> uncp;
    filein >> uncr;
    filein >> uncv;
    filein >> mea_err;
    filein.close();
    double GM = 398600.4415e9;
    double a = 8788000.;
    double w = 2.0*M_PI/(23*3600.+56*60.+4.);
    double t0 = 0.;
    double tt  = UTC2TT(utc);//TT
    double ut1 = utc+0.1577184/24./3600.;//UT1 from IERS BULLETIN B
    double JD_ut1 = CAL2JUL(ut1,+4);//julian day number of the ut1 date relative to J2000 (+4), The second argument of GMST
    ut1 = floor(ut1);
    double JD_ut1at0 = CAL2JUL(ut1,+4);//julian day number of at zero o'clock at the ut1 date relative to J2000 (+4), the first argument of GMST
    double JD_tt = CAL2JUL(tt,+4);//precesion nutation
//gmst(GMST) gast(eci2ecef)
    double gmst = GMST(JD_ut1at0,JD_ut1);
    double G0 = GAST(JD_tt,gmst);//greenwicch apparent sidereal time at utc above
//transformation matrix from Jdate -> J2000
    vector<vector<double> > MJDJ2 = eci2ecef(JD_tt,gmst,0.,0.,1100,-1);

//******* true trajectory propagation******//
    vector<double> xreal(nvp,0.);//true trajectory
    xreal[0] = -0.68787*a;
    xreal[1] = -0.39713*a;
    xreal[2] =  0.28448*a;
    xreal[3] = -0.51331*sqrt(GM/a);
    xreal[4] = +0.98266*sqrt(GM/a);
    xreal[5] = +0.37661*sqrt(GM/a);
    xreal[6] = 1.5e-6;
    xreal[7] = 1.2e-6;
    xreal[8] = 1.8e-6;
    double thrust = sqrt(xreal[6]*xreal[6]+xreal[7]*xreal[7]+xreal[8]*xreal[8]);

//******* HNEKF propagation******//
//inintial state
    vector<double> dx(nvp,0.);
    dx[0] = 1.2*uncr;
    dx[1] = -1.3*uncr;
    dx[2] = 0.8*uncr;
    dx[3] = 1.2*uncv;
    dx[4] = 1.4*uncv;
    dx[5] = 1.5*uncv;
    dx[6] = 1.0*uncp;
    dx[7] = -1.1*uncp;
    dx[8] = 1.5*uncp;
    vector<double> x(xreal);
    for(int i=0;i<9;++i)
        x[i] = xreal[i]+dx[i];
//initial error covariance
    vector< vector <double> > P(nvp,vector<double>(nvp,0.));
    for(int i=0;i<9;++i)
        P[i][i] = dx[i]*dx[i];
//process noise corvariance matrix
    //no process noise
    vector<vector <double> > Q(nvp,vector<double>(nvp,0.));
    Q[3][3] = 1.2e-13;
    Q[4][4] = 1.2e-13;
    Q[5][5] = 1.2e-13;
//measurement noise corvariance matrix
    vector<vector <double> > R(nobs,vector<double>(nobs,0.));
    R[0][0] = mea_err*mea_err;
    R[1][1] = mea_err*mea_err;
    R[2][2] = mea_err*mea_err;
    
//propagation
//initialize the initial polynomials corresponding to the initial combination of states(parameters) and uncertainty  
    vector<double> unc_x(nvp,1.);
    vector<taylor_polynomial<double> > x_p0, x_pf;
    for(int i=0;i<nvp;i++)
        x_p0.push_back(taylor_polynomial<double>(nvp, degree, i, x[i]-unc_x[i],x[i]+unc_x[i]));
//dynamcal system
    //ACCURATE MODEL
    //dynamics::cartesian<taylor_polynomial<double> >dyn(G0,GM,w,MJDJ2,degree,tt,CR,areatomass,flag_har,flag_srp,flag_third);//without parameter, no bracket
    //INACCURATE MODEL
    dynamics::cartesian<taylor_polynomial<double> >dyn(G0,GM,w,MJDJ2,degree,tt,CR,areatomass,flag_har,0,flag_third);//without parameter, no bracket
    dynamics::cartesianNum<double> dyn_nume(G0,GM,w,MJDJ2,tt,CR,areatomass,flag_har,flag_srp,flag_third);//without parameter, no bracket
//integration scheem 
    integrator::rk78<taylor_polynomial<double> > integrator(&dyn);
    integrator::rk78num<double> integratornum(&dyn_nume);
//HNKEF
    vector<double> x_estimate(nvp,0.);
    vector<double> z_estimate(nobs,0.);
    vector<vector<double> > p_estimate(nvp,vector<double>(nvp,0.));
    vector<vector<double> > pzz_estimate(nobs,vector<double>(nobs,0.));
    vector<vector<double> > pzzinv_estimate(nobs,vector<double>(nobs,0.));
    vector<vector<double> > pxz_estimate(nvp,vector<double>(nobs,0.));
    vector<vector<double> > K(nvp,vector<double>(nobs,0.));
    vector<double> deltam(nvp,0.);//delta m{-}{k+1}
    vector<double> deltan(nobs,0.);//delta n{-}{k+1}
    vector<double> xf(nvp,0.);
    vector<double> xn0(nvp,0.);
    vector<double> zf(nobs,0.);
    for(int i=0;i<nvp;i++)
        xf[i] = xreal[i];
    vector<double> output_step(14,0.);
    vector<vector<double> > output;
//store initial of output 
    for(int j=0;j<nvp;j++){
        output_step[j] = x[j] - xreal[j]; 
    }


    output_step[9] = sqrt(output_step[0]*output_step[0]+output_step[1]*output_step[1]+output_step[2]*output_step[2])/1000.;
    output_step[10] = sqrt(output_step[3]*output_step[3]+output_step[4]*output_step[4]+output_step[5]*output_step[5]);
    output_step[11] = sqrt(output_step[6]*output_step[6]+output_step[7]*output_step[7]+output_step[8]*output_step[8]);
    output_step[12] = output_step[11]/thrust;
    output_step[13] = 0.;
    output.push_back(output_step);
   
    double T = 1500.;
    t0=0.;//initial time for one prediction
    double t0n=0.;//initial time for one prediction
    double tf=0.;//final time for one prediction
    double h=0.2;
    double hn=0.2;
    double hdid=0.;
    double tol=1.e-11;//10;
    double hmin = 0.001;
    double hmax = 200.;
    double hminn = 0.001;
    double hmaxn = 2000.;
    int num = T/period;
    for(int i=0;i<num;++i){
        tf += period;
        cout << tf <<  "\n";
        while(t0<tf){
            if(fabs(tf-t0)<h)
                h = tf-t0;
            integrator.integrate(t0,x_p0,h,hdid,hmin,hmax,tol,x_pf);
        }

        while(t0n<tf){
            if(fabs(tf-t0n)<hn)
                hn = tf-t0n;
            integratornum.integrate(t0n,xf,hn,hdid,hminn,hmaxn,tol,xn0);
        }

        //assume measurement = true trajectory + gaussian noise
        vector<double> rand_samp(nobs,0.);
        for(int j=0;j<nobs;j++)
            rand_samp[j] = random_num[i][j];
        for(int j=0;j<3;++j)
            zf[j] = xf[j]+mea_err*rand_samp[j]; 
        //state prediction
        for(int j=0;j<nvp;j++){
            vector<double> exp_dif(2,0.);
            exp_dif = p1_expectation(x_p0[j],P);
            x_estimate[j] = exp_dif[0];
            deltam[j] = exp_dif[1];//delta m{-}{k+1}
        }
        //covariance prediction
        for(int j=0;j<nvp;j++)
            for(int k=0;k<nvp;k++)
                p_estimate[j][k] = p2_expectation(x_p0[j],x_p0[k],P)-deltam[j]*deltam[k]+Q[j][k];
        //measurement prediction
        vector<taylor_polynomial<double> > z;
        for(int j=0;j<3;++j)
            z.push_back(x_p0[j]);
        for(int j=0;j<nobs;j++){
            vector<double> exp_dif(2,0.);
            exp_dif = p1_expectation(z[j],P);
            z_estimate[j] = exp_dif[0];
            deltan[j] = exp_dif[1];//delta n{-}{k+1}
        }
        //correction
        //covariance pzz
        for(int j=0;j<nobs;j++)
            for(int k=0;k<nobs;k++)
                pzz_estimate[j][k] = p2_expectation(z[j],z[k],P)-deltan[j]*deltan[k]+R[j][k];
        //covariance pxz
        for(int j=0;j<nvp;j++)
            for(int k=0;k<nobs;k++)
                pxz_estimate[j][k] = p2_expectation(x_p0[j],z[k],P)-deltam[j]*deltan[k];
        //K kalman gain
        pzzinv_estimate = inv_mat(pzz_estimate);
        K = mat_mul(pxz_estimate,pzzinv_estimate);

        //state updated
        vector<vector<double> > diff(nobs,vector<double>(1,0.));
        vector<vector<double> > diff_c;
        for(int j=0;j<nobs;j++){
            diff[j][0] = zf[j]-z_estimate[j];
        }
        diff_c = mat_mul(K,diff);
        for(int j=0;j<nvp;j++){
            x_estimate[j] += diff_c[j][0];
        }
        //covariance updated
        vector<vector<double> > matkp(nvp,vector<double>(nobs,0.));
        vector<vector<double> > kt(nobs,vector<double>(nvp,0.));//transpose
        vector<vector<double> > matkpk(nvp,vector<double>(nvp,0.));
        matkp = mat_mul(K,pzz_estimate);
        kt = transposition(K);
        matkpk = mat_mul(matkp,kt);
        for(int j=0;j<nvp;j++)
            for(int k=0;k<nvp;k++)
                P[j][k] =p_estimate[j][k]-matkpk[j][k];



        for(int j=0;j<nvp;j++)
            output_step[j] = fabs(x_estimate[j] - xn0[j]); 
        output_step[9] = sqrt(output_step[0]*output_step[0]+output_step[1]*output_step[1]+output_step[2]*output_step[2])/1000.;
        output_step[10] = sqrt(output_step[3]*output_step[3]+output_step[4]*output_step[4]+output_step[5]*output_step[5]);
        output_step[11] = sqrt(output_step[6]*output_step[6]+output_step[7]*output_step[7]+output_step[8]*output_step[8]);
        output_step[12] = output_step[11]/thrust;
        output_step[13] = tf;//hour
        output.push_back(output_step);

        //the next step
        x_p0.clear();
        for(int i=0;i<nvp;i++)
            x_p0.push_back(taylor_polynomial<double>(nvp, degree, i, x_estimate[i]-unc_x[i],x_estimate[i]+unc_x[i]));
    }
     //printing 
     ofstream file;
     if(degree ==1)
     file.open ("firstorderresult.txt");
     if(degree ==2)
     file.open ("secondorderresult.txt");
     if(degree ==3)
     file.open ("thirdorderresult.txt");
     if(degree ==4)
     file.open ("fourthorderresult.txt");
     for(unsigned int k=0;k<output.size();k++){
         for(unsigned int kk=9;kk<output[k].size();kk++)
             file << setw(16) << setprecision(10) << output[k][kk] << " ";
         file << "\n";
     }
     file << "\n\n\n\n";
     file.close();
    return 0;
}

