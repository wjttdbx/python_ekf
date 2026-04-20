#ifndef _CONSTANT_H_	
#define _CONSTANT_H_

//===================================================================================================================
// Math
static const double pi = 3.14159265358979324;   //¦Ð
static const double e = 2.71828182845904524;   //e

// Tolerance
static const double tor_integrate = 1e-13;
static const double tor_Discrimination = 1e-13;

//-------------------------------------------------------------------------------------------------------------------


const size_t MotionModel_dim = 3;  // The dimension of the state vector is defaulted to 3 dimensions (x,y,z)
const size_t Kepler_Orbit_dim = 6;  // The dimension of the state vector is defaulted to 3 dimensions (x,y,z)


//===================================================================================================================
// Earth(WGS-84)
  // WGS-84 ellipsoid
static const double mu = 398600.4418;	// Gravitational constant   Unit:km^3/s^2
static const double Re = 6378.137;	// Equatorial radius   Unit:km
static const double Re_polar = 6356.752314245;	// Polar radius   Unit:km
static const double Re_mean = 6371.0088;		 // Average radius of the Earth   Unit:km
static const double rotational = 7.2921150e-5; //Spin velocity   Unit:rad/s
static const double J2 = 1.082626683553e-3;        // J2 perturbation item
static const double J3 = -2.532410518567e-6;          // J3 perturbation item
static const double oblateness = 1.0 / 298.257223563;   // Earth flattening
static const double gravitational_modified = 1.93185138639e-3; // Correction coefficient of the gravitational difference between the equator and the polar regions
static const double eccentricity_squared = 6.6943799901413165e-3;   // Square of the first eccentricity of the ellipsoid
  //´óÆø
const double g_eq = 9.7803253359;		// Theoretical gravitational acceleration at the equatorial sea level   Unit:m/s^2
const double g_0 = 9.80665;			 // Standard gravitational acceleration   Unit:m/s^2
const double Atmospheric_pressure_0 = 101.325; // Standard atmospheric pressure at sea level   Unit:kPa
const double Temperature_0 = 288.15;   // Standard temperature at sea level   Unit:K
const double air_mol = 28.9647e-3;		// Molar mass of dry air   Unit:kg/mol
const double dry_Gas = 287.053;			// Gas constant of dry air   Unit:J/(mol¡¤K) 
//-------------------------------------------------------------------------------------------------------------------


#endif
