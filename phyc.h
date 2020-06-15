#ifndef PHYCONSTEXPR_H
#define PHYCONSTEXPR_H

/* --- PHYSICAL AND CONVERSION CONSTEXPRANTS IN CGS --- 
 *     Source: NIST adjusted values from 2010 (?)
 *
 *     Jaime de la Cruz Rodriguez (ISP-SU 2015)
 *     
 */

namespace phyc{
  static constexpr double BK = 1.3806488E-16;               // Boltzmann [erg K]
  static constexpr double HH = 6.62606957E-27;              // Planck [erg s]
  static constexpr double EE = 4.80320441E-10;              // Electron charge
  static constexpr double CC = 2.99792458E10;               // Speed of light
  static constexpr double PI = 3.14159265358979323846;      // Pi
  static constexpr double SQPI = 1.7724538509055159;        // sqrt(PI)
  static constexpr double ME = 9.10938188E-28;              // mass of electron
  static constexpr double MP = 1.672621777E-24;             // mass of proton
  static constexpr double AMU = 1.660538921E-24;            // Atomic mass unit
  static constexpr double R0 = 5.2917721092E-11;            // Bohr radius
  static constexpr double ISQRTPI = 0.5641895835477563;     // 1 / sqrt(pi)
  static constexpr double DTOR = PI / 180.0;                // deg to rad
  static constexpr double RADEG = 180.0 / PI;               // rad to deg
  static constexpr double EV = 1.602176565E-12;             // Electron Volt to erg
  static constexpr double CM1_TO_EV = HH*CC/EV;             // CM^-1 to eV
  static constexpr double EV_TO_CM1 = EV / (HH*CC);         // eV to CM^-1
  
  /* ---                       --- */
}

#endif
