#ifndef LINE_H
#define LINE_H

/* -------------------------------------------------------
   
   Line class contains the atomic info required for the ME class
   Coded by J. de la Cruz Rodriguez (ISP-SU, 2020)
   
   ------------------------------------------------------- */


#include <cstdio>
#include <cmath>
#include <iostream>
#include "phyc.h"

namespace ln{
  static constexpr double zeeman_const = (phyc::EE / (4.0e8 * phyc::PI * phyc::ME * phyc::CC * phyc::CC));

  template<class T> struct line{

    double w0, dw;
    int nZ;
    int l0, l1;
    T gf;
    std::vector<T> strength, splitting;
    std::vector<int> iL;
    
    line():w0(0), dw(0.5), nZ(0), l0(0), l1(0), gf(0){};
    line(const T j1, const T j2, const T g1, const T g2, const T  igf, const double lam0, bool anomalous = true, T const dw_in = 0.5):
      w0(0), dw(dw_in), nZ(0), l0(0), l1(0), gf(igf)
    {
      if(anomalous) setSplitting(j1, j2, g1, g2, lam0);
      else          setSplittingNorm(j1, j2, g1, g2, lam0);
      
      fprintf(stderr, "line::line: Initialized [%f] -> %3d Zeeman components\n", w0, nZ);
      
    }
    void setSplitting(const T &j1, const T &j2, const T &g1, const T &g2, const double &lam0)
    {
      nZ = 0, w0 = lam0;
      const double w02 = w0*w0;

      std::vector<T> normalization(3,T(0.0));
      
      int nup = int(2 * j2) + 1, delta_j = int(j2 - j1);      
      for(int iup = 1; iup <= nup; iup++){
	T Mup = j2 + 1 - iup;
	for(int ilow = 1; ilow <= 3; ilow++){
	  T Mlow = Mup - 2 + ilow;
	  if(std::abs(Mlow) <= j1){
	    
	    /* --- Compute relative Zeeman strength, 
	       Landi Degl'innocenti & Landolfi (2004), 
	       table 3.1 - pag. 81 --- */
	    
	    T strength = 0; // If abs(delta_j) > 1 then don't compute Zeeman splitting. 
	    //
	    if(delta_j == 1){
	      
	      if(ilow == 1)      strength = 1.5 * (j1 + Mlow + 1.0) * (j1 + Mlow + 2.0)
				   / ((j1+1.0)*(2.0*j1 + 1.0) * (2.0 * j1 + 3.0));
	      else if(ilow == 2) strength = 3.0 * (j1 - Mlow + 1.0) * (j1 + Mlow + 1.0)
				   / ((j1+1.0)*(2.0*j1 + 1.0) * (2.0 * j1 + 3.0));
	      else               strength = 1.5 * (j1 - Mlow + 1.0) * (j1 - Mlow + 2.0)
				   / ((j1+1.0)*(2.0*j1 + 1.0) * (2.0 * j1 + 3.0));
	      
	    } else if(delta_j == 0){
	      
	      if(ilow == 1)      strength = 1.5 * (j1 - Mlow) * (j1 + Mlow + 1.0)
				   / (j1 * (j1 + 1.0) * (2.0 * j1 + 1.0));
	      else if(ilow == 2) strength = 3.0 * Mlow * Mlow
				   / (j1 * (j1 + 1.0) * (2.0 * j1 + 1.0));
	      
	      else               strength = 1.5 * (j1 + Mlow) * (j1 - Mlow + 1.0)
				   / (j1 * (j1 + 1.0) * (2.0 * j1 + 1.0));
	      
	    } else if(delta_j == -1){
	      
	      if(ilow == 1)      strength = 1.5 * (j1 - Mlow) * (j1 - Mlow - 1.0)
				   / (j1 * (2.0 * j1 - 1.0) * (2.0 * j1 + 1.0));
	      else if(ilow == 2) strength = 3.0 * (j1 - Mlow) * (j1 + Mlow)
				   / (j1 * (2.0 * j1 - 1.0) * (2.0 * j1 + 1.0));
	      else               strength = 1.5 * (j1 + Mlow) * (j1 + Mlow - 1.0)
				   / (j1 * (2.0 * j1 - 1.0) * (2.0 * j1 + 1.0));
	      
	    }
	    
	    
	    /* --- Zeeman splitting and strength ---*/

	    if(strength <= 1.e-5) continue;
	    
	    double const split =  -(g2*Mup - g1*Mlow)*zeeman_const * w02;
	    splitting.push_back(split);
	    this->strength.push_back(strength);
	    iL.push_back(ilow-1);
	    
	    nZ += 1;
	  }
	}
      }

      
    }
    
    // -------------------------------------------------------------------------------------------------------------//
    
    line(const line &lin): w0(lin.w0), dw(lin.dw), nZ(lin.nZ), l0(lin.l0), l1(lin.l1), gf(lin.gf), strength(lin.strength), splitting(lin.splitting), iL(lin.iL){};
    
    // -------------------------------------------------------------------------------------------------------------//
    
    line &operator=(const line &lin){w0 = lin.w0, dw = lin.dw, nZ = lin.nZ, gf = lin.gf, strength = lin.strength, splitting = lin.splitting, iL = lin.iL; l0=lin.l0; l1=lin.l1; return *this;};
    
    // -------------------------------------------------------------------------------------------------------------//

    void setSplittingNorm(const T &j1, const T &j2, const T &g1, const T &g2, const double &lam0)
    {
      static const T dM[3] = {-1.0,0.0,1.0};
      nZ = 0, w0 = lam0;
      
      // --- Compute geff --- //
      
      T d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0), geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d;

      // --- blue comp --- //

      nZ = 0;
      for(int ii=0; ii<3;++ii){
	splitting.push_back(-zeeman_const * lam0*lam0 * geff * dM[ii]);
	strength.push_back(1.0);
	iL.push_back(ii);
	nZ += 1;
      }
      
    }
  };

  // -------------------------------------------------------------------------------------------------------------//
 
}



#endif
