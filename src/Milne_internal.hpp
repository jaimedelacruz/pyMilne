#ifndef MILNEINTERNALHPP
#define MILNEINTERNALHPP
/* -------------------------------------------------------
   
   Helper functions for the Milne class
   Coded by J. de la Cruz Rodriguez (ISP-SU, 2020)
   
   ------------------------------------------------------- */

#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>

#include "profile.hpp"
#include "line.hpp"

namespace ml{
  
  // ******************************************************************************************* //

  template<typename T>
  void compute_K(int const nWav, const double* __restrict__ wav, T* const __restrict__ buff, T const B,
		     T const inc, T const azi, T const vl, T const vD, T const eta, T const dam,
		     std::vector<ln::line<T>> const& lines, T const sinin, T const cosin, T const sin2az,
		     T const cos2az)
  {

    T const sinin2   = sinin*sinin;
    T const cosin2   = cosin*cosin;

    // --- Init pointers to buffer --- //
    
    T* const __restrict__ Ki = &buff[0*nWav];
    T* const __restrict__ Kq = &buff[1*nWav];
    T* const __restrict__ Ku = &buff[2*nWav];
    T* const __restrict__ Kv = &buff[3*nWav];
    T* const __restrict__ Fq = &buff[4*nWav];
    T* const __restrict__ Fu = &buff[5*nWav];
    T* const __restrict__ Fv = &buff[6*nWav];

    T* const __restrict__ H  = &buff[7*nWav];
    T* const __restrict__ F  = &buff[10*nWav];

    T* const __restrict__ h  = &buff[13*nWav];
    T* const __restrict__ f  = &buff[14*nWav];
    
    
    // --- Add continuum contribution to Ki --- //
    for(int ww = 0; ww<nWav; ++ww) Ki[ww] = 1;
 
    

    // --- Loop line Zeeman components --- //

    int const nLines = int(lines.size());
    
    for(int ii=0; ii<nLines; ++ii){
      ln::line<T> const& iLine = lines[ii];

      // --- Zero buffer elements --- //
      std::memset(&buff[7*nWav], 0, 8*nWav*sizeof(T));

      // --- pre-compute some cuantities common to all components --- //
      int const nComp  = int(iLine.splitting.size());

      T const wav0 = iLine.w0;
      T const va   = iLine.w0 * (vl / (phyc::CC*1.e-5));
      
      int const l0 = iLine.l0;
      int const l1 = iLine.l1;
      int const nWav_line = l1-l0+1;

      // --- Loop Zeeman components --- //
      
      for(int kk=0; kk<nComp; ++kk){
	
	int const iL = iLine.iL[kk];
	T const vb   =  -iLine.splitting[kk] * B; // Negative because we work in lambda
	T const str  =   iLine.strength[kk];
	
	// --- Compute profile of this component --- //
	
	pr::compute_profile<T>(nWav_line, &wav[l0], wav0, str, va, dam, vD, vb, &h[l0], &f[l0]);



	// --- Add to the corresponding polarization state --- //

	for(int ww=l0; ww <= l1; ++ww){
	  H[iL*nWav + ww] += h[ww];
	  F[iL*nWav + ww] += 2*f[ww];
	}	
      } // kk (Zeeman components)

      

      // --- Now add to the abs. matrix --- //

      T const eta2 = iLine.gf * eta / 2;
      
      for(int ww=l0; ww <= l1; ++ww){

	T const tmp  = eta2 * (H[nWav+ww] -  (  H[ww] + H[2*nWav+ww])/2) * sinin2 ;
	T const tmp1 = eta2 * (F[nWav+ww] -  (  F[ww] + F[2*nWav+ww])/2) * sinin2 ;
	
	Kq[ww] += tmp * cos2az;
	Ku[ww] += tmp * sin2az;
	Fq[ww] += tmp1 * cos2az;
	Fu[ww] += tmp1 * sin2az;
      }

      for(int ww=l0; ww <= l1; ++ww)
	Ki[ww] += eta2 * (H[nWav+ww]*sinin2 + (H[ww] + H[2*nWav+ww])*(1+cosin2) / 2);
      
      for(int ww=l0; ww <= l1; ++ww)
	Kv[ww] += eta2*(H[2*nWav+ww] - H[ww]) * cosin;
      
      for(int ww=l0; ww <= l1; ++ww)
	Fv[ww] += eta2*(F[2*nWav+ww] - F[ww]) * cosin;
      

      
    } // ii (lines)


  }


  // ******************************************************************************************* //

  template<typename T>
  void compute_Stokes(int const nWav, const T* const __restrict__ buffer, T* const __restrict__ out, T const S0, T const S1, T const mu)
  {

    const T* const __restrict__ ki = &buffer[0*nWav];
    const T* const __restrict__ kq = &buffer[1*nWav];
    const T* const __restrict__ ku = &buffer[2*nWav];
    const T* const __restrict__ kv = &buffer[3*nWav];
    const T* const __restrict__ fq = &buffer[4*nWav];
    const T* const __restrict__ fu = &buffer[5*nWav];
    const T* const __restrict__ fv = &buffer[6*nWav];

    T* const __restrict__ I = &out[0     ];
    T* const __restrict__ Q = &out[1*nWav];
    T* const __restrict__ U = &out[2*nWav];
    T* const __restrict__ V = &out[3*nWav];

    
    for(int ww=0; ww<nWav; ++ww){

      T const ki2 = ki[ww]*ki[ww];
      T const fq2 = fq[ww]*fq[ww];
      T const fu2 = fu[ww]*fu[ww];
      T const fv2 = fv[ww]*fv[ww];
      
      T const GP3 = ki2 + fq2 + fu2 + fv2;
      T const GP1 = GP3 - ((kq[ww]*kq[ww]) + (ku[ww]*ku[ww]) + (kv[ww]*kv[ww]));
      T const GP2 = kq[ww]*fq[ww] + ku[ww]*fu[ww] + kv[ww]*fv[ww];
       
      T const del   = ki2 * GP1 - GP2*GP2;
      T const S1mud = S1 * (mu / del);

      T const GP4 = ki2 * kq[ww] + ki[ww] * (kv[ww]*fu[ww] - ku[ww]*fv[ww]);
      T const GP5 = ki2 * ku[ww] + ki[ww] * (kq[ww]*fv[ww] - kv[ww]*fq[ww]);
      T const GP6 = ki2 * kv[ww] + ki[ww] * (ku[ww]*fq[ww] - kq[ww]*fu[ww]);

      I[ww] = S0 + (S1mud * ki[ww] * GP3);
      Q[ww] = -S1mud * (GP4 + fq[ww] * GP2);
      U[ww] = -S1mud * (GP5 + fu[ww] * GP2);
      V[ww] = -S1mud * (GP6 + fv[ww] * GP2);
      
    } // ww
  }
  
  
  // ******************************************************************************************* //

  
  template<typename T>
  void ME_Synth(std::vector<double> const &wav, std::vector<ln::line<T>> const& lines,
		    const T* const __restrict__ m, T* const __restrict__ out, T const mu)
  {

    int const nWav = int(wav.size());
    std::memset(out, 0, 4*nWav*sizeof(T));
    
    
    // --- Copy to local scalars ---//
    
    T const B   = m[0];
    T const inc = m[1];
    T const azi = m[2];
    T const vl  = m[3];
    T const vD  = m[4];
    T const eta = m[5];
    T const dam = m[6];
    T const S0  = m[7];
    T const S1  = m[8];

    T const sinin = sin(inc);
    T const cosin = cos(inc);
    T const sin2az= sin(2*azi);
    T const cos2az= cos(2*azi);

    
    // --- Allocate memory buffer ---//
    
    T* const __restrict__ buff = new T [nWav*15]();
    

    
    // --- Compute absorption matrix elements --- //
    
    ml::compute_K<T>(nWav, &wav[0], buff, B, inc, azi, vl, vD, eta, dam, lines, sinin, cosin, sin2az, cos2az);

    

    // --- Compute emerging spectra --- //
    
    ml::compute_Stokes<T>(nWav, buff, out, S0, S1, mu);



    // --- clean-up --- //

    delete [] buff;
  }

  // ******************************************************************************************* //

  template<typename T> inline
  void fillDopacDF(T lineop, T const sinin2, T const cosin,
	      T const cos2az, T const sin2az, T const dH0, T const dH1,
	      T const dH2, T const dF0, T const dF1, T const dF2, T &dki, T &dkq,
	      T &dku, T &dkv, T &dfq, T &dfu, T &dfv)
  {
    {
      lineop /= 2;
      
      T const dum = lineop * (dH1 - (dH0 + dH2)/2)* sinin2;
    
      dki += lineop * (dH1 * sinin2 + (dH0 + dH2) * (cosin*cosin+1)/2);
      dkq += dum * cos2az;
      dku += dum * sin2az;
      dkv += lineop * (dH2 - dH0) * cosin;
    }
    {
      T const dum = lineop * (dF1 -  (dF0 + dF2)/2) * sinin2;
      dfq += dum * cos2az;
      dfu += dum * sin2az;
      dfv += lineop * (dF2 - dF0) * cosin;
    }
  }
  
  // ******************************************************************************************* //

  template<typename T>
  void compute_K_der(int const nWav, const double* __restrict__ wav, T* const __restrict__ buff, T const B,
		     T const inc, T const azi, T const vl, T const vD, T const eta, T const dam,
		     std::vector<ln::line<T>> const& lines, T const sinin, T const cosin, T const sin2az,
		     T const cos2az)
  {

    T const sinin2   = sinin*sinin;
    //T const cosin2   = cosin*cosin;
    T const sinda    = 2*cos2az;
    T const cosda    = -2*sin2az;
    T const cosdi    = -sinin;
    T const sin2in   = sin(2*inc);
    T const cosin2_1 = ((cosin*cosin) + 1)/2;

    
    // --- Init pointers to buffer --- //
    
    T* const __restrict__ Ki = &buff[0*nWav];
    T* const __restrict__ Kq = &buff[1*nWav];
    T* const __restrict__ Ku = &buff[2*nWav];
    T* const __restrict__ Kv = &buff[3*nWav];
    T* const __restrict__ Fq = &buff[4*nWav];
    T* const __restrict__ Fu = &buff[5*nWav];
    T* const __restrict__ Fv = &buff[6*nWav];

    T* const __restrict__ H  = &buff[7*nWav];
    T* const __restrict__ F  = &buff[10*nWav];

    T* const __restrict__ h  = &buff[13*nWav];
    T* const __restrict__ f  = &buff[14*nWav];
    
    
    // --- Add continuum contribution to Ki --- //
    
    for(int ww = 0; ww<nWav; ++ww) Ki[ww] = 1;


    
    // --- Loop lines --- //
    
    int const nLines = int(lines.size());
    
    for(int ii=0; ii<nLines; ++ii){
    
      ln::line<T> const& iLine = lines[ii];

      // --- Zero buffer elements --- //
      std::memset(&buff[7*nWav], 0, (24+8)*nWav*sizeof(T));

      // --- pre-compute some cuantities common to all components --- //

      int const nComp  = int(iLine.splitting.size());

      T const wav0     = iLine.w0;
      T const va       = iLine.w0 * (vl / (phyc::CC*1.e-5));
      T const dvadvlos = -iLine.w0  / (phyc::CC*1.e-5) / vD;
      
      int const l0 = iLine.l0;
      int const l1 = iLine.l1;
      int const nWav_line = l1-l0+1;
      
      // --- Loop Zeeman components --- //
      
      for(int kk=0; kk<nComp; ++kk){
	
	int const iL = iLine.iL[kk];
	T const vb   = -iLine.splitting[kk] * B; // Negative because we work in lambda
	T const dvB  = -iLine.splitting[kk] / vD;
	T const str  = iLine.strength[kk];
	T const strnpi = str / phyc::SQPI;
	
	
	// --- Compute profile of this component --- //
	
	pr::compute_profile<T>(nWav_line, &wav[l0], wav0, str, va, dam, vD, vb, &h[l0], &f[l0]);


	
	// --- Add to the corresponding polarization state --- //
	{
	  int const off = iL*nWav;
	  for(int ww=l0; ww <= l1; ++ww){
	    H[off + ww] +=     h[ww];
	    F[off + ww] += 2 * f[ww];
	  }
	}

	// --- Buffers for derivatives of the profile respect to B, vl, vD, dam --- //
	{
	  T* const __restrict__ dH0 = &buff[15*nWav + iL*4*nWav         ];
	  T* const __restrict__ dH1 = &buff[15*nWav + iL*4*nWav + 1*nWav];
	  T* const __restrict__ dH2 = &buff[15*nWav + iL*4*nWav + 2*nWav];
	  T* const __restrict__ dH3 = &buff[15*nWav + iL*4*nWav + 3*nWav];
	  
	  T* const __restrict__ dF0 = &buff[15*nWav + 12*nWav + iL*4*nWav         ];
	  T* const __restrict__ dF1 = &buff[15*nWav + 12*nWav + iL*4*nWav + 1*nWav];
	  T* const __restrict__ dF2 = &buff[15*nWav + 12*nWav + iL*4*nWav + 2*nWav];
	  T* const __restrict__ dF3 = &buff[15*nWav + 12*nWav + iL*4*nWav + 3*nWav];	
	  
	  
	  
	  for(int ww=l0; ww <= l1; ++ww){
	    
	    // --- Common terms --- //
	    
	    T const Vld  = (wav[ww] - wav0 - va + vb) / vD;
	    T const dhdv =           4*dam*f[ww] - 2*Vld*h[ww];
	    T const dfdv = 2*(strnpi - dam*h[ww] - 2*Vld*f[ww]);
	    
	    T const dvdop= -Vld / vD;
	    
	    // --- B --- //
	    
	    dH0[ww] += dhdv*dvB;
	    dF0[ww] += dfdv*dvB;
	    
	    
	    // --- vlos --- //
	    
	    dH1[ww] += dhdv*dvadvlos;
	    dF1[ww] += dfdv*dvadvlos;
	    
	    
	    // --- vD --- //
	    
	    dH2[ww] += dhdv * dvdop;
	    dF2[ww] += dfdv * dvdop;
	    
	    
	    // --- Damp --- //
	    
	    dH3[ww] += -dfdv;
	    dF3[ww] +=  dhdv;
	  } // ww
	}// scope
      } // kk

      
      // --- Pointers to derivatives of K --- //

      
      int off = 15*nWav + 24*nWav;
      T* const __restrict__ dK0_dB = &buff[off];
      T* const __restrict__ dK1_dB = &buff[off+nWav];
      T* const __restrict__ dK2_dB = &buff[off+2*nWav];
      T* const __restrict__ dK3_dB = &buff[off+3*nWav];
      T* const __restrict__ dK4_dB = &buff[off+4*nWav];
      T* const __restrict__ dK5_dB = &buff[off+5*nWav];
      T* const __restrict__ dK6_dB = &buff[off+6*nWav];
      
      off += 7*nWav;
      T* const __restrict__ dK0_dinc = &buff[off];
      T* const __restrict__ dK1_dinc = &buff[off+nWav];
      T* const __restrict__ dK2_dinc = &buff[off+2*nWav];
      T* const __restrict__ dK3_dinc = &buff[off+3*nWav];
      T* const __restrict__ dK4_dinc = &buff[off+4*nWav];
      T* const __restrict__ dK5_dinc = &buff[off+5*nWav];
      T* const __restrict__ dK6_dinc = &buff[off+6*nWav];
      
      off += 7*nWav;
      //T* const __restrict__ dK0_dazi = &buff[off];
      T* const __restrict__ dK1_dazi = &buff[off+nWav];
      T* const __restrict__ dK2_dazi = &buff[off+2*nWav];
      //T* const __restrict__ dK3_dazi = &buff[off+3*nWav];
      T* const __restrict__ dK4_dazi = &buff[off+4*nWav];
      T* const __restrict__ dK5_dazi = &buff[off+5*nWav];
      //T* const __restrict__ dK6_dazi = &buff[off+6*nWav];
      
      off += 7*nWav;
      T* const __restrict__ dK0_dvl = &buff[off];
      T* const __restrict__ dK1_dvl = &buff[off+nWav];
      T* const __restrict__ dK2_dvl = &buff[off+2*nWav];
      T* const __restrict__ dK3_dvl = &buff[off+3*nWav];
      T* const __restrict__ dK4_dvl = &buff[off+4*nWav];
      T* const __restrict__ dK5_dvl = &buff[off+5*nWav];
      T* const __restrict__ dK6_dvl = &buff[off+6*nWav];

      off += 7*nWav;
      T* const __restrict__ dK0_dvD = &buff[off];
      T* const __restrict__ dK1_dvD = &buff[off+nWav];
      T* const __restrict__ dK2_dvD = &buff[off+2*nWav];
      T* const __restrict__ dK3_dvD = &buff[off+3*nWav];
      T* const __restrict__ dK4_dvD = &buff[off+4*nWav];
      T* const __restrict__ dK5_dvD = &buff[off+5*nWav];
      T* const __restrict__ dK6_dvD = &buff[off+6*nWav];

      off += 7*nWav;
      T* const __restrict__ dK0_deta = &buff[off];
      T* const __restrict__ dK1_deta = &buff[off+nWav];
      T* const __restrict__ dK2_deta = &buff[off+2*nWav];
      T* const __restrict__ dK3_deta = &buff[off+3*nWav];
      T* const __restrict__ dK4_deta = &buff[off+4*nWav];
      T* const __restrict__ dK5_deta = &buff[off+5*nWav];
      T* const __restrict__ dK6_deta = &buff[off+6*nWav];

      off += 7*nWav;
      T* const __restrict__ dK0_ddam = &buff[off];
      T* const __restrict__ dK1_ddam = &buff[off+nWav];
      T* const __restrict__ dK2_ddam = &buff[off+2*nWav];
      T* const __restrict__ dK3_ddam = &buff[off+3*nWav];
      T* const __restrict__ dK4_ddam = &buff[off+4*nWav];
      T* const __restrict__ dK5_ddam = &buff[off+5*nWav];
      T* const __restrict__ dK6_ddam = &buff[off+6*nWav];
      
      
      // --- Now compute derivatives of K --- //

      T const rat  = iLine.gf;
      T const eta2 = rat * eta;
      int const off1 = 15*nWav;
      int const off2 = 15*nWav + 12*nWav;
      
      for(int ww=l0; ww <= l1; ++ww){

	int const w0   = ww;
	int const w1   = nWav + ww;
	int const w2   = 2*nWav + ww;
	
	T const hsum = H[w0] + H[w2];
	T const fsum = F[w0] + F[w2];
	T const hsub = H[w2] - H[w0];
	T const fsub = F[w2] - F[w0];
	
	T tmp0 = (H[w1] * sinin2 + hsum * cosin2_1)/2; // Note that cosin2_1 is divided by two higher up
	T tmp  = (H[w1] - hsum/2)/2;
	T tmp1 = (F[w1] - fsum/2)/2;
	T tmp2 = hsub/2;
	T tmp3 = fsub/2;

	
	// --- Derivatives with respect to eta_l --- //

	dK0_deta[ww] += rat*tmp0;
	dK1_deta[ww] += rat*(tmp  * cos2az * sinin2);
	dK2_deta[ww] += rat*(tmp  * sin2az * sinin2);
	dK3_deta[ww] += rat*(tmp2 * cosin);
	dK4_deta[ww] += rat*(tmp1 * cos2az * sinin2);
	dK5_deta[ww] += rat*(tmp1 * sin2az * sinin2);
	dK6_deta[ww] += rat*(tmp3 * cosin);
	
	tmp0 *= eta2, tmp *= eta2, tmp1 *= eta2, tmp2 *= eta2, tmp3 *= eta2;

	
	// --- Matrix elements --- //
	
	Ki[ww] += tmp0;
	Kq[ww] += tmp  * cos2az * sinin2;
	Fq[ww] += tmp1 * cos2az * sinin2;
	Ku[ww] += tmp  * sin2az * sinin2;
	Fu[ww] += tmp1 * sin2az * sinin2;
	Kv[ww] += tmp2 * cosin;
	Fv[ww] += tmp3 * cosin;
	

	// --- Derivatives respect to B,vl,vD,dam --- //

	fillDopacDF(eta2, sinin2, cosin, cos2az, sin2az, buff[off1+0*4*nWav+0*nWav+ww],
		    buff[off1+1*4*nWav+0*nWav+ww], buff[off1+2*4*nWav+0*nWav+ww],
		    buff[off2+0*4*nWav+0*nWav+ww], buff[off2+1*4*nWav+0*nWav+ww],
		    buff[off2+2*4*nWav+0*nWav+ww], dK0_dB[ww], dK1_dB[ww], dK2_dB[ww],
		    dK3_dB[ww],dK4_dB[ww],dK5_dB[ww], dK6_dB[ww]);

	fillDopacDF(eta2, sinin2, cosin, cos2az, sin2az, buff[off1+0*4*nWav+1*nWav+ww],
		    buff[off1+1*4*nWav+1*nWav+ww], buff[off1+2*4*nWav+1*nWav+ww],
		    buff[off2+0*4*nWav+1*nWav+ww], buff[off2+1*4*nWav+1*nWav+ww],
		    buff[off2+2*4*nWav+1*nWav+ww], dK0_dvl[ww], dK1_dvl[ww], dK2_dvl[ww],
		    dK3_dvl[ww],dK4_dvl[ww],dK5_dvl[ww], dK6_dvl[ww]);

	fillDopacDF(eta2, sinin2, cosin, cos2az, sin2az, buff[off1+0*4*nWav+2*nWav+ww],
		    buff[off1+1*4*nWav+2*nWav+ww], buff[off1+2*4*nWav+2*nWav+ww],
		    buff[off2+0*4*nWav+2*nWav+ww], buff[off2+1*4*nWav+2*nWav+ww],
		    buff[off2+2*4*nWav+2*nWav+ww], dK0_dvD[ww], dK1_dvD[ww], dK2_dvD[ww],
		    dK3_dvD[ww],dK4_dvD[ww],dK5_dvD[ww], dK6_dvD[ww]);
	
	fillDopacDF(eta2, sinin2, cosin, cos2az, sin2az, buff[off1+0*4*nWav+3*nWav+ww],
		    buff[off1+1*4*nWav+3*nWav+ww], buff[off1+2*4*nWav+3*nWav+ww],
		    buff[off2+0*4*nWav+3*nWav+ww], buff[off2+1*4*nWav+3*nWav+ww],
		    buff[off2+2*4*nWav+3*nWav+ww], dK0_ddam[ww], dK1_ddam[ww], dK2_ddam[ww],
		    dK3_ddam[ww],dK4_ddam[ww],dK5_ddam[ww], dK6_ddam[ww]);


	// --- Azimuth --- //

	dK1_dazi[ww] += tmp  * sinin2 * cosda; 
	dK2_dazi[ww] += tmp  * sinin2 * sinda;
	dK4_dazi[ww] += tmp1 * sinin2 * cosda;
	dK5_dazi[ww] += tmp1 * sinin2 * sinda;


	// --- inclination --- //

	tmp *= sin2in;
	tmp1 *= sin2in;
	//
	dK0_dinc[ww] += tmp;
	dK1_dinc[ww] += tmp * cos2az;
	dK2_dinc[ww] += tmp * sin2az;
	//
	dK4_dinc[ww] += tmp1 * cos2az;
        dK5_dinc[ww] += tmp1 * sin2az;
	//
	dK3_dinc[ww] += tmp2 * cosdi;
	dK6_dinc[ww] += tmp3 * cosdi;	
      }
      
    } // ii
    
  }
  
    // ******************************************************************************************* //
  template<typename T> inline 
  void computeDIDv(const T &ki, const T &kq, const T &ku,
		   const T &kv, const T &fq, const T &fu,
		   const T &fv, const T &GP1, const T &GP2,
		   const T &GP3, const T &GP4, const T &GP5,
		   const T &GP6, const T &del, const T &S1mud,
		   T &rf0, T &rf1,
		   T &rf2,  T &rf3, T const &dki, T const& dkq,
		   T const &dku, T const& dkv, T const &dfq, T const& dfu,
		   T const &dfv)
  {
  
  
  const T ki2 = ki*ki;
  const T kid = ki*dki, kqd = kq*dkq, kud = ku*dku, kvd = kv*dkv;
  const T fqd = fq*dfq, fud =  fu*dfu, fvd = fv*dfv;
  
  
  T const dgp1 = 2 * (kid - kqd - kud - kvd + fqd + fud+ fvd);
  
  const T dgp2 = fq*dkq + kq*dfq +  fu*dku +
    ku*dfu + fv*dkv + kv*dfv;
  
  const T ddt = 2*ki*dki*GP1 + ki2*dgp1 - 2*GP2*dgp2;
  const T dgp3 = 2*(kid + fqd+ fud + fvd);
  
  /* --- dI / dpar --- */
  
  rf0 = S1mud * ((dki*GP3+ki*dgp3)*del - ddt*ki*GP3);
  
  const T dgp4 = dki*(2*ki*kq + kv*fu - ku*fv) + ki2*dkq +
    ki*(fu*dkv+kv*dfu-fv*dku-ku*dfv);
  
      
  /* --- dQ / dpar --- */
  
  rf1 = -S1mud * ((dgp4+dfq*GP2 + fq*dgp2)*del - ddt*(GP4+fq*GP2));
  
  const T dgp5 = dki * (2.0 * ki*ku + kq*fv - kv*fq) + ki2 * dku +
    ki*(fv*dkq + kq*dfv - fq*dkv - kv*dfq);
  
  
  /* --- dU / dpar --- */
  
  rf2= -S1mud * ((dgp5+dfu*GP2 + fu*dgp2)*del - ddt*(GP5+fu*GP2));
  
  const T dgp6 = dki * (2.0*ki*kv + ku*fq - kq*fu) + ki2*dkv +
    ki*(fq*dku + ku*dfq - fu*dkq - kq*dfu);
  
  
  /* --- dV / dpar --- */
  
  rf3 = -S1mud * ((dgp6+dfv*GP2 + fv*dgp2)*del - ddt*(GP6+fv*GP2));
}

  // ******************************************************************************************* //
  
  template<typename T>
  void compute_Stokes_RF(int const nWav, const T* const __restrict__ buffer, T* const __restrict__ out, T* const __restrict__ RF, T const S0, T const S1, T const mu)
  {
    
    const T* const __restrict__ ki = &buffer[0*nWav];
    const T* const __restrict__ kq = &buffer[1*nWav];
    const T* const __restrict__ ku = &buffer[2*nWav];
    const T* const __restrict__ kv = &buffer[3*nWav];
    const T* const __restrict__ fq = &buffer[4*nWav];
    const T* const __restrict__ fu = &buffer[5*nWav];
    const T* const __restrict__ fv = &buffer[6*nWav];

    T* const __restrict__ I = &out[0     ];
    T* const __restrict__ Q = &out[1*nWav];
    T* const __restrict__ U = &out[2*nWav];
    T* const __restrict__ V = &out[3*nWav];
    

    T* const __restrict__ dI_dB = &RF[0];
    T* const __restrict__ dQ_dB = &RF[1*nWav];
    T* const __restrict__ dU_dB = &RF[2*nWav];
    T* const __restrict__ dV_dB = &RF[3*nWav];

    int off = 4*nWav;
    T* const __restrict__ dI_dinc = &RF[off];
    T* const __restrict__ dQ_dinc = &RF[off+1*nWav];
    T* const __restrict__ dU_dinc = &RF[off+2*nWav];
    T* const __restrict__ dV_dinc = &RF[off+3*nWav];

    off += 4*nWav;
    T* const __restrict__ dI_dazi = &RF[off];
    T* const __restrict__ dQ_dazi = &RF[off+1*nWav];
    T* const __restrict__ dU_dazi = &RF[off+2*nWav];
    T* const __restrict__ dV_dazi = &RF[off+3*nWav];

    off += 4*nWav;
    T* const __restrict__ dI_dvl = &RF[off];
    T* const __restrict__ dQ_dvl = &RF[off+1*nWav];
    T* const __restrict__ dU_dvl = &RF[off+2*nWav];
    T* const __restrict__ dV_dvl = &RF[off+3*nWav];

    off += 4*nWav;
    T* const __restrict__ dI_dvD = &RF[off];
    T* const __restrict__ dQ_dvD = &RF[off+1*nWav];
    T* const __restrict__ dU_dvD = &RF[off+2*nWav];
    T* const __restrict__ dV_dvD = &RF[off+3*nWav];

    off += 4*nWav;
    T* const __restrict__ dI_deta = &RF[off];
    T* const __restrict__ dQ_deta = &RF[off+1*nWav];
    T* const __restrict__ dU_deta = &RF[off+2*nWav];
    T* const __restrict__ dV_deta = &RF[off+3*nWav];

    off += 4*nWav;
    T* const __restrict__ dI_ddam = &RF[off];
    T* const __restrict__ dQ_ddam = &RF[off+1*nWav];
    T* const __restrict__ dU_ddam = &RF[off+2*nWav];
    T* const __restrict__ dV_ddam = &RF[off+3*nWav];

    off += 4*nWav;
    T* const __restrict__ dI_dS0 = &RF[off];
    //T* const __restrict__ dQ_dS0 = &RF[off+1*nWav];
    //T* const __restrict__ dU_dS0 = &RF[off+2*nWav];
    //T* const __restrict__ dV_dS0 = &RF[off+3*nWav];

    off += 4*nWav;
    T* const __restrict__ dI_dS1 = &RF[off];
    T* const __restrict__ dQ_dS1 = &RF[off+1*nWav];
    T* const __restrict__ dU_dS1 = &RF[off+2*nWav];
    T* const __restrict__ dV_dS1 = &RF[off+3*nWav];


    // --- Source function S0 derivatives --- //
    
    for(int ww=0; ww<nWav; ++ww)
      dI_dS0[ww] = 1;


    // --- Buffer for precomputed constants --- //
    T* const __restrict__ GP = new T [nWav*8]();
    
    T* const __restrict__ GP1= GP;
    T* const __restrict__ GP2 = &GP[nWav];
    T* const __restrict__ GP3 = &GP[2*nWav];
    T* const __restrict__ GP4 = &GP[3*nWav];
    T* const __restrict__ GP5 = &GP[4*nWav];
    T* const __restrict__ GP6 = &GP[5*nWav];
    T* const __restrict__ S1mud = &GP[6*nWav];
    T* const __restrict__ del   = &GP[7*nWav];

    for(int ww=0; ww<nWav; ++ww){

      T const ki2 = ki[ww]*ki[ww];
      T const fq2 = fq[ww]*fq[ww];
      T const fu2 = fu[ww]*fu[ww];
      T const fv2 = fv[ww]*fv[ww];
      
      GP3[ww] = ki2 + fq2 + fu2 + fv2;
      GP1[ww] = GP3[ww] - ((kq[ww]*kq[ww]) + (ku[ww]*ku[ww]) + (kv[ww]*kv[ww]));
      GP2[ww] = kq[ww]*fq[ww] + ku[ww]*fu[ww] + kv[ww]*fv[ww];
       
      del[ww]   = ki2 * GP1[ww] - GP2[ww]*GP2[ww];

      GP4[ww] = ki2 * kq[ww] + ki[ww] * (kv[ww]*fu[ww] - ku[ww]*fv[ww]);
      GP5[ww] = ki2 * ku[ww] + ki[ww] * (kq[ww]*fv[ww] - kv[ww]*fq[ww]);
      GP6[ww] = ki2 * kv[ww] + ki[ww] * (ku[ww]*fq[ww] - kq[ww]*fu[ww]);

      
      S1mud[ww] = S1 * (mu / del[ww]);
    }

    for(int ww=0; ww<nWav; ++ww){
	I[ww] = S0 + (S1mud[ww] * ki[ww] * GP3[ww]);
	Q[ww] = -S1mud[ww] * (GP4[ww] + fq[ww] * GP2[ww]);
	U[ww] = -S1mud[ww] * (GP5[ww] + fu[ww] * GP2[ww]);
	V[ww] = -S1mud[ww] * (GP6[ww] + fv[ww] * GP2[ww]);
    }
	
    // --- S1 --- //
    for(int ww=0; ww<nWav; ++ww){
	dI_dS1[ww] = ki[ww]*GP3[ww]*mu/del[ww];
	dQ_dS1[ww] = -mu * (GP4[ww] + fq[ww]* GP2[ww]) / del[ww];
	dU_dS1[ww] = -mu * (GP5[ww] + fu[ww]* GP2[ww]) / del[ww];
	dV_dS1[ww] = -mu * (GP6[ww] + fv[ww]* GP2[ww]) / del[ww];
    }

    for(int ww=0; ww<nWav; ++ww)
      S1mud[ww] /= del[ww];

      
      
      // --- other parameters, do explicit calls so the compiler can vectorize the
      //     entire loop over wavelength --- //

    int off3 = 15*nWav + 24*nWav;
    {
      const T* const __restrict__ dki = &buffer[off3];
      const T* const __restrict__ dkq = &buffer[off3+nWav];
      const T* const __restrict__ dku = &buffer[off3+2*nWav];
      const T* const __restrict__ dkv = &buffer[off3+3*nWav];
      const T* const __restrict__ dfq = &buffer[off3+4*nWav];
      const T* const __restrict__ dfu = &buffer[off3+5*nWav];
      const T* const __restrict__ dfv = &buffer[off3+6*nWav];
      
      
      for(int ww=0; ww<nWav; ++ww)
	computeDIDv(ki[ww], kq[ww], ku[ww], kv[ww], fq[ww], fu[ww], fv[ww],
		    GP1[ww], GP2[ww], GP3[ww], GP4[ww], GP5[ww], GP6[ww], del[ww], S1mud[ww],
		    dI_dB[ww], dQ_dB[ww], dU_dB[ww], dV_dB[ww], dki[ww], dkq[ww], dku[ww], dkv[ww],
		    dfq[ww], dfu[ww], dfv[ww]);
      
    }

    off3 += 7*nWav;
    {
      const T* const __restrict__ dki = &buffer[off3];
      const T* const __restrict__ dkq = &buffer[off3+nWav];
      const T* const __restrict__ dku = &buffer[off3+2*nWav];
      const T* const __restrict__ dkv = &buffer[off3+3*nWav];
      const T* const __restrict__ dfq = &buffer[off3+4*nWav];
      const T* const __restrict__ dfu = &buffer[off3+5*nWav];
      const T* const __restrict__ dfv = &buffer[off3+6*nWav];
      
      for(int ww=0; ww<nWav; ++ww)
	computeDIDv(ki[ww], kq[ww], ku[ww], kv[ww], fq[ww], fu[ww], fv[ww],
		    GP1[ww], GP2[ww], GP3[ww], GP4[ww], GP5[ww], GP6[ww], del[ww], S1mud[ww],
		    dI_dinc[ww], dQ_dinc[ww], dU_dinc[ww], dV_dinc[ww], dki[ww], dkq[ww], dku[ww], dkv[ww],
		    dfq[ww], dfu[ww], dfv[ww]);
    }

    off3 += 7*nWav;
    {
      const T* const __restrict__ dki = &buffer[off3];
      const T* const __restrict__ dkq = &buffer[off3+nWav];
      const T* const __restrict__ dku = &buffer[off3+2*nWav];
      const T* const __restrict__ dkv = &buffer[off3+3*nWav];
      const T* const __restrict__ dfq = &buffer[off3+4*nWav];
      const T* const __restrict__ dfu = &buffer[off3+5*nWav];
      const T* const __restrict__ dfv = &buffer[off3+6*nWav];
      
      for(int ww=0; ww<nWav; ++ww)
	computeDIDv(ki[ww], kq[ww], ku[ww], kv[ww], fq[ww], fu[ww], fv[ww],
		    GP1[ww], GP2[ww], GP3[ww], GP4[ww], GP5[ww], GP6[ww], del[ww], S1mud[ww],
		    dI_dazi[ww], dQ_dazi[ww], dU_dazi[ww], dV_dazi[ww], dki[ww], dkq[ww], dku[ww], dkv[ww],
		    dfq[ww], dfu[ww], dfv[ww]);
    }

    off3 += 7*nWav;
    {
      const T* const __restrict__ dki = &buffer[off3];
      const T* const __restrict__ dkq = &buffer[off3+nWav];
      const T* const __restrict__ dku = &buffer[off3+2*nWav];
      const T* const __restrict__ dkv = &buffer[off3+3*nWav];
      const T* const __restrict__ dfq = &buffer[off3+4*nWav];
      const T* const __restrict__ dfu = &buffer[off3+5*nWav];
      const T* const __restrict__ dfv = &buffer[off3+6*nWav];
      
      for(int ww=0; ww<nWav; ++ww)
	computeDIDv(ki[ww], kq[ww], ku[ww], kv[ww], fq[ww], fu[ww], fv[ww],
		    GP1[ww], GP2[ww], GP3[ww], GP4[ww], GP5[ww], GP6[ww], del[ww], S1mud[ww],
		    dI_dvl[ww], dQ_dvl[ww], dU_dvl[ww], dV_dvl[ww], dki[ww], dkq[ww], dku[ww], dkv[ww],
		    dfq[ww], dfu[ww], dfv[ww]);
    }

        off3 += 7*nWav;
    {
      const T* const __restrict__ dki = &buffer[off3];
      const T* const __restrict__ dkq = &buffer[off3+nWav];
      const T* const __restrict__ dku = &buffer[off3+2*nWav];
      const T* const __restrict__ dkv = &buffer[off3+3*nWav];
      const T* const __restrict__ dfq = &buffer[off3+4*nWav];
      const T* const __restrict__ dfu = &buffer[off3+5*nWav];
      const T* const __restrict__ dfv = &buffer[off3+6*nWav];
      
      for(int ww=0; ww<nWav; ++ww)
	computeDIDv(ki[ww], kq[ww], ku[ww], kv[ww], fq[ww], fu[ww], fv[ww],
		    GP1[ww], GP2[ww], GP3[ww], GP4[ww], GP5[ww], GP6[ww], del[ww], S1mud[ww],
		    dI_dvD[ww], dQ_dvD[ww], dU_dvD[ww], dV_dvD[ww], dki[ww], dkq[ww], dku[ww], dkv[ww],
		    dfq[ww], dfu[ww], dfv[ww]);
    }


    off3 += 7*nWav;
    {
      const T* const __restrict__ dki = &buffer[off3];
      const T* const __restrict__ dkq = &buffer[off3+nWav];
      const T* const __restrict__ dku = &buffer[off3+2*nWav];
      const T* const __restrict__ dkv = &buffer[off3+3*nWav];
      const T* const __restrict__ dfq = &buffer[off3+4*nWav];
      const T* const __restrict__ dfu = &buffer[off3+5*nWav];
      const T* const __restrict__ dfv = &buffer[off3+6*nWav];
      
      for(int ww=0; ww<nWav; ++ww)
	computeDIDv(ki[ww], kq[ww], ku[ww], kv[ww], fq[ww], fu[ww], fv[ww],
		    GP1[ww], GP2[ww], GP3[ww], GP4[ww], GP5[ww], GP6[ww], del[ww], S1mud[ww],
		    dI_deta[ww], dQ_deta[ww], dU_deta[ww], dV_deta[ww], dki[ww], dkq[ww], dku[ww], dkv[ww],
		    dfq[ww], dfu[ww], dfv[ww]);
    }

    off3 += 7*nWav;
    {
      const T* const __restrict__ dki = &buffer[off3];
      const T* const __restrict__ dkq = &buffer[off3+nWav];
      const T* const __restrict__ dku = &buffer[off3+2*nWav];
      const T* const __restrict__ dkv = &buffer[off3+3*nWav];
      const T* const __restrict__ dfq = &buffer[off3+4*nWav];
      const T* const __restrict__ dfu = &buffer[off3+5*nWav];
      const T* const __restrict__ dfv = &buffer[off3+6*nWav];
      
      for(int ww=0; ww<nWav; ++ww)
	computeDIDv(ki[ww], kq[ww], ku[ww], kv[ww], fq[ww], fu[ww], fv[ww],
		    GP1[ww], GP2[ww], GP3[ww], GP4[ww], GP5[ww], GP6[ww], del[ww], S1mud[ww],
		    dI_ddam[ww], dQ_ddam[ww], dU_ddam[ww], dV_ddam[ww], dki[ww], dkq[ww], dku[ww], dkv[ww],
		    dfq[ww], dfu[ww], dfv[ww]);
    }

    
    
    // --- I,V have zero response to the azimuth --- //
    
    memset(dI_dazi,0,nWav*sizeof(T)); 
    memset(dV_dazi,0,nWav*sizeof(T));


    
    delete [] GP;
  }

  
  
  // ******************************************************************************************* //

  template<typename T>
  void ME_Synth_RF(std::vector<double> const &wav, std::vector<ln::line<T>> const& lines,
		   const T* const __restrict__ m, T* const __restrict__ out, T* const __restrict__ RF,
		   T const mu)
  {

    int const nWav = int(wav.size());
    std::memset(out, 0, 4*nWav*sizeof(T));
    std::memset(RF,  0, 36*nWav*sizeof(T));


    // --- Copy to local scalars ---//
    
    T const B   = m[0];
    T const inc = m[1];
    T const azi = m[2];
    T const vl  = m[3];
    T const vD  = m[4];
    T const eta = m[5];
    T const dam = m[6];
    T const S0  = m[7];
    T const S1  = m[8];

    T const sinin = sin(inc);
    T const cosin = cos(inc);
    T const sin2az= sin(2*azi);
    T const cos2az= cos(2*azi);

    
    
    // --- Allocate memory buffer ---//
    
    T* const __restrict__ buff = new T [nWav*15 + nWav*7*7 + 24*nWav]();
    

    
    // --- Compute abs. matrix and derivatives --- //

    ml::compute_K_der<T>(nWav, &wav[0], buff, B, inc, azi, vl, vD, eta, dam, lines, sinin, cosin, sin2az, cos2az);

    
    // --- Compute emerging spectra and its derivatives --- //
    
    ml::compute_Stokes_RF<T>(nWav, buff, out, RF, S0, S1, mu);

    
    // --- clean-up --- //

    delete [] buff;
  }
}

#endif
