#ifndef PROFILEHPP
#define PROFILEHPP

//#include <complex>
#include <algorithm>
#include <cmath>
#include "myComplex.hpp"

namespace pr{

  using namespace mth; // replace by std if you want to use std::complex
  
  template<typename T> constexpr inline T S(T const& v){return static_cast<T>(v);}

  // ---------------------------------------------------------------------------------------- //
  
template<typename T>
inline T vf(T const& damp, T const& vv,  T &F)
{
  // --- This is part of Hui's approximation for damp > 1.e-3 and it is vectorizable when
  // --- looping in wavelength. If damp <= 1.e-3 use Humlicek's approximation
  
  constexpr static const T  A[7] =  {122.607931777104326, 214.382388694706425, 
				     181.928533092181549, 93.155580458138441, 30.180142196210589, 5.912626209773153,
				     0.564189583562615 };
  constexpr static const T B[7] = {122.60793177387535, 352.730625110963558, 457.334478783897737, 
				   348.703917719495792, 170.354001821091472, 53.992906912940207, 10.479857114260399};
  
  complex<T> const Z(damp, -std::abs<T>(vv));
    
  complex<T> const Z1 = ((((((A[6]*Z+A[5])*Z+A[4])*Z+A[3])*Z+A[2])*Z+A[1])*Z+A[0]) /
    (((((((Z+B[6])*Z+B[5])*Z+B[4])*Z+B[3])*Z+B[2])*Z+B[1])*Z+B[0]);
  
  
  T const tmp = Z1.imag();
  
  F = ((vv < 0) ? -0.5f : 0.5f) * tmp;
  return Z1.real();  
}
  
  // ---------------------------------------------------------------------------------------- //

  template<typename T>
  inline T voigt_complex(T const a, T const v, T &far)
  {
      
    T const sav = std::abs<T>(v) + a;
    complex<T> const tav(a, -v);
    complex<T> const uav = tav*tav;
    complex<T> w4;


    
    /* --- HUMLICEK'S APPROXIMATION --- */
    if(sav >=  static_cast<T>(15)){
      w4 = tav * static_cast<T>(0.5641896) / ( static_cast<T>(0.5) + uav);
    } else if(sav >=  static_cast<T>(5.5)){
      w4 = tav * ( static_cast<T>(1.410474) + uav *  static_cast<T>(0.5641896)) / ( static_cast<T>(0.75)+uav * (static_cast<T>(3) + uav));
    } else if(a >= ( static_cast<T>(0.195) * std::abs<T>(v) -  static_cast<T>(0.176))){
      w4 = (static_cast<T>(16.4955) + tav * (static_cast<T>(20.20933) + tav * (static_cast<T>(11.96482) + tav * (static_cast<T>(3.778987) + tav * static_cast<T>(0.5642236))))) / (static_cast<T>(16.4955) + tav * (static_cast<T>(38.82363) + tav * (static_cast<T>(39.27121) + tav * (static_cast<T>(21.69274) + tav * (static_cast<T>(6.699398) + tav)))));
    } else{
      w4 = tav * (static_cast<T>(36183.31) - uav * (static_cast<T>(3321.9905) - uav * (static_cast<T>(1540.787) -  uav * (static_cast<T>(219.0313) - uav * (static_cast<T>(35.76683) - uav * (static_cast<T>(1.320522) - uav * static_cast<T>(0.56419)))))));
      
      complex<T> const v4 = (static_cast<T>(32066.6) - uav * (static_cast<T>(24322.84) - uav * (static_cast<T>(9022.228) -  uav * (static_cast<T>(2186.181) - uav * (static_cast<T>(364.2191) - uav * (static_cast<T>(61.57037) - uav * (static_cast<T>(1.841439) - uav)))))));
      w4 = exp(uav) - w4 / v4;
    }
    
    /* ---  Note that FVGT below is in fact 2 * (Faradey-Voigt function) ---*/
    
    far = w4.imag()*T(0.5);

    return w4.real();
  }

    // ---------------------------------------------------------------------------------------- //

  template<typename T>
  inline void compute_profile(int const nWav, const double*  __restrict__ wav, double const wav0,  T const str,
			      T const va, T const damp, T const dlnu, T const vb, T* __restrict__ H, T* __restrict__ F)
  {

    // --- Assume this is given from outside: --- //
    // vb = split * B;
    // va = (nu0 * vlos) / phyc::CC<T>;

    // if(damp > 1.e-3f){
      // --- Hui's vectorized function --- //
    // for(int ii=0; ii<nWav; ++ii)
    //H[ii] = vf<T>(damp, (static_cast<T>(wav[ii]-wav0) - va + vb)/dlnu, F[ii]);
    //}else{
    // --- Humlicek's approximation (slower) --- //
    for(int ii=0; ii<nWav; ++ii){
      H[ii] = voigt_complex<T>(damp, (static_cast<T>(wav[ii]-wav0 - va  + vb))/dlnu, F[ii]);
    }
    //}
    
    for(int ii=0; ii<nWav; ++ii){
      H[ii] *= str;
      F[ii] *= str;
    }
    
    
  }
    // ---------------------------------------------------------------------------------------- //
 
}

#endif

