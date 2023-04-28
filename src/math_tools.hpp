#ifndef MATHTOOLS
#define MATHTOOLS

/* --- 
   Math utilities in template form
   Coded by J. de la Cruz Rodriguez (ISP-SU 2019-2022)
   --- */

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdio>

namespace mth{
  // *********************************************************************************** //

  template<typename T>
  inline constexpr T SQ(T const& v){return v*v;}
  
  // *********************************************************************************** //
  
  template<class T>
  inline T signFortran(T const &val)
  {return ((val < static_cast<T>(0)) ? static_cast<T>(-1) : static_cast<T>(1));}
  
  // *********************************************************************************** //

  template<class T> inline T harmonic_derivative_Steffen_one(T const xu, T const x0, T const xd, T const yu, T const y0, T const yd)
  {
    // ---
    // High order harmonic derivatives
    // Ref: Steffen (1990), A&A..239..443S
    //
    // Arguments:
    //   Assuming three consecutive points of a function yu, y0, yd and the intervals between them odx, and dx:
    //      odx: x0 - xu
    //       dx: xd - x0
    //       yu: Upwind point
    //       y0: Central point
    //       yd: downwind point
    //
    // ---

    T const dx = (xd - x0);
    T const odx = (x0 - xu);
    T const S0 = (yd - y0) / dx;
    T const Su = (y0 - yu) / odx;
    T const P0 = std::abs((Su*dx + S0*odx) / (odx+dx)) * 0.5;
    
    return (signFortran(S0) + signFortran(Su)) * std::min<T>(std::abs(Su),std::min<T>(std::abs(S0), P0));
  }
  // *********************************************************************************** //
  
  template<typename T, std::size_t N> struct mPow{
    inline constexpr static T run(T const& v){return v * mPow<T,N-1>::run(v);}
  };

  template<typename T> struct mPow<T,1>{
    inline constexpr static T run(T const& v){return v;}
  };

  template<typename T> struct mPow<T,2>{
    inline constexpr static T run(T const& v){return v*v;}
  };
  
  template<typename T> struct mPow<T,3>{
    inline constexpr static T run(T const& v){return v*v*v;}
  };

  template<typename T> struct mPow<T,4>{
    inline constexpr static T run(T const& v){return SQ<T>(v*v);}
  };

  template<typename T> struct mPow<T,5>{
    inline constexpr static T run(T const& v){return SQ<T>(v*v)*v;}
  };

  template<typename T> struct mPow<T,6>{
    inline constexpr static T run(T const& v){return SQ<T>(mPow<T,3>::run(v));}
  };
  
  template<typename T> struct mPow<T,7>{
    inline constexpr static T run(T const& v){return v*mPow<T,6>::run(v);}
  };

  template<typename T> struct mPow<T,8>{
    inline constexpr static T run(T const& v){return SQ(mPow<T,4>::run(v));}
  };
  
  // *********************************************************************************** //

  template<typename T, size_t N> inline constexpr
  T Pow(T const& v){
    return mPow<T,N>::run(v);
  }
  
  // *********************************************************************************** //
  
  template<typename T> inline
  T E1(T const &x)
  {
    static constexpr const std::size_t N_53 = 6;
    static constexpr const std::size_t N_56 = 4;
    
    static constexpr const std::array<T,N_53> a53 = {-0.57721566,  0.99999193, -0.24991055,
      0.05519968, -0.00976004,  0.00107857 };
    
    static constexpr const std::array<T,N_56> a56 = { 8.5733287401, 18.0590169730,
      8.6347608925,  0.2677737343 };
    
    static constexpr const std::array<T,N_53> b56 = { 9.5733223454, 25.6329561486,
      21.0996530827,  3.9584969228 };
    T res = 0;
    
    if (x <= T(0)) {
      fprintf(stderr,"[error] mth::E1: Exponential integral E1 of x = %e\n", x);
      exit(1);
    } else if (x > T(0)  &&  x <= T(1)) {
      res = -log(x) + std::get<0>(a53) + x*(std::get<1>(a53) + x*(std::get<2>(a53) + x*(std::get<3>(a53) + x*(std::get<4>(a53) + x*std::get<5>(a53)))));
    } else if (x > T(1)  &&  x <= T(80)) {
      res  = std::get<3>(a56)/x +  std::get<2>(a56) + x*(std::get<1>(a56) + x*(std::get<0>(a56) + x));
      res /= std::get<3>(b56) + x*(std::get<2>(b56) + x*(std::get<1>(b56) + x*(std::get<0>(b56) + x)));
      res *= exp(-x);
    }
    return res;
  }
  
  // *********************************************************************************** //

  template<typename T, typename U = T>
  T KSum(size_t const n, const T* const __restrict__ arr)
  {
    // ---
    // simple Kahan summation of the elements of a raw array
    // Source: Wikipedia ...
    // ---
    
    U sum = 0, c = 0;
    
    for(size_t kk = 0; kk<n; ++kk){
      U const y = static_cast<U>(arr[kk]) - c;
      U const t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    
    return T(sum);
  }
  
  // *********************************************************************************** //

  template<typename T, typename U = T>
  T KSum2(size_t const n, const T* const __restrict__ arr)
  {
    // ---
    // simple Kahan summation of the elements of a raw array
    // Source: Wikipedia ...
    // ---
    
    U sum = 0, c = 0;
    
    for(size_t kk = 0; kk<n; ++kk){
      U const y = SQ<U>(static_cast<U>(arr[kk])) - c;
      U const t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    
    return T(sum);
  }
  
  // *********************************************************************************** //

  template<typename T>
  inline T pow10(T const& var){
    return exp(static_cast<T>(2.302585092994046) * var);
  }
  
  // *********************************************************************************** //

  // Compile-time sqrt using a Newton-Raphson scheme, useful for calculating inlined constants
  
  template<typename T>
  constexpr T sqrtBabylonian(T const &var, T const x, T const xprev)
  {
    return (x == xprev) ? x : sqrtBabylonian(var,  (x + var / x) / T(2), x);
  }
  
  template<typename T>
  constexpr T Sqrt(T const &x)
  {
    return x >= T(0) && x < std::numeric_limits<T>::infinity()
      ? sqrtBabylonian(x, (x > T(2)) ? x/T(2) : x, T(0))
      : std::numeric_limits<double>::quiet_NaN();
  }
  
  // *********************************************************************************** //

}

#endif
