#ifndef MYCOMPLEX
#define MYCOMPLEX

/* --- 
   Complex numbers class

   Certain versions of g++, clang++ and icpc generate much faster code with 
   this implementation than with the std::complex class in
   my tests with the Voigt-Faraday profiles. Not sure why.
   This claim is architecture and compiler version dependent.
   But in my tests it never performed slower than std::complex<T>.

   Optimized using as much constexpr as possible but will require
   to compile with -std=c++17 to take full advantage of inlining
   of constants. The speedup is at least 20% and as high as 200%
   depending on the application when compared to std::complex<T>.

   Disclaimer: I have only implemented the methods that could be needed
   for computing approximations to the Faddeeva function. Using std::get
   allows evaluating some expressions at compile time with constrexpr constants, 
   even if the notation becomes less obvious.

   Some of the methods have been implemented so that it is a drop-in replacement
   for std::complex<T>, for example the real() and imag() methods and the 
   real(x) and imag(x) assignment methods.

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2021)

   --- */

#include <array>
#include <cmath>

#include "math_tools.hpp"

namespace mth{

  template<typename T>
  struct complex{
    std::array<T,2> d;

    inline constexpr complex(): d(){};
    inline constexpr complex(T const a, T const b): d({a,b}){};
    inline constexpr complex(T const a): d({a,T(0)}){};
    inline constexpr complex(std::array<T,2> const& in): d(in){};
    inline constexpr complex(std::array<T,2> && in): d(std::move(in)){};

    inline constexpr complex(complex<T> const& a): d(a.d){};
    inline constexpr complex(complex<T> && a): d(std::move(a.d)){};

    inline constexpr complex<T> &operator=(complex<T> && a){
      d = std::move(a.d);
      return *this;
    }
	  
    inline constexpr complex<T> &operator=(complex<T> const& a)
    {
      std::get<0>(d) = std::get<0>(a.d);
      std::get<1>(d) = std::get<1>(a.d);
      return *this;
    }
    
    inline constexpr T &real(){return std::get<0>(d);}
    inline constexpr T real()const{return std::get<0>(d);}

    
    inline constexpr T &imag(){return std::get<1>(d);}
    inline constexpr T imag()const{return std::get<1>(d);}

    inline constexpr void imag(T const& a){ std::get<1>(d) = a;}
    inline constexpr void real(T const& a){ std::get<0>(d) = a;}
    
    inline constexpr complex<T> conjugate()const
    {
      return complex<T>(std::get<0>(d), -std::get<1>(d));
    }
    
    inline constexpr complex<T> &operator+=(complex<T> const& b)
    {
      std::get<0>(d) += std::get<0>(b.d);
      std::get<1>(d) += std::get<1>(b.d);
      return *this;
    }

    inline constexpr complex<T> &operator-=(complex<T> const& b)
    {
      std::get<0>(d) -= std::get<0>(b.d);
      std::get<1>(d) -= std::get<1>(b.d);
      return *this;
    }
    inline constexpr complex<T> operator-()const
    {
      return complex<T>(-std::get<0>(d), -std::get<1>(d));
    }
    inline constexpr complex<T> &operator*=(complex<T> const&b)
    {
      std::get<0>(d) = std::get<0>(d)*std::get<0>(b.d) - std::get<1>(d)*std::get<1>(b.d);
      std::get<1>(d) = std::get<0>(d)*std::get<1>(b.d) + std::get<1>(d)*std::get<0>(b.d);
      return *this;
    }
    
    inline constexpr complex<T> &operator/=(complex<T> const&b)
    {
      T const scl =  abs2(b);
	
      std::get<0>(d) = (std::get<0>(d)*std::get<0>(b.d) + std::get<1>(d)*std::get<1>(b.d)) / scl;
      std::get<1>(d) = (std::get<1>(d)*std::get<0>(b.d) - std::get<0>(d)*std::get<1>(b.d)) / scl;

      return *this;
    }

    inline constexpr complex<T> &operator+=(T const& b)
    {
      std::get<0>(d) += b;
      
      return *this;
    }

    inline constexpr complex<T> &operator-=(T const& b)
    {
      std::get<0>(d) -= b;
      
      return *this;
    }
    
    inline constexpr complex<T> &operator*=(T const& b)
    {
      std::get<0>(d) *= b;
      std::get<1>(d) *= b;
      
      return *this;
    }

    inline constexpr complex<T> &operator/=(T const& b)
    {
      std::get<0>(d) /= b;
      std::get<1>(d) /= b;
      
      return *this;
    }
  };

  // ********************************************************************************************** //
  
  template<typename T> inline constexpr
  T abs2(complex<T> const& a){return mth::SQ(std::get<0>(a.d)) + mth::SQ(std::get<1>(a.d));}

  template<typename T> inline
  T abs(complex<T> const& a){return std::sqrt(abs2(a));}

  template<typename T>
  inline constexpr complex<T> operator+(complex<T> const& a, complex<T> const&b)
  {
    return complex<T>(std::get<0>(a.d)+std::get<0>(b.d),std::get<1>(a.d)+std::get<1>(b.d));
  }
  template<typename T>
  inline constexpr complex<T> operator+(T const& a, complex<T> const&b)
  {
    return complex<T>(std::get<0>(b.d) + a, std::get<1>(b.d));
  }
  template<typename T>
  inline constexpr complex<T> operator+(complex<T> const&b, T const& a)
  {
    return complex<T>(std::get<0>(b.d) + a,std::get<1>(b.d));
  }

  template<typename T>
  inline constexpr complex<T> operator-(complex<T> const& a, complex<T> const&b)
  {
    return complex<T>(std::get<0>(a.d)-std::get<0>(b.d),std::get<1>(a.d)-std::get<1>(b.d));
  }
  template<typename T>
  inline constexpr complex<T> operator-(T const& a, complex<T> const&b)
  {
    return complex<T>(a-std::get<0>(b.d) , -std::get<1>(b.d));
  }
  template<typename T>
  inline constexpr complex<T> operator-(complex<T> const&b, T const& a)
  {
    return complex<T>(std::get<0>(b.d) - a,std::get<1>(b.d));
  }

  template<typename T>
  inline constexpr complex<T> operator*(complex<T> const& a, complex<T> const&b)
  {
    return complex<T>( std::get<0>(a.d)*std::get<0>(b.d) - std::get<1>(a.d)*std::get<1>(b.d),
	               std::get<0>(a.d)*std::get<1>(b.d) + std::get<1>(a.d)*std::get<0>(b.d));    
  }  
  template<typename T>
  inline constexpr complex<T> operator*(T const& a, complex<T> const&b)
  {
    return complex<T>(a*std::get<0>(b.d) , a*std::get<1>(b.d));
  }
  template<typename T>
  inline constexpr complex<T> operator*(complex<T> const&b, T const& a)
  {
    return complex<T>(std::get<0>(b.d)*a, a*std::get<1>(b.d));
  }
  

  template<typename T>
  inline constexpr complex<T> operator/(complex<T> const& a, complex<T> const&b)
  {
    T const scl =  abs2(b);
    // (a+bi)/(c+di)=(a+bi)*(c-di)/((c+di)*(c-di))=(ac-adi+bci+db)/(cc+dd)
	
    return complex<T>((std::get<0>(a.d)*std::get<0>(b.d) + std::get<1>(a.d)*std::get<1>(b.d)) / scl,
		      (std::get<1>(a.d)*std::get<0>(b.d) - std::get<0>(a.d)*std::get<1>(b.d)) / scl);
  
  }
  template<typename T>
  inline constexpr complex<T> operator/(T const a, complex<T> const&b)
  {
    T const scl = a/abs2(b);
	
    return complex<T>( std::get<0>(b.d) *scl, -std::get<1>(b.d) *scl);
  
  }
  template<typename T>
  inline constexpr complex<T> operator/(complex<T> const &b, T const a)
  {
    return complex<T>(std::get<0>(b.d)/a, std::get<1>(b.d)/a);
  }
  template<typename T>
  inline constexpr complex<T> exp(complex<T> const& a)
  {
    // Remember:
    // Exp[a+bi] = Exp[a]*Exp[bi] = Exp[a]*(cos(b) + i sin(b))
    T const eps = std::exp(std::get<0>(a.d));
    return complex<T>(eps*std::cos(std::get<1>(a.d)), eps*std::sin(std::get<1>(a.d)));
  }
  
}


#endif
