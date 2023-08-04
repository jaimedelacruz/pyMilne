/* ----
   
   Expression templates for Array delayed evaluation.
   Basic routines for +,-,*,/ inspired on the implementation presented by:
   C++ Templates - The Complete Guide, 2nd Edition
   by David Vandevoorde, Nicolai M. Josuttis, and Douglas Gregor
   
   Other operators and mathematical functions are homebrewed.

   Coded by J. de la Cruz Rodriguez (ISP-SU 2019)
   
   --- */

#ifndef ARRAYSEXPR
#define ARRAYSEXPR

#include <cmath>
#include <algorithm>
#include <cassert>
#include <functional>

namespace mem{

  // --- wrapper class for scalars --- //
  template<typename T>
  class Scalar {
  private:
    T const& s;  
  public:
    constexpr Scalar (T const &v): s(v) {};
    //template<class U>
    //constexpr Scalar (U const &v): s(T(v)) {};

    constexpr T const& operator[] (std::size_t)const{return s;}
    constexpr std::size_t size() const {return 0;}
  };
  
  // --- Traits array - scalar --- //
  template<typename T>
  struct E_Traits {
    using ExprRef = T const&; 
  };
  
  template<typename T>
  struct E_Traits<Scalar<T>> {
    using ExprRef = Scalar<T>;  
  };
  
  // --- Wrapper class for operations --- //
  
  template<typename T, typename OP1, typename OP2, size_t op> class Operation;


  // --- Sum --- //
  template<typename T, typename OP1, typename OP2>
  class Operation<T,OP1,OP2,0> {
  private:
    typename E_Traits<OP1>::ExprRef op1;    // first operand
    typename E_Traits<OP2>::ExprRef op2;    // second operand

  public: 
    Operation (OP1 const& a, OP2 const& b) : op1(a), op2(b) {};
    
    M_INLINE T operator[] (std::size_t idx) const {return op1[idx] + op2[idx];}
   
    std::size_t size() const {
      
#if defined(DEBUG_ARRAYS)
      assert (op1.size()==0 || op2.size()==0
                || op1.size()==op2.size());
#endif
        return op1.size()!=0 ? op1.size() : op2.size();
    }
  };
  

  // --- subtraction --- //
  template<typename T, typename OP1, typename OP2>
  class Operation<T,OP1,OP2,1> {
  private:
    typename E_Traits<OP1>::ExprRef op1;    // first operand
    typename E_Traits<OP2>::ExprRef op2;    // second operand

  public: 
    Operation (OP1 const& a, OP2 const& b) : op1(a), op2(b) {};
    
    M_INLINE T operator[] (std::size_t idx) const {return op1[idx] - op2[idx];}
   
    std::size_t size() const {
      
#if defined(DEBUG_ARRAYS)
      assert (op1.size()==0 || op2.size()==0
                || op1.size()==op2.size());
#endif
        return op1.size()!=0 ? op1.size() : op2.size();
    }
  };

  // --- multiplication --- //
  template<typename T, typename OP1, typename OP2>
  class Operation<T,OP1,OP2,2> {
  private:
    typename E_Traits<OP1>::ExprRef op1;    // first operand
    typename E_Traits<OP2>::ExprRef op2;    // second operand

  public: 
    Operation (OP1 const& a, OP2 const& b) : op1(a), op2(b) {};
    
    M_INLINE T operator[] (std::size_t idx) const {return op1[idx] * op2[idx];}
   
    std::size_t size() const {
      
#if defined(DEBUG_ARRAYS)
      assert (op1.size()==0 || op2.size()==0
                || op1.size()==op2.size());
#endif
        return op1.size()!=0 ? op1.size() : op2.size();
    }
  };
  
  // --- division --- //
  template<typename T, typename OP1, typename OP2>
  class Operation<T,OP1,OP2,3> {
  private:
    typename E_Traits<OP1>::ExprRef op1;    // first operand
    typename E_Traits<OP2>::ExprRef op2;    // second operand

  public: 
    Operation (OP1 const& a, OP2 const& b) : op1(a), op2(b) {};
    
    M_INLINE T operator[] (std::size_t idx) const {return op1[idx] / op2[idx];}
    
    std::size_t size() const {
      
#if defined(DEBUG_ARRAYS)
      assert (op1.size()==0 || op2.size()==0
                || op1.size()==op2.size());
#endif
        return op1.size()!=0 ? op1.size() : op2.size();
    }
  };

  
  // --- Operations array - array --- //
  
  template<typename T, size_t N, typename R1, typename R2>
  M_INLINE Array<T,N,Operation<T,R1,R2,0>>
  operator+ (Array<T,N,R1> const& a, Array<T,N,R2> const& b) {
    return Array<T,N,Operation<T,R1,R2,0>>(Operation<T,R1,R2,0>(a.rep(),b.rep()));
  }

  template<typename T, size_t N, typename R1, typename R2>
  M_INLINE Array<T,N,Operation<T,R1,R2,1>>
  operator- (Array<T,N,R1> const& a, Array<T,N,R2> const& b) {
    return Array<T,N,Operation<T,R1,R2,1>>(Operation<T,R1,R2,1>(a.rep(),b.rep()));
  }
  
  template<typename T, size_t N, typename R1, typename R2>
  M_INLINE Array<T,N,Operation<T,R1,R2,2>>
  operator* (Array<T,N,R1> const& a, Array<T,N,R2> const& b) {
    return Array<T,N,Operation<T,R1,R2,2>>(Operation<T,R1,R2,2>(a.rep(),b.rep()));
  }

  template<typename T, size_t N, typename R1, typename R2>
  M_INLINE Array<T,N,Operation<T,R1,R2,3>>
  operator/ (Array<T,N,R1> const& a, Array<T,N,R2> const& b) {
    return Array<T,N,Operation<T,R1,R2,3>>(Operation<T,R1,R2,3>(a.rep(),b.rep()));
  }


  // --- Operations scalar - array --- //

  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, Operation<T,Scalar<T>,R2,0>>
  operator+ (T const &s, Array<T,N,R2> const& b) {
    return Array<T,N,Operation<T,Scalar<T>,R2,0>>
      (Operation<T,Scalar<T>,R2,0>(Scalar<T>(s), b.rep()));
  }

  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, Operation<T,R2,Scalar<T>,0>>
  operator+ (Array<T,N,R2> const& b, T const &s) {
    return Array<T,N,Operation<T,R2,Scalar<T>,0>>
      (Operation<T,R2,Scalar<T>,0>(b.rep(),Scalar<T>(s)));
  }

  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, Operation<T,Scalar<T>,R2,1>>
  operator- (T const &s, Array<T,N,R2> const& b) {
    return Array<T,N,Operation<T,Scalar<T>,R2,1>>
      (Operation<T,Scalar<T>,R2,1>(Scalar<T>(s), b.rep()));
  }

  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, Operation<T,R2,Scalar<T>,1>>
  operator- (Array<T,N,R2> const& b, T const &s) {
    return Array<T,N,Operation<T,R2,Scalar<T>,1>>
      (Operation<T,R2,Scalar<T>,1>(b.rep(),Scalar<T>(s)));
  }

  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, Operation<T,Scalar<T>,R2,2>>
  operator* (T const &s, Array<T,N,R2> const& b) {
    return Array<T,N,Operation<T,Scalar<T>,R2,2>>
      (Operation<T,Scalar<T>,R2,2>(Scalar<T>(s), b.rep()));
  }

  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, Operation<T,R2,Scalar<T>,2>>
  operator* (Array<T,N,R2> const& b, T const &s) {
    return Array<T,N,Operation<T,R2,Scalar<T>,2>>
      (Operation<T,R2,Scalar<T>,2>(b.rep(),Scalar<T>(s)));
  }
  
  template< typename T, size_t N, typename R2>
  M_INLINE Array<T, N, Operation<T,Scalar<T>,R2,3>>
  operator/ (T const &s, Array<T,N,R2> const& b) {
    return Array<T,N,Operation<T,Scalar<T>,R2,3>>
      (Operation<T,Scalar<T>,R2,3>(Scalar<T>(s), b.rep()));
  }

  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, Operation<T,R2,Scalar<T>,3>>
  operator/ (Array<T,N,R2> const& b, T const &s) {
    return Array<T,N,Operation<T,R2,Scalar<T>,3>>
      (Operation<T,R2,Scalar<T>,3>(b.rep(),Scalar<T>(s)));
  }

  
  // --- Unary Operation wrapper function --- //
  
  template<typename T, typename OP1, typename F> class SOperation{
  private:
    typename E_Traits<OP1>::ExprRef op1;
  public: 
    SOperation (OP1 const& a) : op1(a) {};
    M_INLINE T operator[] (std::size_t idx) const {return F::run(op1[idx]);}
    constexpr M_INLINE std::size_t size()const{return op1.size();}
  };

  // --- Unary Operation + arg wrapper function --- //
  
  template<typename T, typename U, typename OP1, typename F> class SOperationExtra{
  private:
    typename E_Traits<OP1>::ExprRef op1;
    U extra;
  public: 
    SOperationExtra (OP1 const& a, U const& extin) : op1(a), extra(extin) {};
    M_INLINE T operator[] (std::size_t idx) const {return F::run(op1[idx], extra);}
    constexpr M_INLINE std::size_t size()const{return op1.size();}
  };
  
  // --- Mathematical expressions --- //


  // --- sqrt --- //
  
  template<typename T>
  struct Sqrt{
    static M_INLINE const T run(T const& var){return std::sqrt(var);} 
  };

  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Sqrt<T>>>
  sqrt(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Sqrt<T>>>(SOperation<T,R2,Sqrt<T>>(b.rep()));
  }
 
  // --- abs --- //
  
  template<typename T>
  struct Abs{
    static M_INLINE const T run(T const& var){return std::abs(var);} 
  };

  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Abs<T>>>
  abs(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Abs<T>>>(SOperation<T,R2,Abs<T>>(b.rep()));
  }

  // --- exp --- //
  
  template<typename T>
  struct Exp{
    static M_INLINE const T run(T const& var){return std::exp(var);} 
  };

  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Exp<T>>>
  exp(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Exp<T>>>(SOperation<T,R2,Exp<T>>(b.rep()));
  }
  
  // --- log --- //
  
  template<typename T>
  struct Log{
    static M_INLINE const T run(T const& var){return std::log(var);} 
  };

  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Log<T>>>
  log(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Log<T>>>(SOperation<T,R2,Log<T>>(b.rep()));
  }
  
  // --- log10 --- //
  
  template<typename T>
  struct Log10{
    static M_INLINE const T run(T const& var){return std::log10(var);} 
  };

  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Log10<T>>>
  log10(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Log10<T>>>(SOperation<T,R2,Log10<T>>(b.rep()));
  }
  // --- sin --- //

  template<typename T>
  struct Sin{
    static M_INLINE const T run(T const& var){return std::sin(var);} 
  };

  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Sin<T>>>
  sin(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Sin<T>>>(SOperation<T,R2,Sin<T>>(b.rep()));
  }
  
  // --- cos --- //

  template<typename T>
  struct Cos{
    static M_INLINE const T run(T const& var){return std::cos(var);} 
  };
  
  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Cos<T>>>
  cos(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Cos<T>>>(SOperation<T,R2,Cos<T>>(b.rep()));
  }
  
  // --- tan --- //
  
  template<typename T>
  struct Tan{
    static M_INLINE const T run(T const& var){return std::tan(var);} 
  };
  
  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Tan<T>>>
  tan(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Tan<T>>>(SOperation<T,R2,Tan<T>>(b.rep()));
  }

  
  // --- asin --- //

  template<typename T>
  struct Asin{
    static M_INLINE const T run(T const& var){return std::asin(var);} 
  };
  
  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Asin<T>>>
  asin(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Asin<T>>>(SOperation<T,R2,Asin<T>>(b.rep()));
  }
  
  // --- acos --- //

  template<typename T>
  struct Acos{
    static M_INLINE const T run(T const& var){return std::acos(var);} 
  };
  
  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Acos<T>>>
  acos(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Acos<T>>>(SOperation<T,R2,Acos<T>>(b.rep()));
  }

  // --- atan --- //
  
  template<typename T>
  struct Atan{
    static M_INLINE const T run(T const& var){return std::atan(var);} 
  };
  
  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Atan<T>>>
  atan(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Atan<T>>>(SOperation<T,R2,Atan<T>>(b.rep()));
  }
  

  // --- pow (array^const)--- //

  template<typename T, typename U>
  struct Pow{
    static M_INLINE const T run(T const& var, U const& p){return std::pow(var,p);} 
  };
  
  
  template<typename T, size_t N, typename U, typename R2>
  M_INLINE Array<T, N, SOperationExtra<T,U,R2,Pow<T,U>>>
  pow(Array<T,N,R2> const& b, U const& p)
  {
    return Array<T,N,SOperationExtra<T,U,R2,Pow<T,U>>>(SOperationExtra<T,U,R2,Pow<T,U>>(b.rep(),p));
  }

  // --- pow (const^array)--- //

  template<typename T, typename U>
  struct Pow2{
    static M_INLINE const T run(T const& var, U const& p){return std::pow(p,var);} 
  };
  
  
  template<typename T, size_t N, typename U, typename R2>
  M_INLINE Array<T, N, SOperationExtra<T,U,R2,Pow2<T,U>>>
  pow(U const& p, Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperationExtra<T,U,R2,Pow2<T,U>>>(SOperationExtra<T,U,R2,Pow2<T,U>>(b.rep(),p));
  }
  

  // --- 10^Array --- //

  template<typename T>
  struct E10{
    static M_INLINE const T run(T const& var){return std::pow(static_cast<T>(10),var);} 
  };
  
  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,E10<T>>>
  e10(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,E10<T>>>(SOperation<T,R2,E10<T>>(b.rep()));
  }


  // --- Minus operator without arguiment --- //

  template<typename T>
  struct Minus{
    static M_INLINE const T run(T const& var){return -var;} 
  };
  
  
  template<typename T, size_t N, typename R2>
  M_INLINE Array<T, N, SOperation<T,R2,Minus<T>>>
  operator-(Array<T,N,R2> const& b)
  {
    return Array<T,N,SOperation<T,R2,Minus<T>>>(SOperation<T,R2,Minus<T>>(b.rep()));
  }
  
}

#endif
