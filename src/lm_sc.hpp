#ifndef LMSC
#define LMSC

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/IterativeSolvers>

#include "spatially_coupled_helper.hpp"
#include "spatially_coupled_tools.hpp"


namespace spa{


  
  // ******************************************************************************* //
  
  /* ---

     Simple Chi2_t class to store the result from the merit function and regularization

     --- */
  
  template<typename T>
  struct Chi2_t{
    T chi, reg;
    
    inline Chi2_t(): chi(0), reg(0){}

    inline Chi2_t(Chi2_t<T> const& in): chi(in.chi), reg(in.reg){}

    inline Chi2_t(T const& ichi, T const& ireg): chi(ichi), reg(ireg){}

    T get()const{return chi+reg;}

    void operator+=(Chi2_t<T> const& in){chi+=in.chi, reg+=in.reg;}

    Chi2_t<T> &operator=(Chi2_t<T> const& in)
    {
      chi = in.chi;
      reg = in.reg;
      return *this;
    }


    
  };


  // --- Define some operators --- //
  
  template<typename T>
  inline bool operator<(Chi2_t<T> const& a, Chi2_t<T> const& b){return ((a.get()<b.get())? true : false);}
  
  template<typename T>
  inline bool operator<=(Chi2_t<T> const& a, Chi2_t<T> const& b){return ((a.get()<=b.get())? true : false);}

  template<typename T>
  inline bool operator>(Chi2_t<T> const& a, Chi2_t<T> const& b){return ((a.get()>b.get())? true : false);}

  template<typename T>
  inline bool operator>=(Chi2_t<T> const& a, Chi2_t<T> const& b){return ((a.get()>=b.get())? true : false);}

  template<typename T>
  inline bool operator==(Chi2_t<T> const& a, Chi2_t<T> const& b){return ((a.get()==b.get())? true : false);}
  
  template<typename T>
  inline bool operator!=(Chi2_t<T> const& a, Chi2_t<T> const& b){return ((a.get()!=b.get())? true : false);}
  
  template<typename T>
  inline Chi2_t<T> operator+(Chi2_t<T> const& a, Chi2_t<T> const& b){return Chi2_t<T>(a.chi+b.chi,a.reg+b.reg);}

  template<typename T>
  inline Chi2_t<T> operator*(T const a, Chi2_t<T> const& b){return a*b.get();}

  template<typename T>
  inline Chi2_t<T> operator*(Chi2_t<T> const& a, T const b){return a.get()*b;}
  
  
  // ******************************************************************************* //


  template<typename T, typename U, typename ind_t>
  Chi2_t<T> getChi2(spa::Data<T,U,ind_t> &dat, mem::Array<T,4> const& syn, Eigen::SparseMatrix<U,Eigen::RowMajor, ind_t> const& L, mem::Array<T,3> &m)
  {

    T const Chi2 = getChi2obs(dat, syn, int(dat.me.size()));
    Eigen::Matrix<T,Eigen::Dynamic,1> gam = dat.getGamma(m);

    ind_t const nGam = gam.size();
    
    T sum = 0.0;
    for(ind_t ii=0;ii<nGam; ++ii)
      sum += SQ(gam[ii]);
    
    return Chi2_t<T>(Chi2, sum);
  }
  
  // ******************************************************************************* //
  
  
  template<typename T, typename U = T, typename ind_t = long>
  class LMsc{
  protected:
    ind_t npar, ny, nx;
    Eigen::SparseMatrix<U,Eigen::RowMajor, ind_t> L, LL; 
    Eigen::Matrix<U, Eigen::Dynamic, 1> Diagonal;
  public:

    LMsc(): npar(0), ny(0), nx(0), L(), LL(), Diagonal(){}

    LMsc(ind_t const inpar, ind_t const iny, ind_t const inx): npar(inpar), ny(iny), nx(inx), L(), LL(), Diagonal(){}

    
    Chi2_t<T> invert(spa::Data<T,U,ind_t> &dat,
		     mem::Array<T,3> &m,
		     mem::Array<T,4> &syn,
		     int const max_niter,
		     T const fx_thres,
		     int const method,
		     T const init_Lambda);
    
    spa::Chi2_t<T> getCorrection(int const iter, Chi2_t<T> const& bestChi2,
				 mem::Array<T,3> &m, Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> &A,
				 Eigen::Matrix<U,Eigen::Dynamic,1> const& B, T Lam, spa::Data<T,U,ind_t> &dat)const;
    

    
  };

  

}

#endif
