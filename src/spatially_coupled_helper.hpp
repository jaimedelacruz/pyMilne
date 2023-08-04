#ifndef SPAT2HELPHPP
#define SPAT2HELPHPP

#include <vector>
#include <memory>

#include <Eigen/SparseCore>
#include <Eigen/Core>
#include "Arrays.hpp"
#include "Milne.hpp"
#include "spatially_regularized_tools.hpp"
#include "LinearOperators.hpp"

namespace spa{
  
  // ******************************************************************************** //

  template<typename T, typename ind_t = long>
  struct SpatRegion{
    int ny, nx, wl, wh, y0, y1, x0, x1;
    T clip_threshold;
    
    mem::Array<double,1> wav;
    
    mem::Array<T,4> obs;
    mem::Array<T,4> syn;
    mem::Array<T,3> r;
    mem::Array<T,2> sig;
    mem::Array<int,3> rsum;
    mem::Array<T,2> psf;
    mem::Array<T,2> pixel_weight;
    
    Eigen::SparseMatrix<T,Eigen::RowMajor, ind_t> Op;
    Eigen::SparseMatrix<T,Eigen::RowMajor, ind_t> OpT;
    Eigen::SparseMatrix<T,Eigen::RowMajor, ind_t> cc;
    std::vector<mem::Array<T,2>> cc1;
    std::vector<std::array<int,4>> matrix_elements;

    // ---------------------------------------------------------------- //
    
    void prepareHessianElements(ind_t const ny, ind_t const nx, int const nthreads);
    
    // ---------------------------------------------------------------- //

    inline SpatRegion(): ny(0), nx(0), wl(0), wh(-1), y0(0), y1(-1), x0(0), x1(-1),
		  clip_threshold(T(1)), wav(),  obs(), syn(), r(),
		  sig(), rsum(),  psf(), pixel_weight(), Op(), OpT(), cc(), cc1(),  matrix_elements(){};

    // ---------------------------------------------------------------- //

    inline SpatRegion(SpatRegion<T,ind_t> const& in):
      ny(in.ny), nx(in.nx), wl(in.wl), wh(in.wh), y0(in.y0), y1(in.y1), x0(in.x0), x1(in.x1),
      clip_threshold(in.clip_threshold), wav(in.wav), obs(in.obs), syn(), r(), sig(in.sig),
      rsum(in.rsum), psf(in.psf), pixel_weight(in.pixel_weight), Op(in.Op), OpT(in.OpT), cc(in.cc),
      cc1(in.cc1), matrix_elements(in.matrix_elements)
    {
      r.resize(ny,nx,4*(wh-wl+1));
      syn.resize(ny,nx,4,wh-wl+1);
    }
						   

    // ---------------------------------------------------------------- //

    SpatRegion(ind_t const iny, ind_t const inx,
	       ind_t const iny1, ind_t const inx1, ind_t const iwl, ind_t const iwh,
	       ind_t const iy0, ind_t const iy1, ind_t const ix0, ind_t const ix1,
	       T const clip_thres, const T* const iwav, const T* const isig,
	       ind_t const pny, ind_t const pnx, const T* const iPsf, T* const iObs,
	       int const nthreads);
    
    // ---------------------------------------------------------------- //

    inline SpatRegion<T,ind_t> &operator=(SpatRegion<T,ind_t> const& in)
    {
      ny=in.ny, nx=in.nx, wl=in.wl, wh=in.wh, y0=in.y0, y1=in.y1, x0=in.x0, x1=in.x1,
	clip_threshold=in.clip_threshold, wav=in.wav, obs=in.obs, sig=in.sig,
	rsum=in.rsum, psf=in.psf, pixel_weight=in.pixel_weight, Op=in.Op, OpT=in.OpT, cc=in.cc,
	matrix_elements=in.matrix_elements;

      r.resize(ny,nx,4*(wh-wl+1));
      syn.resize(ny,nx,4,wh-wl+1);
      
    }

    // ---------------------------------------------------------------- //
    
    
  };
  
  // **************************************************************** //

  template<typename T, typename U = T, typename ind_t = long>
  struct Data{
    ind_t ny, nx, npar, ns, nw;
    T mu, nData;

    std::vector<T> sig_total;
    std::vector<std::shared_ptr<SpatRegion<T>>> regions;
    std::vector<ml::Milne<T>*> me;
    std::vector<Par<T>> Pinfo;
    
    
    // ---------------------------------------------------------------- //
    
    inline Data():
      ny(0), nx(0), npar(0), ns(0), nw(0),  mu(T(1)), nData(T(0)), sig_total(), regions(), me(),
      Pinfo(){};

    // ---------------------------------------------------------------- //

    inline void reset_regions(){regions.clear();}
    
    // ---------------------------------------------------------------- //
    
    void synthesize(mem::Array<T,3> &m, mem::Array<T,4> &syn)const;
    
    // ---------------------------------------------------------------- //
    
    void synthesize_rf(mem::Array<T,3> &m, mem::Array<T,4> &syn, mem::Array<T,5> &J)const;
    
    // ---------------------------------------------------------------- //

    void NormalizePars(mem::Array<T,3> &m)const;
    
    // ---------------------------------------------------------------- //

    void ScalePars(mem::Array<T,3> &m)const;
    // ---------------------------------------------------------------- //

    void CheckPars(mem::Array<T,3> &m)const;
    
    // ---------------------------------------------------------------- //

    Eigen::SparseMatrix<U, Eigen::RowMajor, ind_t> get_L(ind_t const npar)const;
    
    // ---------------------------------------------------------------- //

    Eigen::Matrix<U,Eigen::Dynamic,1> getGamma(mem::Array<T,3> const& m)const;

    // ---------------------------------------------------------------- //

  };
}

#endif
