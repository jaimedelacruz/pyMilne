/* -------------------------------------
   
  Various fftw based routines in template format

  Coded by J. de la Cruz Rodriguez (ISP-SU 2016)

  -------------------------------------- */

#ifndef FFTTOOLS_H
#define FFTTOOLS_H

#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <complex>
#include <fftw3.h>
#include <cstdio>

namespace mfft{
  
  /* ------------------------------------------------------------------------------- */
  
  /* --- Internal memory routines --- */

  template <class T> T **mat2d(size_t nx1, size_t nx2, bool zero = false){
    T **p = new T* [nx1];
    //
    if(zero) p[0] = new T [nx1 * nx2]();
    else     p[0] = new T [nx1 * nx2];
    //
    for(size_t x1=1;x1<nx1;++x1) p[x1] = p[x1-1] + nx2;
    return p;
  }
  
  template <class T> void del_mat(T **p){
    delete[] (p[0]);
    delete[] (p);
  }
  
  template <class T> T **var2dim(T *data, size_t nx1, size_t nx2){
    T **p = new T* [nx1];
    p[0] = data;
    for(size_t x1=1;x1<nx1;++x1) p[x1]=p[x1-1] + nx2;
    return p;
  }


  
  /* ------------------------------------------------------------------------------- */

  
  
  /* --- 
     1D FFTW convolution class, useful to perform many convolutions with the
     same PSF (e.g., inversions) because the PSF is only transformed once
     --- */
  
  template <class T> struct fftconv1D {
    size_t npad, n, n1, nft;
    std::complex<double> *otf, *ft;
    fftw_plan fplan, bplan;
    double *padded;
    bool started_plans;
    /* ------------------------------------------------------------------------------- */
    
    fftconv1D(): npad(0), n(0), n1(0), nft(0), otf(NULL), ft(NULL), fplan(0), bplan(0), padded(NULL), started_plans(false){};
    
    /* ------------------------------------------------------------------------------- */

    fftconv1D(fftconv1D<T> const& in): fftconv1D(){
      *this = in;
    }

    /* ------------------------------------------------------------------------------- */

    fftconv1D<T> &operator=(fftconv1D<T> const& in){

            
      // -- copy dimensions --- //
      npad = in.npad;
      n    = in.n;
      n1   = in.n1;
      nft  = in.nft;

      // --- allocate pointers --- //
      padded = new double [npad]();
      ft     = new std::complex<double> [nft]();
      otf    = new std::complex<double> [nft]();

      // --- init plans --- //
      fplan = fftw_plan_dft_r2c_1d(npad, padded, (fftw_complex*)ft, FFTW_MEASURE);
      bplan = fftw_plan_dft_c2r_1d(npad, (fftw_complex*)ft, padded, FFTW_MEASURE);

      started_plans = true;


      // --- copy data --- //
      memcpy(padded, in.padded, npad*sizeof(double));
      memcpy(ft,     in.ft,     nft*sizeof(double));
      memcpy(otf,    in.otf,    nft*sizeof(double));
      
      return *this;
    }

    /* ------------------------------------------------------------------------------- */

  fftconv1D(const int n_in, const int n_psf, const T *psf):
    fftconv1D()
    {
      
      if(n_psf == 0){
	return;
      }
    
      /* --- define dimensions --- */
      
      n = n_in, n1 = n_psf, npad = ((n1/2)*2 == n1) ? n1+n : n1+n-1;
      nft = npad/2 + 1;

      /* --- allocate arrays --- */
      
      double *ppsf   = new double [npad]();
      padded         = new double [npad]();
      //
      ft  = new std::complex<double> [nft]();
      otf = new std::complex<double> [nft]();

      
      /* --- shift PSF 1/2 of the elements of the PSF cyclicly. Apply normalizations --- */
      
      double psf_tot = 0.0;
      for(size_t ii=0; ii<n1; ii++) psf_tot += psf[ii];
      psf_tot = 1.0 / (psf_tot * npad);
      //
      for(size_t ii = 0; ii<n1; ii++) ppsf[ii] = (double)psf[ii] * psf_tot;
      std::rotate(&ppsf[0], &ppsf[n1/2], &ppsf[npad]);

      
      /* --- Init forward and backward plans --- */

      fplan = fftw_plan_dft_r2c_1d(npad, padded, (fftw_complex*)ft, FFTW_MEASURE);
      bplan = fftw_plan_dft_c2r_1d(npad, (fftw_complex*)ft, padded, FFTW_MEASURE);
      started_plans = true;
      

      /* --- transform psf --- */
      
      fftw_execute_dft_r2c(fplan, ppsf, (fftw_complex*)otf);


      
      /* --- take the conjugate --- */

      for(size_t ii=0; ii<nft; ++ii) otf[ii] = std::conj(otf[ii]);

      

      /* --- clean-up --- */
      
      delete [] ppsf;
    }
    /* ------------------------------------------------------------------------------- */
    
    ~fftconv1D(){
      
      if(ft)  delete [] ft;
      if(otf) delete [] otf;
      if(padded) delete [] padded;

      if(started_plans){
	fftw_destroy_plan(fplan);
	fftw_destroy_plan(bplan);
      }

      ft = NULL, otf = NULL, padded = NULL, started_plans = false;
      n = 0, n1 = 0, npad = 0, nft = 0;
    }
  /* ------------------------------------------------------------------------------- */
    
    inline void convolve(size_t n_in, T *d)const{

      if(npad == 0){
	return;
      }
      
      if(n_in != n){
	fprintf(stderr, "error: fftconvol1D::convolve: n_in [%ld] != n [%ld], not convolving!\n", n_in, n);
	return;
      }

      
      /* --- copy data to padded array --- */
      
      for(size_t ii = 0; ii<n; ii++)         padded[ii] = (double)d[ii];
      for(size_t ii = n; ii<n+n1/2; ii++)    padded[ii] = (double)d[n-1];
      for(size_t ii = n+n1/2; ii<npad; ii++) padded[ii] = (double)d[0];

      
      /* --- Forward transform --- */

      fftw_execute_dft_r2c(fplan, (double*)padded, (fftw_complex*)ft);

      
      /* --- Convolve --- */
      
      for(size_t ii = 0; ii<nft; ii++) ft[ii] *= otf[ii];

      
      /* --- Backwards transform --- */

      fftw_execute(bplan);


      /* --- Copy back data (inplace) --- */

      for(size_t ii = 0; ii<n; ii++) d[ii] = (T)padded[ii];

    }

    /* ------------------------------------------------------------------------------- */
    
    
  }; // fftconvol1D class
  
 
  
}//namespace

#endif
