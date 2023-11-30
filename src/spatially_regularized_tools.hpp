#ifndef SPATHTPP
#define SPATHTPP

// -------------------------------------------------------
//   
//   Spatially-regularized Levenberg Marquardt algorithm
//   Coded by J. de la Cruz Rodriguez (ISP-SU, 2020)
//
//   Reference: de la Cruz Rodriguez (2019):
//   https://ui.adsabs.harvard.edu/abs/2019A%26A...631A.153D/abstract
//
//   -------------------------------------------------------

#include <cmath>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "phyc.h"

namespace spa{

  // ************************************************************** //
  
  template<class T> inline T SQ(T const &var){return var*var;}
  
  // ************************************************************** //

  template <class T, typename U > U ksum(const size_t n, const T* const  __restrict__ arr){
    
    U sum = 0, c = 0;
    
    for(size_t kk = 0; kk<n; ++kk){
      U const y = static_cast<U>(arr[kk]) - c;
      U const t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    
    return sum;
  }
  
  // ************************************************************** //

  template <class T, typename U > U ksum2(const size_t n, const T* const __restrict__ arr){
    
    U sum = 0, c = 0;
    
    for(size_t kk = 0; kk<n; ++kk){
      U const y = SQ<U>(static_cast<U>(arr[kk])) - c;
      U const t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    
    return sum;
  }
  // ************************************************************** //

  template <class T, typename U > U ksumMult(const size_t n, const T* const __restrict__ arr, const T* const __restrict__ arr1){
    
    U sum = 0, c = 0;
    
    for(size_t kk = 0; kk<n; ++kk){
      U const y = static_cast<U>(arr[kk]*arr1[kk]) - c;
      U const t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    
    return sum;
  }
  // ************************************************************** //

  template<typename T>
  struct Par{
    bool isCyclic;
    bool limited;
    T scale;
    T limits[2];
    T alpha;
    T alpha_t;
    T beta;

    Par(): isCyclic(false), limited(false), scale(1.0), limits{0,0}, alpha(0), alpha_t(0), beta(0){};
    Par(bool const cyclic, bool const ilimited, T const scal, T const mi, T const ma, T const alp, T const alpt, T const bet):
      isCyclic(cyclic), limited(ilimited), scale(scal), limits{mi,ma}, alpha(alp), alpha_t(alpt), beta(bet){};

    Par(Par<T> const& in):
      isCyclic(in.isCyclic) ,limited(in.limited), scale(in.scale), limits{in.limits[0],
      in.limits[1]}, alpha(in.alpha), alpha_t(in.alpha_t), beta(in.beta){};

    Par<T> &operator=(Par<T> const& in)
    {
      isCyclic = in.isCyclic, limited = in.limited, scale=in.scale, limits[0]=in.limits[0];
      limits[1]=in.limits[1], alpha = in.alpha, alpha_t=in.alpha_t, beta=in.beta;
      return *this;
    }

    // ---------------------------------------------------------- //
    
    inline void Normalize(T &val)const{val /= scale;};
    
    // ---------------------------------------------------------- //

    inline void Scale(T &val)const{val *= scale;};
    
    // ---------------------------------------------------------- //

    inline void Check(T &val)const{
      if(!limited) return;
      if(isCyclic){
	if(val > limits[1]) val -= 3.1415926f;
	if(val < limits[0]) val += 3.1415926f;
      }
      val = std::max<T>(std::min<T>(val, limits[1]),limits[0]);
    }
    
    // ---------------------------------------------------------- //

    inline void CheckNormalized(T &val)const{
      if(!limited) return;
      Scale(val);
      Check(val);
      Normalize(val);
    }
    
  };
  
  // ************************************************************** //

  template<typename T> struct Chi2{
    T chi2;
    T pen2;

    Chi2(T const ichi2, T const ipen2): chi2(ichi2), pen2(ipen2){};

    Chi2(): chi2(0), pen2(0){};

    Chi2(Chi2<T> const& in): chi2(in.chi2), pen2(in.pen2){};

    Chi2<T> &operator=(Chi2<T> const& in){chi2 = in.chi2; pen2 = in.pen2; return *this;}

    inline T value()const{return chi2 + pen2;}
    
    // ---------------------------------------------------------- //

    std::string formatted()const{
      char bla[50];
      sprintf(bla, "%13.5f (%13.5f + %13.5f)", value(), chi2, pen2);
      return std::string(bla);
    }
    
    // ---------------------------------------------------------- //

    T operator()()const{return value();}
    
  };

  // ************************************************************** //
  
  template<typename T> struct container{
    long nDat, nt, ny, nx, Nreal;
    T mu;
    Eigen::TensorMap<Eigen::Tensor<T,4, Eigen::RowMajor>> obs;
    Eigen::Matrix<T,Eigen::Dynamic,1> sig;
    Eigen::Tensor<T,1, Eigen::RowMajor> pwe;

    std::vector<Par<T>> Pinfo;
    std::vector<const ml::Milne<T>*> Me;

    long getNreal()const{return Nreal;}
    
    container(): nDat(0), nt(0), ny(0), nx(0), Nreal(1), mu(1),   sig(), pwe(), Pinfo(), Me(){};

    // ------------------------------------------------------------ //
    
    container(long const i_nt, long const iny, long const inx, T const iMu, long const inDat, T* const __restrict__ iobs,
	      const T* const __restrict__ isig,
	      std::vector<Par<T>> const& Pinfo_in, std::vector<ml::Milne<T>> const& iMe):
      nDat(inDat), nt(i_nt), ny(iny), nx(inx), mu(iMu), obs(iobs, i_nt, iny, inx, inDat), sig(nDat), pwe(i_nt*inx*iny), Pinfo(Pinfo_in)
    {

      // --- References to Me class ---//
      int const nthreads = (int)iMe.size();
      for(int ii=0; ii<nthreads; ++ii)
	Me.emplace_back(static_cast<const ml::Milne<T>*>(&iMe[ii]));

      
      // --- sigma --- //
      Nreal = 0;
      for(long ii=0; ii<nDat; ++ii){
	sig[ii] = 1.0/isig[ii];

	if(isig[ii] < 1.e10)
	  ++Nreal;
      }

      // --- pixel weight --- //
      long const npix = inx*iny;
      long const ntot = npix*nt;
      long const nwav = inDat / 4;
      double meanpwe = 0.0;
      long nsum=0;
      
      for(long ii = 0; ii<ntot; ++ii){
	
	double sum = 0.0;
	const T* const __restrict__ iI = iobs + ii*inDat;
	
	for(long jj=0; jj<nwav;++jj){
	  sum += iI[jj];
	}

	sum /= nwav;
	
	if(sum < 1.e-6){
	  pwe[ii] = 1.e32;// intensity is zero, ignore pixel
	}else{
	  pwe[ii] = std::abs(sum);
	  meanpwe += pwe[ii];
	  ++nsum;
	}
      }

      T const mpwe = meanpwe/nsum;
      pwe = (mpwe / pwe).sqrt();
      
    }
    
    // ------------------------------------------------------------ //

    int getNthreads()const{return int(Me.size());}
    
    // ------------------------------------------------------------ //
    
    void NormalizePars(long const nPar,  T* const __restrict__ par)const
    {
      long const ny1 = ny;
      long const nx1 = nx;
      long const nt1 = nt;
      long const npix = ny1*nx1;
      
      for(long tt=0; tt<nt1; ++tt)
	for(long ipix=0; ipix<npix; ++ipix){
	  long const off = (tt*npix+ipix)*nPar;
	  for(long pp=0; pp<nPar; ++pp)
	    par[off + pp] /= Pinfo[pp].scale;
	}
    }
    
    // ------------------------------------------------------------ //
    
    void ScalePars(long const nPar,  T* const __restrict__ par)const
    {
      long const ny1 = ny;
      long const nx1 = nx;
      long const nt1 = nt;
      long const npix = ny1*nx1;

      for(long tt=0; tt<nt1; ++tt)
	for(long ipix=0; ipix<npix; ++ipix){
	  long const off = (tt*npix+ipix)*nPar;
	    for(long pp=0; pp<nPar; ++pp)
	      par[off + pp] *= Pinfo[pp].scale;
	}
    }
    
    // ------------------------------------------------------------ //

    void checkPars(long const nPar,  T* const __restrict__ par)const
    {
      long const ny1 = ny;
      long const nx1 = nx;
      long const nt1 = nt;
      long const npix = ny1*nx1;
      
      for(long pp=0; pp<nPar; ++pp){
	
	if(!Pinfo[pp].limited) continue;
	
	T const imin = Pinfo[pp].limits[0];
	T const imax = Pinfo[pp].limits[1];
	
	for(long tt=0; tt<nt1; ++tt)
	  for(long ipix=0; ipix<npix; ++ipix){
	    long const off = (tt*npix+ipix)*nPar+pp;
	    
	    T &iPar = par[off];
	    iPar = std::min<T>(std::max<T>(iPar, imin), imax);
	  } //xx
      } // pp
    }
    
    
    // ------------------------------------------------------------ //

    void synthesize(long const nPar,  T* const __restrict__ par,  T* const __restrict__ syn,  T* const __restrict__ r)const
    {
      int const nthreads = Me.size();
      long const npix = nx*ny;
      long ipix = 0, tid = 0, ww= 0, tt=0;
      long const ndat = nDat;
      T const scl = 1.0 / sqrt(T(Nreal*npix*nt));
      
      const T* const __restrict__ o = static_cast<const T* const>(&obs(0,0,0,0));

#pragma omp parallel default(shared) firstprivate(ipix, tt, tid, ww) num_threads(nthreads)

      {
	tid = omp_get_thread_num();
#pragma omp for collapse(2)
	for(tt=0; tt<nt;++tt){
	  for(ipix = 0; ipix<npix; ++ipix){

	    long const off = tt*npix+ipix;
	    
	    // --- scale parameters --- //
	    for(ww=0; ww<nPar; ++ww){
	      Pinfo[ww].Scale(par[off*nPar + ww]);
	      Pinfo[ww].Check(par[off*nPar + ww]);
	    }
	    
	    Me[tid]->synthesize(&par[off*nPar], &syn[nDat*off], mu);
	    
	    for(ww=0; ww<ndat; ++ww)
	      r[off*ndat + ww] = pwe[off]*(o[off*ndat + ww] - syn[off*ndat + ww]) * sig[ww] * scl;      
	    
	    for(ww=0; ww<nPar; ++ww)
	      Pinfo[ww].Normalize(par[off*nPar + ww]);
	    
	  } // ipix
	} // tt
      } // parallel block
      
    }
 
    
    // ------------------------------------------------------------ //

    T fx(long const nPar, T* const __restrict__ par,  T* const __restrict__ syn_in, T* const __restrict__ r)const 
    {

      // --- Synthesize without derivatives --- //
      synthesize(nPar, par, syn_in, r);

      long const nEl = long(nDat)*long(nx*ny*nt);
      return ksum2<T,long double>(nEl, r);
    }
    
    
    // ------------------------------------------------------------ //

    T getChi2( T* const __restrict__ r)const
    {
      long const ndat = nDat;
      long const npix = nx*ny;
      long const nthreads = Me.size();

      std::vector<double> chi2(nthreads,T(0));

      // --- split the work in threads, each stores its own count --- //
      long ipix=0, tid=0, tt=0;
#pragma omp parallel default(shared) firstprivate(ipix, tid, tt) num_threads(nthreads)      
      {
	tid = omp_get_thread_num();
#pragma omp for collapse(2)
	for(tt=0; tt<nt;++tt){
	  for(ipix=0; ipix<npix; ++ipix){
	    
	    chi2[tid] += ksum2<T,long double>(ndat, &r[(tt*npix+ipix)*ndat]);
	    
	  } // ipix
	} // tt
      }// parallel block

      // --- add chi2 from all threads --- //
      T chitot = chi2[tid];
      
      for(long ii=1; ii<nthreads; ++ii)
	chitot += chi2[ii];

      return chitot;
    }

    // ------------------------------------------------------------ //
    template<typename iType = long>
    Eigen::Matrix<T,Eigen::Dynamic,1> getGamma(long const npar, T* const __restrict__ par)const
    {
      long const nthreads = Me.size();
      iType const npix = long(nx)*long(ny);
      iType const nTot = npix*nt;
      
      // --- how many penalty functions do we need? Npixels-1 have penalties, (0,0) doesn't ---//

      long const nPen = 4*nTot*npar; // maximum number of penalty functions per pixel
      T const sqr_nPen = sqrt(T(nPen));

      Eigen::Matrix<T,Eigen::Dynamic,1> Gam(nPen);
      Gam.setZero();

      Eigen::TensorMap<Eigen::Tensor<T,4,Eigen::RowMajor>> m(par, nt, ny, nx, npar);
      T const normAzi = 3.1415926 / Pinfo[2].scale;

      
      // --- Scaling factors are sqrt-ed so when squared we get the right number --- //
      
      T* const  __restrict__ sq_alpha  = new T [npar]();
      T* const  __restrict__ sq_alphat = new T [npar]();
      T* const  __restrict__ sq_beta   = new T [npar]();

      for(long ii=0; ii<npar; ++ii){
	sq_alpha[ii]  = sqrt(Pinfo[ii].alpha)   / sqr_nPen;
	sq_alphat[ii] = sqrt(Pinfo[ii].alpha_t) / sqr_nPen;
	sq_beta[ii]   = sqrt(Pinfo[ii].beta)    / sqr_nPen;
      }
      
#pragma omp parallel default(shared) num_threads(nthreads)      
      {
#pragma omp for
	for(long idat=0; idat<nTot; ++idat){
	  
	  long const tt = idat / npix;
	  long const ipix = idat - tt*npix;
	  long const yy = ipix / nx;
	  long const xx = ipix - yy*nx;  
	  long const off = idat*npar;
	  
	  
	  // -- Time-reg --- //
	  
	  if(tt > 0){
	    for(long pp=0; pp<npar; ++pp){
		Gam[(off+pp)*4] = sq_alphat[pp] * (m(tt,yy,xx,pp) - m(tt-1,yy,xx,pp));
	    }
	    
	    // --- check azimuth --- //
	    
	    long const pp = 2;
	    T const azi = (m(tt,yy,xx,pp) - m(tt-1,yy,xx,pp));
	    
	    if     (fabs(azi-normAzi) < fabs(azi)) Gam[(off+pp)*4] =  sq_alphat[pp]*(azi-normAzi);
	    else if(fabs(azi+normAzi) < fabs(azi)) Gam[(off+pp)*4] =  sq_alphat[pp]*(azi+normAzi);
	    
	  } // tt > 0
	  
	  
	    // --- spatial-reg (y-axis) --- //
	  
	  if(yy > 0){
	    for(long pp=0; pp<npar; ++pp)
	      Gam[(off+pp)*4+1] += sq_alpha[pp] * (m(tt,yy,xx,pp) - m(tt,yy-1,xx,pp));
	    
	    // --- check azimuth --- //
	    
	    long const pp = 2;
	    T const azi = (m(tt,yy,xx,pp) - m(tt,yy-1,xx,pp));
	    
	    if     (fabs(azi-normAzi) < fabs(azi)) Gam[(off+pp)*4+1] =  sq_alpha[pp]*(azi-normAzi);
	    else if(fabs(azi+normAzi) < fabs(azi)) Gam[(off+pp)*4+1] =  sq_alpha[pp]*(azi+normAzi);
	  }
	  
	  
	  // --- spatial-reg (x-axis) --- //
	  
	  if(xx > 0){
	    for(long pp=0; pp<npar; ++pp)
	      Gam[(off+pp)*4+2] = sq_alpha[pp] * (m(tt,yy,xx,pp) - m(tt,yy,xx-1,pp));
	    
	    
	    // --- check azimuth --- //
	    
	    long const pp = 2;
	    T const azi = (m(tt,yy,xx,pp) - m(tt,yy,xx-1,pp));
	    
	    if     (fabs(azi-normAzi) < fabs(azi)) Gam[(off+pp)*4+2] =  sq_alpha[pp]*(azi-normAzi);
	    else if(fabs(azi+normAzi) < fabs(azi)) Gam[(off+pp)*4+2] =  sq_alpha[pp]*(azi+normAzi);
	  }


	  // --- Local low-norm --- //
	  
	  for(long pp=0; pp<npar; ++pp)
	    Gam[(off+pp)*4+3] = sq_beta[pp] * m(tt,yy,xx,pp);

	  
	  // --- in the case of inclination, prefer vertical fields --- //

	  T const quant = ((m(tt,yy,xx,1) <= phyc::PI/T(2)) ? T(0) : phyc::PI);
	  Gam[(off+1)*4+3] -= sq_beta[1] * quant;
	  
	} // ipix
      }// parallel
      
      delete [] sq_alpha;
      delete [] sq_alphat;
      delete [] sq_beta;

      return Gam;
    }

    
    // ------------------------------------------------------------ //
    
    void synthesize_der_one(long const npar, T* __restrict__ par, T* __restrict__ r,
			    T* __restrict__ J, long const tid, long const tpix)const
    {
      
      T const scl = sqrt(T(Nreal)*T(nx*ny*nt));
      T iScl = 0;
      long const ndat = nDat;
      
      const T* const __restrict__ o = static_cast<const T* const>(&obs(0,0,0,0));
      
      for(long pp=0; pp<npar; ++pp){
	Pinfo[pp].Scale(par[pp]);
	//Pinfo[pp].Check(par[pp]);
      }
	    
      Me[tid]->synthesize_rf(par, r, J, mu);
      
      
      for(long ww=0; ww<ndat; ++ww)
	r[ww] = pwe[tpix]*((o[tpix*ndat+ww] - r[ww]) * sig[ww] / scl);  
      
	  
      // --- scale J and compute r--- //
      for(long pp=0; pp<npar; ++pp){
	iScl =  pwe[tpix] * Pinfo[pp].scale / scl;
	
	for(long ww=0; ww<ndat; ++ww)
	  J[pp*ndat+ww] *= iScl * sig[ww];
	
	Pinfo[pp].Normalize(par[pp]);

      } // pp
    
    }
    
    // ------------------------------------------------------------ //

    T fx_dx(long const nPar,  T* const __restrict__ par,  T* const __restrict__ syn_in,  T* const __restrict__ r,
	    T* const __restrict__ J)const 
    {
      
      synthesize_rf(nPar, par, syn_in, r, J);
      
      long const nEl = long(nDat)*long(nx*ny*nt);
      return ksum2<T,long double>(nEl, r);
    }
    
    // ------------------------------------------------------------ //

    template<typename iType = long>
    Eigen::SparseMatrix<T, Eigen::RowMajor, iType> get_L(long const npar,  T* const __restrict__ par)const
    {
      
      iType const npix = nx*ny;
      iType const Nx = nx;
      iType const Nt = nt;
      iType const nthreads = Me.size();
      iType const nTot = npix*Nt;

      iType const nPen = 4*nTot*npar;
      T const sqr_nPen = sqrt(T(nPen));
      
      std::vector<T> iAlpha(npar, T(0));
      std::vector<T> iAlphat(npar, T(0));
      std::vector<T> iBeta(npar, T(0));

      for(long ii=0;ii<npar; ++ii){
	iAlpha[ii]  = sqrt(Pinfo[ii].alpha)   / sqr_nPen;
	iAlphat[ii] = sqrt(Pinfo[ii].alpha_t) / sqr_nPen;
	iBeta[ii]   = sqrt(Pinfo[ii].beta)    / sqr_nPen;
      }
      
	
      // --- get matrix dimensions --- //

      long const nrows = nPen; // Some of them are zero at the edge of the domain, but anyhow...
      long const ncols = npar*npix*Nt;

      Eigen::SparseMatrix<T,Eigen::RowMajor, iType> L(nrows, ncols);

      
      // --- Get number of elements per row --- //

      long const Elements_per_row = 2; // We only have two non-zero elements per Reg function at most
      Eigen::VectorXi nElements_per_row = Eigen::VectorXi::Constant(nrows, Elements_per_row); // 1D vector of integers


      
      // --- reserve elements in the sparse matrix --- //
      
      L.reserve(nElements_per_row);


      // --- Fill Matrix in parallel --- //
      
#pragma omp parallel default(shared) num_threads(nthreads)      
      {
#pragma omp for
	for(long idat = 0; idat<nTot; ++idat){
	  
	  long const tt = idat / npix;
	  long const ipix = idat - tt*npix;
	  long const yy = ipix / Nx;
	  long const xx = ipix - yy*Nx; 
	  long const off = idat*npar;
	  
	  
	  // --- Each thread fills all regularization derivatives for one pixel (all parameters) --- //
	  
	  // -- Time-reg --- //
	  
	  if(tt > 0){
	    for(long pp=0; pp<npar; ++pp){
	      long const y = (off+pp)*4;
	      
	      L.insert(y,((tt-1)*npix+ipix)*npar+pp) = -iAlphat[pp];
	      L.insert(y,off+pp)                     = iAlphat[pp];
	      
	    }
	  } // tt > 0
	  
	  
	    // --- spatial-reg (y-axis) --- //
	  
	  if(yy > 0){
	    for(long pp=0; pp<npar; ++pp){
	      long const y = (off+pp)*4;
	      
	      L.insert(y+1,(tt*npix+(yy-1)*nx+xx)*npar+pp) = -iAlpha[pp];
	      L.insert(y+1,off+pp)                         = iAlpha[pp];
	    }
	  }
	  
	  
	  // --- spatial-reg (x-axis) --- //
	  
	  if(xx > 0){
	    for(long pp=0; pp<npar; ++pp){
	      long const y = (off+pp)*4;
	      
	      L.insert(y+2,(tt*npix+yy*nx+xx-1)*npar+pp) = -iAlpha[pp];
	      L.insert(y+2,off+pp)                       = iAlpha[pp];
	    }
	  }


	  // --- low-norm --- //
	  
	  for(long pp=0; pp<npar; ++pp){
	    long const y = (off+pp)*4;
	    L.insert(y+3,off+pp)                       = iBeta[pp];
	  }
	  
	} // ipix
      } // parallel
      
      return L;
    }

  };
}

#endif
