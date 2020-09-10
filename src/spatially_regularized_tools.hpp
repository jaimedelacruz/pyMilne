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

    Par(): isCyclic(false), limited(false), scale(1.0), limits{0,0}, alpha(0){};
    Par(bool const cyclic, bool const ilimited, T const scal, T const mi, T const ma, T const alp):
      isCyclic(cyclic), limited(ilimited), scale(scal), limits{mi,ma}, alpha(alp){};

    Par(Par<T> const& in): isCyclic(in.isCyclic) ,limited(in.limited), scale(in.scale), limits{in.limits[0], in.limits[1]}, alpha(in.alpha){};

    Par<T> &operator=(Par<T> const& in)
    {
      isCyclic = in.isCyclic, limited = in.limited, scale=in.scale, limits[0]=in.limits[0], limits[1]=in.limits[1], alpha = in.alpha;
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
	if(val < limits[0]) val += 3.1416026f;
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
    int nDat, ny, nx, Nreal;
    T mu;
    Eigen::TensorMap<Eigen::Tensor<T,3, Eigen::RowMajor>> obs;
    Eigen::Matrix<T,Eigen::Dynamic,1> sig;
    
    std::vector<Par<T>> Pinfo;
    std::vector<const ml::Milne<T>*> Me;

    int getNreal()const{return Nreal;}
    
    container(): nDat(0), ny(0), nx(0), Nreal(1), mu(1),   sig(), Pinfo(), Me(){};

    // ------------------------------------------------------------ //
    
    container(int const iny, int const inx, T const iMu, int const inDat, T* const __restrict__ iobs,
	      const T* const __restrict__ isig,
	      std::vector<Par<T>> const& Pinfo_in, std::vector<ml::Milne<T>> const& iMe):
      nDat(inDat), ny(iny), nx(inx), mu(iMu), obs(iobs, iny, inx, inDat), sig(nDat), Pinfo(Pinfo_in)
    {

      // --- References to Me class ---//
      int const nthreads = (int)iMe.size();
      for(int ii=0; ii<nthreads; ++ii) Me.emplace_back(static_cast<const ml::Milne<T>*>(&iMe[ii]));

      
      // --- sigma --- //
      Nreal = 0;
      for(int ii=0; ii<nDat; ++ii){
	sig[ii] = 1/isig[ii];

	if(isig[ii] < 1.e10)
	  ++Nreal;
	
      }
    }
    
    // ------------------------------------------------------------ //

    int getNthreads()const{return int(Me.size());}
    
    // ------------------------------------------------------------ //
    
    void NormalizePars(int const nPar,  T* const __restrict__ par)const
    {
      int const ny1 = ny;
      int const nx1 = nx;
      
      for(int yy=0; yy<ny1; ++yy)
	for(int xx=0; xx<nx1; ++xx)
	  for(int pp=0; pp<nPar; ++pp)
	    par[yy*nx*nPar + xx*nPar + pp] /= Pinfo[pp].scale;
    }
    
    // ------------------------------------------------------------ //
    
    void ScalePars(int const nPar,  T* const __restrict__ par)const
    {
      int const ny1 = ny;
      int const nx1 = nx;
      
      for(int yy=0; yy<ny1; ++yy)
	for(int xx=0; xx<nx1; ++xx)
	  for(int pp=0; pp<nPar; ++pp)
	    par[yy*nx*nPar + xx*nPar + pp] *= Pinfo[pp].scale;
    }

    // ------------------------------------------------------------ //

    void checkPars(int const nPar,  T* const __restrict__ par)const
    {
      int const ny1 = ny;
      int const nx1 = nx;
      
      for(int pp=0; pp<nPar; ++pp){

	if(!Pinfo[pp].limited) continue;
	
	T const imin = Pinfo[pp].limits[0];
	T const imax = Pinfo[pp].limits[1];
	
	for(int yy=0; yy<ny1; ++yy)
	  for(int xx=0; xx<nx1; ++xx){
	    T &iPar = par[yy*nx*nPar + xx*nPar + pp];
	    iPar = std::min<T>(std::max<T>(iPar, imin), imax);
	  } //xx
      } // pp
    }

    
    // ------------------------------------------------------------ //

    void synthesize(int const nPar,  T* const __restrict__ par,  T* const __restrict__ syn,  T* const __restrict__ r)const
    {
      int const nthreads = Me.size(); int const npix = nx*ny;
      int ipix = 0, tid = 0, ww= 0;
      int const ndat = nDat;
      T const scl = 1.0 / sqrt(T(Nreal*npix));
      const T* const __restrict__ o = static_cast<const T* const>(&obs(0,0,0));

#pragma omp parallel default(shared) firstprivate(ipix, tid, ww) num_threads(nthreads)

      {
	tid = omp_get_thread_num();
#pragma omp for
	for(ipix = 0; ipix<npix; ++ipix){
	  
	  // --- scale parameters --- //
	  for(ww=0; ww<nPar; ++ww){
	    Pinfo[ww].Scale(par[ipix*nPar + ww]);
	    Pinfo[ww].Check(par[ipix*nPar + ww]);
	  }
	  
	  Me[tid]->synthesize(&par[ipix*nPar], &syn[nDat*ipix], mu);

	  for(ww=0; ww<ndat; ++ww)
	    r[ipix*ndat + ww] = (o[ipix*ndat + ww] - syn[ipix*ndat + ww]) * sig[ww] * scl;      

	  for(ww=0; ww<nPar; ++ww)
	    Pinfo[ww].Normalize(par[ipix*nPar + ww]);
	  
	}
      } // parallel block
      
    }
 
    
    // ------------------------------------------------------------ //

    T fx(int const nPar, T* const __restrict__ par,  T* const __restrict__ syn_in, T* const __restrict__ r)const 
    {

      // --- Synthesize without derivatives --- //
      synthesize(nPar, par, syn_in, r);

      long const nEl = long(nDat)*long(nx*ny);
      return ksum2<T,double>(nEl, r);
    }
    
    
    // ------------------------------------------------------------ //

    T getChi2( T* const __restrict__ r)const
    {
      int const ndat = nDat;
      int const npix = nx*ny;
      int const nthreads = Me.size();

      std::vector<double> chi2(nthreads,T(0));

      // --- split the work in threads, each stores its own count --- //
      int ipix=0, tid=0;
#pragma omp parallel default(shared) firstprivate(ipix, tid) num_threads(nthreads)      
      {
	tid = omp_get_thread_num();
#pragma omp for
	for(ipix=0; ipix<npix; ++ipix){
	  
	  chi2[tid] += ksum2<T,double>(ndat, r[ipix*ndat]);
      
	} // ipix
      }// parallel block

      // --- add chi2 from all threads --- //
      T chitot = chi2[tid];
      
      for(int ii=1; ii<nthreads; ++ii)
	chitot += chi2[ii];

      return chitot;
    }

    // ------------------------------------------------------------ //
    template<typename iType = long>
    Eigen::Matrix<T,Eigen::Dynamic,1> getGamma(int const npar, T* const __restrict__ par)const
    {
      int const nthreads = Me.size();
      iType const npix = nx*ny;
      iType const ndat = nDat;
      
      // --- how many penalty functions do we need? Npixels-1 have penalties, (0,0) doesn't ---//


      int const nPen = 2*npix*npar; //
      T const sqr_nPen = sqrt(T(nPen));
      Eigen::Matrix<T,Eigen::Dynamic,1> Gam(nPen); Gam.setZero();
      Eigen::TensorMap<Eigen::Tensor<T,3,Eigen::RowMajor>> m(par, ny, nx, npar);
      T const normAzi = 3.1415926 / Pinfo[2].scale;

      // --- Scaling factors are sqrt-ed so when squared we get the right number --- //
      
      T* const  __restrict__ sq_alpha = new T [npar]();
      for(int ii=0; ii<npar; ++ii) sq_alpha[ii] = sqrt(Pinfo[ii].alpha) / sqr_nPen;
      
      
      int ipix=0, tid=0, xx=0, yy=0, pp=0;
#pragma omp parallel default(shared) firstprivate(ipix, tid, xx, yy, pp) num_threads(nthreads)      
      {
	tid = omp_get_thread_num();
#pragma omp for
for(ipix=1; ipix<npix; ++ipix){
	  
	  yy = ipix / nx;
	  xx = ipix - yy*nx;

	  if((yy-1) >= 0){
	    for(pp=0; pp<npar; ++pp)
	      Gam[ipix*npar*2 + pp*2] += sq_alpha[pp] * (m(yy,xx,pp) - m(yy-1,xx,pp));

	    // --- check azimuth --- //
	    pp = 2;
	    T const azi = (m(yy,xx,pp) - m(yy-1,xx,pp));
	    if     (fabs(azi-normAzi) < fabs(azi)) Gam[ipix*npar*2 + pp*2] =  sq_alpha[pp]*(azi-normAzi);
	    else if(fabs(azi+normAzi) < fabs(azi)) Gam[ipix*npar*2 + pp*2] =  sq_alpha[pp]*(azi+normAzi);
	  }
	  if((xx-1) >= 0){
	    for(pp=0; pp<npar; ++pp)
	      Gam[ipix*npar*2 + pp*2 + 1] += sq_alpha[pp] * (m(yy,xx,pp) - m(yy,xx-1,pp));
	    
	    // --- check azimuth --- //
	    pp = 2;
	    T const azi = (m(yy,xx,pp) - m(yy,xx-1,pp));
	    if     (fabs(azi-normAzi) < fabs(azi)) Gam[ipix*npar*2 + pp*2 + 1] =  sq_alpha[pp]*(azi-normAzi);
	    else if(fabs(azi+normAzi) < fabs(azi)) Gam[ipix*npar*2 + pp*2 + 1] =  sq_alpha[pp]*(azi+normAzi);
	  }

	}// ipix
      }// parallel

      delete [] sq_alpha;
      
      return Gam;
    }

    
    // ------------------------------------------------------------ //
    
    void synthesize_der_one(int const npar, T* __restrict__ par, T* __restrict__ r,
			    T* __restrict__ J, int const tid, int const ipix)const
    {

      T const scl = sqrt(T(Nreal)*T(nx*ny));
      T iScl = 0;
      int const ndat = nDat;

      const T* const __restrict__ o = static_cast<const T* const>(&obs(0,0,0));
      
      for(int pp=0; pp<npar; ++pp){
	Pinfo[pp].Scale(par[pp]);
	//Pinfo[pp].Check(par[pp]);
      }
	    
      Me[tid]->synthesize_rf(par, r, J, mu);
      
      
      for(int ww=0; ww<ndat; ++ww)
	r[ww] = (o[ipix*ndat+ww] - r[ww]) * sig[ww] / scl;  
      
	  
      // --- scale J and compute r--- //
      for(int pp=0; pp<npar; ++pp){
	iScl = Pinfo[pp].scale / scl;
	
	for(int ww=0; ww<ndat; ++ww)
	  J[pp*ndat+ww] *= iScl * sig[ww];
	
	Pinfo[pp].Normalize(par[pp]);

      } // pp
    
    }
    
    // ------------------------------------------------------------ //

    T fx_dx(int const nPar,  T* const __restrict__ par,  T* const __restrict__ syn_in,  T* const __restrict__ r,
	    T* const __restrict__ J)const 
    {
      
      synthesize_rf(nPar, par, syn_in, r, J);
      
      long const nEl = long(nDat)*long(nx*ny);
      return ksum2<T,double>(nEl, r);
    }
    
    // ------------------------------------------------------------ //

    template<typename iType = long>
    Eigen::SparseMatrix<T, Eigen::RowMajor, iType> get_L(int const npar,  T* const __restrict__ par)const
    {

      iType const npix = nx*ny;
      iType const ndat = nDat;
      iType const Nx = nx;
      iType const Ny = ny;
      iType const nthreads = Me.size();


      iType const nPen = npix*2*npar;
      T const sqr_nPen = sqrt(T(nPen));
      std::vector<T> iAlpha(npar, T(0));
      for(int ii=0;ii<npar; ++ii) iAlpha[ii] = sqrt(Pinfo[ii].alpha) / sqr_nPen;
      
      // --- get matrix dimensions --- //

      int const nrows = 2*npar*npix;
      int const ncols = npix*npar;

      Eigen::SparseMatrix<T,Eigen::RowMajor, iType> L(nrows, ncols);

      
      // --- Get number of elements per row --- //

      int const Elements_per_row = 2;
      Eigen::VectorXi nElements_per_row = Eigen::VectorXi::Constant(2*npix*npar, Elements_per_row); // 1D vector of integers



      // --- correct numbers for first column and first row --- //
      
      for(int pp=0; pp<npar; ++pp){
	
	for(int yy = 0; yy<Ny; ++yy){
	  int const iPix = (yy*nx + 0);
	  nElements_per_row[2*iPix * npar + 2*pp+1] = 0;
	} // yy
	
	for(int xx = 0; xx<Nx; ++xx){
	  int const iPix = (0*Nx + xx);
	  nElements_per_row[2*iPix * npar + 2*pp ] = 0;
	} // xx

      }// pp
      


      
      // --- reserve elements in the sparse matrix --- //
      
      L.reserve(nElements_per_row);



      
      // --- Fill Matrix in parallel --- //
      
      iType ipix=0, tid=0, xx=0, yy=0, pp=0;
#pragma omp parallel default(shared) firstprivate(ipix, tid, xx, yy, pp) num_threads(nthreads)      
      {
	tid = omp_get_thread_num();
#pragma omp for
	for(ipix=1; ipix<npix; ++ipix){
	  yy = ipix / nx;
	  xx = ipix - yy*nx;
	  
	  // --- Each thread fills all regularization derivatives for one pixel (all parameters) --- //
	  
	  if((yy-1) >= 0)
 	    for(pp = 0; pp<npar; ++pp){
	      L.insert(2*npar*ipix + 2*pp    , ipix*npar + pp - npar*nx) = -iAlpha[pp]; // One pixel below
	      L.insert(2*npar*ipix + 2*pp    , ipix*npar + pp)           =  iAlpha[pp]; // the pixel itself.	      
	    }
	  
	  if((xx-1) >= 0)
	    for(pp = 0; pp<npar; ++pp){
	      L.insert(2*npar*ipix + 2*pp+1  , ipix*npar + pp - npar)    = -iAlpha[pp]; // One pixel to the left
	      L.insert(2*npar*ipix + 2*pp+1  , ipix*npar + pp)           =  iAlpha[pp]; // the pixel itself.
	    }
	} //ipix	
      }// parallel

      return L;
    }

  };
}

#endif
