#include <omp.h>
#include <cmath>

#include "spatially_coupled_helper.hpp"


// ******************************************************************************** //

template<typename T> inline
std::array<int,4> clipY(mem::Array<T,2> const& Y, T const threshold_in)
{
  /* ---
     This function calculates the innermost part of the autocorrelation
     function where a given % of the power is contained. We can use 
     this trick to make the Hessian matrix more sparse and faster to 
     solve / invert
     
     Coded by J. de la Cruz Rodriguez (ISP-SU, 2019)
     
     --- */
  
  T const threshold = threshold_in * Y.sum();
  
  // --- should be square IIRC, but anyhow ... --- //
  
  int const pny = Y.shape(0);
  int const pnx = Y.shape(1);
  
  int const y0 = Y.offset(0);
  int const y1 = y0 + pny-1;
  
  int const x0 = Y.offset(1);
  int const x1 = x0 + pnx-1;
  
  // --- locate coordinates of the maximum element -- //
  
  int xref = x0;
  int yref = y0;
  
  T maxval = Y(y0,x0);

  
  for(int yy=y0; yy<=y1;++yy){
    for(int xx=x0; xx<=x1; ++xx){
      T const iY = Y(yy,xx);
      if(iY > maxval){
	maxval = iY;
	xref = xx;
	yref = yy;
      }
    }
  }
  
  // --- now integrate outwards from that point --- //

  
  std::array<int,4> res = {y0, y1, x0, x1};


  int const rmax = std::max(std::max(yref-y0, y1-yref), std::max(xref-x0, x1-xref));


  for(int rr=0; rr<rmax; ++rr){
    
    int const j0 = std::max(yref - rr, y0);
    int const j1 = std::min(yref + rr, y1);
    int const i0 = std::max(xref - rr, x0);
    int const i1 = std::min(xref + rr, x1);

    T sum = T(0);


    for(int jj=j0; jj<=j1; ++jj)
      for(int ii=i0; ii<=i1; ++ii)
	sum += Y(jj,ii);
    
    if(sum >= threshold){
      res[0] = j0, res[1] = j1, res[2] = i0, res[3] = i1;
      return res;
    }
  }

  
  return res;
}

// ******************************************************************************** //

template<typename T, typename ind_t = long>
mem::Array<T,2> operator2autoCorr2(Eigen::SparseMatrix<T,Eigen::RowMajor, ind_t> const& cc,
				   ind_t const yy, ind_t const xx, ind_t const ny, ind_t const nx)
{
  /* ---
     This function estimates the spread of the coupling terms in the Hessian matrix.
     It only does it for a pixel in the center of the FOV. The ouput of this function
     can be used to trim the spatial coupling of the Hessian matrix, but removing 
     the very smallest terms. This approach was suggested by M. van Noort (2012) 
     and we keep it optional
     
     Coded by J. de la Cruz Rodriguez (ISP-SU, 2019 and 2023)
     --- */
  
  
  // --- Matrix vector multiplication, the vector is the row corresponding to our pixel --- //
  
  Eigen::SparseVector<T> const& res = cc.row(yy*nx+xx);
  
  ind_t x0=nx;
  ind_t x1=0;
  ind_t y0=ny;
  ind_t y1=0;
  
  // --- Do a pass through the result to get dimensions --- //
  
  for (typename Eigen::SparseVector<T>::InnerIterator it(res); it; ++it){
    
    ind_t const idx = it.index();
    ind_t const yy = idx / nx;
    ind_t const xx = idx - yy*nx;
    
    x0 = std::min(xx, x0);
    x1 = std::max(xx, x1);
    y0 = std::min(yy, y0);
    y1 = std::max(yy, y1);
  }
  
  
  // --- second pass to actually copy it into output array --- //
  
  mem::Array<T,2> aut(y0,y1,x0,x1); aut.Zero();
  
  for (typename Eigen::SparseVector<T>::InnerIterator it(res); it; ++it){
    
    ind_t const idx = it.index();
    ind_t const yy = idx / nx;
    ind_t const xx = idx - yy*nx;
    
    aut(yy,xx) = it.value();
  }
  
  return aut;
}

// ******************************************************************************** //

template<typename T, typename ind_t>
void spa::SpatRegion<T,ind_t>::prepareHessianElements(ind_t const ny, ind_t const nx, int const nthreads)
{
  /* --- 
     Pre-calculate the number of elements that we need to allocate in the
     sparse Hessian matrix per region. In principle, if we want to keep 100%
     of the power of the degradation operator in the cross-correlation, we 
     should just look at the number of non-zero elements in reg.rr per row.
     But in reality we can define a threshold to cut the extent of the operator
     in the LHS (van Noort 2012).
     
     Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
     
     --- */
  ind_t const npix = ny*nx;
  
  
#pragma omp parallel default(shared)  num_threads(nthreads)
  {
#pragma omp for
    for(ind_t ipix = 0; ipix<npix; ++ipix){
      ind_t const yy = ipix/nx;
      ind_t const xx = ipix - yy*nx;
      
      
      this->matrix_elements[ipix] = clipY(operator2autoCorr2(this->cc, yy, xx, ny, nx), this->clip_threshold);
    }
  } // parallel 
  
}

template void spa::SpatRegion<double,long>::prepareHessianElements(long const ny, long const nx, int const nthreads);

// ******************************************************************************** //

template<typename T, typename ind_t>
spa::SpatRegion<T,ind_t>::SpatRegion(ind_t const iny, ind_t const inx,
			    ind_t const iny1, ind_t const inx1, ind_t const iwl, ind_t const iwh,
			    ind_t const iy0, ind_t const iy1, ind_t const ix0, ind_t const ix1,
			    T const clip_thres, const T* const iwav, const T* const isig,
			    ind_t const pny, ind_t const pnx, const T* const iPsf, T* const iObs,
			    int const nthreads):
  ny(iny1), nx(inx1), wl(iwl), wh(iwh), y0(iy0), y1(iy1), x0(ix0), x1(ix1),
  clip_threshold(clip_thres), wav(wh-wl+1), obs(iObs,iny1,inx1,4,wh-wl+1),  syn(iny1,inx1,4,wh-wl+1),
  r(iny1, inx1, 4*(wh-wl+1)), sig(4,wh-wl+1), psf(pny,pnx), pixel_weight(iny1,inx1)
{
  /* ---
     Initialize Sparse region. In order to accelerate the calculations within eath iteration,
     we store the total operator, its transpose (used to compute the RHS) and the cross-correlation
     matrix. All these are much smaller than the Hessian matrix in memory, so they don't add
     significant memory consumption.


     Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
     
     --- */
  
  using namespace Eigen;
  
  if(nthreads > 1){
    Eigen::initParallel();
    Eigen::setNbThreads(nthreads);
  }
  
  ind_t const nw = wh-wl+1;
  ind_t const ndat = nw*4;
  

  // --- init pixel weight --- //

  T fmean = T(0);
  for(ind_t yy = 0; yy<ny; ++yy){
    for(ind_t xx = 0; xx<nx; ++xx){
      T sum = T(0);
      const T* const __restrict__ iDat = &obs(yy,xx,0,0);

      for(ind_t dd=0;dd<ndat;++dd)
	sum += iDat[dd];

      pixel_weight(yy,xx) = sum/ndat; // sqrt of the mean
      fmean += pixel_weight(yy,xx);
    }
  }
  
  fmean /= ny*nx;
  pixel_weight = sqrt(fmean / pixel_weight);

  
  // --- Copy the sigma array --- //
  
  for(ind_t ii=0; ii<ndat; ++ii)
    sig[ii] = isig[ii];
  
  
  
  // --- Copy the PSF --- //
  
  ind_t const npsf = pny*pnx;
  for(ind_t ii=0; ii<npsf; ++ii)
    psf[ii] = iPsf[ii];
  
  
  fprintf(stderr,"[info] SparseRegion: Initializing degradation operator and cross-correlation matrix ... ");
  
  // --- Init degradation operators --- //
  
  
  if((iny == iny1) && (inx == inx1))
    Op = spa::psf_to_operator(psf, iny, inx);
  else
    Op = spa::downscale_to_operator<T,long>(iny, inx, iny1, inx1, true) *
      spa::psf_to_operator(psf, iny, inx);
  

  {
    Eigen::SparseMatrix<T,Eigen::RowMajor,ind_t> tmp = Op;
    
    ind_t const npix = iny1*inx1;// number of pixels in spatial region
    
    for(ind_t ii=0; ii<npix; ++ii)
      tmp.row(ii) *= pixel_weight[ii]; // add the pixel weight to the Hessian terms
    
  
  
    // --- Calculate cross-correlation of the total degradation operator,
    //     we can get it by computing cc = Op.T * Op, but it is faster to express
    //     Op = Op.T as the columns of Op would be consecutive in memory --- //

    OpT = tmp.transpose(); //  includes the pixel weight
    cc = OpT*tmp;
  }
  
  
  // --- Init matrix_elements --- //
  
  std::array<int,4> ex={int(iny)-1, 0, int(inx)-1,0};
  
  matrix_elements.resize(iny*inx);
  for(auto &it: matrix_elements)
    it = ex;
  
  
  
  // --- count Hessian elements --- //
  
  this->prepareHessianElements(iny, inx, nthreads);

  fprintf(stderr,"done\n");

  
}

template spa::SpatRegion<double,long>::SpatRegion(long const iny, long const inx,
						  long const iny1, long const inx1, long const iwl, long const iwh,
						  long const iy0, long const iy1, long const ix0, long const ix1,
						  double const clip_thres, const double* const iwav, const double* const isig,
						  long const pny, long const pnx, const double* const iPsf, double* const iObs,
						  int const nthreads);

// ******************************************************************************** //

template<typename T, typename U, typename ind_t>
void spa::Data<T,U,ind_t>::synthesize_rf(mem::Array<T,3> &m, mem::Array<T,4> &syn, mem::Array<T,5> &J)const
{ 
  ind_t const nthreads = me.size();
  ind_t const ny1 = m.shape(0);
  ind_t const nx1 = m.shape(1);
  ind_t const npar1 = m.shape(2);
  ind_t const npix = ny1*nx1;
  ind_t const ndat = nw*ns;
  ind_t const njac = ndat*npar;
  
  T* const __restrict__ model = &m(0,0,0);
  T* const __restrict__ synthetic = &syn(0,0,0,0);
  T* const __restrict__ Jac = &J(0,0,0,0,0);


  // --- Scale pars --- //
  
  ScalePars(m);

  
#pragma omp parallel default(shared) num_threads(nthreads)
  {
    ind_t const tid = omp_get_thread_num();
#pragma omp for schedule(dynamic,2)
    for(ind_t ipix = 0; ipix<npix; ++ipix){

      
      T* const __restrict__ iJ = &Jac[ipix*njac];
      

      // --- Synthesize spectrum + RF --- //
      
      me[tid]->synthesize_rf(&model[ipix*npar1], &synthetic[ndat*ipix], iJ, mu);
      
      
      
      // --- scale J --- //
      
      for(ind_t pp = 0; pp<npar1; ++pp){
	T const iscl = Pinfo[pp].scale;
	
	for(ind_t dd=0; dd<ndat;++dd)
	  iJ[pp*ndat+dd] *= iscl*sig_total[dd]; // Scale Jac
      } // pp
    } // ipix
  } // parallel block


  // --- Normalize pars --- //

  NormalizePars(m);
  
}

template void spa::Data<double,double,long>::synthesize_rf(mem::Array<double,3> &m, mem::Array<double,4> &syn, mem::Array<double,5> &J)const;

// ******************************************************************************** //

template<typename T, typename U, typename ind_t>
void spa::Data<T,U,ind_t>::synthesize(mem::Array<T,3> &m, mem::Array<T,4> &syn)const
{
  
  
  ind_t const nthreads = me.size();
  ind_t const ny1 = m.shape(0);
  ind_t const nx1 = m.shape(1);
  ind_t const npar1 = m.shape(2);
  ind_t const npix = ny1*nx1;
  ind_t const ndat = nw*ns;
  
  T* const __restrict__ model = &m(0,0,0);
  T* const __restrict__ synthetic = &syn(0,0,0,0);


  // --- scale parameters --- //

  ScalePars(m);
  
  
#pragma omp parallel default(shared) num_threads(nthreads)
  {
    ind_t const tid = omp_get_thread_num();
#pragma omp for schedule(dynamic,2)
    for(ind_t ipix = 0; ipix<npix; ++ipix){
      
      
      // --- Synthesize spectrum --- //
      
      me[tid]->synthesize(&model[ipix*npar1], &synthetic[ndat*ipix], mu);
      

    } // ipix
  } // parallel block


  // --- Normalize parameters --- //

  NormalizePars(m);
  
}


template void spa::Data<double,double,long>::synthesize(mem::Array<double,3> &m, mem::Array<double,4> &syn)const;

// ******************************************************************************** //

template<typename T, typename U, typename ind_t>
void spa::Data<T,U,ind_t>::NormalizePars(mem::Array<T,3> &m)const{

  ind_t const ny1  = m.shape(0);
  ind_t const nx1  = m.shape(1);
  ind_t const nPar = m.shape(2);
  
  
  for(ind_t yy=0; yy<ny1; ++yy)
    for(ind_t xx=0; xx<nx1; ++xx)
      for(ind_t pp=0; pp<nPar; ++pp)
	m(yy,xx,pp) /= Pinfo[pp].scale;
}

template void spa::Data<double,double,long>::NormalizePars(mem::Array<double,3> &m)const;


// ******************************************************************************** //

template<typename T, typename U, typename ind_t>
void spa::Data<T,U,ind_t>::ScalePars(mem::Array<T,3> &m)const{
	  
  ind_t const ny1  = m.shape(0);
  ind_t const nx1  = m.shape(1);
  ind_t const nPar = m.shape(2);
  
  
  for(ind_t yy=0; yy<ny1; ++yy)
    for(ind_t xx=0; xx<nx1; ++xx)
      for(ind_t pp=0; pp<nPar; ++pp)
	m(yy,xx,pp) *= Pinfo[pp].scale;
}

template void spa::Data<double,double,long>::ScalePars(mem::Array<double,3> &m)const;

// ******************************************************************************** //

template<typename T, typename U, typename ind_t>
void spa::Data<T,U,ind_t>::CheckPars(mem::Array<T,3> &m)const{
      
  ind_t const ny1  = m.shape(0);
  ind_t const nx1  = m.shape(1);
  ind_t const nPar = m.shape(2);
  
  for(ind_t pp=0; pp<nPar; ++pp){
    
    if(!Pinfo[pp].limited) continue;
    
    T const imin = Pinfo[pp].limits[0];
    T const imax = Pinfo[pp].limits[1];
    
    if(Pinfo[pp].isCyclic){
      
      for(ind_t yy=0; yy<ny1; ++yy)
	for(ind_t xx=0; xx<nx1; ++xx){
	  T &iPar = m(yy,xx,pp);
	  
	  if(iPar < imin) iPar += imax;
	  if(iPar > imax) iPar -= imax;
	  
	  // --- Just in case, force the parameter to be within limits --- //
	  
	  iPar = std::min<T>(std::max<T>(iPar, imin), imax);
	  
	}
      
    }else{ // not cyclic
      for(ind_t yy=0; yy<ny1; ++yy)
	for(ind_t xx=0; xx<nx1; ++xx){
	  T &iPar = m(yy,xx,pp);
	  iPar = std::min<T>(std::max<T>(iPar, imin), imax);
	} //xx
    }// isCyclic
    
  } // pp
}

template void spa::Data<double,double,long>::CheckPars(mem::Array<double,3> &m)const;

// ******************************************************************************** //

template<typename T, typename U, typename ind_t>
Eigen::SparseMatrix<U, Eigen::RowMajor, ind_t>
spa::Data<T,U,ind_t>::get_L(ind_t const npar)const
{
  constexpr static int const Elements_per_row = 2;
  
  // ---- Init dimensions --- //
  
  if(npar != ind_t(Pinfo.size())){
    fprintf(stderr,"[error] spa::Data::get_L: Parinfo array has different number of elements than the input model, fix your code!");
    exit(1);
  }
  
  ind_t const npix = ny*nx;
  ind_t const Ny = ny;
  ind_t const Nx = nx;
  
  ind_t const nPen = npix*Elements_per_row*npar;
  T const sqrt_nPen = sqrt(T(nPen/npar));
  std::vector<T> iAlpha(Pinfo.size(), T(0));

  
  // --- Init alpha array --- //
  
  for(ind_t ii = 0; ii<npar; ++ii)
    iAlpha[ii] = sqrt(Pinfo[ii].alpha) / sqrt_nPen;
  
  
  // --- get matrix dimensions --- //
  
  ind_t const nrows = Elements_per_row*npar*npix;
  ind_t const ncols = npix * npar;
  
  Eigen::SparseMatrix<T,Eigen::RowMajor, ind_t> L(nrows, ncols);
  
  
  
  // --- get number of elements per row --- //
  
  Eigen::VectorXi nElements_per_row = Eigen::VectorXi::Constant(nrows, Elements_per_row);
  
  
  // --- correct numbers for first column and first row --- //
  
  for(ind_t pp=0; pp<npar; ++pp){
    
    for(ind_t yy = 0; yy<Ny; ++yy){
      ind_t const iPix = (yy*nx + 0);
      nElements_per_row[2*iPix * npar + 2*pp+1] = 0;
    } // yy
    
    for(ind_t xx = 0; xx<Nx; ++xx){
      ind_t const iPix = (0*Nx + xx);
      nElements_per_row[2*iPix * npar + 2*pp ] = 0;
    } // xx
  }// pp
  
  
  // --- reserve elements in sparse matrix --- //
  
  L.reserve(nElements_per_row);
  
  
  
  // --- fill matrix with regularization terms --- //
  
  for(ind_t ipix =1; ipix<npix; ++ipix){
    
    ind_t const yy = ipix / nx;
    ind_t const xx = ipix - yy*nx;
    
    if((yy-1) >= 0)
      for(ind_t pp = 0; pp<npar; ++pp){
	L.insert(2*npar*ipix + 2*pp    , ipix*npar + pp - npar*nx) = -iAlpha[pp]; // One pixel below
	L.insert(2*npar*ipix + 2*pp    , ipix*npar + pp)           =  iAlpha[pp]; // the pixel itself.	      
      }
    
    if((xx-1) >= 0)
      for(ind_t pp = 0; pp<npar; ++pp){
	L.insert(2*npar*ipix + 2*pp+1  , ipix*npar + pp - npar)    = -iAlpha[pp]; // One pixel to the left
	L.insert(2*npar*ipix + 2*pp+1  , ipix*npar + pp)           =  iAlpha[pp]; // the pixel itself.
      }
    
  } // ipix
  
  return L;
}


template Eigen::SparseMatrix<double,Eigen::RowMajor,long> spa::Data<double,double,long>::get_L(long const npar)const;

// ******************************************************************************** //

template<typename T, typename U, typename ind_t>
Eigen::Matrix<U,Eigen::Dynamic,1> spa::Data<T,U,ind_t>::getGamma(mem::Array<T,3> const& m)const
{
  static constexpr const ind_t nC = 2;
  
  ind_t const nthreads = me.size();
  ind_t const nPar = m.shape(2);
  ind_t const Ny = m.shape(0);
  ind_t const Nx = m.shape(1);
  ind_t const npix = Ny*Nx;
  ind_t const nPen = nC*npix*npar;
  T const sqr_nPen = sqrt(T(nPen/npar));
  
  Eigen::Matrix<U,Eigen::Dynamic,1> Gam(nPen); Gam.setZero();
  
  T const normAzi = 3.14159265358979323846 / Pinfo[2].scale;

  
  // --- Scaling factors are sqrt-ed so when squared we get the right number --- //
  
  T* const  __restrict__ sq_alpha = new T [nPar]();
  for(ind_t ii=0; ii<nPar; ++ii)
    sq_alpha[ii] = sqrt(Pinfo[ii].alpha) / sqr_nPen;

#pragma omp parallel default(shared)  num_threads(nthreads)      
  {
	//tid = omp_get_thread_num();
#pragma omp for
    for(ind_t ipix=1; ipix<npix; ++ipix){
      
      ind_t const yy = ipix / nx;
      ind_t const xx = ipix - yy*nx;
      
      if((yy-1) >= 0){
	for(ind_t pp=0; pp<nPar; ++pp)
	  Gam[ipix*npar*nC + pp*nC] += sq_alpha[pp] * (m(yy,xx,pp) - m(yy-1,xx,pp));
	
	// --- check azimuth --- //
	ind_t pp = 2;
	T const azi = (m(yy,xx,pp) - m(yy-1,xx,pp));
	if     (fabs(azi-normAzi) < fabs(azi)) Gam[ipix*npar*nC + pp*nC] =  sq_alpha[pp]*(azi-normAzi);
	else if(fabs(azi+normAzi) < fabs(azi)) Gam[ipix*npar*nC + pp*nC] =  sq_alpha[pp]*(azi+normAzi);
      }
      if((xx-1) >= 0){
	for(ind_t pp=0; pp<nPar; ++pp)
	  Gam[ipix*npar*nC + pp*nC + 1] += sq_alpha[pp] * (m(yy,xx,pp) - m(yy,xx-1,pp));
	
	// --- check azimuth --- //
	ind_t pp = 2;
	T const azi = (m(yy,xx,pp) - m(yy,xx-1,pp));
	if     (fabs(azi-normAzi) < fabs(azi)) Gam[ipix*npar*nC + pp*nC + 1] =  sq_alpha[pp]*(azi-normAzi);
	else if(fabs(azi+normAzi) < fabs(azi)) Gam[ipix*npar*nC + pp*nC + 1] =  sq_alpha[pp]*(azi+normAzi);
      }
      
    }// ipix
  }// parallel

  
  delete [] sq_alpha;
  
  return Gam;
}

template Eigen::Matrix<double,Eigen::Dynamic,1> spa::Data<double,double,long>::getGamma(mem::Array<double,3> const& m)const;

// ******************************************************************************** //
