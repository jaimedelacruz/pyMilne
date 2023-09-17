#ifndef SPATTOHPP
#define SPATTOHPP

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <array>
#include <omp.h>

#include "spatially_coupled_helper.hpp"

namespace spa{
  
  // ******************************************************************************** //

  template<typename T, typename U, typename ind_t = long>
  void count_Hessian(ind_t const ny, ind_t const nx, ind_t const npar, spa::Data<T,U,ind_t> const& dat,
		     Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> &A,  int const nthreads)
  {
    
    // --- let's calculate in parallel the number of elements in the Hessian --- //

    Eigen::VectorXi n_elements(ny*nx*npar); n_elements.setZero();

    
    // --- Loop over pixels and count elements --- //

    ind_t const npix     = ny*nx;
    ind_t const nregions = dat.regions.size();

#pragma omp parallel default(shared) num_threads(nthreads)
    {
#pragma omp for
      for(ind_t ipix = 0; ipix<npix; ++ipix){
	
	int xx0 = nx;
	int xx1 = 0;
	int yy0 = ny;
	int yy1 = 0;
	
	for(ind_t ireg=0; ireg<nregions; ++ireg){
	  
	  //spa::SpatRegion<T,ind_t> const& ir = *.get();
	  std::array<int,4> const& ma = dat.regions[ireg]->matrix_elements[ipix];

	  yy0 = std::min<int>(ma[0], yy0);
	  yy1 = std::max<int>(ma[1], yy1);
	  xx0 = std::min<int>(ma[2], xx0);
	  xx1 = std::max<int>(ma[3], xx1);

	  
	} // ireg
	
	int const nel = (yy1-yy0+1)*(xx1-xx0+1)*npar;
	
	for(ind_t ii=0; ii<npar; ++ii)
	  n_elements[ipix*npar + ii] = nel;
	
      } // ipix
    } // parallel
    
    // --- now make allocation --- //

    A.reserve(n_elements);

    fprintf(stderr,  "populating ... ");

    // --- now actually insert the elements in the matrix explicitly --- //

#pragma omp parallel default(shared) num_threads(nthreads)
    {
#pragma omp for
      for(ind_t ipix = 0; ipix<npix; ++ipix){
	
	int xx0 = nx;
	int xx1 = 0;
	int yy0 = ny;
	int yy1 = 0;
	
	for(ind_t ireg=0; ireg<nregions; ++ireg){
	  
	  //spa::SpatRegion<T,ind_t> const& ir = *dat.regions[ireg].get();
	  //std::array<int,4> const& ma = ir.matrix_elements[ipix];
	  std::array<int,4> const& ma = dat.regions[ireg]->matrix_elements[ipix];

	  yy0 = std::min<int>(ma[0], yy0);
	  yy1 = std::max<int>(ma[1], yy1);
	  xx0 = std::min<int>(ma[2], xx0);
	  xx1 = std::max<int>(ma[3], xx1);

	  
	} // ireg
	
	
	// --- now insert that patch in the Hessian for all parameters --- //

	ind_t const y0 = yy0;
	ind_t const y1 = yy1;
	ind_t const x0 = xx0;
	ind_t const x1 = xx1;
	ind_t const poff = ipix*npar;
	
	for(ind_t jj = y0; jj<=y1; ++jj){
	  ind_t const joff = jj*nx*npar;
	  
	  for(ind_t ii = x0; ii<= x1; ++ii){
	    ind_t const ioff = joff + ii*npar;
	    
	    for(ind_t pp0=0; pp0<npar; ++pp0)
	      for(ind_t pp1=0; pp1<npar; ++pp1)
		A.insert(poff + pp0, ioff + pp1) = T(0);
	      
	  } // ii
	} // jj
      } // ipix
    } // parallel
  }
  
  // ******************************************************************************** //

  template<typename T, typename U, typename ind_t = long>
  void fillResiduePixel(mem::Array<T,5> const& J, mem::Array<T,3> const& r,
			U* const __restrict__ B, ind_t const ipix,
			ind_t const ny, ind_t const nx, ind_t const npar, ind_t const w0,
			ind_t const w1, ind_t const nw1, spa::SpatRegion<T,ind_t> const& reg)
  {
    /* ---
       Calculates the RHS of the LM system for a given pixel by calculating J.t * r, 
       where J.t is the transpose of the spatially coupled Jacobian and r is the residue vector.
       
       Coded by J. de la Cruz Rodriguez (ISP-SU, 2019)
       --- */
    
    ind_t const nw = w1-w0+1;
    ind_t const ns = reg.obs.shape(2);


    // --- Coordinates in the finest grid --- //
    
    ind_t const yy = ipix / nx;
    ind_t const xx = ipix - yy*nx;

    
    // --- Iterate over non-zero elements of the operator in the degraded grid --- //
    
    for(typename Eigen::SparseMatrix<T,Eigen::RowMajor,ind_t>::InnerIterator it(reg.OpT,ipix); it; ++it){

      
      // --- Coordinates in the degraded grid --- //
      
      ind_t const idx1 = it.index();
      ind_t const jj = idx1 / reg.nx;
      ind_t const ii = idx1 - jj*reg.nx;

      T const iDegradation = it.value();

      for(ind_t pp=0; pp<npar; ++pp){
	
	T sum = T(0);
	
	for(ind_t ss=ns-1; ss>=0; --ss){
	  
	  const T* const __restrict__ iJ = &J(yy,xx,pp,ss,w0);
	  const T* const __restrict__ iR = &r(jj,ii,ss*nw);

	  for(ind_t dd=0; dd<nw; ++dd)
	    sum += iJ[dd] * iR[dd];
	} // ss
	
	B[pp] += iDegradation * sum;
	
      } // pp
    } // it
  }
  
  // ******************************************************************************** //

  template<typename T, typename ind_t = long>
  void degradeOnePixel(ind_t const ny1, ind_t const nx1, ind_t const ns1, ind_t const nw1,
		       ind_t const ny, ind_t const nx, ind_t const ipix, ind_t const wl, ind_t const wh,
		       Eigen::SparseVector<T,Eigen::RowMajor> const& Op,
		       mem::Array<T,4> const& syn, T* const __restrict__ result)
  {
    /* ---
       This function applies the degradation operator to compute the observed intensity in one 
       pixel at all frequencies of a given spectral window. By making use of sparse matrix, 
       we only cycle through the non-zero elements of the operator. 

       Coded by J. de la Cruz Rodriguez (ISP-SU, 2019 and 2023)
       
       --- */

    
    // --- Apply one row of the degradation operator --- //
    
    for (typename Eigen::SparseVector<T, Eigen::RowMajor>::InnerIterator it(Op); it; ++it){
      
      // --- Get coordinates in the fine (model) grid --- //

      ind_t const idx = it.index();
      T const iDegradation = it.value();

      // --- Coordinates in the undegraded grid --- //
      
      ind_t const jj = idx / nx;
      ind_t const ii = idx - jj*nx;

      // --- The degradation is the same for all wavelengths in this spectral window --- //

      for(ind_t ss = 0; ss<ns1; ++ss){
	
	const T* const __restrict__ s = &syn(jj,ii,ss,wl);
	T* const __restrict__ iResult = &result[ss*nw1];

	for(ind_t ww=0; ww<nw1;++ww)
	  iResult[ww] += iDegradation * s[ww];
	
      } // ss
    }// it
  }

  // ******************************************************************************** //

  template<typename T, typename ind_t = long>
  void degradeRegion(spa::SpatRegion<T,ind_t> &reg, mem::Array<T,4> const& syn, int const nthreads)
  {
    /* --- 
       This function splits the work of applying the degradation operator over many threads.
       Each thread can process all frequencies in a spectral window for a given pixel.

       Coded by J. de la Cruz Rodriguez (ISP-SU, 2019)
       --- */
    
    ind_t const ny1 = reg.ny;
    ind_t const nx1 = reg.nx;
    ind_t const nw1 = reg.obs.shape(3);
    ind_t const ns1 = reg.obs.shape(2);
    ind_t const npix = ny1*nx1;
    
    ind_t const ny = syn.shape(0);
    ind_t const nx = syn.shape(1);

    ind_t const w0 = reg.wl;
    ind_t const w1 = reg.wh;
    

    // --- Visualize the synthetic cube as a 2D one with dimensions (npix, nd) --- //
    
    mem::Array<T,2> iSyn(&reg.syn(0,0,0,0), npix, nw1*ns1);
    iSyn.Zero();

    // --- compute in parallel --- //
    
#pragma omp parallel default(shared)  num_threads(nthreads)
    {
#pragma omp for schedule(static)
      for(ind_t ipix=0; ipix<npix; ++ipix){
	degradeOnePixel<T,ind_t>(ny1,nx1,ns1,nw1,ny,nx,ipix,w0,w1,reg.Op.row(ipix), syn, &iSyn(ipix,0));
      } // ipix
    }// parallel block
  }

  // ******************************************************************************** //
 
  template<typename T, typename ind_t = long>
  void getResidue(spa::SpatRegion<T, ind_t>  &reg, mem::Array<T,4> const& syn, int const nthreads)
  {
    /* --- 
       Computes the residue in the grid of the observations for a given region.
       First it applies the degradation operator in "degradeRegion" and then it 
       calculates the traditional residue as r_i = (o_i - s_i) / sig_i

       Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
       --- */

    ind_t const ny1 = reg.obs.shape(0);
    ind_t const nx1 = reg.obs.shape(1);
    ind_t const ns = reg.obs.shape(2);
    ind_t const nw = reg.obs.shape(3);
    ind_t const nd = nw*ns;

    
    // --- degrade synthetic spectra to the observed grid in the region --- //

    degradeRegion<T,ind_t>(reg, syn, nthreads);
    
    
    // --- Now compute the residue, this part is fast and vectorized, so no multithreading needed --- //
    
    reg.r.Zero();

    const T* const __restrict__ sig = &reg.sig(0,0);

    
    for(ind_t yy=0; yy<ny1; ++yy){
      for(ind_t xx=0; xx<nx1; ++xx){

	// --- use raw pointers in the inner loop for better performance --- //

	T const pweight = reg.pixel_weight(yy,xx);
	T* const __restrict__ r       = &reg.r(yy,xx,0);
	const T* const __restrict__ o = &reg.obs(yy,xx,0,0);
	const T* const __restrict__ s = &reg.syn(yy,xx,0,0);

	for(ind_t dd=0; dd<nd; ++dd){
	  r[dd] = (o[dd] - s[dd]) * sig[dd]*pweight;
	} //dd
	
      } // xx
    } // yy
  }
  
  // ******************************************************************************** //

  template<typename T, typename U, typename ind_t = long>
  void addCoupledResidue(spa::Data<T,U,ind_t> &dat, mem::Array<T,5> const& J,
			 Eigen::Matrix<U,Eigen::Dynamic,1> &B, mem::Array<T,4> const& syn,
			 int const nthreads)
  {
    /* --- 
       This function calculates the spatially coupled RHS of the LM system.
       First is degrades the synthetic spectra using the deagradation operator
       of each spectral region (in "getResidue") and then it fills the RHS
       which is stores in B.

       Coded by J. de la Cruz Rodriguez (ISP-SU, 2019 and 2023)
       --- */
    
    std::vector<std::shared_ptr<spa::SpatRegion<T,ind_t>>> &regions = dat.regions;
    
    int const nRegions = regions.size();
    ind_t const ny = dat.ny;
    ind_t const nx = dat.nx;
    ind_t const npar = dat.npar;
    ind_t const npix = nx*ny;

    
    // --- pre-compute the residue in the grid of each spectral window --- //

    for(ind_t ii=0; ii<nRegions; ++ii){
      getResidue<T,ind_t>(*regions[ii].get(), syn, nthreads);
    }


    // --- now add to the RHS of the LM system, need to compute OP.T * r for all regions --- //

#pragma omp parallel default(shared) num_threads(nthreads)
    {
      for(int rr = 0; rr<nRegions; ++rr){
	spa::SpatRegion<T,ind_t> &reg = *regions[rr].get();
	
	ind_t const w0 = reg.wl;
	ind_t const w1 = reg.wh;
	ind_t const nw1 = w1-w0+1;
	
#pragma omp for schedule(static)
	for(ind_t ipix=0; ipix<npix; ++ipix){
	    fillResiduePixel<T,U,ind_t>(J, reg.r, &B[ipix*npar],
					ipix, ny, nx, npar, w0,
					w1, nw1, reg);
	  
	}
	
      } // rr, all threads work on all regions
    } // parallel block
  }

  // ******************************************************************************** //

  template<typename T, typename U, typename ind_t = long>
  T getChi2obs(spa::Data<T,U,ind_t> &dat, mem::Array<T,4> const& syn, int const nthreads)
  {
    /* --- 
       Computes Chi2 by cycling through all spectral windows and 
       adding the square of the residue.
       
       Just in case, for large FOVs, it keeps the contribution from 
       each Stokes parameter separate and then adds them all at the end.


       Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
       --- */
    
    std::array<T,4> iChi2 = {T(0), T(0), T(0), T(0)};

    
    // --- cycle through regions and get Chi2 from the degraded residue --- //
    
    for(auto &it: dat.regions){
      ind_t const ny1 = it->ny;
      ind_t const nx1 = it->nx;
      ind_t const ns = it->obs.shape(2);
      ind_t const nw = it->obs.shape(3);
      
      // --- pre-compute the residue in the grid of each spectral window --- //

      spa::getResidue(*it.get(), syn, nthreads);
      

      
      // --- Since we are computing chi2 over the entire FOV, first compute the contribution
      // --- from each Stokes parameter separately and then add up all contributions

      for(ind_t yy=0; yy<ny1; ++yy)
	for(ind_t xx=0; xx<nx1; ++xx)
	  for(ind_t ss=ns-1; ss>=0; --ss){
	    
	    const T* const __restrict__ ir = &it->r(yy,xx,ss*nw);
	    
	    T sum = T(0);
	    
	    for(ind_t ww=0; ww<nw; ++ww)
	      sum += SQ(ir[ww]);
	    
	    iChi2[ss] += sum;
	  }
    } // region
    
    
    
    // --- add contributions from (Q+U) + V + I --- //
    
    T Chi2 = std::get<1>(iChi2) + std::get<2>(iChi2);
    Chi2 += std::get<3>(iChi2);
    Chi2 += std::get<0>(iChi2);

    return Chi2;
  }
  

  // ******************************************************************************** //

  template<typename T, typename ind_t = long>
  Eigen::VectorXi countRegionElements(spa::SpatRegion<T,ind_t> const& reg,
				      ind_t const ny, ind_t const nx, ind_t const npar)
  {
    /* --- 
       This function counts the elements present in the clipped auto-correlation function
       of the degradation operator.

       Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
    --- */
       
    ind_t const npix = nx*ny;
    Eigen::VectorXi n_elements(npix*npar);
    n_elements.setZero();
      
    for(ind_t ipix=0; ipix<npix; ++ipix){
      
      std::array<int,4> const& me = reg.matrix_elements[ipix];
      ind_t const n_elements_pix = (me[1] - me[0] + 1) * (me[3] - me[2] + 1) * npar;

      
      // --- repeat the same value for the entire block ot npar parameters --- //
      
      for(ind_t par=0; par<npar; ++par)
	n_elements[ipix*npar+par] = n_elements_pix;  
    }
    
    return n_elements;
  }
    
  
  // ******************************************************************************** //

  template<typename T, typename U, typename ind_t = long>
  void fillHessianOne(Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> &A, mem::Array<T,5> const& J,
		      ind_t const w0, ind_t const w1, ind_t const yy, ind_t const xx,
		      ind_t const ny, ind_t const nx, ind_t const npar, ind_t const j0, ind_t const j1,
		      ind_t const i0, ind_t const i1, Eigen::SparseMatrix<T,Eigen::RowMajor, ind_t> const& cc,
		      mem::Array<T,2> const& pweight, ind_t const ir)
  {
    /* --- 
       For a given pixel (xx,yy) this function fills the influence of the degradation operator
       for all parameters (a subspace of npar x npar). The auto-correlation Y of the degradation
       operator has been pre-computed, so there is no need to recompute it in every iteration.


       Coded by J. de la Cruz Rodriguez (ISP-SU, 2019 and 2023)
       --- */
    
    ind_t const ns = J.shape(3);
    ind_t const offy = npar*(yy*nx+xx);

    
    // --- loop through the vecinity of the pixel with coordinates (yy,xx) --- //

    for(ind_t dy=j0; dy<=j1; ++dy){
      for(ind_t dx=i0; dx<=i1; ++dx){
	
	
	// --- cross-correlation coeff, reuse it in the entire sub-block of npar x npar --- //
	
	T const iY = cc.coeff(yy*nx+xx, dy*nx+dx); 
	ind_t const offx = npar*(dy*nx+dx);
	
	
	// --- Calculate the product of Jacobians --- //
	
	for(ind_t pp0=0; pp0<npar; ++pp0){
	  for(ind_t pp1=0; pp1<npar; ++pp1){
	    
	    T sum = T(0);
	    
	    for(ind_t ss=ns-1; ss>=0; --ss){
	      
	      // --- Use raw pointers to ensure vectorization of the inner loop --- //
	      
	      const T* const __restrict__ jJ = &J(yy,xx,pp0,ss,0);
	      const T* const __restrict__ iJ = &J(dy,dx,pp1,ss,0);
	      
	      for(ind_t dd=w0; dd<=w1; ++dd)
		sum += iJ[dd]*jJ[dd];
	    }
	    
	    // --- insert in the sparse matrix --- //
	    
	    A.coeffRef(offy+pp0,offx+pp1) += sum*iY;
	    
	  }// pp1
	} // pp0
	
      }// dx
    } //dy
  }
  
  // ******************************************************************************** //

  template<typename T, typename U, typename ind_t = long>
  void fillHessianTerms(spa::Data<T,U,ind_t> const& dat,
			Eigen::SparseMatrix<U,Eigen::RowMajor, ind_t> &A, mem::Array<T,5> const& J,
			ind_t const nthreads)
  {
    /* ---
       Wrapper function that distributes the task of filling the Hessian matrix
       terms among nthreads.

       Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
       --- */
    ind_t const ny = dat.ny;
    ind_t const nx = dat.nx;
    ind_t const npix = ny*nx;
    ind_t const npar = dat.npar;
    ind_t const nRegions = dat.regions.size();

    for(ind_t ir=0; ir<nRegions;++ir){

      spa::SpatRegion<T,ind_t> const& iR = *dat.regions[ir].get();
      Eigen::VectorXi elements_per_row = spa::countRegionElements<T,ind_t>(iR, ny, nx, npar);

#pragma omp parallel default(shared) num_threads(nthreads)
      {
#pragma omp for
	for(ind_t ipix = 0; ipix<npix; ++ipix){
	  
	  ind_t const yy = ipix/nx;
	  ind_t const xx = ipix - yy*nx;
	  
	  std::array<int,4> const& me = iR.matrix_elements[ipix];
	  
	  fillHessianOne<T,U,ind_t>(A, J, iR.wl, iR.wh, yy, xx, ny, nx, npar,
				    std::get<0>(me), std::get<1>(me), std::get<2>(me),
				    std::get<3>(me), iR.cc, iR.pixel_weight, ir);
	  
	} // ipix
      } // parallel block
      
      
	
    }// ir
    

  }
  
  // ******************************************************************************** //

}


#endif

