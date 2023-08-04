#ifndef LINOPHPP
#define LINOPHPP

/* ---
   Linear Operator construction. 
   These routines construct the spatial transformation operators using Eigen sparse matrices

   So far only implemented:
   1) a convolution with a PSF
   2) an integral downscale. Similar to a rebin but can handle fractional integration factors
   3) FOV cut. Trivially cuts the edges of a FOV.
   
   
   Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)

   --- */

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "Arrays.hpp"

namespace spa{
  
  // *************************************************************************************** //
  // --- Given a 2D spatial psf, construct the (sparse) convolution operator over a FOV of Ny x Nx pixels --- //
  
  template<typename T, typename ind_t = long>
  Eigen::SparseMatrix<T, Eigen::RowMajor, ind_t>
  psf_to_operator(mem::Array<T,2> psf, ind_t const ny, ind_t const nx)
  {
    using Tri = Eigen::Triplet<T,ind_t>;

    ind_t const npix = ny*nx;

    ind_t const pny = psf.shape(0);
    ind_t const pnx = psf.shape(1);
    ind_t const pny2 = pny / 2;
    ind_t const pnx2 = pnx / 2;
    ind_t const nOpPSF = pnx*pny*npix;

    
    // --- reverse PSF --- //

    {
      mem::Array<T,2> ipsf = psf;
      
      for(ind_t jj=0; jj<pny; ++jj)
	for(ind_t ii=0; ii<pnx; ++ii)
	  psf(jj,ii) = ipsf(pny-jj-1,pnx-ii-1);
    }
    
    // --- Center indexes of PSF array in the central element, but keep the number of elements! --- //
    
    psf.reshape(-pny2, pny2, -pnx2, pnx2);
    

    
    // --- Count PSF elements inside the FOV at all locations --- //

    std::vector<Tri> triplets;
    triplets.reserve(nOpPSF); // it will be smaller than this because of the truncation at the edges
    
    for(ind_t yy=0; yy<ny; ++yy){
      for(ind_t xx=0; xx<nx; ++xx){

	// --- PSF ranges at (yy,xx) --- //
	
	ind_t const j0 = std::max<ind_t>(0,yy-pny2);
	ind_t const j1 = std::min<ind_t>(ny-1,yy+pny2);
	ind_t const i0 = std::max<ind_t>(0,xx-pnx2);
	ind_t const i1 = std::min<ind_t>(nx-1,xx+pnx2);
	
	ind_t const pix = yy*nx + xx;

	// --- integrate psf to get boundary condition --- //

	T sum = T(0);
	for(ind_t jj=j0; jj<=j1; ++jj){
	  ind_t const iy = -(jj-yy);
	  for(ind_t ii=i0; ii<=i1; ++ii)
	    sum += psf(iy,-(ii-xx));
	}
 
	// --- Boundary condition --- //

	sum = (T(1) - psf(0,0)) / (sum - psf(0,0));


	// --- Create Operator row --- //

	for(ind_t jj=j0; jj<=j1; ++jj){

	  bool const isY0 = (((jj-yy)==0) ? true : false);
	  ind_t const iy = -(jj-yy);
	  
	  for(ind_t ii=i0; ii<=i1; ++ii){

	    ind_t const pix1 = jj*nx + ii;
	    T const scl = ((isY0 && ((ii-xx)==0)) ? T(1) : sum);
	    triplets.emplace_back(Tri(pix, pix1, psf(iy, -(ii-xx))*scl));
	  } // ii
	} // jj 
      } // xx
    } // yy

    
    // --- Now create a sparse matrix from the triplets --- //
    
    Eigen::SparseMatrix<T, Eigen::RowMajor, ind_t> OP(npix,npix);
    OP.setFromTriplets(triplets.begin(), triplets.end());

    return OP;
  }

  // *************************************************************************************** //

  template<typename T, typename ind_t = long>
  Eigen::SparseMatrix<T, Eigen::RowMajor, ind_t>
  downscale_to_operator(ind_t const ny, ind_t const nx,
			ind_t const ny1, ind_t const nx1, bool const normalize)
  {
    using Tri = Eigen::Triplet<T,ind_t>;

    
    // --- Step --- //

    T const dy = T(ny) / T(ny1);
    T const dx = T(nx) / T(nx1);

    
    // --- How many fine-grid pixels are included in the integral for one coarse-grid pixel? --- //
    
    ind_t const xnstep_max = ind_t(std::ceil(dx)) + 1;
    ind_t const ynstep_max = ind_t(std::ceil(dy)) + 1;

    
    // --- precompute integration weights as their are regular --- //
    
    mem::Array<T,2> weiy(ny1,ynstep_max); weiy.Zero();
    mem::Array<T,2> weix(nx1,xnstep_max); weix.Zero();
    mem::Array<T,2> area(ny1,nx1); area.Zero(); // do we need this?
    std::vector<int> nY(ny1,0), nX(nx1, 0);

    
    // --- precompute integration weights along each axis --- //
    
    for(ind_t yy=0; yy<ny1; ++yy){
      T const y0 = yy*dy;
      T const y1 = std::min(y0+dy, T(ny));

      ind_t iy0 = ind_t(std::floor(y0));
      ind_t iy1 = ind_t(std::floor(y1));

      nY[yy] = ind_t(std::ceil(std::ceil(y1)-y0));

      weiy(yy,0) = T(1) - (y0-iy0);
      weiy(yy,iy1-iy0) = y1-iy1;

      ind_t const n_middle = iy1-iy0;
      for(ind_t ii=1; ii<n_middle; ++ii)
	weiy(yy,ii) = T(1);
      
    } // yy
    
    for(ind_t xx=0; xx<nx1; ++xx){
      T const x0 = xx*dx;
      T const x1 = std::min(x0+dx, T(nx));

      ind_t ix0 = ind_t(std::floor(x0));
      ind_t ix1 = ind_t(std::floor(x1));

      nX[xx] = ind_t(std::ceil(std::ceil(x1)-x0));
      ind_t const n_middle = ix1-ix0;

      weix(xx,0) = T(1) - (x0-ix0);
      weix(xx,n_middle) = x1-ix1;

      for(ind_t ii=1; ii<n_middle; ++ii)
	weix(xx,ii) = T(1);
      
    }// xx
    
    

    // --- Calculate normalization by the area --- //

    if(normalize)
      for(ind_t yy=0; yy<ny1; ++yy)
	for(ind_t xx=0; xx<nx1; ++xx)
	  for(ind_t jj=0; jj<ynstep_max; ++jj)
	    for(ind_t ii=0; ii<xnstep_max; ++ii)
	      area(yy,xx) += weiy(yy,jj) * weix(xx,ii);
    else
      area = T(1);
    
    
    // --- Preallocate maximum number of Triplets to store the non-zero elements --- //

    std::vector<Tri> triplets;
    triplets.reserve(nx1*ny1*ynstep_max*xnstep_max);

    
    
    // --- Build Operator --- //

    for(ind_t yy=0; yy<ny1; ++yy){
      
      ind_t const y0 = ind_t(std::floor(yy*dy));
      ind_t const iNy = nY[yy];
      
      for(ind_t xx=0; xx<nx1; ++xx){
	ind_t const x0 = ind_t(std::floor(xx*dx));
	ind_t const iNx = nX[xx];
	
	for(ind_t jj=0; jj<iNy; ++jj){
	  for(ind_t ii=0; ii<iNx; ++ii){
	    
	    T const weight = (weix(xx,ii)*weiy(yy,jj)) / area(yy,xx);
	    
	    ind_t const pix  = (y0+jj)*nx + x0+ii;
	    ind_t const pix1 = (yy*nx1+xx);
	    
	    triplets.emplace_back(Tri(pix1, pix, weight));

	  }
	}
      }
    }


    // --- Create the matrix from arrays of triplets --- //
    
    Eigen::SparseMatrix<T, Eigen::RowMajor, ind_t> OP(nx1*ny1,nx*ny);
    OP.setFromTriplets(triplets.begin(), triplets.end());

    return OP;
  }
  
  // *************************************************************************************** //
  
    template<typename T, typename ind_t = long>
    Eigen::SparseMatrix<T, Eigen::RowMajor, ind_t>
    clipFOV_to_operator(ind_t const ny, ind_t const nx, ind_t const y0, ind_t const y1,
			ind_t const x0, ind_t const x1)
    {

      // --- Dimensions of the clipped FOV --- //
      
      ind_t const nx1 = x1-x0+1;
      ind_t const ny1 = y1-y0+1;

      
      // --- Create operator, only one element per row is non-zero --- //
      
      Eigen::SparseMatrix<T, Eigen::RowMajor, ind_t> OP(nx1*ny1,nx*ny);
      Eigen::VectorXi nElements_per_row = Eigen::VectorXi::Constant(ny1*nx1, 1); // 1D vector of integers
      OP.reserve(nElements_per_row);

      
      // --- Assign operator elements --- //
      
      for(ind_t yy=y0; yy<=y1; ++yy){
	for(ind_t xx=x0; xx<=x1; ++xx){
	  
	  ind_t const ipix1 = (yy-y0)*nx1 + (xx-x0); // coordinates in the new FOV
	  ind_t const ipix = yy*nx+xx; // coordinates in the old FOV
	  
	  OP.insert(ipix1, ipix) = T(1); // Convolution with a delta
	}
      }
      return OP;
    }
  
  // *************************************************************************************** //

   
  template<typename T, typename ind_t = long>
  Eigen::SparseMatrix<T, Eigen::RowMajor, ind_t>
  DiagonalOperator(ind_t const ny, ind_t const nx)
  {
    ind_t const npix = ny*nx;
    Eigen::SparseMatrix<T, Eigen::RowMajor, ind_t> A(npix,npix);

    Eigen::VectorXi nElements_per_row = Eigen::VectorXi::Constant(npix, 1); // 1D vector of integers
    A.reserve(nElements_per_row);


    for(ind_t ii=0; ii<npix; ++ii)
      A.insert(ii,ii) = T(1);
    
    return A;
  }
  
}

#endif
