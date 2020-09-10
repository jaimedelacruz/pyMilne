#ifndef SPATHPP
#define SPATHPP

// -------------------------------------------------------
//   
//   Spatially-regularized Levenberg Marquardt algorithm
//   Coded by J. de la Cruz Rodriguez (ISP-SU, 2020)
//
//   Reference: de la Cruz Rodriguez (2019):
//   https://ui.adsabs.harvard.edu/abs/2019A%26A...631A.153D/abstract
//
//   ------------------------------------------------------- 

#include <omp.h>
#include <vector>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <chrono>

#include "line.hpp"
#include "Milne.hpp"
#include "lm.hpp"
#include "spatially_regularized_tools.hpp"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace spa{

  // ************************************************************** //

  template<typename T, typename iType = long>
  class lms{
  protected:
    iType npar, ny, nx;
    
    Eigen::SparseMatrix<T, Eigen::RowMajor, iType> A;
    
    Eigen::Matrix<T,Eigen::Dynamic, 1> B;
    
    Eigen::SparseMatrix<T,Eigen::RowMajor, iType> L;
    Eigen::SparseMatrix<T,Eigen::RowMajor, iType> LL;
    
  public:
    lms(int const inpar, int const iny, int const inx):
      npar(inpar), ny(iny), nx(inx),  A(), B(){};
        
    // ------------------------------------------------------------ //
    
    static inline T checkLambda(T val, T const &mi, T const& ma)
    {return std::max<T>(std::min<T>(ma, val), mi);}

    // ------------------------------------------------------------ //

    inline static T get_one_JJ(int const ndat, const T* const __restrict__ Jy, const T* const __restrict__ Jx)
    {
      return static_cast<T>(ksumMult<T,double>(ndat, Jy, Jx));
    }
    
    // ------------------------------------------------------------ //

    void construct_system(int const npar, container<T> const& cont, T* const __restrict__ m, 
			  T* const __restrict__ r,  Eigen::Matrix<T,Eigen::Dynamic,1> const& Reg_RHS)
    {

      iType const npix = cont.ny*cont.nx;
      iType const ndat = cont.nDat;
      iType const nthreads = cont.getNthreads();
      iType const nx = cont.nx;
      iType const ny = cont.ny;
      iType const Jstride = npar*ndat;

      
      // --- Build Sparse system --- //

      B.resize(npix*npar); B.setZero();
      A.resize(0,0); A.data().squeeze(); A.resize(npix*npar, npix*npar);
      A.reserve(Eigen::VectorXi::Constant(npix*npar,npar));

      
      T* __restrict__ iJ = NULL;
      iType ipix=0, tid=0, pp=0, ii=0, jj=0;
      T iSum = 0;

      // --- parallel block --- //
      
#pragma omp parallel default(shared) firstprivate(ipix, tid, iSum, pp, ii, jj, iJ) num_threads(nthreads)      
      {
	
	tid = omp_get_thread_num();
	iJ = new T [npar*ndat](); // Allocate thread buffer for derivatives
	
#pragma omp for
	for(ipix=0; ipix<npix; ++ipix){

	  // --- synthesize_one pixel with derivatives --- //
	  
	  cont.synthesize_der_one(npar, &m[ipix*npar], &r[ipix*ndat], iJ, tid, ipix);


	  // --- Fill in subspace in sparse Hessian matrix --- //

	  for(jj=0; jj<npar; ++jj){

	    // --- RHS of the equation --- //
	    
	    B[ipix*npar+jj] = ksumMult<T,double>(ndat, &iJ[jj*ndat], &r[ipix*ndat]) - Reg_RHS[ipix*npar+jj];
	    
	    for(ii=0; ii<=jj;++ii){

	      // --- Matrix subspaces --- //
	      
	      iSum = get_one_JJ(ndat, &iJ[jj*ndat], &iJ[ii*ndat]);
	      A.insert(ipix*npar + jj, ipix*npar + ii) = iSum;
	      
	      if(ii != jj) // The matrix is symmetric but avoid inserting the diagonal term twice
	       	A.insert(ipix*npar + ii, ipix*npar + jj) = iSum;
	      
	    } // ii
	  } // jj
	  
	  
	} // ipix
	
	delete [] iJ;
	iJ = NULL;
	
      }// parallel
      
    }


    // ------------------------------------------------------------ //

    Chi2<T> getCorrection(container<T> const& cont, T* const __restrict__ m,
			  T* const __restrict__ syn, T* const __restrict__ r, T iLam, int const method)const 
    {


      iType const npix = cont.ny*cont.nx;
      iType const ndat = cont.nDat;

      Eigen::SparseMatrix<T,Eigen::RowMajor,iType> Atot = A+LL;

      
      // --- damp diagonal and get correction --- //
      
      iType const nDiag = iType(npar)*iType(npix);
      for(iType kk =0; kk<nDiag; ++kk)
	Atot.coeffRef(kk,kk) *= (1+iLam);

      Eigen::Matrix<T,Eigen::Dynamic,1> dx;
      
      // --- Solve for corrections --- //
      
      if(method == 0){
	Eigen::ConjugateGradient<Eigen::SparseMatrix<T,Eigen::RowMajor,iType>, Eigen::Lower| Eigen::Upper> solver(Atot);
	dx = solver.solve(B);
      }else if(method == 1){
	Eigen::BiCGSTAB<Eigen::SparseMatrix<T,Eigen::RowMajor,iType>> solver(Atot);
	dx = solver.solve(B);
      }else if(method == 2){
	Eigen::SparseLU<Eigen::SparseMatrix<T,Eigen::RowMajor,iType>> solver(Atot);
	dx = solver.solve(B);
      }

      
      // --- Check corrections --- //
      
      for(iType pp=0; pp<npar; ++pp)
	for(iType ipix = 0; ipix<npix; ++ipix){
	  m[ipix*npar+pp] += dx[ipix*npar+pp];
	  cont.Pinfo[pp].CheckNormalized(m[ipix*npar+pp]);
	  
	}

      // --- compute chi2 --- //

      
      std::vector<T> rnew(ndat*npix,0);
      cont.fx(npar, m, syn, &rnew[0]);
      Eigen::Matrix<T, Eigen::Dynamic, 1> Gam = cont.getGamma(npar, m);
      
      Chi2<T> chi2(ksum2<T,double>(npix*ndat, &rnew[0]), ksum2<T,double>(Gam.size(), &Gam[0]));
      
      return chi2;
    }
    
    // ------------------------------------------------------------ //

    Chi2<T> getStep(container<T> const& cont, T* const __restrict__ m, 
		    T* const __restrict__ syn, T* const __restrict__ r, T& iLam, bool bracket,
		    T const minLam, T const maxLam, T const Lam_step,  Eigen::Matrix<T,Eigen::Dynamic,1> const& Reg_RHS,
		    Chi2<T> const& bestChi2, int const method)
    {
      
      // --- if no bracketing just compute one correction --- //
      
      if(!bracket){
	return getCorrection(cont, m, syn, r, iLam, method);
      }else{

	// --- Bracketing optimal lambda value --- //
	
	int const npix = cont.nx*cont.ny;
	std::vector<Chi2<T>> iChi2;
	std::vector<T> Lambdas;

	int idx = 0;

	Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,1>> input_model(m, npix*npar);
	Eigen::Matrix<T,Eigen::Dynamic,1> Model = input_model;

	Chi2<T> chi2 = getCorrection(cont, &Model[0], syn, r, iLam, method);
	if(chi2.value() > bestChi2.value()) return chi2;
	
	Eigen::Matrix<T,Eigen::Dynamic,1> bestModel = Model;

	iChi2.emplace_back(chi2);
	Lambdas.emplace_back(iLam);

	// --- First try to bracket by decreasing lambda --- //
	int iter = 0;
	while((iter < 1) || ((iter < 4) && (iChi2[iter].value()<iChi2[iter-1].value()) && (Lambdas[iter] > minLam))){
	  iLam = checkLambda(iLam / Lam_step, minLam, maxLam);
	  Model = input_model;

	  Lambdas.emplace_back(iLam);
	  iChi2.emplace_back(getCorrection(cont, &Model[0], syn, r, iLam, method));
	  if(iChi2[iter+1].value() < iChi2[idx].value()){
	    idx = iter+1;
	    bestModel = Model;
	  }
	  
	  ++iter;
	  
	}// while
	
	// --- if the best Chi2 is not in the first element we consider it bracketed --- //

	if(idx == 0){
	  // --- Go in the opposite direction, increasing lambda --- //
	  iter = 0;
	  while((iter == 0) ||( (iter++ <= 5) && (Lambdas[0] < maxLam))){
	    Model = input_model;
	    iLam = checkLambda(iLam * Lam_step*Lam_step, minLam, maxLam);

	    Lambdas.insert(Lambdas.begin(), iLam);
	    iChi2.insert(iChi2.begin(), getCorrection(cont, &Model[0], syn, r, iLam, method) );

	    if(iChi2[0].value() < iChi2[1].value()){
	      bestModel = Model;
	      idx = 0;
	    }else{
	      idx += 1;
	      break;
	    }
	    
	  }// while
	}
	
	input_model = bestModel;
	iLam = Lambdas[idx];
	return iChi2[idx];
      }
      
    }

    // ------------------------------------------------------------ //

    T fitData(container<T> const& cont, int const npar, T* __restrict__ bestModel,
	      T* __restrict__ bestSyn, int const max_iter = 20, T iLam = 10,
	      T const Chi2_thres = 1.0, T const fx_thres = 2.e-3, int const delay_braket = 2,
	      bool verbose = true, int const method = 0)
    {
      int const nthreads = int(cont.Me.size());
      Eigen::initParallel();
      Eigen::setNbThreads(nthreads);

      static constexpr T const facLam = 3.1622776601683795;
      static constexpr T const maxLam = 1000.;
      static constexpr T const minLam =  3.1622776601683795e-3;
      static constexpr int const max_n_reject = 6;

      
      // --- Init temporary variables --- //

      Chi2<T> bestChi2(1.e34,1.e34); 
      Chi2<T> chi2 = bestChi2;
      
      iType const npix = cont.ny*cont.nx;
      iType const ndat = cont.nDat;
      iType const nJ = long(npix)*long(npar)*long(ndat);

      
      T* const __restrict__ r   = new T [npix*ndat]();
      T* const __restrict__ m   = new T [npix*npar]();
      

      
      // --- check pars --- //

      cont.checkPars(npar, bestModel);
      cont.NormalizePars(npar, bestModel);
      std::memcpy(m, bestModel, npix*npar*sizeof(T));

      

      // --- Init residue "r" --- //
      
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      cont.fx(npar, m, bestSyn, r);
      

      
      // --- precompute L (only needed once) --- //
      
      if(verbose)
	fprintf(stdout, "lms::fitData: pre-computing regularization derivatives matrix ... ");
      
      L  = cont.get_L(npar, m);
      LL = L.transpose()*L;
      Eigen::Matrix<T,Eigen::Dynamic,1> Reg_RHS;
      
      // --- Init total Chi2 --- //
      {
	Eigen::Matrix<T, Eigen::Dynamic, 1> Gam = cont.getGamma(npar, m);
	bestChi2 = Chi2<T>(ksum2<T,double>(npix*ndat, r), ksum2<T,double>(Gam.size(), &Gam[0]));
	Reg_RHS = L.transpose()*Gam; // Init RHS regularization term. Vector that only needs to be computed once per successfull iteration.
      }

      

      // --- Initialize sparse linear system for the first iteration --- //
      
      construct_system(npar, cont, m, r, Reg_RHS);
   
      
      if(verbose)
	fprintf(stdout,"done\n");
      

      
      // --- Init iteration --- //

      if(verbose)
	fprintf(stderr, "\nlms::fitData: [Init] Chi2=%s\n", bestChi2.formatted().c_str());
      int iter = 0, n_rejected = 0;
      bool quit = false, tooSmall = false;
      T oLam = 0, dfx = 0;
      
      iLam = checkLambda(iLam, minLam, maxLam);


      

      // --- Iterate the solution --- //
      
      while(iter < max_iter){

	bool do_bracket =  ((delay_braket > iter)? false : true);
	
      	oLam = iLam;
	std::memcpy(m, bestModel, npar*npix*sizeof(T));

	
	// --- Get model correction --- //

	chi2 = getStep(cont, m, bestSyn, r, iLam, do_bracket, minLam, maxLam, facLam, Reg_RHS, bestChi2, method);


	// --- have we improved? --- //

	if(chi2.value() < bestChi2.value()){

	  oLam = iLam;
	  dfx = (bestChi2.value() - chi2.value()) / bestChi2.value();
	  bestChi2 = chi2;

	  std::memcpy(bestModel,   m, npar*npix*sizeof(T));


	  if(!do_bracket)
	    if(iLam*1.00001 > minLam)
	      iLam = checkLambda(iLam/facLam, minLam, maxLam);
	    else
	      iLam *= facLam*facLam;
	  else
	    iLam = facLam*iLam;
	    	    
	  if(dfx < fx_thres){
	    if(tooSmall) quit = true;
	    else tooSmall = true;
	  }
	  n_rejected = 0;

	}else{
	  
	  iLam = checkLambda(iLam*SQ<T>(facLam), minLam, maxLam);
	  n_rejected += 1;
	  if(verbose)
	    fprintf(stderr,"lms::fitData: ----> Chi2=%s > %s -> Increasing lambda %f -> %f\n",  chi2.formatted().c_str(), bestChi2.formatted().c_str(), oLam, iLam);
	  
	  if(n_rejected<max_n_reject) continue;
	  

	} // else


	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	begin = end;
	
	// --- Check what has happened with Chi2 --- //
	
	if(n_rejected >= max_n_reject){
	  if(verbose)
	    fprintf(stderr, "lms::fitData: maximum number of rejected iterations reached, finishing inversion");
	  break;
	}

	if(verbose)
	  fprintf(stderr, "lms::fitData: [%3d] Chi2=%s, lambda=%e, dtime=%6.1fs\n", iter, chi2.formatted().c_str(), oLam, dt/1000.);

	if(bestChi2.value() < Chi2_thres){
	  if(verbose)
	    fprintf(stderr, "lms::fitData: Chi2 (%f) < Chi2_threshold (%f), finishing inversion", bestChi2.value(), Chi2_thres);
	  break;
	}

	if(quit){
	  if(verbose)
	    fprintf(stderr, "lms::fitData: Chi2 improvement too small for 2-iterations, finishing inversion\n");
	  break;
	}
	
	iter++;
	if(iter >= max_iter){
	  break;
	}
	
	// --- init next iteration --- //
	
	std::memcpy(m, bestModel, npix*npar*sizeof(T));
	
	{
	  Eigen::Matrix<T, Eigen::Dynamic, 1> Gam = cont.getGamma(npar, m);
	  Reg_RHS = L.transpose()*Gam;
	}


	// --- Construct sparse matrix with the new model estimate --- //
	
	construct_system(npar, cont, m, r,  Reg_RHS);
      }

      
      // --- Synthesize with best model --- //

      cont.fx(npar, bestModel, bestSyn, r);


      
      // --- scale model parameters ---- //

      cont.ScalePars(npar, bestModel);

      

      // --- Clean up --- //
      
      delete [] r;
      delete [] m;


      return bestChi2.value();
    }
    
    // ------------------------------------------------------------ //

  };
  
}



#endif
