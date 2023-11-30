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
#include <unsupported/Eigen/IterativeSolvers>

namespace spa{

  // ************************************************************** //

  template<typename T, typename iType = long>
  class lms{
  protected:
    iType npar, nt, ny, nx;
    
    Eigen::SparseMatrix<T, Eigen::RowMajor, iType> A;
    Eigen::Matrix<T,Eigen::Dynamic, 1> B;
    
    Eigen::SparseMatrix<T,Eigen::RowMajor, iType> L;
    Eigen::SparseMatrix<T,Eigen::RowMajor, iType> LL;
    
  public:
    lms(long const inpar, long const i_nt, long const iny, long const inx):
      npar(inpar), nt(i_nt), ny(iny), nx(inx),  A(), B(){};
        
    // ------------------------------------------------------------ //
    
    static inline T checkLambda(T val, T const &mi, T const& ma)
    {return std::max<T>(std::min<T>(ma, val), mi);}

    // ------------------------------------------------------------ //

    inline static T get_one_JJ(long const ndat, const T* const __restrict__ Jy, const T* const __restrict__ Jx)
    {
      return static_cast<T>(ksumMult<T,long double>(ndat, Jy, Jx));
    }
    
    // ------------------------------------------------------------ //

    void construct_system(long const npar, container<T> const& cont, T* const __restrict__ m, 
			  T* const __restrict__ r,  Eigen::Matrix<T,Eigen::Dynamic,1> const& Reg_RHS)
    {

      
      iType const npix = cont.ny*cont.nx;
      iType const ndat = cont.nDat;
      iType const nthreads = cont.getNthreads();
      iType const nTot = nt*npix;

      
      // --- Build Sparse system --- //

      B.resize(nTot*npar); B.setZero();
      A.resize(0,0); A.data().squeeze(); A.resize(nTot*npar, nTot*npar);
      A.reserve(Eigen::VectorXi::Constant(nTot*npar,npar));

      
      iType tid=0;

      // --- parallel block --- //
      
#pragma omp parallel default(shared) firstprivate( tid) num_threads(nthreads)      
      {
	
	tid = omp_get_thread_num();
	T* const __restrict__ iJ = new T [npar*ndat](); // Allocate thread buffer for derivatives
	
#pragma omp for 
	for(iType tpix=0; tpix<nTot;++tpix){
	  
	  iType const Elem = tpix*npar;
	  
	  // --- synthesize_one pixel with derivatives --- //
	  
	  cont.synthesize_der_one(npar, &m[Elem], &r[tpix*ndat], iJ, tid, tpix);
	  
	  
	    
	    // --- Fill in subspace in sparse Hessian matrix --- //
	    
	    for(iType jj=0; jj<npar; ++jj){
	      

	      // --- RHS of the equation --- //
	      
	      B[Elem+jj] = ksumMult<T,long double>(ndat, &iJ[jj*ndat], &r[tpix*ndat]) - Reg_RHS[Elem+jj];

	      
	      for(iType ii=0; ii<npar;++ii){
		
		// --- Matrix subspaces --- //
		
		T const iSum = get_one_JJ(ndat, &iJ[jj*ndat], &iJ[ii*ndat]);
		A.insert(Elem + jj, Elem + ii) = iSum;
		
		//if(ii != jj) // The matrix is symmetric but avoid inserting the diagonal term twice in the diagonal
		// A.insert(Elem + ii, Elem + jj) = iSum;
		
	      } // ii
	    } // jj    
	} // itime
	
	delete [] iJ;
	
      }// parallel
      
    }


    // ------------------------------------------------------------ //

    void getInitialSolution(iType const nDiag, T const iLam, Eigen::SparseMatrix<T,Eigen::RowMajor,iType> &Atot,
			    Eigen::Matrix<T,Eigen::Dynamic,1> &dx)const
    {

      // --- store diagonal --- //

      Eigen::Matrix<T,Eigen::Dynamic,1> Diagonal = Atot.diagonal();

      
      // --- damp diagonal with a large value and get correction --- //
      
      T one_ilam = T(1);
      
      if(iLam >= T(10))    one_ilam +=  iLam*2;
      else if(iLam > T(1)) one_ilam +=  T(10) + iLam;
      else                 one_ilam +=  T(1)  + iLam;
      
      for(iType kk =0; kk<nDiag; ++kk)
	Atot.coeffRef(kk,kk) *= one_ilam;

      Eigen::BiCGSTAB<Eigen::SparseMatrix<T,Eigen::RowMajor,iType>> solver(Atot);
      dx = solver.solve(B);

      
      // --- Restore diagonal --- //
      
      Atot.diagonal() = Diagonal;
      
    }

    // ------------------------------------------------------------ //

    Chi2<T> getCorrection(container<T> const& cont, T* const __restrict__ m,
			  T* const __restrict__ syn, T* const __restrict__ r, T const iLam, int const method)const 
    {


      iType const npix = cont.ny*cont.nx;
      iType const nt   = cont.nt;
      iType const ndat = cont.nDat;
      iType const nTot = nt*npix;
      iType const nDiag = iType(npar)*nTot;
      
      Eigen::SparseMatrix<T,Eigen::RowMajor,iType> Atot = A+LL;
      Eigen::Matrix<T,Eigen::Dynamic,1> dx;

	

      // --- get initial solution with a large value of lambda --- //

      if(method != 2){
	getInitialSolution(nDiag, iLam, Atot, dx);
      }
      
      
      // --- damp diagonal and get correction --- //
      
      T const one_ilam = 1.0 + iLam;
      for(iType kk =0; kk<nDiag; ++kk)
	Atot.coeffRef(kk,kk) *= one_ilam;

      
      // --- Solve for corrections --- //
      
      if(method == 0){
      	Eigen::ConjugateGradient<Eigen::SparseMatrix<T,Eigen::RowMajor,iType>, Eigen::Lower| Eigen::Upper> solver(Atot);
      	dx = solver.solveWithGuess(B, dx);
      }else if(method == 1){
	Eigen::BiCGSTAB<Eigen::SparseMatrix<T,Eigen::RowMajor,iType>> solver(Atot);
	dx = solver.solveWithGuess(B, dx);
      }else if(method == 2){
	Eigen::SparseLU<Eigen::SparseMatrix<T,Eigen::RowMajor,iType>> solver(Atot);
	dx = solver.solve(B);
      }else{
	Eigen::GMRES<Eigen::SparseMatrix<T,Eigen::RowMajor,iType>> solver(Atot);
	dx = solver.solveWithGuess(B, dx);
      }
   

      
      // --- Check corrections --- //
      
      for(iType tpix=0; tpix<nTot; ++tpix){
	for(iType pp=0; pp<npar; ++pp){
	  m[tpix*npar+pp] += dx[tpix*npar+pp];
	  cont.Pinfo[pp].CheckNormalized(m[tpix*npar+pp]);
	} // pp
      } // tpix

	

	
      // --- compute chi2 --- //
	
      std::vector<T> rnew(nt*ndat*npix,0);
      cont.fx(npar, m, syn, &rnew[0]);
      Eigen::Matrix<T, Eigen::Dynamic, 1> Gam = cont.getGamma(npar, m);
      
      Chi2<T> chi2(ksum2<T,long double>(nt*npix*ndat, &rnew[0]), ksum2<T,long double>(Gam.size(), &Gam[0]));

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
	
	iType const npix = cont.nx*cont.ny;
	iType const nt = cont.nt;
	std::vector<Chi2<T>> iChi2;
	std::vector<T> Lambdas;

	int idx = 0;

	Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,1>> input_model(m, npix*npar*nt);
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
	      T const Chi2_thres = 1.0, T const fx_thres = 1.e-3, int delay_braket = 2,
	      bool verbose = true, int const method = 0)
    {
      int const nthreads = int(cont.Me.size());
      Eigen::initParallel();
      Eigen::setNbThreads(nthreads);

      static constexpr T const facLam = 2.75;
      static constexpr T const maxLam = 1E4;
      static constexpr T const minLam =  1e-4;
      static constexpr int const max_n_reject = 6;

      
      // --- Init temporary variables --- //

      Chi2<T> bestChi2(1.e34,1.e34); 
      Chi2<T> chi2 = bestChi2;
      
      iType const npix = cont.nt*cont.ny*cont.nx;
      iType const ndat = cont.nDat;
       
      //iType const nJ = long(npix)*long(npar)*long(ndat);

      
      T* const __restrict__ r   = new T [npix*ndat]();
      T* const __restrict__ m   = new T [npix*npar]();
      

      std::vector<T> lambdas(max_iter, T(0));
      
      
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
	bestChi2 = Chi2<T>(ksum2<T,long double>(npix*ndat, r), ksum2<T,long double>(Gam.size(), &Gam[0]));
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
	  lambdas[iter] = iLam;

	  if(!do_bracket)
	    if(iLam*1.00001 > minLam)
	      iLam = checkLambda(iLam/facLam, minLam, maxLam);
	    else
	      iLam *= facLam*facLam;
	  else{
	    if(iter > 2){
	      int repeated = 0;
	      for(int ii=iter-2; ii<=iter; ++ii)
		if(lambdas[ii] == minLam) repeated += 1;

	      if(repeated == 3){
		iLam = 1;
		lambdas[iter] = 1;
	      }
	      else
		iLam = facLam*facLam*facLam*iLam;
	    }else{
	      iLam = facLam*facLam*iLam;
	    }
	  }	    
	  if(dfx < fx_thres){
	    if(tooSmall) quit = true;
	    else tooSmall = true;
	  }
	  n_rejected = 0;

	}else{
	  
	  iLam = checkLambda(iLam*SQ<T>(facLam), minLam, maxLam);
	  n_rejected += 1;
	  if(verbose)
	    fprintf(stderr,"lms::fitData: ----> Chi2=%s > %s -> Increasing lambda %f -> %f\n",
		    chi2.formatted().c_str(), bestChi2.formatted().c_str(), oLam, iLam);
	  
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
	Reg_RHS = L.transpose()*cont.getGamma(npar, m);
	


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
