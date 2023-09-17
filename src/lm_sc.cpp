/* ---

   Implementation of the spatially coupled LM routines.
   Many of the routines to construct the spatially coupled Hessian and residue
   can be found in spatially_coupled_tools.hpp and spatially_coupled_helper.{hpp,cpp}

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
   
   --- */
#include <cmath>
#include <chrono>

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>


#include "spatially_coupled_tools.hpp"
#include "lm_sc.hpp"

// *********************************************************************************************** //

using namespace spa;

// *********************************************************************************************** //

template<typename T, typename U, typename ind_t>
void construct_linear_system(spa::Data<T,U,ind_t> &dat, mem::Array<T,3> &m, 
			Eigen::SparseMatrix<U, Eigen::RowMajor,ind_t> const& L,
			Eigen::SparseMatrix<U, Eigen::RowMajor,ind_t> const& LL,
			Eigen::Matrix<U,Eigen::Dynamic,1> &B,
			Eigen::Matrix<U,Eigen::Dynamic,1> &diagonal,
			Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> &A)

{
  
  int const nthreads = int(dat.me.size());
  ind_t const ny = dat.ny;
  ind_t const nx = dat.nx;
  ind_t const npar = dat.npar;
  ind_t const npix = ny*nx;
  ind_t const nw = dat.nw;
  ind_t const ns = dat.ns;
  ind_t const dim = npar*npix;

  fprintf(stderr,"\n       [Constructing coupled linear system");
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


  
  mem::Array<T,5> J(ny,nx,npar,ns,nw); J.Zero();
  mem::Array<T,4> syn(ny,nx,ns,nw); syn.Zero();
  
  B.resize(npix*npar); B.setZero();


  
  // --- Add regularization terms to RHS --- //
  
  B -= L.transpose() * dat.getGamma(m);
 
  
  //Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> A(B.size(),B.size());

  

  
  // --- Get synthetic spectrum and non-coupled Jacobian --- //

  dat.synthesize_rf(m, syn, J);
  

  

  // --- Get the RHS, including spatially coupled terms --- //

  
  spa::addCoupledResidue<T,U,ind_t>(dat, J, B, syn, nthreads);


  

  // --- Get spatially coupled Hessian matrix --- //


  spa::fillHessianTerms<T,U,ind_t>(dat, A, J, nthreads);



  // --- Add regularization Hessian --- //

  for(ind_t dd=0; dd<dim;++dd){ // for all rows

    // --- iterate over all non-zero elements in that row, pray that they are allocated in the Hessian ... --- //
    
    for(typename Eigen::SparseMatrix<T,Eigen::RowMajor,ind_t>::InnerIterator it(LL,dd); it; ++it){
      
      ind_t const idx = it.index();
      A.coeffRef(dd,idx) += it.value();
    }
  }
  
  // --- Copy diagonal so we don't need to rebuild the matrix if we change Lambda --- //


  ind_t const nDiag = npix * npar;
  diagonal.resize(nDiag);
  
  for(ind_t ii=0; ii<nDiag; ++ii){
    diagonal[ii] = A.coeff(ii,ii);
  }

  
  // --- Add clock printout --- //
  
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double dt = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())*0.001;
  fprintf(stderr," -> dt=%lf s]", dt);
  
}


// *********************************************************************************************** //

template<typename T> T checkLambda(T const L, T const Lmin, T const Lmax)
{
  return std::min<T>(std::max<T>(L, Lmin), Lmax);
}


// *********************************************************************************************** //

template<typename T, typename U, typename ind_t>
void InitSolutionSystem( T const Lam,
			 Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> &A,
			 Eigen::Matrix<T,Eigen::Dynamic,1> const& B,
			 Eigen::Matrix<T,Eigen::Dynamic,1> const& Diagonal,
			 spa::Data<T,U,ind_t> &dat,
			 Eigen::Matrix<T,Eigen::Dynamic,1> &dm)
{
  /* ---
     Iterative solvers usually assume x_0 = 0. For small Lambda damping values,
     this can lead to stalling in the iterative process. We can provide an initial
     guess by solving the system with a larger Lambda value and getting a solution.
     Not great, but it works.

     Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
     --- */
  
  ind_t const N = dat.nx*dat.ny*dat.npar;
  U const ULam = std::min<U>(std::max<U>(1.0+10.0, 1.0+Lam*10), 5000.0);


  fprintf(stderr," initial solution");
  

  // --- Apply diagonal damping --- //
  
  for(ind_t ii=0; ii<N; ++ii)
    A.coeffRef(ii,ii) = ULam * Diagonal[ii];


  // --- Solve linear system for a larger Lambda value --- //
  
  Eigen::BiCGSTAB<Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t>> solver(A);
  dm = solver.solve(B);
  
}

// *********************************************************************************************** //


template<typename T, typename U, typename ind_t>
spa::Chi2_t<T> SolveSystem( T const Lam,
			    mem::Array<T,3> &m,
			    Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> &A,
			    Eigen::Matrix<T,Eigen::Dynamic,1> const& B,
			    Eigen::Matrix<T,Eigen::Dynamic,1> const& Diagonal,
			    int const method,
			    spa::Data<T,U,ind_t> &dat,
			    Eigen::SparseMatrix<T,Eigen::RowMajor,ind_t> const& L)
{

  ind_t const N = dat.nx*dat.ny*dat.npar;
  U const Lam_1 = U(1) + U(Lam);

  
  fprintf(stderr,"\n       [Solving coupled linear system, Lambda=%le:", Lam);
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  
  
  // --- Apply diagonal damping --- //
  
  for(ind_t ii=0; ii<N; ++ii)
    A.coeffRef(ii,ii) = Lam_1 * Diagonal[ii];


  
  // --- Solve linear system, try obtaining an initial guess --- //

  Eigen::Matrix<U,Eigen::Dynamic,1> dm;
  
  InitSolutionSystem(Lam, A, B, Diagonal, dat, dm);

  fprintf(stderr,", solving system");
  Eigen::BiCGSTAB<Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t>> solver(A);
  dm = solver.solveWithGuess(B, dm);



  

  // --- Add correction to model --- //

  for(ind_t ii=0; ii<N; ++ii)
    m[ii] += T(dm[ii]);

  dat.ScalePars(m);
  dat.CheckPars(m);
  dat.NormalizePars(m);

  

  // --- Synthesize spectrum to get Chi2 --- //

  mem::Array<T,4> isyn(dat.ny, dat.nx, dat.ns, dat.nw); isyn.Zero();
  dat.synthesize(m, isyn);


  
  // --- Add clock printout --- //
  
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double dt = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())*0.001;
  fprintf(stderr," -> dt=%lf s]", dt);
  
  
  return getChi2(dat, isyn, L, m);
}

// *********************************************************************************************** //

template<typename T, typename U, typename ind_t>
spa::Chi2_t<T> LMsc<T,U,ind_t>::getCorrection(int const iter, Chi2_t<T> const& bestChi2,
					      mem::Array<T,3> &m, Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> &A,
					      Eigen::Matrix<U,Eigen::Dynamic,1> const& B, T Lam, spa::Data<T,U,ind_t> &dat)const
{
  return SolveSystem(Lam, m, A, B, Diagonal, 0, dat, L);  
}

template spa::Chi2_t<double> LMsc<double,double,long>::getCorrection(int const iter, Chi2_t<double> const& bestChi2,
								     mem::Array<double,3> &m,
								     Eigen::SparseMatrix<double,Eigen::RowMajor,long> &A,
								     Eigen::Matrix<double,Eigen::Dynamic,1> const& B,
								     double Lam,
								     spa::Data<double,double,long> &dat)const;


// *********************************************************************************************** //

template<typename T, typename U, typename ind_t>
spa::Chi2_t<T> LMsc<T,U,ind_t>::invert(spa::Data<T,U,ind_t> &dat, mem::Array<T,3> &BestModel, mem::Array<T,4> &BestSyn, int const max_niter, T const fx_thres, int const method, T const init_Lambda)
{
  int const nthreads = int(dat.me.size());
  Eigen::initParallel();
  Eigen::setNbThreads(nthreads);
  
  static constexpr T const facLam = 2.75;//sqrt(10.0);
  static constexpr T const maxLam = 10000.;
  static constexpr T const minLam =  1.0e-4;//3.1622776601683795e-3;
  static constexpr int const max_n_reject = 6;
  

  
  // --- check dimensions of the model vs Data struct --- //

  if((dat.ny != BestModel.shape(0)) || (dat.nx != BestModel.shape(1)) || (dat.npar != BestModel.shape(2))){
    fprintf(stderr, "[error] LM_sc::invert: provided model does not have the same dimensions as those in Data struct, fix your code!");
    return Chi2_t<T>(T(1.e30),T(0));
  }


  // --- Init some temporary variables --- //

  T Lam = checkLambda(init_Lambda, minLam, maxLam), oLam = 0;
  
  
  // --- check model parameter and compress them --- //
  
  dat.CheckPars(BestModel);
  dat.NormalizePars(BestModel);

  mem::Array<T,3> m = BestModel;



  // --- Get regularization matrices --- //

  this->L = dat.get_L(dat.npar);
  this->LL = L.transpose() * L;

  Eigen::Matrix<T,Eigen::Dynamic, 1> B(m.size()); B.setZero();


  // --- Preallocate Hessian matrix --- //

  Eigen::SparseMatrix<U,Eigen::RowMajor,ind_t> A(dat.ny*dat.nx*dat.npar, dat.ny*dat.nx*dat.npar);
  fprintf(stderr,"[info] invert: preallocating Hessian matrix ... ");
  spa::count_Hessian(dat.ny, dat.nx, dat.npar, dat, A, nthreads);
  fprintf(stderr,"done\n");

  double siZ = (double(A.nonZeros())*(sizeof(U) +sizeof(ind_t)) + double(dat.ny*dat.nx*dat.npar)*sizeof(ind_t))*1.e-9;
  fprintf(stderr,"\n[info] invert: sparse Hessian size -> %lf GBytes\n\n", siZ);

  
  
  // --- Get initial Chi2 --- // 

  mem::Array<T,4> syn(dat.ny, dat.nx, dat.ns, dat.nw); syn.Zero();
  dat.synthesize(m, syn);

  Chi2_t<T> BestChi2 = spa::getChi2(dat, syn, L, m);
  Chi2_t<T> Chi2 = BestChi2;
  
  fprintf(stderr,"[info] invert: iter=%3d, Chi2=%13.5f (%e)", 0, Chi2.get(), Chi2.reg);

  
  // --- Construct coupled system --- //

  construct_linear_system<T,U,ind_t>(dat, m, this->L, this->LL, B, this->Diagonal, A);
  
  int failed = 0;
  
  // --- Iterate --- //

  int iter = 1;
  while(iter <= max_niter){
    
    m = BestModel;
    Chi2 = getCorrection(iter, BestChi2, m, A, B, Lam, dat);

    if(Chi2 < BestChi2){
      
      BestChi2 = Chi2;
      BestModel = m;
    
      fprintf(stderr,"\n\n[info] invert: iter=%3d, Chi2=%13.5f (Reg=%e), Lambda=%le", iter, Chi2.get(), Chi2.reg, Lam);

      oLam = Lam;
      Lam = checkLambda(Lam/facLam, minLam, maxLam);
      
      if((Lam == oLam) && (Lam < 0.1)) Lam = 0.31622776601683794;
      
      failed = 0;
      iter+=1;
      
    }else{

      fprintf(stderr,"\n       [Chi2=%13.5f > BestChi2=%13.5f -> increasing Lambda parameter]", Chi2.get(), BestChi2.get());

      Lam = checkLambda(Lam*facLam*facLam, minLam, maxLam);
      failed += 1;

      if(failed >= max_n_reject){
	fprintf(stderr,"\n[info] invert: too many consecutive failed attempts, ending inversion with BestChi2=%15.5f\n", BestChi2.get());

	break;
      }
    } // rejected


    
    // --- recompute linear system? --- ?

    if((failed == 0) && (iter <= max_niter)){
      A.coeffs() *= U(0);
      construct_linear_system<T,U,ind_t>(dat, m, this->L, this->LL, B, this->Diagonal, A);
    }

  } // iter
  

  // --- Copy the best model to the output variable, synthesize and scale parameters --- /
  
  dat.synthesize(BestModel, BestSyn);
  dat.ScalePars(BestModel);

  fprintf(stderr,"\n[info] invert: inversion finalized, Chi2=%le\n", BestChi2.get());
  
  
  return BestChi2;
}

template spa::Chi2_t<double> LMsc<double,double,long>::invert(spa::Data<double,double,long> &dat, mem::Array<double,3> &m,  mem::Array<double,4> &BestSyn, int const max_niter, double const fx_thres,  int const method, double const init_Lambda);

// *********************************************************************************************** //

