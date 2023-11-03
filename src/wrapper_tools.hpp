#ifndef WRAPPERHPP
#define WRAPPERHPP

#include <omp.h>
#include <vector>
#include <iostream>

#include "line.hpp"
#include "Milne.hpp"
#include "lm.hpp"

#include "spatially_regularized_tools.hpp"
#include "spatially_regularized.hpp"

namespace wr{
  
  template<typename T>
  void SynMany(std::vector<ml::Milne<T>> const& ME,  T* __restrict__ m,
	       T* __restrict__ stokes_in, long const ny, long const nx, T const mu)
  {
    
    // --- Number of threads and pixels --- //
    
    long const nthreads = int(ME.size());
    long const nPix     = nx*ny;
    long const sStride  = 4*ME[0].get_number_of_wavelength();
    
    // --- Parallel loop --- //
    
    long ipix = 0, tid = 0;
#pragma omp parallel default(shared) firstprivate(ipix, tid) num_threads(nthreads)  
    {
      tid = omp_get_thread_num();
      
#pragma omp for
      for(ipix = 0; ipix<nPix; ++ipix){
	ME[tid].checkParameters(&m[ipix*9]);
	ME[tid].synthesize(&m[ipix*9], &stokes_in[sStride*ipix], mu);
      }
      
    } // parallel block
    
    
  }
  
  // ********************************************************************* //
  template<typename T>
  void SynManyRF(std::vector<ml::Milne<T>> const& ME,  T* __restrict__ m,
		 T* __restrict__ stokes_in, T* __restrict__ rf_in, long const ny, long const nx, T const mu)
  {
    
    // --- Number of threads and pixels --- //
    
    long const nthreads = int(ME.size());
    long const nPix     = nx*ny;
    long const sStride  = 4*ME[0].get_number_of_wavelength();
    long const rfStride = sStride*9;
    
    // --- Parallel loop --- //
    
    long ipix = 0, tid = 0;
#pragma omp parallel default(shared) firstprivate(ipix, tid) num_threads(nthreads)  
    {
      tid = omp_get_thread_num();
      
#pragma omp for
      for(ipix = 0; ipix<nPix; ++ipix){
	ME[tid].checkParameters(&m[ipix*9]);
	ME[tid].synthesize_rf(&m[ipix*9], &stokes_in[sStride*ipix], &rf_in[rfStride*ipix],mu);
      }
      
    } // parallel block
  }

  // ********************************************************************* //

  template<typename T> T fitOne(ml::Milne<T> const& ME,  lm::LevMar<T> const& fit, T* __restrict__ m,
				T* __restrict__ syn, const T* __restrict__ obs, const T* __restrict__ sig, long const nDat,
			        int const nRandom, int const niter, T const chi2_thres, T const mu, bool verbose)
  {

    // --- we need to re-invert as many as nRandom --- //

    T* __restrict__ bestM = new T [9]();
    T* __restrict__ bestS = new T [nDat]();
    T* __restrict__ Mref  = new T [9]();
    std::memcpy(Mref,m,9*sizeof(T));
    
    
    T bestChi2 = 1.e32;

    for(int iter = 0; iter < nRandom; ++iter){

      // --- Add perturbation to initial parameters --- //
      
      if(iter > 0){
	if((iter != (nRandom-1)) || (nRandom <= 2)){
	  std::memcpy(m,Mref,9*sizeof(T));
	  ml::randomizeParameters<T>(m);
	}else{
	  std::memcpy(m,bestM,9*sizeof(T));
	  ml::randomizeParameters<T>(m, 0.2);
	}

	if(m[7] > m[8]){
	  T const bla = m[8];
	  m[8] = m[7];
	  m[7] = bla;
	}
      }

      // --- fit data --- //
      
      T chi2 = fit.fitData(ME, nDat, obs, syn, sig, m, mu, niter, sqrt(10.), chi2_thres, 3.e-3, 2, verbose);

      if(chi2 < bestChi2){
	memcpy(bestM,m,     9*sizeof(T));
	memcpy(bestS,syn,nDat*sizeof(T));
	bestChi2 = chi2;
      }
      
      if(bestChi2 < chi2_thres) break;
    }

    // --- copy back best guessed model --- //

    std::memcpy(m,   bestM,    9*sizeof(T));
    std::memcpy(syn, bestS, nDat*sizeof(T));


    // --- clean up --- //

    delete [] bestM;
    delete [] bestS;
    delete [] Mref;
    
    return bestChi2;
  }
  
  // ********************************************************************* //

  template<typename T>
  void InvertMany(std::vector<ml::Milne<T>> const& ME,  T* __restrict__ m,
		  T* __restrict__ stokes_in, const T* __restrict__ obs, const T* __restrict__ sig,
		  T* __restrict__ bestChi2, long const ny, long const nx, long nDat, int const nRandom,
		  int const niter, T const chi2_thres, T const mu, bool  verbose)
  {

    int const nthreads = int(ME.size());
    if(nthreads > 1 && verbose == true && (nx*ny > 1)){
      verbose = false;
      printf("InvertMany: nthreads > 1 -> verbose mode deactivated!\n");
    }
    
    // --- Init one LevMar object per thread --- //
    
    std::vector<lm::LevMar<T>> fit;
    for(int ii=0; ii<nthreads; ++ii){
      fit.push_back(lm::LevMar<T>(9));
      for(int jj=0; jj<9; ++jj)
	fit[ii].Pinfo[jj] = lm::Par<T>(false, true, ml::pscl<T>[jj], ml::pmin<T>[jj], ml::pmax<T>[jj]);

      fit[ii].Pinfo[2].isCyclic = true; 
    }
     
    //  Par(bool const cyclic, bool const ilimited, T const scal, T const mi, T const ma):



    // --- loop? --- //

    long const nPix     = nx*ny;
    long const sStride  = 4*ME[0].get_number_of_wavelength();
    //long const rfStride = sStride*9;


    
    // --- Parallel loop --- //
    
        
    long oper = -1, per = 0;
#pragma omp parallel default(shared) num_threads(nthreads)  
    {
      long const tid = omp_get_thread_num();

      //

#pragma omp for schedule(dynamic,1)
      for(long ipix = 0; ipix<nPix; ++ipix){
	ME[tid].checkParameters(&m[ipix*9]);	
	bestChi2[ipix] = fitOne<T>(ME[tid], fit[tid], &m[9*ipix], &stokes_in[sStride*ipix],
				   &obs[sStride*ipix], sig, nDat, nRandom, niter, chi2_thres, mu, verbose);

	if(tid == 0){
	  per = int((ipix*100.0)/std::max<double>(nPix-1.0, 1.0)+0.5);
	  if(oper < per){
	    oper = per;
	    fprintf(stderr,"\rInvertMany: Processed -> %3ld%s", per,"%");
	  }
	}
      }
      
    }// parallel loop
    fprintf(stderr,"\rInvertMany: Processed -> %3d%s\n", 100,"%");

  }

  // ********************************************************************* //
  
  template<typename T>
  T invert_spatially_regularized(long const nt, long const ny, long const nx, long const ndat,
				 std::vector<ml::Milne<T>> const& ME,  T* __restrict__ m,
				 T* __restrict__ obs, T* __restrict__ syn, T* __restrict__ sig, int const method,
				 int const nIter, T const chi2_thres, T const mu, T const iLam,
				 const T* const __restrict__ alphas,
				 const T* const __restrict__ alphas_time,
				 const T* const __restrict__ betas, 
				 int const delay_bracket)
  {

    // --- Init parameters info array --- //
    
    std::vector<spa::Par<T>> Pinfo;
    for(int ii=0; ii<9;++ii)
      Pinfo.emplace_back(spa::Par<T>(((ii == 2)? true: false), true, ml::pscl<T>[ii],
				     ml::pmin<T>[ii], ml::pmax<T>[ii], alphas[ii], alphas_time[ii], betas[ii]));
  
    
    // --- init container --- //
    
    spa::container<T> cont(nt, ny, nx, mu, ndat, obs, sig, Pinfo, ME);
    

    
    // --- Init inverter class --- //
    
    spa::lms<T,long> fitter(9, nt, ny, nx);


    // --- Fit data --- //
    
    return fitter.fitData(cont, 9, m, syn, nIter, iLam, chi2_thres,  2.e-3,  delay_bracket,  true, method);
  }
  
  
  // ********************************************************************* //

}

#endif

