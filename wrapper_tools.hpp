#ifndef WRAPPERHPP
#define WRAPPERHPP

#include <omp.h>
#include <vector>
#include <iostream>

#include "line.hpp"
#include "Milne.hpp"
#include "lm.hpp"

namespace wr{
  
  template<typename T>
  void SynMany(std::vector<ml::Milne<T>> const& ME,  T* __restrict__ m,
	       T* __restrict__ stokes_in, int const ny, int const nx, T const mu)
  {
    
    // --- Number of threads and pixels --- //
    
    int const nthreads = int(ME.size());
    int const nPix     = nx*ny;
    int const sStride  = 4*ME[0].get_number_of_wavelength();
    
    // --- Parallel loop --- //
    
    int ipix = 0, tid = 0;
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
		 T* __restrict__ stokes_in, T* __restrict__ rf_in, int const ny, int const nx, T const mu)
  {
    
    // --- Number of threads and pixels --- //
    
    int const nthreads = int(ME.size());
    int const nPix     = nx*ny;
    int const sStride  = 4*ME[0].get_number_of_wavelength();
    int const rfStride = sStride*9;
    
    // --- Parallel loop --- //
    
    int ipix = 0, tid = 0;
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
				T* __restrict__ syn, const T* __restrict__ obs, const T* __restrict__ sig, int const nDat,
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
	if(iter != (nRandom-1)){
	  std::memcpy(m,Mref,9*sizeof(T));
	  ml::randomizeParameters(m);
	}else{
	  std::memcpy(m,bestM,9*sizeof(T));
	  ml::randomizeParameters(m, 0.2);
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
		  T* __restrict__ bestChi2, int const ny, int const nx, int nDat, int const nRandom,
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

    int const nPix     = nx*ny;
    int const sStride  = 4*ME[0].get_number_of_wavelength();
    int const rfStride = sStride*9;


    
    // --- Parallel loop --- //
    
        
    int ipix = 0, tid = 0, oper = -1, per = 0;
#pragma omp parallel default(shared) firstprivate(ipix, tid) num_threads(nthreads)  
    {
      tid = omp_get_thread_num();

      //

#pragma omp for schedule(dynamic,2)
      for(ipix = 0; ipix<nPix; ++ipix){
	ME[tid].checkParameters(&m[ipix*9]);	
	bestChi2[ipix] = fitOne<T>(ME[tid], fit[tid], &m[9*ipix], &stokes_in[sStride*ipix],
				   &obs[sStride*ipix], sig, nDat, nRandom, niter, chi2_thres, mu, verbose);

	if(tid == 0){
	  per = (ipix*100)/std::max<int>(nPix-1, 1);
	  if(oper < per){
	    oper = per;
	    fprintf(stderr,"\rInverMany: Processed -> %3d%s", per,"%");
	  }
	}
      }
      
    }// parallel loop
    fprintf(stderr,"\rInverMany: Processed -> %3d%s\n", 100,"%");

  }

  // ********************************************************************* //

  
}

#endif

