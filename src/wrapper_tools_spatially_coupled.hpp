#ifndef WTSC
#define WTSC

#include "spatially_coupled_helper.hpp"


namespace spa{
  
  //  **************************************************************************************** // 

  double invert_spatially_coupled(double* const __restrict__ im,
				  double* const __restrict__ syn,
				  int const method,
				  int const nIter, double const chi2_thres, double const mu, double const iLam,
				  int const delay_bracket, spa::Data<double,double,long> &dat);
    
  //  **************************************************************************************** // 
  
  void addRegions(spa::Data<double,double,long> &dat,
		  long const ny, long const nx, long const w0, long const w1, long const npy,
		  long const npx, const double* const iPSF, double const clip_thres, const double* const iwav,
		  const double* const sigma, double* const obs, int const nthreads);

  //  **************************************************************************************** // 

  void InitDataContainer(long const ny, long const nx, 
			long const npar, spa::Data<double,double,long> &dat, const double* const alpha,
			std::vector<ml::Milne<double>> &ME, double const mu);

  //  **************************************************************************************** // 


  void init_nData(spa::Data<double,double,long> &dat);

}

#endif

