#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cstdio>

#include "spatially_coupled_helper.hpp"
#include "spatially_coupled_tools.hpp"
#include "wrapper_tools_spatially_coupled.hpp"
#include "Milne.hpp"
#include "lm_sc.hpp"

using namespace spa;

//  **************************************************************************************** // 

void spa::InitDataContainer(long const ny, long const nx, 
			    long const npar, spa::Data<double,double,long> &dat, const double* const alpha,
			    std::vector<ml::Milne<double>> &ME, double const mu)
{
  using T = double;

  
  // --- Init parameters info array --- //
  
  dat.Pinfo.clear();

  
  for(int ii=0; ii<npar;++ii)
    dat.Pinfo.emplace_back(spa::Par<T>(((ii == 2)? true: false), true, ml::pscl<T>[ii],
				       ml::pmin<T>[ii], ml::pmax<T>[ii], alpha[ii], T(0), T(0)));

  
  // --- Fill other quantities --- //

  dat.ny = ny, dat.nx = nx, dat.npar = npar, dat.ns = 4, dat.nw = ME[0].get_number_of_wavelength();
  dat.mu = mu;

  dat.sig_total.resize(dat.nw*dat.ns);

  
  
  // --- Copy ME pointers to dat.me array --- //

  int const nthreads = ME.size();
  dat.me.resize(nthreads);
  
  for(int ii=0;ii<nthreads; ++ii)
    dat.me[ii] = &ME[ii];

  
}

//  **************************************************************************************** // 

void spa::addRegions(spa::Data<double,double,long> &dat,
		long const ny, long const nx, long const w0, long const w1, long const npy,
		long const npx, const double* const iPSF, double const clip_thres, const double* const iwav,
		const double* const sigma, double* const obs, int const nthreads)
{
  static bool firsttime = true;
  
  if(firsttime && (nthreads > 1)){
    Eigen::initParallel();
    Eigen::setNbThreads(nthreads);
    firsttime = false;
  }
  
  dat.regions.emplace_back(std::make_shared<spa::SpatRegion<double,long>>(dat.ny,dat.nx,ny,nx,w0,w1,0,ny-1,0,nx-1,clip_thres,iwav,
									  sigma, npy,npx,iPSF,obs,nthreads));

  // --- copy sigma to sig_total --- //

  long const nw = dat.nw;
  long const ns = dat.ns;
  long const nw1 = w1-w0+1;
  
  for(long ss=0; ss<ns; ++ss){
    long const off = nw*ss;
    for(long ww=w0; ww<=w1; ++ww){
      dat.sig_total[off+ww] = sigma[nw1*ss+ww-w0];
    }
  }
  
}

//  **************************************************************************************** // 

void spa::init_nData(spa::Data<double,double,long> &dat)
{

  static constexpr const double sig_max = 1000.;
  constexpr static const long  ns = 4;

  // --- init nData in Data struct, first pixels --- //

  long nData = 0;
  
  for(auto &it: dat.regions)
    nData += it->ny*it->nx;
    
  
  
  // --- Wavelength points --- //
  
  long valid_wav = 0;
  long const npoints = dat.sig_total.size();

  for(long ii=0; ii<npoints; ++ii)
    if(dat.sig_total[ii] < sig_max)
      valid_wav += 1; 
    
  dat.nData = sqrt(double(nData*valid_wav));

  for(long ii=0; ii<npoints; ++ii)
    if(dat.sig_total[ii] < sig_max)
      dat.sig_total[ii] = 1.0 / (dat.nData * dat.sig_total[ii]);
    else
      dat.sig_total[ii] = 0.0;


  for(auto &it: dat.regions){
    long const w0 = it->wl;
    long const w1 = it->wh;
    long const nw = w1-w0+1;
    
    
    for(long ss=0; ss<ns; ++ss){
      for(long ww=0; ww<nw; ++ww){
	if(it->sig(ss,ww) < sig_max)
	  it->sig(ss,ww) = 1.0 / (it->sig(ss,ww)*dat.nData);
	else
	  it->sig(ss,ww) = 0.0;
      }
    }

  }
}

//  **************************************************************************************** // 

double spa::invert_spatially_coupled(double* const __restrict__ im,
				     double* const __restrict__ isyn,
				     int const method,
				     int const nIter, double const chi2_thres, double const mu, double const iLam,
				     int const delay_bracket, spa::Data<double,double,long> &dat)
{  
  mem::Array<double,3> m(im, dat.ny, dat.nx, dat.npar);
  mem::Array<double,4> syn(isyn, dat.ny, dat.nx, dat.ns, dat.nw);

  spa::LMsc<double,double,long> Inverter(dat.npar,dat.ny,dat.nx);



  spa::Chi2_t<double> chi2 = Inverter.invert(dat, m, syn, nIter, 1.e-3, 1, iLam);
  
  return chi2.get();
}


//  **************************************************************************************** // 
