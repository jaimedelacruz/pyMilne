#ifndef MILNEHPP
#define MILNEHPP
/* ------------------------------------------------------------
   Milne-Eddington class
   Coded by J. de la Cruz Rodriguez (ISP-SU 2020)
   ------------------------------------------------------------ */
#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <functional>
#include <random>

#include "line.hpp"
#include "fft_tools.hpp"
#include "Milne_internal.hpp"
#include "phyc.h"

namespace ml{

  // *************************************************** //

  template<typename T>
  static constexpr T pmax[9] = {6000., phyc::PI, phyc::PI, 13., 0.200, 120.00, 10.,  10, 10};
  
  template<typename T>
  static constexpr T pmin[9] = {0.,           0,        0,-13., 0.001,   0.010, 1.e-5, 0, -10};
  
  template<typename T>
  static constexpr T pscl[9] = {1000., 1.0, 1.0, 1., 0.05, 10., 0.1, 1.0, 1.0};

  // *************************************************** //

  template<typename T> void randomizeParameters(T* const __restrict__ m, T const scl=1.0)
  {
    static std::uniform_int_distribution<std::mt19937::result_type> rand_dist(0,1.e4);
    static constexpr T norms[9] = {2000, 3.1415926/2, 3.1415926/2, 6, 0.02, 100., 0.5, 0.4, 0.4};
    
    static std::mt19937 rng;
    static bool firsttime = true;
    if(firsttime){
      rng.seed(std::random_device()());
      firsttime = false;
    }
    
    for(int ii=0; ii<9; ++ii){
      T rnum = rand_dist(rng)*1.e-4;
      if(ii == 3) rnum-= 0.5;
      
      m[ii] += norms[ii] * rnum *scl;
    }
    
  }
  
  // *************************************************** //

  template<typename T>
  struct Region{
    int nLambda;
    int idx;
    
    std::vector<double> wav;
    mfft::fftconv1D<T> Degradation;

    Region(): nLambda(0), idx(0), wav(), Degradation(){};
    
    Region(int const nLambda_i, int const idx_i, int const nPSF, const T* __restrict__ psf):
      nLambda(nLambda_i), idx(idx_i), wav(nLambda_i), Degradation(nLambda_i, nPSF, psf){};

    
    ~Region(){};

    inline void Degrade(T* const __restrict__ spectrum)const{
      Degradation.convolve(nLambda, &spectrum[idx]);
    }


    Region<T> &operator=(Region<T> const& in){
      
      nLambda = in.nLambda;
      idx     = in.idx;
      wav     = in.wav;
      
      Degradation = in.Degradation;
      return *this;
    }

    
  };

  // *************************************************** //

  template<typename T>
  class Milne{
  protected:
    std::vector<double> wav;
    std::vector<ln::line<T>> lines;
    std::vector<Region<T>> regions;
    
  public:
    const std::vector<double>       &get_wavelength()const{return wav;}
    const std::vector<ln::line<T>>  &get_lines()const{return lines;}
    const std::vector<Region<T>>    &get_regions()const{return regions;}
    
    // --------------------------------------------------- //

    int get_number_of_wavelength()const{return int(wav.size());}
    
    // --------------------------------------------------- //

    Milne(): wav(), lines(), regions(){};

    // --------------------------------------------------- //
    
    Milne(std::vector<Region<T>> const& regions_in, std::vector<ln::line<T>> const& lines_in): wav(), lines(lines_in), regions(regions_in)
    {

      bool regions_overlap = false;
      
      // --- count total number of wavelength points --- //
      int nWav = 0;
      for(auto &it: regions)
	nWav += int(it.wav.size());

      // --- check if regions overlap --- //

      int const nRegions = regions.size();
      for(int ii=1; ii<nRegions; ++ii){
	int const nLast = regions[ii-1].wav.size()-1;
	if(regions[ii-1].wav[nLast] > regions[ii].wav[0]) regions_overlap = true;
      }

      
      // --- Allocate array --- //
      wav.reserve(nWav);

      
      // --- Store wavelengths as a linear array --- //
      int kk = 0;
      for(auto &it: regions){
	
	it.idx = kk;
	int const inWav = int(it.wav.size());
	kk += inWav;
	
	
	for(int ii=0; ii<inWav; ++ii)
	  wav.push_back(it.wav[ii]);

      }

      // --- normalize the line opacity with the ratio relative to the first line of the list --- //
      T const gf_ref = lines[0].gf;
      for(auto &it: lines){
	it.gf /= gf_ref;
      }

      
      // --- If regions do not overlap, accelerate calculations by restricting the
      // --- calculations for each line 

      int const cnwav = nWav;
      if(!regions_overlap){
	for(auto &it: lines){
	  double const w0 = it.w0 - it.dw;
	  double const w1 = it.w0 + it.dw;

	  it.l0 = 0;
	  it.l1 = nWav-1;
	  
	  for(int ii=0; ii<cnwav; ++ii){
	    if(wav[ii] <= w0) it.l0 = ii;
	    if(wav[ii] <= w1) it.l1 = ii;
	  }
	}
      }else{
	
	// --- For the time being compute the profiles of all lines at all wavelengths --- //
	for(auto &it: lines){
	  it.l0 = 0;
	  it.l1 = nWav-1;
	}
      }

      //for(auto &it: lines){
      //	fprintf(stderr,"%lf -> %d %d\n", it.w0, it.l0, it.l1);
      //}
      
    }
    
    // --------------------------------------------------- //

    Milne<T> &operator=(Milne<T> const& in)
    {
      wav     = in.get_wavelength();
      lines   = in.get_lines();
      regions = in.get_regions();

      return *this;
    }
    
    // --------------------------------------------------- //

    Milne(Milne<T> const& in): Milne(){*this = in;}
    
    // --------------------------------------------------- //

    void synthesize(const T* const __restrict__ m_in, T* const __restrict__ out, T const mu)const
    {
      // --- call external function --- //
      
      ml::ME_Synth(wav, lines, m_in, out, mu);

      
      // --- Degrade Spectra --- //
      
      int const nWav = wav.size();
      for(auto &it: regions)
	for(int ss=0; ss<4; ++ss)
	  it.Degrade(&out[ss*nWav]);
      
    }
    
    // --------------------------------------------------- //
    
    void synthesize_rf(const T* const __restrict__ m_in, T* const __restrict__ out, T* const __restrict__ rf, T const mu)const
    {
      // --- call external function --- //
      
      ml::ME_Synth_RF(wav, lines, m_in, out,rf,  mu);

      
      // --- Degrade Spectra --- //
      
      int const nWav = wav.size();
      for(auto &it: regions)
	for(int ss=0; ss<4; ++ss)
	  it.Degrade(&out[ss*nWav]);

      
      // --- Degrade RFs --- //

      for(int ii=0; ii<9; ++ii)
	for(auto &it: regions)
	  for(int ss=0; ss<4; ++ss)
	    it.Degrade(&rf[ii*4*nWav+ss*nWav]);

      
    }
    
    // --------------------------------------------------- //

    inline static void checkParameters(T* const __restrict__ pars)
    {
      for(int ii=0; ii<9; ++ii)
	pars[ii] = std::min<T>(std::max<T>(pmin<T>[ii], pars[ii]), pmax<T>[ii]);
      
    }
  };
  
  // *************************************************** //

  
  
  
  
}

#endif
