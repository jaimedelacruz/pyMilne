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
  static constexpr T pmax[9] = {6000., phyc::PI, phyc::PI, 10., 0.200, 2000.00, 10.,  10, 10};
  
  template<typename T>
  static constexpr T pmin[9] = {0.,           0,        0,-10., 0.001,   0.010, 1.e-5, 0,  0};
  
  template<typename T>
  static constexpr T pscl[9] = {1000., 1.0, 1.0, 1., 0.05, 100., 0.1,1.0, 1.0};

  // *************************************************** //

  template<typename T> void randomizeParameters(T* __restrict__ m, T const scl=1.0)
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

    inline void Degrade(T* __restrict__ spectrum)const{
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

      // --- count total number of wavelength points --- //
      int nWav = 0;
      for(auto &it: regions)
	nWav += int(it.wav.size());

      // --- Allocate array --- //
      wav.reserve(nWav);

      // --- Store wavelengths as a linear array --- //
      int kk = 0;
      for(auto &it: regions){
	
	it.idx = kk;
	int const inWav = int(it.wav.size());
	
	for(int ii=0; ii<inWav; ++ii)
	  wav.push_back(it.wav[ii]);
      }

      // --- normalize the line opacity with the ratio relative to the first line of the list --- //
      T const gf_ref = lines[0].gf;
      for(auto &it: lines){
	it.gf /= gf_ref;
      }

      
      // --- For the time being compute the profiles of all lines at all wavelengths --- //
      for(auto &it: lines){
	it.l0 = 0;
	it.l1 = nWav-1;
      }
			 
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

    void synthesize(const T* __restrict__ m_in, T* __restrict__ out, T const mu)const
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
    
    void synthesize_rf(const T* __restrict__ m_in, T* __restrict__ out, T* __restrict__ rf, T const mu)const
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

    inline static void checkParameters(T* __restrict__ pars)
    {
      for(int ii=0; ii<9; ++ii)
	pars[ii] = std::min<T>(std::max<T>(pmin<T>[ii], pars[ii]), pmax<T>[ii]);
      
    }
  };
  
  // *************************************************** //

  
  
  
  
}

#endif
