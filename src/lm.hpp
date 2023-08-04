#ifndef LMHPP
#define LMHPP
/* -------------------------------------------------------
   
   Levenberg Marquardt algorithm
   Coded by J. de la Cruz Rodriguez (ISP-SU, 2020)
   
   ------------------------------------------------------- */

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

#include "Milne.hpp"

namespace lm{
  
  // -- Simple Gaussian elimination implemented in the header file as template for inlining in the code --- //
  
  template<typename T>
  inline void SolveLinearGauss(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>  &A,
			       Eigen::Matrix<T,Eigen::Dynamic,1> &B)
  {
    
    
    // --- Simple Gaussian elimination with partial pivoting --- //
    
    // A is the matrix with coeffs that multiply X. B is the right hand
    // term (on input). The result is overwritten into B. All operations
    // are done in-place
    
    // The algorithm is a pretty standard Gauss-Jordan algorithm, here optimized a
    // bit for performance.

    //Eigen::Matrix<T,Eigen::Dynamic,1> B_copy = B;
    //Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> A_copy = A;

    int const N = int(B.size());
    int maxrow = 0, swapme = 0;
    
    
    for (int i=0; i<N; ++i) {
      
      // Find pivot
      
      T maxel = std::abs(A(i,i));
      maxrow = i, swapme = 0;
      
      for (int k=i+1; k<N; ++k){
	T  tmp = std::abs(A(k,i));
	if(tmp > maxel){
	  maxel = tmp;
	  maxrow = k, swapme = 1;
	}
      }
      
      // swap
      if(swapme){
	for (int k=i; k<N;++k)
	  std::swap(A(maxrow,k),A(i,k));
	
	std::swap(B[maxrow],B[i]);
      }
      
      // Set to zero relevant columns
      
      T const Aii = -A(i,i);
      
      for (int k=i+1; k<N; ++k){
	T const tmp = A(k,i) / Aii;
	A(k,i) = T(0);
	for (int j=i+1; j<N; ++j) {
	  A(k,j) += tmp * A(i,j);
	}
	B[k] += tmp*B[i];
      }
    }
    
    
    // Solve upper triagonal system and store in-place in B

    for (int i=N-1; i>=0; --i) {
      B[i] /= A(i,i);
      for (int k=i-1;k>=0; --k) {
	B[k] -= A(k,i) * B[i];
      }
    }
  }
  
  // ***************************************** //

  template<typename T>
  struct Par{
    bool isCyclic;
    bool limited;
    T scale;
    T limits[2];

    Par(): isCyclic(false), limited(false), scale(1.0), limits{0,0}{};
    Par(bool const cyclic, bool const ilimited, T const scal, T const mi, T const ma):
      isCyclic(cyclic), limited(ilimited), scale(scal), limits{mi,ma}{};

    Par(Par<T> const& in): isCyclic(in.isCyclic) ,limited(in.limited), scale(in.scale), limits{in.limits[0], in.limits[1]}{};

    Par<T> &operator=(Par<T> const& in)
    {
      isCyclic = in.isCyclic, limited = in.limited, scale=in.scale, limits[0]=in.limits[0], limits[1]=in.limits[1];
      return *this;
    }

    inline void Normalize(T &val)const{val /= scale;};
    
    inline void Scale(T &val)const{val *= scale;};
    
    inline void Check(T &val)const{
      if(!limited) return;
      if(isCyclic){
	if(val > limits[1]) val -= 3.1415926f;
	if(val < limits[0]) val += 3.1416026f;
      }
      val = std::max<T>(std::min<T>(val, limits[1]),limits[0]);
    }
    
    inline void CheckNormalized(T &val)const{
      if(!limited) return;
      Scale(val);
      Check(val);
      Normalize(val);
    }
    
  };
  
  // ***************************************** //

  template<typename T>
  struct container{
    int const nDat;
    T const mu;
    int Nreal;
    const ml::Milne<T>& Me;
    const T* __restrict__ d;
    const T* __restrict__ sig;
    const std::vector<Par<T>> &Pinfo;

    container(int const nd, T const imu, ml::Milne<T> const& iMe, const T* __restrict__ din, const T* __restrict__ sigin, const std::vector<Par<T>> &Pi): nDat(nd), mu(imu), Nreal(1), Me(iMe), d(din), sig(sigin), Pinfo(Pi)
    {
      // --- only account for non-dummy points in the data array --- //
      Nreal = 0;
      for(int ii = 0; ii<nDat; ++ii)
	if(sig[ii] < 1.e20) Nreal += 1;
      
    }
    
  };

  // ***************************************** //

  template<typename T> constexpr inline T SQ(T const v){return v*v;}
  
  // ***************************************** //

  template<typename T>
  T getChi2(int const nDat, const T* __restrict__ r)
  {
    double sum = 0.0;
    int const nDat4 = nDat/4;
    
    if(nDat4*4 == nDat){
      double sumI = 0.0;
      double sumQ = 0.0;
      double sumU = 0.0;
      double sumV = 0.0;
      
      for(int ii=0; ii<nDat4; ++ii){
	sumI += SQ(r[ii*4+0]);
	sumQ += SQ(r[ii*4+1]);
	sumU += SQ(r[ii*4+2]);
	sumV += SQ(r[ii*4+3]);
      }
      sum = sumI + (sumQ + sumU + sumV);
    }else{

      for(int ii=0; ii<nDat;++ii)
	sum += SQ(r[ii]);
    }

    return static_cast<T>(sum);
  }

  // ***************************************** //

  template<typename T>
  T fx(container<T> const& myData, int const nPar, const T* __restrict__ m_in, T* __restrict__ syn, T* __restrict__ r)
  {


    // --- Copy model --- //

    T* __restrict__ m = new T [nPar]();
    std::memcpy(m,m_in,nPar*sizeof(T));


    // --- /// 
    
    int const nDat = myData.nDat;
    const T* __restrict__ dat = myData.d;
    const T* __restrict__ sig = myData.sig;
    
    // --- Scale up model --- //

    for(int ii=0; ii<nPar; ++ii){
      myData.Pinfo[ii].Scale(m[ii]);
    }

    
    // --- calculate spectrum --- //

    myData.Me.synthesize(m, syn, myData.mu);
    

    // --- calculate residue --- //

    T const scl = sqrt(T(myData.Nreal)); 
    for(int ii=0; ii<nDat; ++ii) r[ii] = (dat[ii] - syn[ii]) / (sig[ii] * scl);



    delete [] m;
    
    // --- get Chi2 --- //
    
    return getChi2<T>(nDat, r);  
  }
    // ***************************************** //

  template<typename T>
  T fx_dx(container<T> const& myData, int const nPar, const T* __restrict__ m_in, T* __restrict__ syn, T* __restrict__ r, T* __restrict__ J)
  {

    // --- Copy model --- //

    T* __restrict__ m = new T [nPar]();
    std::memcpy(m,m_in,nPar*sizeof(T));
    
    int const nDat = myData.nDat;
    const T* __restrict__ dat = myData.d;
    const T* __restrict__ sig = myData.sig;
    
    // --- Scale up model --- //

    for(int ii=0; ii<nPar; ++ii){
      myData.Pinfo[ii].Scale(m[ii]);
    }

    
    // --- calculate spectrum --- //

    myData.Me.synthesize_rf(m, syn, J, myData.mu);
    

    
    // --- calculate residue --- //

    T const scl = sqrt(T(myData.Nreal)); 
    for(int ii=0; ii<nDat; ++ii) r[ii] = (dat[ii] - syn[ii]) / (sig[ii] * scl);


    // --- scale J --- //

    for(int ii = 0; ii<nPar; ++ii){

      T const iScl = myData.Pinfo[ii].scale / scl;
      
      for(int ww = 0; ww<nDat; ++ww)
	J[ii*nDat + ww] *=  iScl / sig[ww];
    }
    

    // --- clean up model array --- //
    delete [] m;

    
    // --- get Chi2 --- //
    
    return getChi2<T>(nDat, r);  
  }

  // **************************************** // 
  
  template<typename T>
  struct LevMar{
    int nPar;
    std::vector<T> diag;
    std::vector<Par<T>> Pinfo;

    void set(int const& iPar){
      nPar = iPar;
      diag = std::vector<T>(iPar,0.0);
      Pinfo = std::vector<Par<T>>(iPar, Par<T>());
    }
    
    LevMar(): nPar(0), diag(), Pinfo(){};
    LevMar(int const &nPar_i):LevMar(){set(nPar_i);}

    LevMar(LevMar<T> const& in): LevMar(){nPar = in.nPar, diag=in.diag, Pinfo = in.Pinfo;}
    
    LevMar<T> &operator=(LevMar<T> const& in){nPar = in.nPar, diag=in.diag(), Pinfo = in.Pinfo; return *this;}

    static inline T checkLambda(T val, T const &mi, T const& ma){return std::max<T>(std::min<T>(ma, val), mi);}


    // ------------------------------------------------------------------------------ //

    T getCorrection(container<T> const& myData, T* __restrict__ m, const T* __restrict__ J, T* __restrict__ syn, T* __restrict__ r, T const iLam)const
    {
      
      // --- Simplify the notation --- //
      
      using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
      

      // --- Define some quantities --- //

      int const nDat = myData.nDat;
      int const cPar = nPar;

      Mat A(cPar, cPar); A.setZero();
      Vec B(cPar); B.setZero();
      
      
      // --- get Hessian matrix --- //
       
      for(int jj = 0; jj<cPar; ++jj){

	// --- Compute left-hand side of the system --- //
	
	for(int ii=0; ii<=jj; ++ii){
	  double sum = 0.0;
	  
	  for(int ww=0; ww<nDat; ++ww) sum += J[jj*nDat + ww]*J[ii*nDat + ww];
	  A(jj,ii) = A(ii,jj) = static_cast<T>(sum);
	}//ii
	
		
	// --- Compute right-hand side of the system --- //
	
	double sum = 0.0;
	for(int ww = 0; ww<nDat; ww++) sum += J[jj*nDat+ww] * r[ww];
	B[jj] = static_cast<T>(sum); 
	
	
	A(jj,jj) *= 1.0+iLam;
      } // jj

      
      // --- Solve linear system to get solution --- //
      
      //Eigen::BDCSVD<Mat> sy(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::ColPivHouseholderQR<Mat> sy(A); // Also rank revealing but much faster than SVD
      sy.setThreshold(1.e-9);
      
      Vec dm = sy.solve(B);
      //dm = B;
      //SolveLinearGauss(A, B);
      
      
      
      // --- add to model and check parameters --- //

      for(int ii =0; ii<cPar; ++ii){
	m[ii] += dm[ii];
	Pinfo[ii].CheckNormalized(m[ii]);
      }   
      
      return fx<T>(myData, nPar, m, syn, r);
    }
    
    // ------------------------------------------------------------------------------ //
    
    T getStep(container<T> const& myData, T* __restrict__ m, const T* __restrict__ J,
	      T* __restrict__ syn, T* __restrict__ r, T &iLam, bool braket, T const maxLam,
	      T const minLam)const{

      // if(!braket){
	return getCorrection(myData, m, J, syn, r, iLam);
	// }
      
      
    }
    
    // ------------------------------------------------------------------------------ //

    
    T fitData(ml::Milne<T> const& Me,  int const nDat, const T* __restrict__ dat, T* __restrict__ syn, const T* __restrict__ sig, T* __restrict__ m, T const mu, int const max_iter = 20, T iLam = sqrt(10.0f), T const Chi2_thres = 1.0, T const fx_thres = 2.e-3, int const delay_braket = 2, bool verbose = true)const
    {
      static constexpr T const facLam = 3.1622776601683795;
      static constexpr T const maxLam = 100*facLam;
      static constexpr T const minLam = 1.e-4;
      static constexpr int const max_n_reject = 6;
      
      // --- Init container --- //

      container<T> myData(nDat, mu, Me, dat, sig, Pinfo);


      
      // --- Check initial Lambda value --- //
      
      iLam = checkLambda(iLam, minLam, maxLam);
      

      // --- Init temp arrays and values--- //
      
      int const cPar = nPar;
      T bestChi2     = 1.e32;
      T     Chi2     = 1.e32;
      
      T* __restrict__ bestModel  = new T [cPar]();
      T* __restrict__ bestSyn    = new T [nDat]();      

      T* __restrict__     J      = new T [cPar*nDat]();
      T* __restrict__     r      = new T [nDat]();


      
      // --- Work with normalized quantities --- //

      for(int ii =0; ii<cPar; ++ii){
	Pinfo[ii].Check(m[ii]);
	Pinfo[ii].Normalize(m[ii]);
      }
      
      std::memcpy(bestModel,  m, cPar*sizeof(T));

      
      // --- get derivatives and init Chi2 --- //

      bestChi2 = fx_dx<T>(myData, nPar, bestModel, bestSyn, r, J);

      
      // --- Init iteration --- //

      if(verbose){
	fprintf(stderr, "\nLevDer::fitData: [Init] Chi2=%13.5f\n", bestChi2);
      }


      
      // --- Iterate --- //

      int iter = 0, n_rejected = 0;
      bool quit = false, tooSmall = false;
      T oLam = 0, dfx = 0;

      while(iter < max_iter){
	
	oLam = iLam;
	std::memcpy(m, bestModel, nPar*sizeof(T));

	
	// --- Get model correction --- //

	Chi2 = getStep(myData, m, J, syn, r, iLam, false, minLam, maxLam);


	// --- Did Chi2 improve? --- //

	if(Chi2 < bestChi2){

	  oLam = iLam;
	  dfx = (bestChi2 - Chi2) / bestChi2;
	  
	  bestChi2 = Chi2;
	  std::memcpy(bestModel,   m, cPar*sizeof(T));
	  std::memcpy(bestSyn  , syn, nDat*sizeof(T));

	  if(iLam > 1.0001*minLam)
	    iLam = checkLambda(iLam/facLam, minLam, maxLam);
	  else
	    iLam = 10*minLam;
	    
	  if(dfx < fx_thres){
	    if(tooSmall) quit = true;
	    else tooSmall = true;
	  }
	  
	  n_rejected = 0;
	}else{
	  
	  // --- Increase lambda and re-try --- //
	  
	  iLam = checkLambda(iLam*SQ<T>(facLam), minLam, maxLam);
	  n_rejected += 1;
	  if(verbose)
	    fprintf(stderr,"LevMar::fitData: Chi2=%13.5f > %13.5f -> Increasing lambda %f -> %f\n", Chi2, bestChi2, oLam, iLam);
	  
	  if(n_rejected<max_n_reject) continue;

	}
	
	// --- Check what has happened with Chi2 --- //
	
	if(n_rejected >= max_n_reject){
	  if(verbose)
	    fprintf(stderr, "LevMar::fitData: maximum number of rejected iterations reached, finishing inversion");
	  break;
	}

	if(verbose)
	  fprintf(stderr, "LevMar::fitData [%3d] Chi2=%13.5f, lambda=%e\n", iter, Chi2, oLam);

	if(bestChi2 < Chi2_thres){
	  if(verbose)
	    fprintf(stderr, "LevMar::fitData: Chi2 (%f) < Chi2_threshold (%f), finishing inversion", bestChi2, Chi2_thres);
	  break;
	}

	if(quit){
	  if(verbose)
	    fprintf(stderr, "LevMar::fitData: Chi2 improvement too small for 2-iterations, finishing inversion\n");
	  break;
	}
	
	iter++;
	if(iter >= max_iter){
	  break;
	}

	// --- compute gradient of the new model for next iteration --- //

	std::memcpy(m, bestModel, cPar*sizeof(T));
	fx_dx<T>(myData, nPar, m, syn, r, J);
      }
      
      std::memcpy(m, bestModel, cPar*sizeof(T));
      std::memcpy(syn, bestSyn, nDat*sizeof(T));

      for(int ii=0; ii<cPar; ++ii)
	Pinfo[ii].Scale(m[ii]);
      
      
      // --- Clean-up --- //

      delete [] bestModel;
      delete [] bestSyn;
      delete [] r;
      delete [] J;


      return bestChi2;
    }
    
  };
  
}


#endif
