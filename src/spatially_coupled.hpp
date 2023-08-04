#ifndef SPAT2HPP
#define SPAT2HPP

// -------------------------------------------------------
//   
//   Spatially-coupled Levenberg Marquardt algorithm
//   Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
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

  using iType = long;
  
  template<typename T, typename U = T>
  class lms2{
  protected:
    iType npar, ny, nx;
    Eigen::SparseMatrix<U,Eigen::RowMajor, iType> L;
    Eigen::SparseMatrix<U,Eigen::RowMajor, iType> LL;

  public:
    lms(iType const inpar, iType const iny, iType const inx):
      npar(inpar), ny(iny), nx(inx), L(), LL(){};
    
    
    
    
  };
  
  
  
}


#endif
