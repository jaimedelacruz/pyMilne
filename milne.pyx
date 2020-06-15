"""
CYTHON interface for C++ classes from the cmilne module.
Author: J. de la Cruz Rodriguez (ISP-SU, 2016)

Included MODULES: pyMILNE, pyCdegrade

"""

from libcpp.vector cimport vector
from libcpp.string cimport string
import cython
cimport numpy as np
from numpy cimport ndarray as ar
from numpy import empty, ascontiguousarray, zeros
from libc.stdlib cimport malloc, free, atoi
from libc.string cimport strlen, strcpy,memset, memcpy
from libcpp cimport bool
import os
import sys

__author__="Jaime de la Cruz Rodriguez (ISP-SU 2016)"
__status__="Testing"
__email__="jaime@astro.su.se"

cdef extern from "line.h":
  cppclass "line<float>" line:
    double w0;
    int nZ;
    float gf;
    vector[float]  strength, splitting;
    vector[int] iL;
    line();
    line(const float &j1, const float &j2, const float &g1, const float &g2, const float &igf, const float &lam0, bool anomalou);
    

cdef extern from "milne.h":
  cppclass milne:
    
    
# ---------------------------------------
# Interface for ME routines
# ---------------------------------------

cdef class pyMILNE:
    """
    Class pyMILNE
    
    Purpose: Computes synthetic spectral profiles assuming a Milne-Eddington atmosphere
    Model struct: Each atmosphere consists of 9 or 10 (optional) parameters in the following order.
         par[0] -> B [Gauss]
         par[1] -> inclination [rad]
         par[2] -> azimuth [rad]    
         par[3] -> vlos [km/s]
         par[4] -> Doppler width [\AA]
         par[5] -> line opacity [includes a factor related to the Doppler width]
         par[6] -> Damping parameter [in units of Doppler width]
         par[7] -> S0
         par[8] -> S1


    IMPORTANT NOTE: All models passed to these routines MUST BE contiguous in memory, assuming C ordering of elements.
          Something like this: 
              model = model.astype('float64', order = 'c')              

          ...or if we create a new array:
              model = numpy.zeros((ny, nx, npar), order='c', dtype='float64')


    Example:
         To call the constructor:
              me = pyMILNE.pyMILNE('FeI_6302', [6302.068800, 0.0215, 56], mu = 1.0, nthreads = 5, lines = 'lines.cfg', psf = psf)

         To synthesize profiles:
              stokes = me.synth(model.reshape(ny * nx, npar))


         To synthesize response functions (analytically):
              rf = me.responseFunction(model.reshape(ny * nx, npar))

         To synthesize the profiles and response functions (analytically):
              stokes, rf = me.synth_rf(model.reshape(ny * nx, npar))


         To clip (IN-PLACE) the parameters of a model cube to the min/max accepted values:
              model = model.reshape(ny*nx, npar)
              me.checkParameters(model)    
    
    
    Modifications: 

        2015-11-26: JdlCR - Added LM pixel-to-pixel inversion.

        2015-12-22: JdlCR - Added analytical response functions to the cmilne module, 
                            and adapted the python interface accordingly. Removed
                            vmac, and added intrumental PSF.

        2016-05-17: JdlCR - Removed FFT based convolutions, it is better to not depend on 
                            external libraries.

    """
    cdef vector[cmilne*] at
    cdef iput_t input
    cdef int nt
    cdef public ar wav
    cdef int nlambda
    cdef public ar pscal
    
    def __cinit__(self, bytes linetag, list region, double mu = 1.0, int nthreads=1, str lines= 'lines.cfg', psf = []):
        self.nt = nthreads
        self.at.resize(self.nt)
        cdef int ii

        
        # Init lines
        
        self.input.ilines.push_back(string(linetag))
        self.input.mu = <double>mu
        cdef string dum = lines
        read_lines(dum, self.input, 0)


        
        # add region
        
        self.input.regions.resize(1)
        self.input.regions[0].w0 = inv_convl(<double>region[0])
        self.input.regions[0].dw = <double>region[1]
        self.input.regions[0].nw = <int>region[2]
        self.input.regions[0].cscal = 1.0
        print( "pyMILNE::__cinit__: region -> w0={0}, dw={1}, nw={2}".\
          format(self.input.regions[0].w0, self.input.regions[0].dw, self.input.regions[0].nw))

          
        # PSF ?
          
        cdef int npsf = len(psf)
        cdef ar[double,ndim=1] dpsf
        if(npsf >= 1): dpsf = ascontiguousarray(psf, dtype='float64')
            
        # Init array of solvers, one element per thread
        
        for ii in range(self.nt):
            if(npsf>0):
                self.at[ii] = new cmilne(self.input, npsf, &dpsf[0])
            else:
                self.at[ii] = new cmilne(self.input, 0, NULL)
        print("pyMILNE::__cinit__: Initialized {0} threads".format(self.nt))
        
        
        # Copy lambda array
        
        self.nlambda = <int>self.at[0].nlambda       
        self.wav = zeros(self.nlambda, dtype='float64')
        for ii in range(self.nlambda):
            self.wav[ii] = convl(self.at[0].wav[ii])


            
        # Start scaling of parameters from the cmilne class
        # to be consistent
        
        self.pscal = empty((9), dtype='float64', order='c')
        for ii in range(9):
            self.pscal[ii] = <double>self.at[0].pscal[ii]
        
