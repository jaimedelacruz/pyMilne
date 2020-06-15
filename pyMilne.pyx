"""
CYTHON interface for C++ MilneEddington tools.
Author: J. de la Cruz Rodriguez (ISP-SU, 2020)
"""
cimport numpy as np
from numpy cimport ndarray as ar
from numpy import zeros, abs, sqrt, arctan2, where, pi, float32, float64
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string


__author__="J. de la Cruz Rodriguez"
__status__="Developing"


#
# Expose templates
#
cdef extern from "line.hpp" namespace "ln":
    cdef cppclass line[T]:
        line(const T j1, const T j2, const T g1, const T g2, const T  igf, const T lam0, bool anomalous)
ctypedef line[double] lined



cdef extern from "Milne.hpp" namespace "ml":
    cdef cppclass Region[T]:
        vector[double] wav
        Region(int  nLambda_i, int  idx_i, int  nPSF, T* psf)
ctypedef Region[double] Regiond



cdef extern from "Milne.hpp" namespace "ml":
    cdef cppclass Milne[T]:
        vector[double] wav
        Milne(vector[Regiond]& regions_in, vector[lined]& lines_in)
        int get_number_of_wavelength()const
        vector[double] get_wavelength()const
        
ctypedef Milne[double] Milned



cdef extern from "wrapper_tools.hpp" namespace "wr":
    cdef void SynManyd "wr::SynMany<double>"(vector[Milne[double]]& ME, const double* m, double* stokes, int ny, int nx, double mu)
    cdef void SynManyRFd "wr::SynManyRF<double>"(vector[Milne[double]]& ME, const double* m, double* stokes, double* rf, int ny, int nx, double mu)
    cdef void InvertManyd "wr::InvertMany<double>"(vector[Milne[double]]& ME, double* m, double* stokes, double* obs, double* sig, double* chi2, int ny, int nx, int nDat, int nRandom, int niter, double chi2_thres, double mu, bool verbose)

#
# Wrapper cython classes
#


cdef class pyLines:

    cpdef double j1
    cpdef double j2
    cpdef double g1
    cpdef double g2
    cpdef double w0
    cpdef double gf
    cpdef bool anomalous

    def __cinit__(self, j1=0, j2=0, g1=0, g2=0, gf = 1.0, cw=0.0, bool anomalous = True):
        self.j1 = <double>j1
        self.j2 = <double>j2
        self.g1 = <double>g1
        self.g2 = <double>g2
        self.w0 = <double>cw
        self.gf = <double>gf
        self.anomalous = <bool>anomalous


    cpdef setLine(self, int label):
        
        if(label == 6301):
            self.j1 = 2.0; self.j2 = 2.0; self.g1 = 1.84; self.g2 = 1.50; self.w0 = 6301.4995; self.gf = 10.**-0.718
        elif(label == 6302):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.49; self.g2 = 0.0; self.w0 = 6302.4931; self.gf  = 10.0**-0.968
        elif(label == 6173):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.50; self.g2 = 0.0; self.w0 = 6173.3340; self.gf  = 10.0**-2.880
        else:
            print("pyLines::setLine: Error line with label {0 } is not implented".format(label))

    cpdef getj1(self):
        return self.j1
    cpdef getj2(self):
        return self.j2
    cpdef getg1(self):
        return self.g1
    cpdef getg2(self):
        return self.g2    
    cpdef getw0(self):
        return self.w0
    cpdef getgf(self):
        return self.gf
    cpdef getanomalous(self):
        return self.anomalous
    
cdef class pyMilne: 
    cdef vector[Milned] Me
    cdef vector[lined]  lines
    cdef vector[Regiond] regions
    
    def __cinit__(self, list regions, list lines, int nthreads = 1, bool anomalous = True):

        #
        # Get dimensions
        #
        cdef int nregions = len(regions)
        cdef int nlines   = len(lines)
        cdef int ii = 0
        cdef int ww = 0
        
        #
        # Resize vectors
        #
        self.regions.reserve(nregions)
        self.Me.reserve(nthreads)
        self.lines.reserve(nlines)
        
        cdef int nlambda = 0
        cdef int npsf    = 0

        #
        # Init regions
        #
        
        cdef ar[double, ndim=1] tmp
        
        for ii in range(nregions):
            nlambda = <int>regions[ii][0].size
            if(regions[ii][1] is not None):
                nPsf    = <int>regions[ii][1].size
            else:
                nPsf    = 0
            
            tmp = zeros(nPsf,dtype='float64', order='c')
            for ww in range(nPsf):
                tmp[ww] = regions[ii][1][ww]
                
            if(nPsf != 0):
                self.regions.push_back(Regiond(nlambda, <int>0, nPsf, <double*>tmp.data))
            else:
                self.regions.push_back(Regiond(nlambda, <int>0, nPsf, NULL))

            for ww in range(nlambda):
                self.regions[ii].wav[ww] = regions[ii][0][ww]
            
        #
        # Init lines
        #
        for ii in range(nlines):
            self.lines.push_back(lined(lines[ii].getj1(), lines[ii].getj2(), lines[ii].getg1(), lines[ii].getg2(), lines[ii].getgf(), lines[ii].getw0(), lines[ii].getanomalous()))


        #
        # Init one Me object per thread
        #
        for ii in range(nthreads):
            self.Me.push_back(Milned(self.regions, self.lines))

        
    
    def __dealloc__(self):
        print("pyMilne::dealloc: cleaning-up...")
        self.Me.clear()
        self.lines.clear()
        self.regions.clear()
            
        

    def synthesize(self, ar[double,ndim=3] m, double mu = 1.0):

        cdef int nWav = self.Me[0].get_number_of_wavelength()
        
        cdef int ny   = m.shape[0]
        cdef int nx   = m.shape[1]
        cdef int npar = m.shape[2]

        
        cdef ar[double,ndim=4] Stokes = zeros((ny,nx,4,nWav), dtype='float64', order='c')
        
        if(npar != 9):
            print("pyMilne::synthesize: ERROR, npar != 9, returning")
            return Stokes

        SynManyd(self.Me, <double*>m.data, <double*>Stokes.data, ny, nx, mu)
        
        return Stokes
    
    def synthesize_RF(self, ar[double,ndim=3] m, double mu = 1.0):

        cdef int nWav = self.Me[0].get_number_of_wavelength()
        
        cdef int ny   = m.shape[0]
        cdef int nx   = m.shape[1]
        cdef int npar = m.shape[2]

        
        cdef ar[double,ndim=4] Stokes = zeros((ny,nx,4,nWav), dtype='float64', order='c')
        cdef ar[double,ndim=5] RF     = zeros((ny,nx,9,4,nWav), dtype='float64', order='c')
        
        if(npar != 9):
            print("pyMilne::synthesize: ERROR, npar != 9, returning")
            return Stokes

        SynManyRFd(self.Me, <double*>m.data, <double*>Stokes.data, <double*>RF.data,ny, nx, mu)
        
        return Stokes, RF

    def invert(self, ar[double,ndim=3] m, ar[double,ndim=4] obs, ar[double,ndim=2] sig, double mu = 1.0, int nRandom  = 1, int nIter = 20, double chi2_thres = 1.0, verbose = True):

        #
        # Dimensions
        #
        cdef int ny = m.shape[0]
        cdef int nx = m.shape[1]
        cdef int npar = m.shape[2]
        cdef int nwav = obs.shape[3]
        cdef int nwav1= self.Me[0].get_number_of_wavelength()
        cdef int nDat = nwav*4
        
        #
        # Init output arrays 
        #
        cdef ar[double,ndim=4] syn  = zeros((ny,nx,4,nwav), dtype='float64', order='c')
        cdef ar[double,ndim=2] chi2 = zeros((ny,nx),        dtype='float64', order='c')


        #
        # Check dims
        # 
        if(nwav != nwav1):
            print("pyMilne::invert: ERROR, input obs do not have the same number of wavelength than the regions you provided: {0} != {1}".format(nwav, nwav1))
            return m, syn, chi2

        
        #
        # invert pixels
        # 
        InvertManyd(self.Me, <double*>m.data, <double*>syn.data, <double*>obs.data, <double*>sig.data, <double*>chi2.data, ny, nx, nDat, nRandom, nIter, chi2_thres, mu, <bool>verbose)

        
        return m, syn, chi2

    def get_wavelength_array(self):
        
        cdef vector[double] iwav = self.Me[0].get_wavelength()
        cdef ar[double,ndim=1] res = zeros(iwav.size(), dtype='float64')

        cdef int ii = 0
        cdef int nWav = iwav.size()
        for ii in range(nWav):
            res[ii] = iwav[ii]
            
        return res
