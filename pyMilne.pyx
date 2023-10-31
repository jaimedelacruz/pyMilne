"""
CYTHON interface for C++ MilneEddington tools.
Author: J. de la Cruz Rodriguez (ISP-SU, 2023)
"""
cimport numpy as np
from numpy cimport ndarray as ar
from numpy import zeros, abs, sqrt, arctan2, where, pi, float32, float64, ndarray
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string


__author__="J. de la Cruz Rodriguez"
__status__="Developing"


#
# Expose templates
#

# ********************************************************************************

cdef extern from "line.hpp" namespace "ln":
    cdef cppclass line[T]:
        line(const T j1, const T j2, const T g1, const T g2, const T  igf, const double lam0, bool anomalous, const T dw)
ctypedef line[double] lined
ctypedef line[float] linef

# ********************************************************************************

cdef extern from "Milne.hpp" namespace "ml":
    cdef cppclass Region[T]:
        vector[double] wav
        Region(int  nLambda_i, int  idx_i, int  nPSF, T* psf)
        #Region(Region[T] &in)
       # Region[T] &operator=(Region[T] &in)
        
ctypedef Region[double] Regiond
ctypedef Region[float] Regionf

# ********************************************************************************

cdef extern from "Milne.hpp" namespace "ml":
    cdef cppclass Milne[T]:
        vector[double] wav
        Milne(vector[Region[T]]& regions_in, vector[line[T]]& lines_in)
        int get_number_of_wavelength()const
        vector[double] get_wavelength()const
        
ctypedef Milne[double] Milned
ctypedef Milne[float] Milnef

# ********************************************************************************

cdef extern from "spatially_coupled_helper.hpp" namespace "spa":
    cdef cppclass Data[T,U,ind_t]:
         vector[Milne[T]] me
         void reset_regions();
         Data();

ctypedef Data[double,double,long] Data_d


# ********************************************************************************

cdef extern from "wrapper_tools.hpp" namespace "wr":
    cdef void SynManyd "wr::SynMany<double>"(vector[Milne[double]]& ME, const double* m, double* stokes, long ny, long nx, double mu)
    cdef void SynManyRFd "wr::SynManyRF<double>"(vector[Milne[double]]& ME, const double* m, double* stokes, double* rf, long ny, long nx, double mu)
    cdef void InvertManyd "wr::InvertMany<double>"(vector[Milne[double]]& ME, double* m, double* stokes, double* obs, double* sig, double* chi2, long ny, long nx, long nDat, int nRandom, int niter, double chi2_thres, double mu, bool verbose)

    cdef void SynManyf "wr::SynMany<float>"(vector[Milne[float]]& ME, const float* m, float* stokes, long ny, long nx, float mu)
    cdef void SynManyRFf "wr::SynManyRF<float>"(vector[Milne[float]]& ME, const float* m, float* stokes, float* rf, long ny, long nx, float mu)
    cdef void InvertManyf "wr::InvertMany<float>"(vector[Milne[float]]& ME, float* m, float* stokes, float* obs, float* sig, float* chi2, long ny, long nx, long nDat, int nRandom, int niter, float chi2_thres, float mu, bool verbose)

    
    cdef float invert_spatially_regularized_float "wr::invert_spatially_regularized<float>"(long nt, long ny, long nx, long  ndat, vector[Milne[float]] &ME,  float*  m, float* obs, float* syn, float*  sig, int method, int nIter, float chi2_thres, float  mu, float iLam,  float*  alphas,  float*  alphas_time, int  delay_bracket)
    
    cdef double invert_spatially_regularized_double "wr::invert_spatially_regularized<double>"(long nt, long ny, long nx, long  ndat, vector[Milne[double]] &ME,  double*  m, double* obs, double* syn, double*  sig, int method, int nIter, double chi2_thres, double  mu, double iLam,  double*  alphas, double*  alphas_time, int delay_bracket)

# ********************************************************************************

cdef extern from "wrapper_tools_spatially_coupled.hpp" namespace "spa":
     cdef void addRegions(Data_d &dat, long  ny, long  nx, long  w0, long  w1, long  npy, long  npx, const double* const iPSF, double clip_thres, const double* const iwav, const double* const sigma, double* const obs, int nthreads)
     
     cdef void InitDataContainer(long  ny, long  nx, long  npar, Data_d &dat, const double* const alpha, \
		                 vector[Milne[double]] &ME, double mu)
     
     cdef void addRegions(Data_d &dat, \
		          long  ny1, long  nx1, long  w0, long  w1, long  npy, long  npx, const double* const iPSF, \
                          double  clip_thres, const double* const iwav, \
		          const double* const sigma, double* const obs, int nthreads);

     cdef double invert_spatially_coupled(double* const im, \
				          double* const syn, \
				          int  method, \
				          int nIter, double chi2_thres, double  mu, double  iLam, \
				          int  delay_bracket, Data_d &dat);

     cdef void init_nData(Data_d &dat);

#
# Wrapper cython classes
#


# ******************************************************************************************************



cdef class pyLines:

    cdef double j1
    cdef double j2
    cdef double g1
    cdef double g2
    cdef double w0
    cdef double gf
    cdef bool anomalous
    cdef double dw

    def __cinit__(self, j1=0, j2=0, g1=0, g2=0, gf = 1.0, cw=0.0, bool anomalous = True, double dw = 20):
        self.j1 = <double>j1
        self.j2 = <double>j2
        self.g1 = <double>g1
        self.g2 = <double>g2
        self.w0 = <double>cw
        self.gf = <double>gf
        self.anomalous = <bool>anomalous
        self.dw = <double> <double>dw


    cpdef setLine(self, int label):
        
        if(label == 6301):
            self.j1 = 2.0; self.j2 = 2.0; self.g1 = 1.84; self.g2 = 1.50; self.w0 = 6301.4995; self.gf = 10.**-0.718; self.dw = 20
        elif(label == 6302):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.49; self.g2 = 0.0; self.w0 = 6302.4931; self.gf  = 10.0**-0.968; self.dw = 20
        elif(label == 6173):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.50; self.g2 = 0.0; self.w0 = 6173.3340; self.gf  = 10.0**-2.880; self.dw = 20
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
    cpdef getDw(self):
        return self.dw
    
#
# ******************************************************************************************************
#


cdef class pyLinesf:

    cdef float j1
    cdef float j2
    cdef float g1
    cdef float g2
    cdef double w0
    cdef float gf
    cdef bool anomalous
    cdef float dw
    
    
    def __cinit__(self, j1=0, j2=0, g1=0, g2=0, gf = 1.0, cw=0.0, bool anomalous = True, float dw = 20):
        self.j1 = <float>j1
        self.j2 = <float>j2
        self.g1 = <float>g1
        self.g2 = <float>g2
        self.w0 = <double>cw
        self.gf = <float>gf
        self.anomalous = <bool>anomalous
        self.dw = <float> <float>dw


    cpdef setLine(self, int label):
        
        if(label == 6301):
            self.j1 = 2.0; self.j2 = 2.0; self.g1 = 1.84; self.g2 = 1.50; self.w0 = 6301.4995; self.gf = 10.**-0.718; self.dw = 20
        elif(label == 6302):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.49; self.g2 = 0.0; self.w0 = 6302.4931; self.gf  = 10.0**-0.968; self.dw = 20
        elif(label == 6173):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.50; self.g2 = 0.0; self.w0 = 6173.3340; self.gf  = 10.0**-2.880; self.dw = 20
        else:
            print("pyLinesf::setLine: Error line with label {0 } is not implented".format(label))

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
    cpdef getDw(self):
        return self.dw
    
#
# ******************************************************************************************************
#
cdef addRegionToClass(Data_d &dat, ar[double,ndim=4] obs, ar[double,ndim=2] psf, ar[double,ndim=2] sigma, \
                      int woff, vector[double] wav, double clip_threshold = 1.0, int nthreads = 1):

    cdef long ny1 = obs.shape[0]
    cdef long nx1 = obs.shape[1]
    cdef long ns = obs.shape[2]
    cdef long nw = obs.shape[3]
    cdef long npy = psf.shape[0]
    cdef long npx = psf.shape[1]
    
    addRegions(dat, ny1, nx1, woff, woff+nw-1, npy, npx, <double*>psf.data, \
               clip_threshold, <double*>&wav[0], <double*>sigma.data, <double*>obs.data, nthreads);
    
    return nw
    
#
# ******************************************************************************************************
#

cdef class pyMilne: 
    cdef vector[Milned] Me
    cdef vector[lined]  lines
    cdef vector[Regiond] regions
    cdef int dtype
    
    def __cinit__(self, list regions, list lines, int nthreads = 1, bool anomalous = True):
        
        self.dtype = 8
        
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
            self.lines.push_back(lined(lines[ii].getj1(), lines[ii].getj2(), lines[ii].getg1(), lines[ii].getg2(), lines[ii].getgf(), lines[ii].getw0(), lines[ii].getanomalous(), lines[ii].getDw()))


        #
        # Init one Me object per thread
        #
        for ii in range(nthreads):
            self.Me.push_back(Milned(self.regions, self.lines))

        
    
    def __dealloc__(self):
        print("pyMilne::dealloc: cleaning-up...")
        cdef int sizeMe = self.Me.size()
        cdef int ii = 0
        
        self.Me.clear()
        self.lines.clear()
        self.regions.clear()
        
        

    def synthesize(self, ar[double,ndim=3] m, double mu = 1.0):

        cdef long nWav = self.Me[0].get_number_of_wavelength()
        
        cdef long ny   = m.shape[0]
        cdef long nx   = m.shape[1]
        cdef long npar = m.shape[2]

        
        cdef ar[double,ndim=4] Stokes = zeros((ny,nx,4,nWav), dtype='float64', order='c')
        
        if(npar != 9):
            print("pyMilne::synthesize: ERROR, npar != 9, returning")
            return Stokes

        SynManyd(self.Me, <double*>m.data, <double*>Stokes.data, ny, nx, mu)
        
        return Stokes
    
    def synthesize_RF(self, ar[double,ndim=3] m, double mu = 1.0):

        cdef long nWav = self.Me[0].get_number_of_wavelength()
        
        cdef long ny   = m.shape[0]
        cdef long nx   = m.shape[1]
        cdef long npar = m.shape[2]

        
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
        cdef long ny = m.shape[0]
        cdef long nx = m.shape[1]
        cdef long npar = m.shape[2]
        cdef long nwav = obs.shape[3]
        cdef long nwav1= self.Me[0].get_number_of_wavelength()
        cdef long nDat = nwav*4
        
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


    def invert_spatially_regularized(self, ar[double,ndim=4] m, ar[double,ndim=5] obs, ar[double,ndim=2] sig, ar[double,ndim=1] alphas, ar[double,ndim=1] alphas_time, double mu = 1.0, int nRandom  = 1, int nIter = 20, double chi2_thres = 1.0, verbose = True, int method = 0, double iLam = 10, int delay_bracket = 2):

        #
        # Dimensions
        #
        cdef long nt = m.shape[0]
        cdef long ny = m.shape[1]
        cdef long nx = m.shape[2]
        cdef long npar = m.shape[3]
        cdef long nwav = obs.shape[4]
        cdef long nwav1= self.Me[0].get_number_of_wavelength()
        cdef long nDat = nwav*4
        
        #
        # Init output arrays 
        #
        cdef ar[double,ndim=5] syn  = zeros((nt,ny,nx,4,nwav), dtype='float64', order='c')
        cdef double chi2 = 1.e32
        
        
        #
        # Check dims
        # 
        if(nwav != nwav1):
            print("pyMilne::invert_spatially_regularized_double: ERROR, input obs do not have the same number of wavelength than the regions you provided: {0} != {1}".format(nwav, nwav1))
            return m, syn, chi2

        
        #
        # invert pixels
        #                                                              
        chi2 = invert_spatially_regularized_double(nt, ny, nx, nDat, self.Me,  <double*>m.data, <double*>obs.data, <double*>syn.data, <double*>sig.data, <int>method, <int>nIter, <double>chi2_thres, <double>mu, <double>iLam,  <double*>alphas.data,  <double*>alphas_time.data,<int>delay_bracket)
                                                                        
        
        return m, syn, chi2                                                                        
    
    def get_dtype(self):
        return self.dtype


    def invert_Spatially_Coupled(self, ar[double,ndim=3] m, list coupled_regions, \
                                 ar[double,ndim=1] alphas, double mu = 1.0,  int nIter = 20, \
                                 double chi2_thres = 1.0,  int method = 0, double iLam = 1.0, \
                                 int delay_bracket = 2):

        cdef long ny = m.shape[0]
        cdef long nx = m.shape[1]
        cdef long npar = m.shape[2]
        cdef int nthreads = self.Me.size()
        
        if(npar != 9):
            print("[error] pyMilne::invert_spatially_coupled: the model must have 9 parameters, exiting")
            return None

        
        # Init C++ data struct for spatially coupled inversions

        cdef Data_d dat
        InitDataContainer(ny,nx,npar,dat,<double*>&alphas[0],self.Me, mu)

        
        # add regions to data struct

        cdef int nreg = len(coupled_regions)
        cdef int ii = 0

        cdef int off = 0
        cdef double thres = 1.0

        cdef vector[double] wav = self.Me[0].get_wavelength()

        dat.reset_regions()

        for ii in range(nreg):
            thres = 1.0
            if(len(coupled_regions[ii]) == 4):
                thres = float(coupled_regions[ii][3])
        
            off += addRegionToClass(dat,coupled_regions[ii][0],coupled_regions[ii][2],coupled_regions[ii][1], off, wav, thres, nthreads)

        
        # init nData
        
        init_nData(dat)

        
        # call C++ inverter
        
        cdef ar[double,ndim=4] syn = zeros((ny, nx, 4, wav.size()), dtype='float64')
        
        invert_spatially_coupled(<double*>m.data, \
				 <double*>syn.data, \
        			 <int>method, \
				 <int>nIter, <double>chi2_thres, <double>mu, <double>iLam, \
        			 <int>delay_bracket, dat);


        return m, syn
#
# ******************************************************************************************************
#

    
cdef class pyMilne_float: 
    cdef vector[Milnef] Me
    cdef vector[linef]  lines
    cdef vector[Regionf] regions
    cdef int dtype
    
    def __cinit__(self, list regions, list lines, int nthreads = 1, bool anomalous = True):
        self.dtype = 4

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
        
        cdef ar[float, ndim=1] tmp
        
        for ii in range(nregions):
            nlambda = <int>regions[ii][0].size
            if(regions[ii][1] is not None):
                nPsf    = <int>regions[ii][1].size
            else:
                nPsf    = 0
            
            tmp = zeros(nPsf,dtype='float32', order='c')
            for ww in range(nPsf):
                tmp[ww] = regions[ii][1][ww]
                
            if(nPsf != 0):
                self.regions.push_back(Regionf(nlambda, <int>0, nPsf, <float*>tmp.data))
            else:
                self.regions.push_back(Regionf(nlambda, <int>0, nPsf, NULL))

            for ww in range(nlambda):
                self.regions[ii].wav[ww] = regions[ii][0][ww]
            
        #
        # Init lines
        #
        for ii in range(nlines):
            self.lines.push_back(linef(lines[ii].getj1(), lines[ii].getj2(), lines[ii].getg1(), lines[ii].getg2(), lines[ii].getgf(), lines[ii].getw0(), lines[ii].getanomalous(), lines[ii].getDw()))


        #
        # Init one Me object per thread
        #
        for ii in range(nthreads):
            self.Me.push_back(Milnef(self.regions, self.lines))

        
    
    def __dealloc__(self):
        print("pyMilne::dealloc: cleaning-up...")
        self.Me.clear()
        self.lines.clear()
        self.regions.clear()
            
        

    def synthesize(self, ar[float,ndim=3] m, float mu = 1.0):

        cdef int nWav = self.Me[0].get_number_of_wavelength()
        
        cdef long ny   = m.shape[0]
        cdef long nx   = m.shape[1]
        cdef long npar = m.shape[2]

        
        cdef ar[float,ndim=4] Stokes = zeros((ny,nx,4,nWav), dtype='float32', order='c')
        
        if(npar != 9):
            print("pyMilne::synthesize: ERROR, npar != 9, returning")
            return Stokes

        SynManyf(self.Me, <float*>m.data, <float*>Stokes.data, ny, nx, mu)
        
        return Stokes
    
    def synthesize_RF(self, ar[float,ndim=3] m, float mu = 1.0):

        cdef long nWav = self.Me[0].get_number_of_wavelength()
        
        cdef long ny   = m.shape[0]
        cdef long nx   = m.shape[1]
        cdef long npar = m.shape[2]

        
        cdef ar[float,ndim=4] Stokes = zeros((ny,nx,4,nWav), dtype='float32', order='c')
        cdef ar[float,ndim=5] RF     = zeros((ny,nx,9,4,nWav), dtype='float32', order='c')
        
        if(npar != 9):
            print("pyMilne::synthesize: ERROR, npar != 9, returning")
            return Stokes

        SynManyRFf(self.Me, <float*>m.data, <float*>Stokes.data, <float*>RF.data,ny, nx, mu)
        
        return Stokes, RF

    def invert(self, ar[float,ndim=3] m, ar[float,ndim=4] obs, ar[float,ndim=2] sig, float mu = 1.0, int nRandom  = 1, int nIter = 20, float chi2_thres = 1.0, verbose = True):

        #
        # Dimensions
        #
        cdef long ny = m.shape[0]
        cdef long nx = m.shape[1]
        cdef long npar = m.shape[2]
        cdef long nwav = obs.shape[3]
        cdef long nwav1= self.Me[0].get_number_of_wavelength()
        cdef long nDat = nwav*4
        
        #
        # Init output arrays 
        #
        cdef ar[float,ndim=4] syn  = zeros((ny,nx,4,nwav), dtype='float32', order='c')
        cdef ar[float,ndim=2] chi2 = zeros((ny,nx),        dtype='float32', order='c')


        #
        # Check dims
        # 
        if(nwav != nwav1):
            print("pyMilne::invert: ERROR, input obs do not have the same number of wavelength than the regions you provided: {0} != {1}".format(nwav, nwav1))
            return m, syn, chi2

        
        #
        # invert pixels
        # 
        InvertManyf(self.Me, <float*>m.data, <float*>syn.data, <float*>obs.data, <float*>sig.data, <float*>chi2.data, ny, nx, nDat, nRandom, nIter, chi2_thres, mu, <bool>verbose)

        
        return m, syn, chi2

    def get_wavelength_array(self):
        
        cdef vector[double] iwav = self.Me[0].get_wavelength()
        cdef ar[double,ndim=1] res = zeros(iwav.size(), dtype='float64')

        cdef int ii = 0
        cdef int nWav = iwav.size()
        for ii in range(nWav):
            res[ii] = iwav[ii]
            
        return res

    def invert_spatially_regularized(self, ar[float,ndim=4] m, ar[float,ndim=5] obs, ar[float,ndim=2] sig, ar[float,ndim=1] alphas, ar[float,ndim=1] alphas_time, float mu = 1.0,  int nIter = 20, float chi2_thres = 1.0,  int method = 0, float iLam = 10, int delay_bracket = 2):

        #
        # Dimensions
        #
        cdef long nt = m.shape[0]
        cdef long ny = m.shape[1]
        cdef long nx = m.shape[2]
        cdef long npar = m.shape[3]
        cdef long nwav = obs.shape[4]
        cdef long nwav1= self.Me[0].get_number_of_wavelength()
        cdef long nDat = nwav*4
        
        #
        # Init output arrays 
        #
        cdef ar[float,ndim=5] syn  = zeros((nt,ny,nx,4,nwav), dtype='float32', order='c')
        cdef float chi2 = 1.e32
        
        
        #
        # Check dims
        # 
        if(nwav != nwav1):
            print("pyMilne::invert_spatially_regularized_float: ERROR, input obs do not have the same number of wavelength than the regions you provided: {0} != {1}".format(nwav, nwav1))
            return m, syn, chi2

        
        #
        # invert pixels
        #                                                              
        chi2 = invert_spatially_regularized_float(nt, ny, nx, nDat, self.Me,  <float*>m.data, <float*>obs.data, <float*>syn.data, <float*>sig.data, <int>method, <int>nIter, <float>chi2_thres, <float>mu, <float>iLam,  <float*>alphas.data, <float*>alphas_time.data, <int>delay_bracket)
                                                                        
        
        return m, syn, chi2                                                                        

    def get_dtype(self):
        return self.dtype


