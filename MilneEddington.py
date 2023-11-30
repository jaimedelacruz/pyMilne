import numpy as np
import pyMilne


class MilneEddington:
    """
    MilneEddington class

    Purpose: Implementation of a parallel Milne-Eddington solver with analytical response functions
    Coded in C++/python by J. de la Cruz Rodriguez (ISP-SU, 2020)
    
    References:
           Landi Degl'Innocenti & Landolfi (2004)
           Orozco Suarez & del Toro Iniesta (2007)

    """

    # *************************************************************************************************

    def _initLine(self, label, anomalous, dw, precision):

        # 6302 log gf from Socas-Navarro (2011), the rest from VALD.
        
        if(precision == 'float64'):
            if(label == 6301):
                return pyMilne.pyLines(j1 = 2.0, j2 = 2.0, g1 = 1.84, g2 = 1.50, cw = 6301.4995, gf = 10.**-0.718, anomalous = anomalous, dw = dw)
            elif(label == 6302):
                return pyMilne.pyLines(j1 = 1.0, j2 = 0.0, g1 = 2.49, g2 = 0.00, cw = 6302.4931, gf = 10.**-1.160, anomalous = anomalous, dw = dw)
            elif(label == 6173):
                return pyMilne.pyLines(j1 = 1.0, j2 = 0.0, g1 = 2.50, g2 = 0.00, cw = 6173.3340, gf = 10.**-2.880, anomalous = anomalous, dw = dw)
            else:
                print("pyLines::setLine: Error line with label {0 } is not implented".format(label))
                return pyMilne.pyLines()
        else:
            if(label == 6301):
                return pyMilne.pyLinesf(j1 = 2.0, j2 = 2.0, g1 = 1.84, g2 = 1.50, cw = 6301.4995, gf = 10.**-0.718, anomalous = anomalous, dw = dw)
            elif(label == 6302):
                return pyMilne.pyLinesf(j1 = 1.0, j2 = 0.0, g1 = 2.49, g2 = 0.00, cw = 6302.4931, gf = 10.**-1.160, anomalous = anomalous, dw = dw)
            elif(label == 6173):
                return pyMilne.pyLinesf(j1 = 1.0, j2 = 0.0, g1 = 2.50, g2 = 0.00, cw = 6173.3340, gf = 10.**-2.880, anomalous = anomalous, dw = dw)
            else:
                print("pyLines::setLine: Error line with label {0 } is not implented".format(label))
                return pyMilne.pyLinesf()
        
    # *************************************************************************************************

    def _get_dtype(self):
        num = self.Me.get_dtype()

        if(num == 4): return 'float32'
        else:         return 'float64'
    
    # *************************************************************************************************

    def _getLines(self, labels, anomalous, dw, precision):

        nLines = len(labels)
        lines  = [None]*nLines

        for ii in range(nLines):
            lines[ii] = self._initLine(labels[ii], anomalous, dw, precision)

        return lines
    
    # *************************************************************************************************

    def __init__(self, regions, lines, anomalous=True, dw_lines = 20,  nthreads=1, precision = 'float64'):
        """
        __init__ method
        
        Arguments:
             regions:   it is a list that contains lists with region information [[wav1, psf1], [wav2, psf2]]
                        where wav1, wav2, psf1, psf2 are float64 numpy arrays. If no PSF is desired, use None.

             lines:     list with the labels of lines to be used (defined in _initLine).

             anomalous: If True, all Zeeman components are calculated for each spectral lines.

             dw_lines: spectral window +/- dw from line center to compute the line profile. Outside that window the profile won't be calculated.
                       Given in km/s (default 20 km/s)

             nthreads:  number of threads to be used when synthesizing or inverting. Only relevant if there is 
                        more than 1 pixel.
             
        """
        error = False
        
        # check regions
        for ii in range(len(regions)):
            if(len(regions[ii]) != 2):
                print("MilneEddington::__init__: ERROR, region {0} has {1} elements, should have 2!".format(ii, len(regions[ii])))
                error = True

        if(error):
            return None
                
        # Init C++ object
        pyLines =  self._getLines(lines, anomalous, dw_lines, precision)

        if(precision == 'float32'):
            self.Me = pyMilne.pyMilne_float(regions, pyLines, nthreads=nthreads, anomalous=anomalous)
        else:
            self.Me = pyMilne.pyMilne(regions, pyLines, nthreads=nthreads, anomalous=anomalous)


    # *************************************************************************************************

    def synthesize(self, model, mu = 1.0):
        """
        synthesize spectra for a given model at a mu angle
        Arguments:
              model: 1D [9] or 3D array [ny,nx,9] with the parameters of the model
              mu:    heliocentric angle for the synthesis

        The model parameters are: [|B| [G], inc [rad], azi [rad], vlos [km/s], vDop [\AA], eta_l, damp, S0, S1]

        Returns:
              4D array [ny,nx,4,nwaw] with the emerging intensity
        """
        ndim = len(model.shape)
        dtype = self._get_dtype()
        
        if(ndim == 1):
            model1 = np.ascontiguousarray(model.reshape((1,1,model.size)), dtype=dtype)
        elif(ndim == 3):
            model1 = model
        else:
            print("MilneEddington::synthesize: ERROR, the input model must have 1 or 3 dimensions")
            return None

        if(model1.shape[2] != 9):
            print("MilneEddington::synthesize: ERROR, input model has npar={0}, should be 9".format(model1.shape[2]))
            return None

        isContiguous = model1.flags['C_CONTIGUOUS']
        if(not isContiguous or model1.dtype != dtype):
            model1 = np.ascontiguousarray(model1, dtype=dtype)

        
            
            
        return self.Me.synthesize(model1, mu=mu)


    # *************************************************************************************************


    def get_wavelength_array(self):
        """
        get_wavelength_array returns the total wavelength array 1D (regions are concatenated)
        
        """
        return self.Me.get_wavelength_array()


    # *************************************************************************************************
    
    def synthesize_rf(self, model, mu=1.0):
        """
        synthesize the spectra and analytical response functions for a given model at a mu angle
        Arguments:
              model: 1D [9] or 3D array [ny,nx,9] with the parameters of the model
              mu:    heliocentric angle for the synthesis

        The model parameters are: [|B| [G], inc [rad], azi [rad], vlos [km/s], vDop [\AA], eta_l, damp, S0, S1]

        Returns:
              a tuple  (spectra, response_function)
                 spectra: 4D array [ny,nx,4,nwaw] with the emerging intensity
                 response_function: 5D array [ny, ny, 9, 4, nwav]
        """
        ndim = len(model.shape)
        dtype = self._get_dtype()
        
        if(ndim == 1):
            model1 = np.ascontiguousarray(model.reshape((1,1,model.size)), dtype=dtype)
        elif(ndim == 3):
            model1 = model
        else:
            print("MilneEddington::synthesize_rf: ERROR, the input model must have 1 or 3 dimensions")
            return None

        if(model1.shape[2] != 9):
            print("MilneEddington::synthesize_rf: ERROR, input model has npar={0}, should be 9".format(model1.shape[2]))
            return None

        isContiguous = model1.flags['C_CONTIGUOUS']
        if(not isContiguous or model1.dtype != dtype):
            model1 = np.ascontiguousarray(model1, dtype=dtype)


            
        return self.Me.synthesize_RF(model1, mu=mu)

    # *************************************************************************************************
      
    def invert(self, model, obs, sig = 1.e-3, mu = 1.0, nRandom = 3, nIter = 20, chi2_thres = 1.0, verbose = False):
        """
        invert observations acquired at a given mu angle
        Arguments:
              model: 1D [9] or 3D array [ny,nx,9] with the parameters of the model
                obs: 2D [4,nwav] or 4D array [ny,nx,4,nwav] with the observed profiles. Should be normalized to the mean continuum.
                sig: scalar or 2D array [4,nwav] with the noise estimate

                 mu:    heliocentric angle for the synthesis
            nRandom: if larger than 1, the input model parameters will be randomized and more inversion will be performed
                     to avoid converging to a local minimum. The best fit will be returned
              nIter: maximum number of Levenberg Marquardt iterations per inversion
         chi2_thres: stop inversion if Chi2 <= chi2_thres
            verbose: only used if nthreads=1, printsout info of each LM iteration

        The model parameters are: [|B| [G], inc [rad], azi [rad], vlos [km/s], vDop [\AA], eta_l, damp, S0, S1]

        Returns:
              a tuple  (spectra, response_function)
                 spectra: 4D array [ny,nx,4,nwaw] with the emerging intensity
                 response_function: 5D array [ny, ny, 9, 4, nwav]
        """
        #
        # Check guessed model properties
        #
        ndim = len(model.shape)
        dtype = self._get_dtype()

        
        if(ndim == 1):
            model1 = np.ascontiguousarray(model.reshape((1,1,model.size)), dtype=dtype)
        elif(ndim == 3):
            model1 = model
        else:
            print("MilneEddington::synthesize: ERROR, the input model must have 1 or 3 dimensions")
            return None, None, None

        if(model1.shape[2] != 9):
            print("MilneEddington::synthesize: ERROR, input model has npar={0}, should be 9".format(model1.shape[2]))
            return None, None, None

        isContiguous = model1.flags['C_CONTIGUOUS']
        if(not isContiguous or model1.dtype != dtype):
            model1 = np.ascontiguousarray(model1, dtype=dtype)


        
        #
        # Check observations
        #
        ndim = len(obs.shape)

        if(ndim == 2):
            obs1 = np.ascontiguousarray(model.reshape((1,1,obs.shape[0], obs.shape[1])), dtype=dtype)
        elif(ndim == 4):
            obs1 = obs
        else:
            print("MilneEddington::invert: ERROR, the input observations must have 2 or 4 dimensions")
            return None, None, None

        
        wav = self.Me.get_wavelength_array()
        nwav = wav.size
        if(obs1.shape[3] != nwav):
            print("MilneEddington::invert: ERROR, input observations has nwav={0}, should be nwav={1}".format(obs1.shape[3], nwav))
            return None, None, None

        isContiguous = obs1.flags['C_CONTIGUOUS']
        if(not isContiguous or obs1.dtype != dtype):
            obs1 = np.ascontiguousarray(obs1, dtype=dtype)

        
        
        #
        # Check sigma
        #
        if isinstance(sig, np.ndarray):
            if(sig.shape[1] != nwav):
                print("MilneEddington::invert: sigma array has nwav={0}, but it should be {1}".format(sigma.shape[1], nwav))
                return None, None, None

            sig1 = np.zeros((4,nwav), dtype=dtype, order='c')
            sig1[:] = sig

        else:
            sig1 = np.zeros((4,nwav), dtype=dtype, order='c')
            sig1[:] = sig  
            
            

        #
        # Call C++ module
        #
        return self.Me.invert(model1, obs1, sig1, mu=mu, nRandom=nRandom, nIter = nIter, chi2_thres = chi2_thres, verbose=verbose)
    
    # *************************************************************************************************

    def get_a_guessed_model(self, ny=1, nx=1):
        iPar = np.float64([750, 1.0, 0.39, 0.25, 0.02, 30., 0.1, 0.8, 0.2])
        dtype = self._get_dtype()

        res = np.zeros((ny, nx, 9), dtype = dtype, order='c')
        for ii in range(9):
            res[:,:,ii] = iPar[ii]
        return res
    
    # *************************************************************************************************

    def repeat_model(self, m_in, ny, nx, nt=None):
        """
        This routine repeats a 1D model over an entire FOV with dimensions ny, nx pixels
        m_in must have 9 elements
        """
        dtype = self._get_dtype()

        if(nt is not None):
            res = np.zeros((nt, ny, nx, 9), dtype = dtype, order='c')
        else:
            res = np.zeros((ny, nx, 9), dtype = dtype, order='c')

        m = m_in.squeeze()
        
        nPar = m.shape[0]
        
        if(nPar != 9):
            print("MilneEddington::repeat_model: Error, input model must have 9 elements!")
            return None

        if(nt is not None):
            for ii in range(9):
                res[:,:,:,ii] = m[ii]
        else:
            for ii in range(9):
                res[:,:,ii] = m[ii]    
            
        return res


    # *************************************************************************************************
    
    def estimate_uncertainties(self, model, obs, sig, mu=1.0):
        """
        estimates uncertainties based on the quality of the fit
        and the parameters sensitivity.

        Model: output model from the inversion [ny, nx, 9]
        Obs  : Observed profiles [ny, nx, 4, nwav]
        sig  : Noise estimate 1D or 2D [4,nwav]

        returns the uncertainty estimate per parameter per pixel [ny, nx, 9]

        Reference: del Toro Iniesta (2003), Eq. 11.30
        """

        
        syn, J = self.synthesize_rf(model, mu=mu)

        error = model*0
        ny, nx = error.shape[0:2]
        
        for yy in range(ny):
            for xx in range(nx):
                
                for kk in range(9):
                    J[yy,xx,kk] /= sig
        

                Hdiag = (J[yy,xx,:]**2).sum(axis=(1,2))
                error[yy,xx,:] = (((obs[yy,xx]-syn[yy,xx]) / sig )**2).sum()

                for kk in range(9):
                    error[yy,xx,kk] /= Hdiag[kk]

        error *= 2.0 / 9.0
        
        return np.sqrt(error)
    
    # *************************************************************************************************
      
    def invert_spatially_regularized(self, model, obs, sig = 1.e-3, mu = 1.0, nIter = 20, chi2_thres = 1.0,
                                     alpha=1.0, alphas=np.ones(9,dtype='float32'),
                                     alpha_time=1.0, alphas_time=np.ones(9,dtype='float32'),
                                     betas = np.zeros(9, dtype='float32'),
                                     method = 1, delay_bracket = 3, init_lambda = 10.0):
        """
        invert_spatially_regularized observations acquired at a given mu angle
        Arguments:
              model: 1D [9], 3D array [ny,nx,9] or 4D array [nt,ny,nx,9] with the parameters of the model
                obs: 2D [4,nwav], 4D array [ny,nx,4,nwav] or 5D array [nt,ny,nx,4,nw] with the observed profiles. Should be normalized to the mean continuum.
                sig: scalar or 2D array [4,nwav] with the noise estimate
                 mu:    heliocentric angle for the synthesis
              nIter: maximum number of Levenberg Marquardt iterations per inversion
         chi2_thres: stop inversion if Chi2 <= chi2_thres
              alpha: global regularization weight that multiplies the value of "alphas" (default = 1).
       x alpha_time: global time regularization weight that multiplies the value of "alphas_time" (default = 1).   
             alphas: the relative scaling of regularization weights for each parameter (default = 1).
        alphas_time: the relative scaling of regularization weights for each parameter in time (default = 1).
              betas: low-norm regularization weight (default = 0).   
             method: Numerical method to solve the sparse system: 0) Conjugate Gradient, 1) BiCGStab, 2) SparseLU (default 1)
      delay_bracket: Delay optimal lambda bracketing for this number of iterations. Avoids taking too large steps in the initial iterations.
        The model parameters are: [|B| [G], inc [rad], azi [rad], vlos [km/s], vDop [\AA], eta_l, damp, S0, S1]

        Returns:
              a tuple  (spectra, response_function)
                 spectra: 4D array [ny,nx,4,nwaw] with the emerging intensity
                 response_function: 5D array [ny, ny, 9, 4, nwav]
        """
        #
        # Check guessed model properties
        #
        ndim = len(model.shape)
        dtype = self._get_dtype()

        if(ndim == 1):
            model1 = np.ascontiguousarray(model.reshape((1,1,1,model.size)), dtype=dtype)
        elif(ndim == 3):
            ny,nx,npar = model.shape
            model1 = model.reshape((1,ny,nx,npar))
        elif(ndim == 4):
            model1 = model
        else:
            print("MilneEddington::invert_spatially_regularized_float: ERROR, the input model must have 1, 3 or 4 dimensions")
            return None, None, None

        if(model1.shape[3] != 9):
            print("MilneEddington::invert_spatially_regularized_float: ERROR, input model has npar={0}, should be 9".format(model1.shape[2]))
            return None, None, None

        isContiguous = model1.flags['C_CONTIGUOUS']
        if(not isContiguous or model1.dtype != dtype):
            model1 = np.ascontiguousarray(model1, dtype=dtype)


        
        #
        # Check observations
        #
        ndim = len(obs.shape)

        if(ndim == 2):
            obs1 = np.ascontiguousarray(model.reshape((1,1,obs.shape[0], obs.shape[1])), dtype=dtype)
        elif(ndim == 4):
            ny,nx,ns,nw = obs.shape
            obs1 = obs.reshape((1,ny,nx,ns,nw))
        elif(ndim == 5):
            obs1 = obs
        else:
            print("MilneEddington::invert_spatially_regularized_float: ERROR, the input observations must have 2, 4 or 5 dimensions")
            return None, None, None

        
        wav = self.Me.get_wavelength_array()
        nwav = wav.size
        if(obs1.shape[4] != nwav):
            print("MilneEddington::invert_spatially_regularized_float: ERROR, input observations has nwav={0}, should be nwav={1}".format(obs1.shape[3], nwav))
            return None, None, None

        isContiguous = obs1.flags['C_CONTIGUOUS']
        if(not isContiguous or obs1.dtype != dtype):
            obs1 = np.ascontiguousarray(obs1, dtype=dtype)

        
        
        #
        # Check sigma
        #
        if isinstance(sig, np.ndarray):
            if(sig.shape[1] != nwav):
                print("MilneEddington::invert_spatially_regularized_float: sigma array has nwav={0}, but it should be {1}".format(sigma.shape[1], nwav))
                return None, None, None

            sig1 = np.zeros((4,nwav), dtype=dtype, order='c')
            sig1[:] = sig

        else:
            sig1 = np.zeros((4,nwav), dtype=dtype, order='c')
            sig1[:] = sig  
            


        #
        # make alphas
        #
        alphas_in      = np.zeros(9,dtype=dtype)
        alphas_time_in = np.zeros(9,dtype=dtype)
        betas_in       = np.zeros(9,dtype=dtype)

        for ii in range(9):
            alphas_in[ii]      = alpha * alphas[ii]
            alphas_time_in[ii] = alpha_time * alphas_time[ii]
            betas_in[ii]       = betas[ii]

        
        #
        # Call C++ module
        #
        return self.Me.invert_spatially_regularized(model1, obs1, sig1, alphas_in, alphas_time_in, betas_in, mu=mu, nIter = nIter, chi2_thres = chi2_thres,  method=method, delay_bracket = delay_bracket, iLam = init_lambda)
    

    # *************************************************************************************************

    def invert_spatially_coupled(self, model, spat_regions, mu = 1.0, nIter = 20, \
                                 chi2_thres = 1.0, alpha=1.0, alphas=np.ones(9,dtype='float32'),
                                 delay_bracket = 3, init_lambda = 10.0):
        """
        invert_spatially_regularized observations acquired at a given mu angle
        Arguments:
              model: 1D [9] or 3D array [ny,nx,9] with the parameters of the model
       spat_regions: list of lists [[obs1,sigma1,psf1], obs2,sigma2,psf2] with the observations and their PSF
                sig: scalar or 2D array [4,nwav] with the noise estimate
                 mu:    heliocentric angle for the synthesis
              nIter: maximum number of Levenberg Marquardt iterations per inversion
         chi2_thres: stop inversion if Chi2 <= chi2_thres
              alpha: global regularization weight that multiplies the value of "alphas" (default = 1).
             alphas: the relative scaling of regularization weights for each parameter (default = 1).
      delay_bracket: Delay optimal lambda bracketing for this number of iterations. Avoids taking too large steps in the initial iterations. (TODO)
        The model parameters are: [|B| [G], inc [rad], azi [rad], vlos [km/s], vDop [\AA], eta_l, damp, S0, S1]

        Returns:
              a tuple  (spectra, response_function)
                 spectra: 4D array [ny,nx,4,nwaw] with the emerging intensity
                 response_function: 5D array [ny, ny, 9, 4, nwav]
        """
        
        #
        # Check guessed model properties
        #
        ndim = len(model.shape)
        dtype = self._get_dtype()

        if(ndim == 1):
            model1 = np.ascontiguousarray(model.reshape((1,1,model.size)), dtype=dtype)
        elif(ndim == 3):
            model1 = model
        else:
            print("MilneEddington::invert_spatially_regularized_float: ERROR, the input model must have 1 or 3 dimensions")
            return None, None, None

        if(model1.shape[2] != 9):
            print("MilneEddington::invert_spatially_regularized_float: ERROR, input model has npar={0}, should be 9".format(model1.shape[2]))
            return None, None, None

        isContiguous = model1.flags['C_CONTIGUOUS']
        if(not isContiguous or model1.dtype != dtype):
            model1 = np.ascontiguousarray(model1, dtype=dtype)
        
        
        #
        # make alphas
        #
        alphas_in = np.zeros(9,dtype=dtype)

        for ii in range(9):
            alphas_in[ii] = alpha * alphas[ii]

        

        
        return self.Me.invert_Spatially_Coupled(model1, spat_regions, alphas_in,  mu=mu, nIter = nIter, chi2_thres = chi2_thres,  method=0, delay_bracket = delay_bracket, iLam = init_lambda)
