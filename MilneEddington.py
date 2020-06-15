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
    def _initLine(self, label, anomalous):
        if(label == 6301):
            return pyMilne.pyLines(j1 = 2.0, j2 = 2.0, g1 = 1.84, g2 = 1.50, cw = 6301.4995, gf = 10.**-0.718, anomalous = anomalous)
        elif(label == 6302):
            return pyMilne.pyLines(j1 = 1.0, j2 = 0.0, g1 = 2.49, g2 = 0.00, cw = 6302.4931, gf = 10.**-0.968, anomalous = anomalous)
        elif(label == 6173):
            return pyMilne.pyLines(j1 = 1.0, j2 = 0.0, g1 = 2.50, g2 = 0.00, cw = 6173.3340, gf = 10.**-2.880, anomalous = anomalous)
        else:
            print("pyLines::setLine: Error line with label {0 } is not implented".format(label))
            return pyMilne.pyLines()
    
    
    def _getLines(self, labels, anomalous):

        nLines = len(labels)
        lines  = [None]*nLines

        for ii in range(nLines):
            lines[ii] = self._initLine(labels[ii], anomalous)

        return lines
    
    def __init__(self, regions, lines, anomalous=True, nthreads=1):
        """
        __init__ method
        
        Arguments:
             regions:   it is a list that contains lists with region information [[wav1, psf1], [wav2, psf2]]
                        where wav1, wav2, psf1, psf2 are float64 numpy arrays. If no PSF is desired, use None.

             lines:     list with the labels of lines to be used (defined in _initLine).

             anomalous: If True, all Zeeman components are calculated for each spectral lines.

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
        pyLines =  self._getLines(lines, anomalous)

        self.Me = pyMilne.pyMilne(regions, pyLines, nthreads=nthreads, anomalous=anomalous)
        


        
        
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
        
        if(ndim == 1):
            model1 = np.ascontiguousarray(model.reshape((1,1,model.size)), dtype='float64')
        elif(ndim == 3):
            model1 = model
        else:
            print("MilneEddington::synthesize: ERROR, the input model must have 1 or 3 dimensions")
            return None

        if(model1.shape[2] != 9):
            print("MilneEddington::synthesize: ERROR, input model has npar={0}, should be 9".format(model1.shape[2]))
            return None

        isContiguous = model1.flags['C_CONTIGUOUS']
        if(not isContiguous or model1.dtype != 'float64'):
            model1 = np.ascontiguousarray(model1, dtype='float64')

            
            
        return self.Me.synthesize(model, mu=mu)




    def get_wavelength_array(self):
        """
        get_wavelength_array returns the total wavelength array 1D (regions are concatenated)
        
        """
        return self.Me.get_wavelength_array()


    
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
        
        if(ndim == 1):
            model1 = np.ascontiguousarray(model.reshape((1,1,model.size)), dtype='float64')
        elif(ndim == 3):
            model1 = model
        else:
            print("MilneEddington::synthesize_rf: ERROR, the input model must have 1 or 3 dimensions")
            return None

        if(model1.shape[2] != 9):
            print("MilneEddington::synthesize_rf: ERROR, input model has npar={0}, should be 9".format(model1.shape[2]))
            return None

        isContiguous = model1.flags['C_CONTIGUOUS']
        if(not isContiguous or model1.dtype != 'float64'):
            model1 = np.ascontiguousarray(model1, dtype='float64')

            
            
        return self.Me.synthesize_RF(model, mu=mu)

        
    
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
        
        if(ndim == 1):
            model1 = np.ascontiguousarray(model.reshape((1,1,model.size)), dtype='float64')
        elif(ndim == 3):
            model1 = model
        else:
            print("MilneEddington::synthesize: ERROR, the input model must have 1 or 3 dimensions")
            return None, None, None

        if(model1.shape[2] != 9):
            print("MilneEddington::synthesize: ERROR, input model has npar={0}, should be 9".format(model1.shape[2]))
            return None, None, None

        isContiguous = model1.flags['C_CONTIGUOUS']
        if(not isContiguous or model1.dtype != 'float64'):
            model1 = np.ascontiguousarray(model1, dtype='float64')


        
        #
        # Check observations
        #
        ndim = len(obs.shape)

        if(ndim == 2):
            obs1 = np.ascontiguousarray(model.reshape((1,1,obs.shape[0], obs.shape[1])), dtype='float64')
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
        if(not isContiguous or obs1.dtype != 'float64'):
            obs1 = np.ascontiguousarray(obs1, dtype='float64')

        
        
        #
        # Check sigma
        #
        if isinstance(sig, np.ndarray):
            if(sig.shape[1] != nwav):
                print("MilneEddington::invert: sigma array has nwav={0}, but it should be {1}".format(sigma.shape[1], nwav))
                return None, None, None

            sig1 = np.zeros((4,nwav), dtype='float64', order='c')
            sig1[:] = sig

        else:
            sig1 = np.zeros((4,nwav), dtype='float64', order='c')
            sig1[:] = sig  
            
            
        #
        # Call C++ module
        #
        return self.Me.invert(model1, obs1, sig1, mu=mu, nRandom=nRandom, nIter = nIter, chi2_thres = chi2_thres, verbose=verbose)
    

    def get_a_guessed_model(self, ny=1, nx=1):
        iPar = np.float64([750, 1.0, 0.39, 0.25, 0.02, 30., 0.1, 0.8, 0.2])

        res = np.zeros((ny, nx, 9), dtype = 'float64', order='c')
        for ii in range(9):
            res[:,:,ii] = iPar[ii]
        return res
