import MilneEddington 
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #
    # Init Milne-Eddington object, let's use 6301, 6302 with no degradation PSF
    # Adjust the number of threads as you wish!
    #
    regions = [[np.arange(201, dtype='float64')*0.01 + 6301.0, None]]
    lines   = [6301, 6302]

    #
    # Alternatively we could have provided a PSF array, which is used for all pixels
    # Assuming that the PSF is contained in an array called psf:
    #
    # regions = [[np.arange(201, dtype='float64')*0.01 + 6301.0, psf]]
    
    me = MilneEddington.MilneEddington(regions, lines, nthreads=2)


    
    #
    # First example, let's create a model and synthesize the spectra
    # Let's assume a field of view of ny=1, nx=2 pixels
    #
    #
    # We will repeat the same model in all pixels of the FOV
    # The model is organized as:
    # [|B| [G], inc [rad], azi [rad], vlos [km/s], vDop [Angstrom], eta_l, damping, S0, S1]
    #
    m_in = np.float64([1000., 1.0, 0.39, 0.1, 0.02, 30., 0.1, 0.2, 0.8])

    ny = 1
    nx = 2
    model = me.repeat_model(m_in, ny, nx)

    syn, rf = me.synthesize_rf(model, mu = 1.0)


    #
    # we can add Gaussian noise to those synthetic profiles and try to invert them
    #
    noise_level = 5.e-3
    syn += np.random.normal(loc = 0.0, scale=noise_level, size = syn.shape)

    #
    # The noise estimate can be provided as a scalar or as a 2D array with the
    # noise estimate for each Stokes parameter and wavelength
    #
    sig = np.zeros((4, me.get_wavelength_array().size), dtype='float64', order='c')
    sig += noise_level

    #
    # provide initial model with different parameters than the ones used in the synthesis
    #
    iGuessed = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
    guessed_initial  = me.repeat_model(iGuessed, ny, nx)

    
    #
    # invert
    #
    model_out, syn_out, chi2 = me.invert(guessed_initial, syn, sig, nRandom = 5, nIter=20, chi2_thres=1.0, verbose=False)


    
    #
    # Estimate uncertainties of the results
    #
    errors = me.estimate_uncertainties(model_out, syn, sig, mu=1.0)
    

    #
    # Print obtained parameters and compare with the original ones
    #
    print('Real parameters -----> Inverted values +/- uncertainty')
    for ii in range(9):
        print('{0:13.5f} -----> {1:13.5f} +/- {2:8.5f}'.format(m_in[ii], model_out[0,0,ii], errors[0,0,ii]))
    
    
    #
    # Make plots of the results for the first pixel
    #
    plt.close("all"); plt.ion()
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6), sharex=True)
    ax1 = ax.flatten()

    wav = me.get_wavelength_array() - 6302.4931
    labels = ['I','Q', 'U', 'V']
    for ii in range(4):
        if(ii == 0):
            ax1[ii].plot(wav, syn[0,0,ii], 'k-', label='Obs')
            ax1[ii].plot(wav, syn_out[0,0,ii], '-', color='orangered', label='Fit')
            ax1[ii].legend(loc='lower right')
        else:
            ax1[ii].plot(wav, syn[0,0,ii], 'k-')
            ax1[ii].plot(wav, syn_out[0,0,ii], '-', color='orangered')
        ax1[ii].set_ylabel(labels[ii])

    ax1[2].set_xlabel('Wavelength - 6302.4931')
    f.set_tight_layout(True)
    f.show()
