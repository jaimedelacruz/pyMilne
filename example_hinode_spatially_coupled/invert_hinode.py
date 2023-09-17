import MilneEddington as ME
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import utils as ut


# **************************************************************

def doubleGrid(obs):
    """
    The Hinode spectra are undersampled by almost a factor x2
    in the spectral direction. We need to synthesize the spectra
    in a critically sampled grid in order to perform the convolution
    with the instrumental profile. We will ignore these fake points
    by giving them zero weight in the inversion.
    """
    ny, nx, ns, nw = obs.shape
    obs1 = np.zeros((ny, nx, ns, nw*2))
    obs1[:,:,:,0::2] = obs

    wav = (np.arange(nw*2, dtype='float64')-nw)*0.010765 + 6302.08
    
    return wav, obs1

# **************************************************************

def loadData(clip_threshold = 0.99):
    
    wav, obs = doubleGrid(ut.readFits('hinode_170x170_deep.fits', ext=1).transpose((1,2,0)).reshape((170,170,4,112)) / 30000.911001378045)
    tr = np.float64([0.00240208, 0.00390950, 0.0230995, 0.123889, 0.198799,0.116474,0.0201897,0.00704875,0.00277027]) # source A. Asensio 
    psf = ut.readFits('hinode_psf_0.16.fits')

    sig = np.zeros((4, 112*2)) + 1.e32
    sig[:,0::2] = 1.e-3
    sig[1:3] /= 4.0
    sig[3] /= 3.0

    return [[wav, tr/tr.sum()]], [[obs, sig, psf/psf.sum(), clip_threshold]]
    

# **************************************************************

if __name__ == "__main__":

    nthreads = 32 # adapt this number to the number of cores that are available in your machine


    # Sanity check
    bla = 'n'
    bla = input("Has your machine at least 50 GB of RAM in order to run this inversion? [n/y] ")
    if(bla != 'y'):
        sys.exit("exiting ... ")
    
    # Load data
    region, sregion = loadData()


    # Init ME inverter
    me = ME.MilneEddington(region, [6301, 6302], nthreads=nthreads)
    
    # generate initial model
    ny, nx = sregion[0][0].shape[0:2]
    Ipar = np.float64([1000, 1,1,0.01,0.02,20.,0.1, 0.2,0.7])
    m = me.repeat_model(Ipar, ny, nx)
    

    # Invert pixel by pixel
    mpix, syn, chi2 = me.invert(m, sregion[0][0], sregion[0][1], nRandom=8, nIter=15, chi2_thres=1.0, mu=0.96)
    ut.writeFits("modelout_pixel-to-pixel.fits", mpix)

    # smooth model
    m = ut.smoothModel(mpix, 4)


    # invert spatially-coupled with initial guess from pixel-to-pixel (less iterations)
    m1, chi = me.invert_spatially_coupled(m, sregion, mu=0.96, nIter=10, alpha=100., \
                                     alphas = np.float64([1,1,1,0.01,0.01,0.01,0.01,0.01,0.01]),\
                                     init_lambda=10.0)

    

    # smooth model with very narrow PSF and restart with less regularization (lower alpha)
    m = ut.smoothModel(m1, 2)

    
    # invert spatially-coupled 
    m1, chi = me.invert_spatially_coupled(m, sregion, mu=0.96, nIter=20, alpha=10., \
                                          alphas = np.float64([2,2,2,0.01,0.01,0.01,0.01,0.01,0.01]),\
                                          init_lambda=1.0)
    
    ut.writeFits("modelout_spatially_coupled.fits", m1)

    
