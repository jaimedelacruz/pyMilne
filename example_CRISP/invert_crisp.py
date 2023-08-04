import numpy as np
import matplotlib.pyplot as plt
import MilneEddington as ME
import crisp
import imtools as im
import time
from astropy.io import fits

# ***********************************************************

def loadFits(name):
    return np.ascontiguousarray(fits.open(name, 'readonly')[0].data, dtype='float64')

# ***********************************************************

def findgrid(w, dw, extra = 5):
    """
    Findgrid creates a regular wavelength grid 
    with a step of dw that includes all points in 
    input array w. It adds extra points at the edges
    for convolution purposes

    Returns the new array and the positions of the
    wavelengths points from w in the new array
    """
    nw = np.int32(np.rint(w/dw))
    nnw = nw[-1] - nw[0] + 1 + 2*extra
    
    iw = np.arange(nnw, dtype='float64')*dw - extra*dw + w[0]

    idx = np.arange(w.size, dtype='int32')
    for ii in range(w.size):
        idx[ii] = np.argmin(np.abs(iw-w[ii]))

    return iw, idx

# ***********************************************************

class container:
    def __init__(self):
        pass
    
# ***********************************************************

if __name__ == "__main__":
    #
    # Decide to work in float32 or float64
    #
    dtype = 'float32'
    nthreads = 8
    
    #
    # Load data, wavelength array and cmap
    #
    l = container()
    container.iwav = loadFits('crisp.6301_6302.2019-05-10_wave.fits')
    container.d    = loadFits('crisp.6301_6302.2019-05-10_data.fits')
    container.cmap = loadFits('crisp.6301_6302.2019-05-10_cmap.fits')

    
    # The inversions need to account for the instrumental
    # profile, which involve convolutions. The convolutions
    # must be done in a wavelength grid that is at least
    # 1/2 of the FWHM of the instrumental profile. In the
    # case of CRISP that would be ~55 mA / 2 = ~27.5 mA
    #
    # Get finer grid for convolutions purposes
    # Since we only observed at the lines, let's create
    # two regions, one for each line
    #
    # The observed line positions are not equidistant, the
    # Fe I 6301 points only fit into a regular grid of 5 mA
    # whereas the Fe I 6302 can fit into a 15 mA grid
    #
    iw1, idx1 = findgrid(l.iwav[0:17], 0.005) # Fe I 6301
    iw2, idx2 = findgrid(l.iwav[17::], 0.015) # Fe I 6302


    #
    # Now we can concatenate both regions for plotting and
    # manipulating the data
    #
    iw  = np.append(iw1,iw2)
    idx = np.append(idx1, idx2 + iw1.size) 

    #
    # Now we need to create a data cube with the fine grid
    # dimensions. All observed points will contribute to the
    # inversion. The non-observed ones will have zero weight
    # but will be used internally to properly perform the
    # convolution of the synthetic spectra
    #
    ny,nx = l.d.shape[0:2]
    obs = np.zeros((ny,nx,4,iw.size), dtype=dtype, order='c')

    for ss in range(4):
        for ii in range(idx.size):
            obs[:,:,ss,idx[ii]] = l.d[:,:,ss,ii]

    #
    # Create sigma array with the estimate of the noise for
    # each Stokes parameter at all wavelengths. The extra
    # non-observed points will have a very large noise (1.e34)
    # (zero weight) compared to the observed ones (3.e-3)
    #
    sig= np.zeros((4,iw.size), dtype=dtype) + 1.e32
    sig[:,idx] = 3.e-3

    #
    # Since the amplitudes of Stokes Q,U and V are very small
    # they have a low imprint in Chi2. We can artificially
    # give them more weight by lowering the noise estimate.
    #
    sig[1:3, idx] /= 10
    sig[3, idx ] /= 3.5

    
    #
    # Init Me class. We need to create two regions with the
    # wavelength arrays defined above and a instrumental profile
    # for each region in with the same wavelength step
    #
    tw1 = (np.arange(75, dtype=dtype)-75//2)*0.005
    tw2 = (np.arange(25, dtype=dtype)-25//2)*0.015
    
    tr1 = crisp.crisp(6302.0).dual_fpi(tw1, erh=-0.001)
    tr2 = crisp.crisp(6302.0).dual_fpi(tw2, erh=-0.001)

    regions = [[iw1 + 6302.4931,tr1/tr1.sum()], [iw2 + 6302.4931, tr2/tr2.sum()]]
    lines = [6301,6302]
    me = ME.MilneEddington(regions, lines, nthreads=nthreads, precision=dtype)


    #
    # Init model parameters 
    #
    iPar = np.float64([1500, 2.2, 1.0, -0.5, 0.035, 50., 0.1, 0.24, 0.7])
    Imodel   = me.repeat_model(iPar, ny, nx)


    #
    # Run a first cycle with 4 inversions of each pixel (1 + 3 randomizations)
    # 
    t0 = time.time()
    Imodel, syn, chi2 = me.invert(Imodel, obs, sig, nRandom=4, nIter=25, chi2_thres=1.0, mu=0.93)
    t1 = time.time()
    print("dT = {0}s -> <Chi2> = {1}".format(t1-t0, chi2.mean()))

    
    #
    # Smooth result to remove outlayers
    #
    psf = im.gauss2d(45,15)
    psf /= psf.sum()
    
    for ii in range(9):
        Imodel[:,:,ii] = im.fftconvol2d(Imodel[:,:,ii], psf)

    #
    # Run second cycle, starting from the smoothed guessed model
    #
    t0 = time.time()
    mo, syn, chi2 = me.invert(Imodel, obs, sig, nRandom=4, nIter=25, chi2_thres=1.0, mu=0.93)
    t1 = time.time()
    print("dT = {0}s -> <Chi2> = {1}".format(t1-t0, chi2.mean()))


    #
    # Correct velocities for cavity error map from CRISP
    #
    mo[:,:,3] += l.cmap+0.45 # The 0.45 is a global offset that seems to make the umbra at rest
    
    
    
    #
    # make plots
    #
    #plt.ion()
    f, ax = plt.subplots(nrows=3, ncols=3, figsize=(11,4))
    ax1 = ax.flatten()

    cmaps = ['gist_gray', 'RdGy', 'RdGy', 'bwr', 'gist_gray', 'gist_gray',\
             'gist_gray','gist_gray', 'gist_gray']
    labels = ['B [G]', 'inc [rad]', 'azi [rad]', 'Vlos [km/s]', 'vDop [Angstroms]', 'lineop','damp', 'S0', 'S1']

    extent = np.float32((0, nx, 0, ny))*0.059
    for ii in range(9):
        if(ii != 3):
            a = ax1[ii].imshow(im.histo_opt(mo[:,:,ii]), cmap=cmaps[ii], interpolation='nearest', extent=extent, aspect='equal')
        else:
            a = ax1[ii].imshow(mo[:,:,ii], cmap=cmaps[ii], interpolation='nearest', extent=extent, vmax=4, vmin=-4, aspect='equal')
        f.colorbar(a, ax=ax1[ii], orientation='vertical',label=labels[ii])

        
    for jj in range(3):
        for ii in range(3):
            if(jj!=2): ax[jj,ii].set_xticklabels([])
            if(ii!=0): ax[jj,ii].set_yticklabels([])
    
    f.set_tight_layout(True)
    print("saving figure with results -> fig_results.pdf")
    f.savefig('fig_results.pdf', dpi=250, format='pdf', compression=5)
    f.show()
