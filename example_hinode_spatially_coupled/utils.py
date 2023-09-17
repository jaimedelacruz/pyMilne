import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ***********************************************************************************

def fftconvol2d(im, psf, padding = 1, no_shift = False):
    """
    FFTCONVOL2D: performs the convolution of an image with a PSF, 
                 using FFTs. The rutine arranges the PSF ordering.
    INPUT:
       im  = image (2d numpy array)
       psf = psf (2d numpy array). Should have smaller dims than im (?).

    OPTIONAL KEYWORDS: 
       padding = [int]
          0: Pad extra values with the mean of the image
          1: Pad extra values with the image iteself mirrored (default)
          2: Pad with zeroes
       no_shift = [True/False (default)], don't shift the PSF by half it's size.

    DEPENDENCIES: Numpy

    AUTHOR: J. de la Cruz Rodriguez (ISP-SU 2015)
    """
    # Check dims
    n = np.asarray(im.squeeze().shape)
    n1 = np.asarray(psf.shape)
    #
    if(len(n) != 2 or len(n1) != 2):
        print("fftconvol2d: ERROR, images must be 2 dimensions:")
        print(" IMG -> ", n)
        print(" PSF -> ", n1)
        return(0)

    # Get padded dims
    npad = n + n1
    off = np.zeros(2, dtype = 'int16')
    #
    for ii in range(2):
        if((n1[ii]//2)*2 != n1[ii]):
            npad[ii] -= 1
            off[ii] = 1

    # Create padded arrays
    pim = np.zeros(npad, dtype='float64')
    ppsf= np.zeros(npad, dtype='float64')

    npsf = npad - n
    
    # Copy data to padded arrays
    if(padding == 0): # Pad with the mean of the image
        me = np.mean(im)
        pim[0:n[0], 0:n[1]] = im - me
    elif(padding == 1): # Pad by mirroring a reversed version of the image
        me = 0.0

        pim[0:n[0], 0:n[1]] = im
        
        pim[n[0]:n[0]+npsf[0]//2,0:n[1]] = im[n[0]-1:n[0]-npsf[0]//2-1:-1]
        pim[n[0]+npsf[0]//2:npad[0], 0:n[1]] = im[npsf[0]//2:0:-1,0:n[1]]
        
        pim[:,n[1]:n[1] + npsf[1]//2] = pim[:,n[1]-1:n[1]-npsf[1]//2-1:-1]
        pim[:,n[1]+npsf[1]//2::] = pim[:,npsf[1]//2:0:-1]
        
    elif(padding == 2):
        me = 0.0
        pim[0:n[0], 0:n[1]] = im
    else:
        print("fftconvol2d: ERROR, padding can take values:")
        print("   -> 0: pad the arrays with the average value of the array")
        print("   -> 1: pad the arrays with the same image mirrored (default)")
        print("   -> 2: pad the arrays with zeroes")
        return(0)
    
    # Pad psf and shift
    ppsf[0:n1[0], 0:n1[1]] = psf
    if(not no_shift): 
        ppsf = np.roll(np.roll(ppsf, -n1[0]//2 + off[0] ,axis = 0), -n1[1]//2 + off[1] ,axis = 1)

    
    # Convolve & return
    return(np.fft.irfft2(np.fft.rfft2(pim) * np.conj(np.fft.rfft2(ppsf)))[0:n[0], 0:n[1]] + me)

# ***********************************************************************************

def gauss2d(npsf, fwhm):
    sig  = (fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))))**2
    psf = np.zeros((npsf,npsf), dtype='float64', order='c')
    npsf2 = float(npsf//2)
    for yy in range(npsf):
        for xx in range(npsf):
            psf[yy,xx] = np.exp(-0.5 * ((xx-npsf2)**2 + (yy-npsf2)**2) / sig)
    return(psf)

# **************************************************************

def writeFits(filename, var):
    print("[info] writeFits: writing -> {0}".format(filename))
    io = fits.PrimaryHDU(var)
    io.writeto(filename, overwrite=True)
    
    
# **************************************************************

def readFits(filename, dtype='float64', ext=0):
    print("[info] readFits: reading -> {0}".format(filename))
    io = fits.open(filename, 'readonly')
    res = np.ascontiguousarray(io[ext].data, dtype=dtype)
    io.close()
    return res

# ***********************************************************************************

def mkplot(m_in):

    m = np.copy(m_in)
    m[:,:,0] = m_in[:,:,0] * np.cos(m_in[:,:,1])*0.001
    m[:,:,1] = m_in[:,:,0] * np.sin(m_in[:,:,1])*0.001
    m[:,:,4] *= 1.e3
    
    plt.ion()
    f, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8,6.))
    ax1 = ax.flatten()
    
    mi = np.float64([-2.200,0,0,-4,0,0,0.05,0,0.35])
    ma = np.float64([2.200,1.700, np.pi, 4, 65, 50, 2, 0.5, 1.2])
    cm = ['gist_gray', 'bone', 'RdGy', 'bwr', 'afmhot','gist_gray','gist_gray','gist_gray','gist_gray']
    lab = [r'$B_\parallel$ [kG]', r'$B_\perp$ [kG]', r'$\varphi$ [rad]', r'v$_\mathrm{l.o.s}$ [km/s]', r'v$_\mathrm{turb}$ [m$\mathrm{\AA}$]', r'$\eta_L$', r'$a$', r'$S_0$', r'$S_1$']
    ext = np.float64([0,170,0,170])*0.158
    pl = [None]*9
    cl = [None]*9

    for ii in range(9):
        pl[ii] = ax1[ii].imshow(m[:,:,ii], vmax=ma[ii], vmin=mi[ii], interpolation='nearest', \
                                cmap=cm[ii], extent=ext, origin='lower', aspect=1)
        cl[ii] = f.colorbar(pl[ii], ax=ax1[ii], shrink=0.8, orientation='vertical')
        cl[ii].set_label(lab[ii], fontsize=8)


    for ii in range(3):
        ax[2,ii].set_xlabel("x [arcsec]")
        ax[ii,0].set_ylabel("y [arcsec]")

    f.subplots_adjust(wspace=0.05, hspace=0.01, bottom=0.1, top=0.98, left=0.08, right=0.95)
    
    return f, ax

# **************************************************************

def smoothModel(m, fwhm):

    npix = int(fwhm*2.5)
    if((npix//2)*2 == npix):
        npix -= 1

    psf = gauss2d(npix, fwhm)
    psf /= psf.sum()

    res = np.copy(m)
    ny, nx, npar = res.shape

    for ii in range(npar):
        res[:,:,ii] = fftconvol2d(res[:,:,ii].squeeze(), psf)

    return res
# ***********************************************************************************
