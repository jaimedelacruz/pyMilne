import numpy as np

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

# ***********************************************************************************
