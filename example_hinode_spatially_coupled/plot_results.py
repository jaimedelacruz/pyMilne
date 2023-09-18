import utils as ut
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from importlib import reload
import numpy as np
reload(ut)


# **************************************************************

if __name__ == "__main__":
    
    plt.close("all")
    
    # make plots
    if(os.path.isfile("modelout_pixel-to-pixel.fits")):
        f, ax = ut.mkplot(ut.readFits("modelout_pixel-to-pixel.fits"))
        f.savefig('fig_pixel_to_pixel.pdf', dpi=250, format='pdf')
        del f, ax
    if(os.path.isfile("modelout_spatially_coupled.fits")):
        f, ax = ut.mkplot(ut.readFits("modelout_spatially_coupled.fits"))
        f.savefig('fig_spatially_coupled.pdf', dpi=250, format='pdf')
        del f, ax
    if(os.path.isfile("modelout_spatially_coupled_x2.fits")):
        f, ax = ut.mkplot(ut.readFits("modelout_spatially_coupled_x2.fits"))
        f.savefig('fig_spatially_coupled.pdf', dpi=250, format='pdf')
        del f, ax
    if(os.path.isfile("modelout_spatially_coupled_x2.fits")):
        f, ax = ut.mkplot(ut.readFits("modelout_spatially_coupled_x1.5.fits"))
        f.savefig('fig_spatially_coupled_x1.5.pdf', dpi=250, format='pdf')
        del f, ax   
