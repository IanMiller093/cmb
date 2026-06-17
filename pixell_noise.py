import numpy as np
from pixell import enmap, utils

def make_noise(dec_radius=90, ra_radius=180, res=1, flatsky=True):
    box = np.array([[-1 * dec_radius, ra_radius], [dec_radius, -1 * ra_radius]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')

    # taken from matched filtering Pixell notebook
    pixarea = enmap.pixsizemap(shape, wcs)
    ivar = 10**-2*pixarea/utils.arcmin**2
    noise = enmap.rand_gauss((3, ) + shape, wcs) * np.sqrt(1/ivar)

    return noise