import numpy as np
from pixell import enmap, utils

def make_noise(dec_radius=90, ra_radius=180, res=1, beam=True, fwhm=1, flatsky=True):
    box = np.array([[-1 * dec_radius, ra_radius], [dec_radius, -1 * ra_radius]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')

    # taken from matched filtering Pixell notebook
    pixarea = enmap.pixsizemap(shape, wcs)
    ivar = 10**-2*pixarea/utils.arcmin**2
    noise = enmap.rand_gauss((3, ) + shape, wcs) * np.sqrt(1/ivar)

    if not beam:
        return noise

    if flatsky:
        beamed_noise = enmap.smooth_gauss(noise, fwhm * utils.arcmin / (8*np.log(2))**0.5)
        
    else:
        nyquist_lmax = (60 / res) * 180

        beam_ell = hp.sphtfunc.gauss_beam(fwhm * utils.arcmin, lmax=nyquist_lmax, pol=True).T[:3]
        gen_alms = curvedsky.almxfl(gen_alms, beam_ell)
        beamed_noise = curvedsky.alm2map(alm=gen_alms, map=shape_map)

    return beamed_noise