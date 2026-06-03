# used PySM tutorial as a starting point for imports etc https://pysm3.readthedocs.io/en/latest/basic_use.html
# used Pixell spherical harmonics tutorial for spherical harmonics stuff https://github.com/simonsobs/pixell_tutorials/blob/master/pixell_spherical_harmonics.ipynb

# made by ian to try to learn more about Pixell/PySM and CMB component separation

# imports
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np

from pixell import enmap, reproject, utils, curvedsky
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# sky made up of only CMB and d0 dust, no synchotron or other components 
# don't understand completely, but because of something with the sampling theorem, i need to have nside 
# > lmax / 3
sky = pysm3.Sky(nside=128, preset_strings=["d0", "c1"])

# initial method I made, kinda meh
def plot_fullsky_intensity_power_spectra(f_GHz, sky):
    # get map at the specific frequency from specific sky
    map_w_astropy_coords = sky.get_emission(f_GHz * u.GHz)

    # things needed to get this from HP map to Pixell map
    hp_map = map_w_astropy_coords.value
    shape, wcs = enmap.fullsky_geometry(res=0.5 * utils.degree, proj='car')
    bsize = 12 * sky.nside**2

    # if i understand correctly, we don't need to apodize because the spherical sky is technically periodic
    # but this only applies for the full sky.  For a segment, apodization is necessary.
    imap = reproject.healpix2map(hp_map, shape, wcs, bsize=bsize)
    intensity = imap[0]

    # using this lmax because it was what was used in the Pixell spherical harmonics ipynb
    # i get the idea of higher lmax => more high frequency variation captured
    # but beyond that idk what the "standard" choice for a good lmax is
    lmax = 4000

    alms = curvedsky.map2alm(intensity, lmax=lmax)

    cl = curvedsky.alm2cl(alms)
    l = np.arange(cl.shape[0])

    # exact equation taken from spherical harmonics notebook
    plt.semilogy(l, cl * l *(l+1) / 2 / np.pi, label="intensity power spectrum")
    plt.ylim(1e-1, 1e4)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell / 2\pi$')
    plt.legend()

    # plotting didn't work, so I'm trying to save it as an image
    plt.savefig("cmb_power_spectrum.png", dpi=300)

plot_fullsky_intensity_power_spectra(100, sky)