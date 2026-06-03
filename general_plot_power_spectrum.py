# used PySM tutorial as a starting point for imports etc https://pysm3.readthedocs.io/en/latest/basic_use.html
# used a lot of the Pixell notebooks for this https://github.com/simonsobs/pixell_tutorials/blob/master/README.md 

# imports
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np

from pixell import enmap, reproject, utils, curvedsky, uharm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Plots the power spectra of TT, EE, and BB of a segment of given size of a given simulated PySM sky
# Assumes segment is small enough to be assumed as rectangular
def show_power_spectra(sky, f_GHz, ra_center_degrees, dec_center_degrees, radius_degrees):

    # making the shape so I can take a piece out of the sky without looking at the whole sky
    shape, wcs = enmap.geometry(
        # I like triple checked to make sure dec and ra are correct and the +- signs are right -fixed
        pos=[[(dec_center_degrees - radius_degrees) * utils.degree,
        (ra_center_degrees  + radius_degrees) * utils.degree],
        [(dec_center_degrees + radius_degrees) * utils.degree,
        (ra_center_degrees  - radius_degrees) * utils.degree]],

        # using nside2resol instead of calculating it myself
        res = hp.pixelfunc.nside2resol(sky.nside, arcmin=True) * utils.arcmin,
        proj="car"
    )

    # initialize imap_hp of the given frequency from sky
    imap_hp = sky.get_emission(f_GHz * u.GHz).value

    # isolate only the small segment of the sky
    imap = reproject.healpix2map(imap_hp, shape, wcs)

    # add some random noise?  Gaussian noise taken from matched filtering notebook
    pixarea = enmap.pixsizemap(shape, wcs)
    ivar = 10**-2*pixarea/utils.arcmin**2
    for i in range(3):
        noise = enmap.rand_gauss(shape, wcs) * np.sqrt(1/ivar)
        imap[i] += noise
    
    tapered_imap = imap.copy()
    # calculate the width of the taper mask based on the size of the 
    apod_width = max(5, shape[1] // 8)
    taper_mask = enmap.apod(enmap.ones(imap[0].shape, imap[0].wcs), width=apod_width)
    # removed the division by 4 \pi in the formula, no longer assuming that this is over a sphere
    w2 = np.mean(taper_mask**2)
    for i in range(3):
        tapered_imap[i] = taper_mask * imap[i]

    # lmax is no longer arbitrarily large, changed it to have a set formula
    lmax = 2 * sky.nside

    # Still using sphere stuff under the hood for alms, even if w2 calculations assume = rectangular
    alms = curvedsky.map2alm(tapered_imap, lmax=lmax)
    cl = curvedsky.alm2cl(alms) / w2
    l = np.arange(cl.shape[1])

    for i in range(3):
        plt.semilogy(l, cl[i] * l *(l+1) / 2 / np.pi, label=['TT', 'EE', 'BB'][i])
    plt.ylim(-1e4, 1e4)
    plt.legend()

    plt.savefig("power_spectra.png", dpi=150, bbox_inches="tight")
    plt.close()

# testing it out with sky with d0 dust and CMB
sky = pysm3.Sky(nside=512, preset_strings=["d0", "c1"])
# r = 1 was too small, I think, it led to a 9x9 map lol
show_power_spectra(sky, 100, 0, 0, 5)