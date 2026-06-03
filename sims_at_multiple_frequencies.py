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

def show_power_spectra(sky, f_GHz, ra_center_degrees, dec_center_degrees, radius_degrees, fullsky_bool=False):

    # making the shape so I can take a piece out of the sky without looking at the whole sky
    shape, wcs = enmap.geometry(
        # I like triple checked to make sure dec and ra are correct and the +- signs are right but idk
        pos=[[(ra_center_degrees - radius_degrees) * utils.degree, 
        (dec_center_degrees + radius_degrees) * utils.degree], 
        [(ra_center_degrees + radius_degrees) * utils.degree, 
        (dec_center_degrees - radius_degrees) * utils.degree]], 

        # conversion factor of 1/3600 for sky.nside seeeeems to be correct?
        res = (3600 / sky.nside) * utils.arcmin,
        proj="car"
    )

    # initialize imap_hp from sky
    imap_hp = sky.get_emission(f_GHz * u.GHz).value

    # reproject to get only a tiny segment
    imap = reproject.healpix2map(imap_hp, shape, wcs)

    # add some random noise?  Gaussian noise taken from matched filtering notebook
    # definitely ask Zach about this
    pixarea = enmap.pixsizemap(shape, wcs)
    ivar = 10**-2*pixarea/utils.arcmin**2
    for i in range(3):
        noise = enmap.rand_gauss(shape, wcs) * np.sqrt(1/ivar)
        imap[i] += noise

    # apodize if not fullsky
    # I'm preeeeeeeetty sure you don't have to apodize if it's the full sky?  Cuz it loops around right?
    # if we don't apodize, then w2 ends up being 1 by default, and tapered_imap is just a copy of imap
    w2 = 1
    tapered_imap = imap.copy()
    if not fullsky_bool:
        # VERY VERY unsure about these next 3 lines
        res_arcmin = 3600 / sky.nside
        map_pixels = int((2 * radius_degrees * 60) / res_arcmin)
        apod_width = max(5, map_pixels // 8)
        # from spherical harmonics notebook
        taper_mask = enmap.apod(enmap.ones(imap[0].shape, imap[0].wcs), width=apod_width)
        w2 = np.sum(taper_mask.pixsizemap() * taper_mask**2) / (4*np.pi)
        for i in range(3):
            tapered_imap[i] = taper_mask * imap[i]

    # not so sure about lmax=4000?  I tried to make it arbitrarily large?
    alms = curvedsky.map2alm(tapered_imap, lmax=4000)
    cl = curvedsky.alm2cl(alms) / w2
    l = np.arange(cl.shape[1])

    for i in range(3):
        plt.semilogy(l, cl[i] * l *(l+1) / 2 / np.pi, label=['TT', 'EE', 'BB'][i])
    plt.ylim(1e-1, 1e9)
    plt.legend()

    plt.savefig("power_spectra.png", dpi=150, bbox_inches="tight")
    plt.close()

# testing it out with sky with d0 dust and CMB
sky = pysm3.Sky(nside=256, preset_strings=["d0", "c1"])
show_power_spectra(sky, 100, 0, 0, 1, False)