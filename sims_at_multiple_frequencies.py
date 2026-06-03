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
        # I like triple checked to make sure dec and ra are correct and the +- signs are right -fixed
        pos=[[(dec_center_degrees - radius_degrees) * utils.degree,
        (ra_center_degrees  + radius_degrees) * utils.degree],
        [(dec_center_degrees + radius_degrees) * utils.degree,
        (ra_center_degrees  - radius_degrees) * utils.degree]],

        # fixed
        res = hp.pixelfunc.nside2resol(sky.nside, arcmin=True) * utils.arcmin,
        proj="car"
    )

    # initialize imap_hp from sky
    imap_hp = sky.get_emission(f_GHz * u.GHz).value
    print(imap_hp.shape)

    # reproject to get only a tiny segment
    imap = reproject.healpix2map(imap_hp, shape, wcs)
    print(imap.shape)

    # add some random noise?  Gaussian noise taken from matched filtering notebook
    # definitely ask Zach about this
    # gonna comment this out briefly, I wanna double check and see if the power spectrum is more normal w/o this
    # only some minor things came from commenting it out, to be expected, may as well keep.
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
        # deleted the three lines I was unsure of for this...
        apod_width = max(5, shape[1] // 8)
        # from spherical harmonics notebook
        taper_mask = enmap.apod(enmap.ones(imap[0].shape, imap[0].wcs), width=apod_width)
        # removed the division by 4 \pi in the formula...
        w2 = np.mean(taper_mask**2)
        for i in range(3):
            tapered_imap[i] = taper_mask * imap[i]

    print(tapered_imap.shape)

    # lmax is no longer arbitrarily large, changed it to have a set formula
    lmax = 2 * sky.nside
    alms = curvedsky.map2alm(tapered_imap, lmax=lmax)
    cl = curvedsky.alm2cl(alms) / w2
    l = np.arange(cl.shape[1])

    print(alms.shape)
    print(cl.shape)
    print(l.shape)

    for i in range(3):
        plt.semilogy(l, cl[i] * l *(l+1) / 2 / np.pi, label=['TT', 'EE', 'BB'][i])
    plt.ylim(1e-1, 1e9)
    plt.legend()

    plt.savefig("power_spectra.png", dpi=150, bbox_inches="tight")
    plt.close()

# testing it out with sky with d0 dust and CMB
sky = pysm3.Sky(nside=512, preset_strings=["d0", "c1"])
# r = 1 was too small, I think, it led to a 9x9 map lol
show_power_spectra(sky, 100, 0, 0, 5, False)