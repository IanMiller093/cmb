# imports
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import urllib.request

from pixell import enmap, reproject, utils, curvedsky, powspec, enplot
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# get power spectra .txt
ps_url = "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
urllib.request.urlretrieve(ps_url, "ps.txt")

# set up randomly generated alm using power spectrum from the txt file
ps = powspec.read_spectrum("ps.txt", inds=True, scale=True, expand='diag')
# just checked, uses np seeds under hood, 67 is a valid seed!!!  Hooray!!!
gen_seed = 67
lmax = 5000
gen_alms = curvedsky.rand_alm(ps=ps, seed=gen_seed, lmax=lmax)

# set up map used solely for shape, taken from map manipulation notebook
# adjust res as necessary, higher res is lower resolution
res = 1
box = np.array([[-25, 25], [25, -25]]) * utils.degree
shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')
shape_map = enmap.zeros((3,) + shape, wcs=wcs)

# generate the map from random alm
gen_map = curvedsky.alm2map(alm=gen_alms, map=shape_map)

# enplot was being buggy, so I clauded a fix with Matplotlib, gonna comment out for rn tho
'''
stokes = ['I', 'Q', 'U']
for i in range(3):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(gen_map[i], origin='lower', cmap='RdBu_r')
    ax.set_title(f'Stokes {stokes[i]}')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'my_plot_{stokes[i]}.png', dpi=150)
    plt.close()
    print(f"Saved my_plot_{stokes[i]}.png")
'''

# works for input of imap xor alms
def plot_power_spectrum(imap=None, alms=None, name=None):
    '''
    given an imap xor alms, saves a png of the power spectrum.

    imap: if given an imap of shape (3, _, _), apodizes, calculates cl, l, and w2, and then plots the 
    power specturm

    alms: if given alms of shape (3, _), calculates cl and l and then plots the power spectrum

    if neither imap nor alms is inputted, prints error message.  if both imap and alms is inputted, 
    prints different error message.

    returns nothing, but saves plots to png
    '''
    if (imap is not None and alms is not None):
        print("one of either imap or alms must be None, can't input both")
    
    # imap given
    elif(imap is not None):
        # apodize
        tapered_map = imap.copy()
        apod_width = max(5, shape[1] // 8)
        taper_mask = enmap.apod(enmap.ones(imap[0].shape, imap[0].wcs), width=apod_width)

        # w2 calculation taken DIRECTLY from spherical harmonics notebook this time.
        w2 = np.sum(taper_mask.pixsizemap() * taper_mask**2) / (4*np.pi)
        for i in range(3):
            tapered_map[i] = taper_mask * imap[i]

        calc_alms = curvedsky.map2alm(tapered_map, lmax=lmax)
        cl = curvedsky.alm2cl(calc_alms) / w2
        l = np.arange(cl.shape[1])

        # plot
        for i in range(3):
            plt.semilogy(l, cl[i] * l *(l+1) / 2 / np.pi, label=['TT', 'EE', 'BB'][i])
        plt.ylim(1e-4, 1e4)
        plt.legend()
        
        if name is None:
            plt.savefig("power_spectra.png", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(name + ".png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # alms given
    elif(alms is not None):
        cl = curvedsky.alm2cl(alms)
        l = np.arange(cl.shape[1])

        # plot
        for i in range(3):
            plt.semilogy(l, cl[i] * l *(l+1) / 2 / np.pi, label=['TT', 'EE', 'BB'][i])
        plt.ylim(1e-4, 1e4)
        plt.legend()

        if name is None:
            plt.savefig("power_spectra.png", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(name + ".png", dpi=150, bbox_inches="tight")
        plt.close()

    else:
        print("one of either imap or alms must be inputted, can't leave both blank")

# if everything works, these two should be the same
plot_power_spectrum(imap=gen_map, name="power_spectrum_from_map")
plot_power_spectrum(alms=gen_alms, name="power_spectrum_directly_from_alm")



# note to self-- what frequency am I seeing here with the CMB map?