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

from plot_power_spectrum import plot_ps

# get power spectra .txt
ps_url = "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
urllib.request.urlretrieve(ps_url, "ps.txt")

raw = np.loadtxt("ps.txt").T  # shape (ncols, nrows)
# columns are: ell, TT, TE, EE, BB, PP
ell = raw[0].astype(int)
lmax_file = ell[-1]

ps = np.zeros((3, 3, lmax_file + 1))

# scale factor: 2pi / l(l+1), same as what read_spectrum does with scale=True
scale = np.zeros(lmax_file + 1)
scale[ell] = 2 * np.pi / (ell * (ell + 1))

ps[0, 0, ell] = raw[1] * scale[ell]  # TT
ps[1, 1, ell] = raw[3] * scale[ell]  # EE
ps[2, 2, ell] = raw[4] * scale[ell]  # BB

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

# if everything works, these two should be the same
plot_ps(imap=gen_map, name="power_spectrum_from_map", lmax=5000)
plot_ps(alms=gen_alms, name="power_spectrum_directly_from_alm", lmax=5000)