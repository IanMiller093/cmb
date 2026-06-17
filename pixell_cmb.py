# imports
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import urllib.request
from act_planck_beam import apply_beam

from pixell import enmap, reproject, utils, curvedsky, powspec, enplot
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from plot_power_spectrum import plot_ps

def make_cmb(dec_radius=90, ra_radius=180, ps_txt_filepath="ps.txt", seed=67, res=1, beam=True, beam_telescope="planck", beam_channel=100, beam_pa=5, flatsky=True, beam_type="jitter_cmb", beam_split="coadd"):
    '''
    dec_radius: the radius of the declination in degrees, equivalently 0.5 * the dec dimension.  Default
    is 90, corresponding to fullsky.

    ra_radius: the radius of the right ascension in degrees, equivalently 0.5 * the ra dimension.
    Default is 180, corresponding to fullsky.

    ps_txt_filepath: filepath to go to for rand_alm calculations.  Default is the ps.text file I 
    already have in folder, from "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"

    seed: integer value to kickstart rand_alm and rand_map, for reproducability and whatnot.  Default
    is None.

    res: resolution in arcmin per pixel.  Default is 1.

    returns a tuple containing a simulated CMB map made using Pixell and a boolean representing if the
    map is flat (True) or not (False).
    '''

    raw = np.loadtxt(ps_txt_filepath).T
    # columns are ell, TT, TE, EE, BB, PP
    ell = raw[0].astype(int)
    lmax_file = ell[-1]

    ps = np.zeros((3, 3, lmax_file + 1))

    # scale factor: 2pi / l(l+1), same as what read_spectrum does with scale=True
    scale = np.zeros(lmax_file + 1)
    scale[ell] = 2 * np.pi / (ell * (ell + 1))

    ps[0, 0, ell] = raw[1] * scale[ell]  
    ps[1, 1, ell] = raw[3] * scale[ell] 
    ps[2, 2, ell] = raw[4] * scale[ell] 

    # geometry nonsense
    box = np.array([[-1 * dec_radius, ra_radius], [dec_radius, -1 * ra_radius]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')
    # for the whole sphere
    nyquist_lmax = (60 / res) * 180

    if flatsky:

        gen_map = enmap.rand_map(shape=(3,) + shape, wcs=wcs, cov=ps, seed=seed)

        if beam:
            beamed_map = apply_beam(imap=gen_map, alms=None, lmax=nyquist_lmax, telescope=beam_telescope, channel=beam_channel, pa=beam_pa, beam_type=beam_type, split=beam_split)
            gen_map = beamed_map

    else:

        shape_map = enmap.zeros((3,) + shape, wcs=wcs)

        gen_alms = curvedsky.rand_alm(ps=ps, seed=seed, lmax=nyquist_lmax)

        if beam:
            beamed_map = apply_beam(imap=None, alms=gen_alms, lmax=nyquist_lmax, telescope=beam_telescope, channel=beam_channel, pa=beam_pa, beam_type=beam_type, split=beam_split)
            gen_map = beamed_map
    
    return gen_map