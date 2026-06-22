# imports
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import urllib.request
from act_planck_beam import apply_beam
from pixell import enmap, reproject, utils, curvedsky, powspec, enplot
import matplotlib.pyplot as plt
import camb

import warnings
warnings.filterwarnings("ignore")

def get_camb_cls(lmax=6000):
    
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=67.5,
        ombh2=0.022,
        omch2=0.122,
        tau=0.06,
    )
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)

    results = camb.get_results(pars)

    # [TT, EE, BB, TE]
    powers = results.get_cmb_power_spectra(
        pars,
        lmax=lmax,
        spectra=['lensed_scalar'],
        CMB_unit='muK',
        raw_cl=True,
    )

    cls = powers['lensed_scalar']
    return cls

from pixell import curvedsky, enmap

def make_cmb_ps_matrix(cls):
    
    lmax = cls.shape[0] - 1
    TT = cls[:, 0]
    EE = cls[:, 1]
    BB = cls[:, 2]
    TE = cls[:, 3]

    ps = np.zeros((3, 3, lmax + 1))
    ps[0, 0] = TT
    ps[1, 1] = EE
    ps[2, 2] = BB
    ps[0, 1] = TE
    ps[1, 0] = TE

    return ps

def make_cmb(dec_radius=90, ra_radius=180, seed=67, res=1, beam=True, beam_telescope="planck", beam_channel=100, beam_pa=5, flatsky=True, beam_type="jitter_cmb", beam_split="coadd"):
    '''
    dec_radius: the radius of the declination in degrees, equivalently 0.5 * the dec dimension.  Default
    is 90, corresponding to fullsky.

    ra_radius: the radius of the right ascension in degrees, equivalently 0.5 * the ra dimension.
    Default is 180, corresponding to fullsky.

    seed: integer value to kickstart rand_alm and rand_map, for reproducability and whatnot.  Default
    is None.

    res: resolution in arcmin per pixel.  Default is 1.

    returns a tuple containing a simulated CMB map made using Pixell and a boolean representing if the
    map is flat (True) or not (False).
    '''

    # for the whole sphere
    nyquist_lmax = (60 / res) * 180

    cls = get_camb_cls(lmax=nyquist_lmax)
    ps  = make_cmb_ps_matrix(cls)

    # geometry nonsense
    box = np.array([[-1 * dec_radius, ra_radius], [dec_radius, -1 * ra_radius]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')

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
        else:
            gen_map = alm2map(gen_alms)
    
    return gen_map