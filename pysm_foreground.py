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
from bandpass import bandpass

def make_foreground(dec_radius=90, ra_radius=180, sky_f=150, res=1, foreground_components=["d0"], fwhm=1, beam=True, rot=False, bp=True, bb_telescope="planck", bb_channel=100, bb_pa=None, beam_type="jitter_cmb", beam_split="coadd"):
    '''
    dec_radius: the radius of the declination in degrees, equivalently 0.5 * the dec dimension.  Default
    is 90, corresponding to fullsky.

    ra_radius: the radius of the right ascension in degrees, equivalently 0.5 * the ra dimension.
    Default is 180, corresponding to fullsky.

    sky_f: the frequency to take retrieve a map from the sky at in GHz.  Defauly is 150 GHz.

    res: resolution in arcmin per pixel.  Default is 1.

    foreground_components: the argument to be given to the psym3 Sky constructor, must be an array of
    strings corresponding to different components.  Default is ["d0"], corresponding to a map with just 
    a simple dust component.

    gaussian_noise: boolean governing whether gaussian noise is included in the foreground or not

    returns a Pixell map of the sky at given frequency.  Designed so that when given the same resolution 
    and dimensions as an input to the make_cmb function, the outputs will have the same shape (3, _, _). 
    '''

    # sky_nside has to be the smallest power of two such that its resolution is GREATER than res
    # then that will be like casted down by the shape geometry call
    sky_nside = 1
    while hp.nside2resol(sky_nside, arcmin=True) > res:
        sky_nside *= 2

    box = np.array([[-1 * dec_radius, ra_radius], [dec_radius, -1 * ra_radius]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')
    foreground_sky = pysm3.Sky(nside=sky_nside, preset_strings=foreground_components)
    
    if bp:
        bp_freqs, bp_weights = bandpass(telescope=bb_telescope, channel=bb_channel, pa=bb_pa)

        mask = bp_freqs > 0

        bp_freqs = bp_freqs[mask]
        bp_weights = bp_weights[mask]

        hp_map = foreground_sky.get_emission(bp_freqs * u.GHz, bp_weights)
    else:
        hp_map = foreground_sky.get_emission(sky_f * u.GHz)

    if rot:
        foreground_map = reproject.healpix2map(hp_map, shape, wcs, rot="gal,cel")

    else:
        foreground_map = reproject.healpix2map(hp_map, shape, wcs)

    if beam:
        nyquist_lmax = (60 / res) * 180
        beamed_map = apply_beam(imap=foreground_map, alms=None, lmax=nyquist_lmax, telescope=bb_telescope, channel=bb_channel, pa=bb_pa, beam_type=beam_type, split=beam_split)
        foreground_map = beamed_map
    
    return foreground_map