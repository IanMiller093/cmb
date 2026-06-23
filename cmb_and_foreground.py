from pixell_cmb import make_cmb
from pysm_foreground import make_foreground
from act_planck_beam import apply_beam
from act_planck_noise import accurate_noise

from pixell import enmap, utils
import numpy as np

def make_cmb_and_foreground(dec_radius=90, ra_radius=180, seed=67, res=1, sky_f=150, foreground_components=["d0"], include_noise=True, beam=True, rot=False, bp=True, bb_telescope="planck", bb_channel=100, bb_pa=None, beam_type="jitter_cmb", beam_split="coadd"):
    '''
    pretty much just a wrapper of make_cmb and make_foreground.  I'm too lazy to copy over the comments
    about params from the other files, and the fishies would get mad at me if I used Claude to grab the
    comments, so you will have to go over and read the comments from the other two files yourself :(

    returns make_cmb + make_foreground with the same dimensions
    '''

    cmb = make_cmb(dec_radius=dec_radius, ra_radius=ra_radius, seed=seed, res=res, beam=False, beam_telescope=bb_telescope, beam_channel=bb_channel, beam_pa=bb_pa, beam_type=beam_type, beam_split=beam_split)
    foreground = make_foreground(dec_radius=dec_radius, ra_radius=ra_radius, sky_f=sky_f, res=res, foreground_components=foreground_components, beam=False, rot=rot, bp=bp, bb_telescope=bb_telescope, bb_channel=bb_channel, bb_pa=bb_pa, beam_type=beam_type, beam_split=beam_split)
    assert cmb.shape == foreground.shape

    if beam:
        nyquist_lmax = (60 / res) * 180
        result = apply_beam(imap=(cmb + foreground), lmax=nyquist_lmax, telescope=bb_telescope, channel=bb_channel, pa=bb_pa, beam_type=beam_type, split=beam_split)
    else:
        result = cmb + foreground

    if include_noise:
        box = np.array([[-1 * dec_radius, ra_radius], [dec_radius, -1 * ra_radius]]) * utils.degree
        shape, wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')

        result += accurate_noise(telescope=bb_telescope, channel=bb_channel, shape=shape, wcs=wcs, pa=bb_pa)

    return result