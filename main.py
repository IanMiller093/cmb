from pixell_cmb import make_cmb
from pysm_foreground import make_foreground
from cmb_and_foreground import make_cmb_and_foreground
from plot_power_spectrum import plot_ps
from plot_rectangular_map import plot_rect_map
from plot_ps_compare import plot_ps_compare
from plot_2d_power_spectrum import plot_2d_ps
from bandpass import bandpass
from concurrent.futures import ProcessPoolExecutor
from plot_bandpass_fn import plot_bandpass
from act_planck_beam import apply_beam
from act_planck_noise import accurate_noise

import numpy as np
from pixell import utils, enmap

'''
3) test different scenarios with beams
'''

IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

act_full_sim_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, seed=67, res=1, foreground_components=["d0"], include_noise=True, beam=True, rot=True, bp=True, bb_telescope="act", bb_channel=150, bb_pa=None, beam_type="jitter_cmb", beam_split="coadd")
act_cmb_map = make_cmb(dec_radius=4, ra_radius=8, seed=67, res=1, beam=True, beam_telescope="act", beam_channel=150, beam_pa=5, flatsky=True, beam_type="jitter_cmb", beam_split="coadd")
planck_full_sim_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, seed=67, res=1, foreground_components=["d0"], include_noise=True, beam=True, rot=True, bp=True, bb_telescope="planck", bb_channel=143, bb_pa=None, beam_type="jitter_cmb", beam_split="coadd")
planck_cmb_map = make_cmb(dec_radius=4, ra_radius=8, seed=67, res=1, beam=True, beam_telescope="planck", beam_channel=143, beam_pa=5, flatsky=True, beam_type="jitter_cmb", beam_split="coadd")

plot_rect_map(imap=act_full_sim_map, name=IMG_OUT_PATH+"beam_bp_act_rotated_full_sim_map")
plot_rect_map(imap=act_cmb_map, name=IMG_OUT_PATH+"beam_bp_act_rotated_cmb_map")
plot_rect_map(imap=planck_full_sim_map, name=IMG_OUT_PATH+"beam_bp_planck_rotated_full_sim_map")
plot_rect_map(imap=planck_cmb_map, name=IMG_OUT_PATH+"beam_bp_planck_rotated_cmb_map")