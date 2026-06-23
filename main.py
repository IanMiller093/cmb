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
4) fix beam call in pixell_cmb to get around almxfl
'''


IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

NOISE_DEC_R = 4
NOISE_RA_R = 8

box = np.array([[-1 * NOISE_DEC_R, NOISE_RA_R], [NOISE_DEC_R, -1 * NOISE_RA_R]]) * utils.degree
shape, wcs = enmap.geometry(pos=box, res=1 * utils.arcmin, proj='car')

planck_noise_map = accurate_noise(telescope="planck", channel=143, shape=shape, wcs=wcs)

plot_rect_map(planck_noise_map, IMG_OUT_PATH + "planck_noise")

# plot_ps_compare(
#     imap1=act_full_sim_map,
#     imap2=planck_full_sim_map,
#     name=IMG_OUT_PATH+"beam_bp_rotated_full_sim_planck_act_ps_compare",
#     fullsky=False,
#     map1_label="ACT 150 GHz",
#     map2_label="Planck 143 GHz",
#     lmax=5000,
#     title="CMB Power Spectra Comparison"
# )

# plot_2d_ps(imap=act_full_sim_map, name=IMG_OUT_PATH+"beam_bp_act_rotated_full_sim_map")
# plot_2d_ps(imap=planck_full_sim_map, name=IMG_OUT_PATH+"beam_bp_planck_rotated_full_sim_map")