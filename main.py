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
from cmb_and_foreground import new_make_cmb_and_foreground, make_a_cmb, make_T_and_dust_model

import numpy as np
from pixell import utils, enmap

IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

planck_freqs = [100, 143, 217, 353, 545, 857]
act_freqs = [90, 150, 220]
N_chan_planck = len(planck_freqs)
N_chan_act = len(act_freqs)

# --- Planck ---
a_cmb_1d_planck, shape_planck, wcs_planck = make_a_cmb(
    dec_radius=4, ra_radius=8, seed=67, res_arcmin=1, flatsky=True
)
N_pix_planck = len(a_cmb_1d_planck)

T_planck, dust_model_planck = make_T_and_dust_model(
    N_pix=N_pix_planck, shape=shape_planck, wcs=wcs_planck,
    beam_telescope="planck", rot=True, freqs=planck_freqs,
    dust_list=["d0"], res_arcmin=1
)

d_flat_planck = new_make_cmb_and_foreground(
    freqs=planck_freqs, T=T_planck, a_cmb_1d=a_cmb_1d_planck,
    dust_model=dust_model_planck, shape=shape_planck, wcs=wcs_planck,
    res_arcmin=1, beam=True, beam_telescope="planck", beam_pas=None,
    include_noise=True, rot=True
)

# --- ACT ---
a_cmb_1d_act, shape_act, wcs_act = make_a_cmb(
    dec_radius=4, ra_radius=8, seed=67, res_arcmin=1, flatsky=True
)
N_pix_act = len(a_cmb_1d_act)

T_act, dust_model_act = make_T_and_dust_model(
    N_pix=N_pix_act, shape=shape_act, wcs=wcs_act,
    beam_telescope="act", rot=True, freqs=act_freqs,
    dust_list=["d0"], res_arcmin=1
)

d_flat_act = new_make_cmb_and_foreground(
    freqs=act_freqs, T=T_act, a_cmb_1d=a_cmb_1d_act,
    dust_model=dust_model_act, shape=shape_act, wcs=wcs_act,
    res_arcmin=1, beam=True, beam_telescope="act", beam_pas=None,
    include_noise=True, rot=True
)

ny_p, nx_p = shape_planck[-2:]
ny_a, nx_a = shape_act[-2:]

for idx, freq in enumerate(planck_freqs):
    channel_map = d_flat_planck.reshape(N_chan_planck, ny_p, nx_p)[idx]
    plot_rect_map(channel_map, name=IMG_OUT_PATH+f"vector_reconstructed_I_planck_{freq}GHz")

for idx, freq in enumerate(act_freqs):
    channel_map = d_flat_act.reshape(N_chan_act, ny_a, nx_a)[idx]
    plot_rect_map(channel_map, name=IMG_OUT_PATH+f"vector_reconstructed_I_act_{freq}GHz")

# TODO: plot noise variances for each frequency