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
from cmb_and_foreground import new_make_cmb_and_foreground

import numpy as np
from pixell import utils, enmap

IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

# act_full_sim_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, seed=67, res=1, foreground_components=["d0"], include_noise=True, beam=True, rot=True, bp=True, bb_telescope="act", bb_channel=150, bb_pa=None, beam_type="jitter_cmb", beam_split="coadd")
# act_cmb_map = make_cmb(dec_radius=4, ra_radius=8, seed=67, res=1, beam=True, beam_telescope="act", beam_channel=150, beam_pa=5, flatsky=True, beam_type="jitter_cmb", beam_split="coadd")
# planck_full_sim_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, seed=67, res=1, foreground_components=["d0"], include_noise=True, beam=True, rot=True, bp=True, bb_telescope="planck", bb_channel=143, bb_pa=None, beam_type="jitter_cmb", beam_split="coadd")
# planck_cmb_map = make_cmb(dec_radius=4, ra_radius=8, seed=67, res=1, beam=True, beam_telescope="planck", beam_channel=143, beam_pa=5, flatsky=True, beam_type="jitter_cmb", beam_split="coadd")

# plot_rect_map(imap=act_full_sim_map, name=IMG_OUT_PATH+"beam_bp_act_rotated_full_sim_map")
# plot_rect_map(imap=act_cmb_map, name=IMG_OUT_PATH+"beam_bp_act_rotated_cmb_map")
# plot_rect_map(imap=planck_full_sim_map, name=IMG_OUT_PATH+"beam_bp_planck_rotated_full_sim_map")
# plot_rect_map(imap=planck_cmb_map, name=IMG_OUT_PATH+"beam_bp_planck_rotated_cmb_map")

freqs = [100, 143]
N_chan = len(freqs)

d_flat, T, shape, wcs = new_make_cmb_and_foreground(freqs=freqs, dec_radius=4, ra_radius=8, dust_list=["d0"], res_arcmin=1, beam=True, seed=67, beam_telescope="planck", beam_pas=None, flatsky=True, include_noise=True)

ny, nx = shape[-2:]

f_idx = 0
channel_map_idx_0 = d_flat.reshape(N_chan, ny, nx)[f_idx]
plot_rect_map(channel_map_idx_0, name=IMG_OUT_PATH+f"vector_reconstructed_I_{freqs[f_idx]}GHz")

f_idx = 1
channel_map_idx_1 = d_flat.reshape(N_chan, ny, nx)[f_idx]
plot_rect_map(channel_map_idx_1, name=IMG_OUT_PATH+f"vector_reconstructed_I_{freqs[f_idx]}GHz")