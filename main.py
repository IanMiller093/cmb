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