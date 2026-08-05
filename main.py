from pixell_cmb import make_cmb
from plot_power_spectrum import plot_ps
from plot_rectangular_map import plot_rect_map
from plot_ps_compare import plot_ps_compare
from plot_2d_power_spectrum import plot_2d_ps
from bandpass import bandpass
from concurrent.futures import ProcessPoolExecutor
from plot_bandpass_fn import plot_bandpass
from act_planck_beam import apply_beam
from act_planck_noise import accurate_noise, load_N_multi_channel
from cmb_and_foreground import hp_to_car_wrapper, make_cmb_and_foreground, make_a_cmb, make_T_and_dust_model
from plot_noise_var import plot_noise_variance_by_channel
import pysm3
import pysm3.units as u
import time
from posterior_sampling import posterior_sample
from plot_sample import make_test_prior, run_prior_verification, plot_component_separation, run_global_calibration_check
from plot_bandpass_sed import plot_bandpass_sed_overlay

import numpy as np
from pixell import utils, enmap, reproject

IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

act_freqs = [90, 150, 220]

def get_dust_truth_amplitude(dust_model, nu_0_dust_stokes, shape, wcs, rot):
    amp_maps = []
    for s, nu_ref in enumerate(nu_0_dust_stokes):
        emission = dust_model.get_emission(nu_ref * u.GHz)
        stokes_map = emission[s].to(u.uK_RJ, equivalencies=u.cmb_equivalencies(nu_ref * u.GHz)).value
        stokes_map_car = hp_to_car_wrapper(stokes_map, shape, wcs, rot=rot)
        amp_maps.append(stokes_map_car)
    return np.stack(amp_maps, axis=0)


cmb, shape, wcs = make_a_cmb(dec_radius=4, ra_radius=8, seed=67, res_arcmin=1, flatsky=True)
N_pix = len(cmb[0].flatten())
ny, nx = shape[-2:]

T, dust_model = make_T_and_dust_model(N_pix=N_pix, shape=shape, wcs=wcs, beam_telescope="act", rot=True, freqs=act_freqs, dust_list=["d0"], res_arcmin=1)
d = make_cmb_and_foreground(freqs=act_freqs, T=T, a_cmb_stokes=cmb, dust_model=dust_model, shape=shape, wcs=wcs, res_arcmin=1, beam=False, beam_telescope="act", beam_pas=None, include_noise=True, rot=True, beam_type="jitter_cmb", beam_split="coadd", debug=False, give_dust_early=False)

N = load_N_multi_channel(telescope="act", channels=act_freqs, shape=shape, wcs=wcs, pa=None)

nu_0_dust_stokes = [545.0, 353.0, 353.0]
dust_model_amplitude = get_dust_truth_amplitude(dust_model, nu_0_dust_stokes, shape, wcs, rot=True)
truth = np.stack([cmb.reshape(3, N_pix), dust_model_amplitude.reshape(3, N_pix)], axis=0)

# --- verification run ---
comp = 0          # cmb
stokes = 0        # I
pix = N_pix // 2  # arbitrary central pixel; swap for one you care about

run_global_calibration_check(T=T, d=d, N=N, truth=truth, ny=ny, nx=nx, posterior_sample_fn=posterior_sample, img_out_path=IMG_OUT_PATH, n_draws=1000, comp_labels=None)