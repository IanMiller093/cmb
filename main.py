from pixell_cmb import make_cmb
from pysm_foreground import make_foreground
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
from plot_sample import make_test_prior, run_prior_verification

import numpy as np
from pixell import utils, enmap, reproject

IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

act_freqs = [90, 150, 220]

def get_dust_truth_amplitude(dust_model, nu_0_dust_stokes, shape, wcs, rot):
    """
    Returns dust amplitude map at each Stokes' own reference freq, in uK_RJ,
    reprojected to CAR — matching the convention T's dust scaling expects.
    """
    amp_maps = []
    for s, nu_ref in enumerate(nu_0_dust_stokes):
        emission = dust_model.get_emission(nu_ref * u.GHz)  # (3, npix) healpix, uK_RJ
        stokes_map = emission[s].to(u.uK_RJ, equivalencies=u.cmb_equivalencies(nu_ref * u.GHz)).value
        stokes_map_car = hp_to_car_wrapper(stokes_map, shape, wcs, rot=rot)
        amp_maps.append(stokes_map_car)
    return np.stack(amp_maps, axis=0)  # (N_stokes, N_pix)



cmb, shape, wcs = make_a_cmb(dec_radius=4, ra_radius=8, seed=67, res_arcmin=1, flatsky=True)
N_pix = len(cmb[0].flatten())
ny, nx = shape[-2:]

T, dust_model = make_T_and_dust_model(N_pix=N_pix, shape=shape, wcs=wcs, beam_telescope="act", rot=True, freqs=act_freqs, dust_list=["d0"], res_arcmin=1)
d = make_cmb_and_foreground(freqs=act_freqs, T=T, a_cmb_stokes=cmb, dust_model=dust_model, shape=shape, wcs=wcs, res_arcmin=1, beam=False, beam_telescope="act", beam_pas=None, include_noise=True, rot=True, beam_type="jitter_cmb", beam_split="coadd", debug=False, give_dust_early=False)

N = load_N_multi_channel(telescope="act", channels=act_freqs, shape=shape, wcs=wcs, pa=None)

mu0, S = make_test_prior(T, N, mode='tight')
print("T:", T.shape)
print("d:", d.shape)
print("N:", N.shape)
print("mu0:", mu0.shape)
print("S:", S.shape)
x_sample = posterior_sample(T=T, d=d, N=N, mu0=mu0, S=S)

nu_0_dust_stokes = [545.0, 353.0, 353.0]
dust_model_amplitude = get_dust_truth_amplitude(dust_model, nu_0_dust_stokes, shape, wcs, rot=True)
truth = np.stack([cmb.reshape(3, N_pix), dust_model_amplitude.reshape(3, N_pix)], axis=0)

run_prior_verification(T, d, N, truth, comp=0, stokes=0, pix=1234, ny=ny, nx=nx, posterior_sample_fn=posterior_sample, img_out_path=IMG_OUT_PATH)
