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
from act_planck_noise import accurate_noise
from cmb_and_foreground import make_cmb_and_foreground, make_a_cmb, make_T_and_dust_model
from plot_noise_var import plot_noise_variance_by_channel
import pysm3
import pysm3.units as u
import time

import numpy as np
from pixell import utils, enmap, reproject

IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

planck_freqs = [100]
act_freqs = [90, 150, 220]
N_chan_planck = len(planck_freqs)
N_chan_act = len(act_freqs)
N_stokes = 3

# # --- Planck ---

# t0 = time.time()

# a_cmb_stokes_planck, shape_planck, wcs_planck = make_a_cmb(
#     dec_radius=4, ra_radius=8, seed=67, res_arcmin=1, flatsky=True
# )
# N_pix_planck = len(a_cmb_stokes_planck[0])

# print(f"make_a_cmb took {time.time() - t0} seconds")

# t0 = time.time()

# T_planck, dust_model_planck = make_T_and_dust_model(
#     N_pix=N_pix_planck, shape=shape_planck, wcs=wcs_planck,
#     beam_telescope="planck", rot=True, freqs=planck_freqs,
#     dust_list=["d0"], res_arcmin=1
# )

# print(f"make_T_and_dust_model took {time.time() - t0} seconds")

# t0 = time.time()

# a_dust_stokes_planck = make_cmb_and_foreground(
#     freqs=planck_freqs, T=T_planck, a_cmb_stokes=a_cmb_stokes_planck,
#     dust_model=dust_model_planck, shape=shape_planck, wcs=wcs_planck,
#     res_arcmin=1, beam=True, beam_telescope="planck", beam_pas=None,
#     include_noise=True, rot=True, debug=True, give_dust_early=True
# )

# print(f"make_cmb_and_foreground took {time.time() - t0} seconds")

# for f_index, freq in enumerate(planck_freqs):

#     t0 = time.time()

#     bp_freqs, bp_weights = bandpass(telescope="planck", channel=freq, pa=None)

#     print(f"bandpass took {time.time() - t0} seconds")

#     mask = bp_freqs > 0

#     bp_freqs = bp_freqs[mask]
#     bp_weights = bp_weights[mask]

#     t0 = time.time()

#     hp_map = dust_model_planck.get_emission(bp_freqs * u.GHz, bp_weights)

#     print(f"get_emission took {time.time() - t0} seconds")

#     t0 = time.time()

#     # get_emission returns uK_RJ; convert to uK_CMB (approximate, using bandpass-weighted
#     # effective frequency) so this is comparable to the T-based reconstruction below
#     nu_eff = np.average(bp_freqs, weights=bp_weights) * u.GHz
#     hp_map_uK_CMB = hp_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu_eff))

#     # reproject all 3 Stokes planes jointly so Q/U get correct pol-angle rotation under gal->cel
#     dust_getemission_map = reproject.healpix2map(hp_map_uK_CMB.value, shape_planck, wcs_planck, rot="gal,cel")

#     # T-tensor reconstruction: elementwise SED scaling applied to the reference dust template
#     a_dust_stokes_planck_arr = np.array(a_dust_stokes_planck)  # (N_stokes, N_pix)
#     dust_T_recon = np.einsum('sp,sp->sp', T_planck[f_index, 1, :, :], a_dust_stokes_planck_arr)

#     ny_p, nx_p = shape_planck[-2:]
#     dust_T_recon_map = enmap.enmap(dust_T_recon.reshape(N_stokes, ny_p, nx_p), wcs_planck)

#     diff_map = dust_getemission_map - dust_T_recon_map

#     print(f"einsum and stuff {time.time() - t0} seconds")

#     plot_rect_map(dust_getemission_map, name=IMG_OUT_PATH+f"dust_getemission_planck_{freq}GHz")
#     plot_rect_map(dust_T_recon_map, name=IMG_OUT_PATH+f"dust_T_recon_planck_{freq}GHz")
#     plot_rect_map(diff_map, name=IMG_OUT_PATH+f"dust_diff_planck_{freq}GHz")

# --- ACT ---
t0 = time.time()

a_cmb_stokes_act, shape_act, wcs_act = make_a_cmb(
    dec_radius=4, ra_radius=8, seed=67, res_arcmin=1, flatsky=True
)
N_pix_act = len(a_cmb_stokes_act[0])

print(f"make_a_cmb took {time.time() - t0} seconds")

t0 = time.time()

T_act, dust_model_act = make_T_and_dust_model(
    N_pix=N_pix_act, shape=shape_act, wcs=wcs_act,
    beam_telescope="act", rot=True, freqs=act_freqs,
    dust_list=["d0"], res_arcmin=1
)

print(f"shape of dust_model_act.I_ref is {dust_model_act.I_ref.shape}")

print(f"make_T_and_dust_model took {time.time() - t0} seconds")

t0 = time.time()

a_dust_stokes_act = make_cmb_and_foreground(
    freqs=act_freqs, T=T_act, a_cmb_stokes=a_cmb_stokes_act,
    dust_model=dust_model_act, shape=shape_act, wcs=wcs_act,
    res_arcmin=1, beam=True, beam_telescope="act", beam_pas=None,
    include_noise=True, rot=True, debug=True, give_dust_early=True
)

# TODO: compare to pysm way as healpix maps without reprojecting

print(f"make_cmb_and_foreground took {time.time() - t0} seconds")

for f_index, freq in enumerate(act_freqs):
    
    t0 = time.time()

    bp_freqs, bp_weights = bandpass(telescope="act", channel=freq, pa=None)

    print(f"bandpass took {time.time() - t0} seconds")

    mask = bp_freqs > 0

    bp_freqs = bp_freqs[mask]
    bp_weights = bp_weights[mask]

    t0 = time.time()

    hp_map = dust_model_act.get_emission(bp_freqs * u.GHz, bp_weights)

    print(f"get_emission took {time.time() - t0} seconds")

    t0 = time.time()

    nu_eff = np.average(bp_freqs, weights=bp_weights) * u.GHz
    hp_map_uK_CMB = hp_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu_eff))

    dust_getemission_map = reproject.healpix2map(hp_map_uK_CMB.value, shape_act, wcs_act, rot="gal,cel")

    a_dust_stokes_act_arr = np.array(a_dust_stokes_act)  # (N_stokes, N_pix)
    dust_T_recon = np.einsum('sp,sp->sp', T_act[f_index, 1, :, :], a_dust_stokes_act_arr)

    ny_a, nx_a = shape_act[-2:]
    dust_T_recon_map = enmap.enmap(dust_T_recon.reshape(N_stokes, ny_a, nx_a), wcs_act)

    diff_map = dust_getemission_map - dust_T_recon_map

    print(f"einsum and processing took {time.time() - t0} seconds")

    plot_rect_map(dust_getemission_map, name=IMG_OUT_PATH+f"dust_getemission_act_{freq}GHz")
    plot_rect_map(dust_T_recon_map, name=IMG_OUT_PATH+f"dust_T_recon_act_{freq}GHz")
    plot_rect_map(diff_map, name=IMG_OUT_PATH+f"dust_diff_act_{freq}GHz")

'''
ny_p, nx_p = shape_planck[-2:]
ny_a, nx_a = shape_act[-2:]

# d_planck / d_act are (N_chan, N_stokes, ny, nx) once reshaped;
# plot_rect_map handles the (N_stokes, ny, nx) case per channel itself.
for idx, freq in enumerate(planck_freqs):
    channel_map = d_planck.reshape(N_chan_planck, N_stokes, ny_p, nx_p)[idx]
    plot_rect_map(channel_map, name=IMG_OUT_PATH+f"vector_reconstructed_planck_{freq}GHz")

for idx, freq in enumerate(act_freqs):
    channel_map = d_act.reshape(N_chan_act, N_stokes, ny_a, nx_a)[idx]
    plot_rect_map(channel_map, name=IMG_OUT_PATH+f"vector_reconstructed_act_{freq}GHz")
'''

