from pixell_cmb import make_cmb
from pysm_foreground import make_foreground
from act_planck_beam import apply_beam
from act_planck_noise import accurate_noise
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
from pixell import enmap, reproject, utils
from bandpass import bandpass  
from plot_rectangular_map import plot_rect_map 

'''
new cmb_and_foreground architecture below!!!
'''

# constants yay
h_P = 6.626e-34
k_B = 1.381e-23
T_cmb = 2.7255

# sed scaling function for cmb
def cmb_sed_scaling(nu):
    return 1.0

# sed scaling function for dust
def dust_sed_scaling(nu, nu_0, beta, T_dust):
    nu_hz = nu * 1e9
    nu_0_hz = nu_0 * 1e9

    x = (h_P * nu_hz)   / (k_B * T_cmb)
    x_0 = (h_P * nu_0_hz) / (k_B * T_cmb)
    x_d = (h_P * nu_hz)   / (k_B * T_dust)
    x_d0 = (h_P * nu_0_hz) / (k_B * T_dust)

    return (nu / nu_0)**(beta - 1) * ((np.exp(x_d0) - 1)/(np.exp(x_d) - 1)) * ((np.exp(x) - 1)/(np.exp(x_0) - 1))**2 * (np.exp(x_0)/np.exp(x))

# might implement later
'''
def rj_to_cmb_factor(nu):
    x  = h_P * nu * 1e9 / (k_B * T_cmb)
    ex = np.exp(x)
    return x**2 * ex / (ex - 1)**2
'''

def hp_to_car_wrapper(hp_map, shape, wcs, rot=True):
    rot_str = "gal,cel" if rot else None
    car = np.array(reproject.healpix2map(hp_map, shape[-2:], wcs, rot=rot_str))
    if car.ndim == 3:  # (N_stokes, ny, nx) -> (N_stokes, npix)
        return car.reshape(car.shape[0], -1)
    return car.flatten()

def bandpass_sed_dust(bp_freqs, bp_weights, nu_0, beta_map, T_dust_map):
    norm = np.trapezoid(bp_weights, bp_freqs)
    bp_weights_normed = bp_weights / norm
    
    result = np.zeros(len(beta_map))
    prev_integrand = None
    
    for i, nu in enumerate(bp_freqs):
        mbb = dust_sed_scaling(nu, nu_0, beta_map, T_dust_map)
        curr_integrand = mbb * bp_weights_normed[i] # / rj_to_cmb_factor(nu)
        
        if prev_integrand is not None:
            # trapezoid rule step: 0.5 * (f_i + f_{i-1}) * dx
            result += 0.5 * (prev_integrand + curr_integrand) * (bp_freqs[i] - bp_freqs[i-1])
        
        prev_integrand = curr_integrand
    
    return result

def bandpass_sed_cmb(bp_freqs, bp_weights):
    return 1.0

'''
NEW lacrosse pipeline segue finance: 
cmb w/o beam -> 
dust w/o beam... w/ bandpass? -> 
stack together -> 
apply beams to channels of the same frequency??? ->
add noise ->
return T and d

Now returns full T, Q, U (Stokes) maps, not just intensity.
'''

def make_a_cmb(dec_radius, ra_radius, seed, res_arcmin, flatsky):
    a_cmb = make_cmb(dec_radius=dec_radius, ra_radius=ra_radius, seed=seed, res=res_arcmin, beam=False, beam_telescope=None, flatsky=flatsky)
    shape = a_cmb.shape
    wcs = a_cmb.wcs

    # T, Q, U
    a_cmb_stokes = np.stack([np.array(a_cmb[s]).flatten() for s in range(3)], axis=0)

    return a_cmb_stokes, shape, wcs



def make_T_and_dust_model(N_pix, shape, wcs, beam_telescope, rot, freqs, dust_list=["d0"], res_arcmin=1):

    N_chan = len(freqs)
    N_comp = 2
    N_stokes = 3

    sky_nside  = 1
    while hp.nside2resol(sky_nside, arcmin=True) > res_arcmin:
        sky_nside *= 2

    sky = pysm3.Sky(nside=sky_nside, max_nside=sky_nside, preset_strings=dust_list)
    dust_model = sky.components[0]

    beta_dust = dust_model.mbb_index.value.squeeze()
    T_dust = dust_model.mbb_temperature.value.squeeze()
    # ref freq of 545 for intensity and 353 for polarized components
    nu_0_dust_I = dust_model.freq_ref_I.to("GHz").value
    nu_0_dust_P = dust_model.freq_ref_P.to("GHz").value
    nu_0_dust_stokes = [nu_0_dust_I, nu_0_dust_P, nu_0_dust_P]

    if beta_dust.ndim == 0:
        beta_dust = np.full(N_pix, beta_dust.item())
    else:
        beta_dust = hp_to_car_wrapper(beta_dust, shape, wcs, rot=rot)

    if T_dust.ndim == 0:
        T_dust = np.full(N_pix, T_dust.item())
    else:
        T_dust = hp_to_car_wrapper(T_dust, shape, wcs, rot=rot)

    bandpasses = [bandpass(telescope=beam_telescope, channel=nu) for nu in freqs]

    # actual construct ts T now!!!  Tensor with a bunch of maps as opposed to just one huge matrix
    T = np.zeros((N_chan, N_comp, N_stokes, N_pix))

    for f_index, (bp_freqs, bp_weights) in enumerate(bandpasses):
        cmb_scaling = bandpass_sed_cmb(bp_freqs, bp_weights)
        print("[debug] bp_freqs[0] and bp_freqs[-1]: ")
        print(bp_freqs[0], bp_freqs[-1])
        for s in range(N_stokes):
            T[f_index, 0, s, :] = cmb_scaling
            T[f_index, 1, s, :] = bandpass_sed_dust(bp_freqs, bp_weights, nu_0_dust_stokes[s], beta_dust, T_dust)
    
    print(f"[debug] dust_list={dust_list}, beta_dust ndim={dust_model.mbb_index.value.ndim}, "
        f"beta_dust sample={np.atleast_1d(dust_model.mbb_index.value.squeeze())[:3]}, "
        f"T_dust sample={np.atleast_1d(dust_model.mbb_temperature.value.squeeze())[:3]}")

    return T, dust_model



def make_cmb_and_foreground(freqs, T, a_cmb_stokes, dust_model, shape, wcs, res_arcmin=1, beam=True, beam_telescope="act", beam_pas=None, include_noise=True, rot=True, beam_type="jitter_cmb", beam_split="coadd", debug=False, give_dust_early=True):

    # N_comp * N_chan is the number of individual maps we'll have
    # number of freq channels
    N_chan = len(freqs)
    # for cmb and dust
    N_comp = 2
    N_stokes = 3
    N_pix = len(a_cmb_stokes[0])

    nyquist_lmax = (60 // res_arcmin) * 180 

    nu_ref_I = dust_model.freq_ref_I  # dust model's reference frequency for I
    nu_ref_P = dust_model.freq_ref_P  # dust model's reference frequency for Q/U

    dust_ref_attrs = ["I_ref", "Q_ref", "U_ref"]
    dust_ref_freqs = [nu_ref_I, nu_ref_P, nu_ref_P]

    refs_uK = []
    for attr, nu_ref in zip(dust_ref_attrs, dust_ref_freqs):
        ref = getattr(dust_model, attr, None)
        if ref is None:
            refs_uK.append(np.zeros_like(np.atleast_1d(dust_model.I_ref.value.squeeze())))
        else:
            refs_uK.append(ref.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu_ref)).value.squeeze())

    refs_uK_arr = np.array(refs_uK)  # (3, npix_hp) -- I, Q, U together
    a_dust_stokes_car = hp_to_car_wrapper(refs_uK_arr, shape, wcs, rot=rot)  # single joint call
    a_dust_stokes = [a_dust_stokes_car[s] for s in range(N_stokes)]

    if give_dust_early:
        return a_dust_stokes

    if debug:
        stokes_labels = ["I", "Q", "U"]
        for s, label in enumerate(stokes_labels):
            cmb_rms = np.sqrt(np.mean(a_cmb_stokes[s]**2))
            dust_rms = np.sqrt(np.mean(a_dust_stokes[s]**2))
            dust_is_zero = np.allclose(a_dust_stokes[s], 0.0)
            print(f"[debug] Stokes {label}: cmb_rms={cmb_rms:.4g} uK, "
                  f"dust_rms={dust_rms:.4g} uK"
                  f"{'  <-- dust ref map missing/zero!' if dust_is_zero and label != 'I' else ''}")

    # a: (N_comp, N_stokes, N_pix)
    a = np.stack(
        [np.vstack([a_cmb_stokes[s], a_dust_stokes[s]]) for s in range(N_stokes)],
        axis=1
    )

    # d_tensor: (N_chan, N_stokes, N_pix)
    d_tensor = np.einsum('fcsp,csp->fsp', T, a)

    if debug:
        for s, label in enumerate(stokes_labels):
            sig_rms = np.sqrt(np.mean(d_tensor[:, s, :]**2))
            print(f"[debug] Stokes {label}: pre-beam/noise signal rms across all channels = {sig_rms:.4g} uK")

    if beam:
        ny, nx = shape[-2:]
        beamed_channels = []
        for i in range(N_chan):
            imap_i = enmap.ndmap(d_tensor[i].reshape(N_stokes, ny, nx), wcs)
            beamed_map = apply_beam(imap=imap_i, alms=None, cls=None, lmax=nyquist_lmax, telescope=beam_telescope, channel=freqs[i], pa=(None if beam_pas is None else beam_pas[i]), beam_type=beam_type, split=beam_split)
            beamed_channels.append(np.array(beamed_map).reshape(N_stokes, N_pix))
        d_vector = np.stack(beamed_channels, axis=0)
    else:
        d_vector = d_tensor.copy()

    if include_noise:
        for f_idx, nu in enumerate(freqs):
            pa = beam_pas[f_idx] if beam_pas is not None else None
            noise_map = accurate_noise(telescope=beam_telescope, channel=nu, shape=shape, wcs=wcs, pa=pa)
            noise_arr = np.array(noise_map).reshape(N_stokes, N_pix)

            if debug:
                for s, label in enumerate(stokes_labels):
                    sig_rms = np.sqrt(np.mean(d_vector[f_idx, s, :]**2))
                    noise_rms = np.sqrt(np.mean(noise_arr[s]**2))
                    ratio = sig_rms / noise_rms if noise_rms > 0 else np.inf
                    print(f"[debug] {nu} GHz, Stokes {label}: "
                          f"post-beam signal rms={sig_rms:.4g} uK, "
                          f"noise rms={noise_rms:.4g} uK, signal/noise={ratio:.3g}")

            d_vector[f_idx] += noise_arr

    # d_vector: (N_chan, N_stokes, N_pix)
    return d_vector