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
    x = (h_P * nu) / (k_B * T_cmb)
    x_0 = (h_P * nu_0) / (k_B * T_cmb)
    x_d = (h_P * nu) / (k_B * T_dust)
    x_d0 = (h_P * nu_0) / (k_B * T_dust)
    
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
    car = reproject.healpix2map(hp_map, shape[-2:], wcs, rot=rot_str)
    return np.array(car).flatten()

def bandpass_sed_dust(bp_freqs, bp_weights, nu_0, beta_map, T_dust_map):
    norm = np.trapezoid(bp_weights, bp_freqs)
    bp_weights_normed = bp_weights / norm
    
    result = np.zeros(len(beta_map))
    prev_integrand = None
    
    for i, nu in enumerate(bp_freqs):
        mbb = dust_sed_scaling(nu, nu_0, beta_map, T_dust_map)
        curr_integrand = mbb * bp_weights_normed[i] # / rj_to_cmb_factor(nu) removed
        
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

Currently just look at intensity
'''

def make_a_cmb(dec_radius, ra_radius, seed, res_arcmin, flatsky):
    a_cmb = make_cmb(dec_radius=dec_radius, ra_radius=ra_radius, seed=seed, res=res_arcmin, beam=False, beam_telescope=None, flatsky=flatsky)
    shape = a_cmb.shape
    wcs = a_cmb.wcs
    
    a_cmb_1d = np.array(a_cmb[0]).flatten()

    N_pix = len(a_cmb_1d)
    
    return a_cmb_1d, shape, wcs



def make_T_and_dust_model(N_pix, shape, wcs, beam_telescope, rot, freqs, dust_list=["d0"], res_arcmin=1):

    N_chan = len(freqs)
    N_comp = 2

    sky_nside  = 1
    while hp.nside2resol(sky_nside, arcmin=True) > res_arcmin:
        sky_nside *= 2

    sky = pysm3.Sky(nside=sky_nside, preset_strings=dust_list)
    dust_model = sky.components[0]

    beta_dust = dust_model.mbb_index.value.squeeze()
    T_dust = dust_model.mbb_temperature.value.squeeze()
    # ref freq of 545 for intensity and 353 for polarized components
    nu_0_dust = dust_model.freq_ref_I.to("GHz").value

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
    T = np.zeros((N_chan, N_comp, N_pix))

    for f_index, (bp_freqs, bp_weights) in enumerate(bandpasses):
        T[f_index, 0, :] = bandpass_sed_cmb(bp_freqs, bp_weights)
        T[f_index, 1, :] = bandpass_sed_dust(bp_freqs, bp_weights, nu_0_dust, beta_dust, T_dust)
    
    return T, dust_model



def new_make_cmb_and_foreground(freqs, T, a_cmb_1d, dust_model, shape, wcs, res_arcmin=1, beam=True, beam_telescope="act", beam_pas=None, include_noise=True, rot=True, beam_type="jitter_cmb", beam_split="coadd"):

    # N_comp * N_chan is the number of individual maps we'll have
    # number of freq channels
    N_chan = len(freqs)
    # for cmb and dust
    N_comp = 2
    N_pix = len(a_cmb_1d)

    nyquist_lmax = (60 // res_arcmin) * 180 

    # TODO: check if astropy quantity and there's some << u.uk_CMB or something
    a_dust = dust_model.I_ref.value.squeeze()

    a_dust = hp_to_car_wrapper(a_dust, shape, wcs, rot=rot)

    a = np.vstack([a_cmb_1d, a_dust])

    d_tensor = np.einsum('fcp,cp->fp', T, a)

    if beam:
        ny, nx = shape[-2:]
        beamed_vector = np.array([])
        for i in range(N_chan):
            imap_np = d_tensor[i].reshape(ny, nx)
            imap_3 = enmap.ndmap(np.stack([imap_np]*3), wcs)
            beamed_map = apply_beam(imap=imap_3, alms=None, cls=None, lmax=nyquist_lmax, telescope=beam_telescope, channel=freqs[i], pa=(None if beam_pas is None else beam_pas[i]), beam_type=beam_type, split=beam_split)
            beamed_vector = np.concatenate((beamed_vector, np.array(beamed_map[0]).flatten()))
        d_vector = beamed_vector
    else:
        d_vector = d_tensor.flatten()

    if include_noise:
        d_vector = d_vector.reshape(N_chan, N_pix)
        for f_idx, nu in enumerate(freqs):
            pa = beam_pas[f_idx] if beam_pas is not None else None
            noise_map = accurate_noise(telescope=beam_telescope, channel=nu, shape=shape, wcs=wcs, pa=pa)
            d_vector[f_idx, :] += np.array(noise_map[0]).flatten()
        d_vector = d_vector.flatten()

    return d_vector