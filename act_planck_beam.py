import numpy as np
from astropy.io import fits
from pixell import curvedsky, enmap
import healpy as hp


def load_planck_beam(channel, lmax):
    filepath = "/data6/miller42/cmb_sim/data_files_and_folders/BeamWf_HFI_R3.01/Bl_T_R3.01_fullsky_" + str(channel) + "evenx" + str(channel) + "even.fits"
    
    with fits.open(filepath) as f:
        binary_data  = f[1].data
        bl_file = binary_data.field(0).astype(float)
        
    ell_file = np.arange(len(bl_file))

    # 1 at ell = 0
    bl_file = bl_file / bl_file[0]

    ell_out = np.arange(lmax + 1)
    bl = np.interp(ell_out, ell_file, bl_file, left=bl_file[0], right=0.0)

    # of size 3, lmax
    return np.array([bl, bl, bl])



def load_act_beam(channel, lmax, pa, beam_type="jitter_cmb", split="coadd"):
    default_pas = {
            30 : 7,
            40 : 7,
            90 : 6,
            150 : 6,
            220 : 4
        }
    
    if pa is None:
        pa = default_pas[channel]
    
    filepath = "/data6/miller42/cmb_sim/data_files_and_folders/act_dr6.02_main_beams/main_beams/nominal/" + split + "_pa" + str(pa) + "_f" + str(channel).zfill(3) + "_night_beam_tform_" + beam_type + ".txt"
    
    data = np.loadtxt(filepath)
    ell_file = data[:, 0].astype(int)
    bl_file  = data[:, 1]

    bl_file = bl_file / bl_file[0]

    ell_out = np.arange(lmax + 1)
    bl = np.interp(ell_out, ell_file, bl_file, left=bl_file[0], right=0.0)

    return np.array([bl, bl, bl])



def apply_planck_beam(imap=None, alms=None, cls=None, lmax=5000, channel=143):
    beam_ell = load_planck_beam(channel, lmax)

    if cls is not None:
        bl = beam_ell[0]
        return cls * (bl ** 2)[:, np.newaxis] if cls.ndim == 2 else cls * bl ** 2

    if alms is None:
        alms = curvedsky.map2alm(imap, lmax=lmax)
    alms = curvedsky.almxfl(alms, beam_ell)
    return curvedsky.alm2map(alms, imap.copy())


def apply_act_beam(imap=None, alms=None, cls=None, lmax=5000, pa=5, channel=150, beam_type="jitter_cmb", split="coadd"):
    beam_ell = load_act_beam(channel=channel, lmax=lmax, pa=pa, beam_type=beam_type, split=split)

    if cls is not None:
        bl = beam_ell[0]
        return cls * (bl ** 2)[:, np.newaxis] if cls.ndim == 2 else cls * bl ** 2

    if alms is None:
        alms = curvedsky.map2alm(imap, lmax=lmax)
    alms = curvedsky.almxfl(alms, beam_ell)
    return curvedsky.alm2map(alms, imap.copy())

def apply_beam(imap=None, alms=None, cls=None, lmax=5000, telescope="planck", channel=100, pa=5, beam_type="jitter_cmb", split="coadd"):
    assert telescope in ["planck", "act"]
    
    if cls is not None:
        if (telescope == "planck"):
            return apply_planck_beam(cls=cls, lmax=lmax, channel=channel)

        else:
            return apply_act_beam(cls=cls, lmax=lmax, channel=channel, pa=pa, beam_type=beam_type, split=split)

    elif imap is not None:
        if (telescope == "planck"):
            beamed = apply_planck_beam(imap=imap, lmax=lmax, channel=channel)

        else:
            beamed = apply_act_beam(imap=imap, lmax=lmax, pa=pa, channel=channel, beam_type=beam_type, split=split)

    else:
        if (telescope == "planck"):
            beamed = apply_planck_beam(alms=alms, lmax=lmax, channel=channel)

        else:
            beamed = apply_act_beam(alms=alms, lmax=lmax, pa=pa, channel=channel, beam_type=beam_type, split=split)

    return enmap.apply_window(beamed, order=1)



'''
what to do if given vector?

this would be full sim vector -> full sim map -> alms -> almxfl -> back to map -> back to vector?


'''