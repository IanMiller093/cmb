from astropy.io import fits
import healpy as hp
import numpy as np
from pixell import enmap, reproject

PLANCK_DIR_GPC = "/data5/planck/pr3"
ACT_DIR_GPC = "/data5/act/maps/dr6v4_20230316"

ACT_SETS = [0, 1, 2, 3]


def load_planck_noise(channel):
    ivar_full = None

    for ring in ["evenring", "oddring"]:
        fname = (
            f"{PLANCK_DIR_GPC}/HFI_SkyMap_{channel}_2048_R3.01_full-{ring}_ivar.fits"
        )
        # read all 3 Stokes components (I, Q, U)
        ring_ivar = enmap.read_map(fname)
        ivar_full = ring_ivar if ivar_full is None else ivar_full + ring_ivar

    ivar_full /= 2.0
    return ivar_full


def load_act_noise(channel, pa):
    default_pas = {
        90 : 6,
        150: 6,
        220: 4,
    }

    if pa is None:
        pa = default_pas[channel]

    ivar = None

    for s in ACT_SETS:
        fname = (
            f"{ACT_DIR_GPC}/cmb_night_pa{pa}_f{channel:03d}_3pass_4way_set{s}_ivar.fits"
        )

        split_ivar = enmap.read_map(fname)
        ivar = split_ivar if ivar is None else ivar + split_ivar

    ivar /= len(ACT_SETS)

    return ivar


def planck_noise(channel, shape, wcs):
    # Empirically confirmed (median ivar ~9.5e-6 -> implied std ~324 uK with
    # NO extra conversion) that Planck HFI R3.01 ivar maps from this pipeline
    # are already natively in mu K^(-2)_CMB, not K^(-2)_CMB. Do NOT multiply
    # by 1e12 here, that was based on an unverified assumption about the
    # standard Planck convention and was wrong for these specific files.
    ivar_full = load_planck_noise(channel)

    shape3 = (3,) + shape[-2:]
    ivar = enmap.project(ivar_full, shape3, wcs, order=0)

    ivar_np = np.array(ivar)

    std = np.where(ivar_np > 0, 1.0 / np.sqrt(ivar_np), 0.0)

    return enmap.enmap(np.random.standard_normal(shape3) * std, wcs)


def act_noise(channel, shape, wcs, pa):
    # Empirically confirmed (median ivar ~9.4e-5 -> implied std ~103 uK with
    # no conversion) that ACT DR6v4 ivar maps are natively in mu K^(-2)_CMB.
    # No conversion needed.
    ivar_full = load_act_noise(channel, pa)

    # project to target map geometry (no fake component axis)
    ivar = enmap.project(ivar_full, shape[-2:], wcs, order=0)

    ivar_np = np.array(ivar)

    std = np.where(ivar_np > 0, 1.0 / np.sqrt(np.maximum(ivar_np, 0)), 0.0)

    noise_TQU = np.random.standard_normal((3,) + shape[-2:]) * std

    return enmap.enmap(noise_TQU, wcs)


def accurate_noise(telescope, channel, shape, wcs, pa=None):
    assert telescope in ["act", "planck"]

    if telescope == "planck":
        return planck_noise(channel=channel, shape=shape, wcs=wcs)
    else:
        return act_noise(channel=channel, shape=shape, wcs=wcs, pa=pa)