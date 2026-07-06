import os

import matplotlib.pyplot as plt
import numpy as np
from pixell import enmap, utils

from act_planck_noise import load_planck_noise, load_act_noise

IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

# Default channel sets
PLANCK_CHANNELS = [100, 143, 217, 353, 545, 857]
ACT_CHANNELS = [90, 150, 220]


def _make_geometry(dec_radius, ra_radius, res_arcmin):
    """
    Construct a common CAR geometry for comparing all noise maps.
    """
    box = np.array([
        [-dec_radius,  ra_radius],
        [ dec_radius, -ra_radius]
    ]) * utils.degree

    shape, wcs = enmap.geometry(
        pos=box,
        res=res_arcmin * utils.arcmin,
        proj="car",
    )

    return shape, wcs


def _avg_variance_from_ivar(ivar_np):
    """
    Mean of 1/ivar over pixels where ivar > 0.
    Returns np.nan if there are no valid pixels.
    """
    valid = ivar_np > 0

    if not np.any(valid):
        return np.nan

    return np.mean(1.0 / ivar_np[valid])


def planck_channel_avg_variance(
    channel,
    dec_radius,
    ra_radius,
    res_arcmin,
    rot=False,
):
    """
    Compute average Planck noise variance after projecting onto a common map.

    Returns
    -------
    var_I, var_Q, var_U
        Average variance (uK^2_CMB) of each Stokes component.
    """

    shape, wcs = _make_geometry(
        dec_radius,
        ra_radius,
        res_arcmin,
    )

    ivar_full = load_planck_noise(channel)
    print(channel, ivar_full.shape)

    shape_out = (3,) + shape[-2:]

    ivar = enmap.project(
        ivar_full,
        shape_out,
        wcs,
        order=0,
    )

    ivar_np = np.asarray(ivar)

    n_comp = ivar_np.shape[0] if ivar_np.ndim == 3 else 1

    if ivar_np.ndim == 2:
        ivar_np = ivar_np[np.newaxis, ...]

    var_I = _avg_variance_from_ivar(ivar_np[0])
    var_Q = _avg_variance_from_ivar(ivar_np[1]) if n_comp > 1 else np.nan
    var_U = _avg_variance_from_ivar(ivar_np[2]) if n_comp > 2 else np.nan

    return var_I, var_Q, var_U


def act_channel_avg_variance(
    channel,
    dec_radius,
    ra_radius,
    res_arcmin,
    pa=None,
    rot=False,
):
    """
    Compute average ACT noise variance after projecting onto a common map.

    Returns
    -------
    var_I
        Average variance (uK^2_CMB)
    """

    shape, wcs = _make_geometry(
        dec_radius,
        ra_radius,
        res_arcmin,
    )

    ivar_full = load_act_noise(channel, pa)

    if rot:
        print(
            "Warning: rot=True requested, but loaded ACT noise maps are "
            "already CAR maps. No additional rotation is applied."
        )

    ivar = enmap.project(
        ivar_full,
        shape[-2:],
        wcs,
        order=0,
    )

    ivar_np = np.asarray(ivar)

    return _avg_variance_from_ivar(ivar_np)


def plot_noise_variance_by_channel(
    dec_radius,
    ra_radius,
    res_arcmin,
    rot=False,
    planck_channels=PLANCK_CHANNELS,
    act_channels=ACT_CHANNELS,
    act_pas=None,
    include_planck_pol=True,
    savename="noise_variance_by_channel.png",
):
    """
    Compare average noise variance after projecting every instrument onto the
    same CAR map geometry.

    Parameters
    ----------
    dec_radius : float
        Half-height of output map in degrees.

    ra_radius : float
        Half-width of output map in degrees.

    res_arcmin : float
        Pixel size in arcminutes.

    rot : bool
        Reserved for future Galactic->Celestial rotation support. Currently
        ignored because the loaded noise maps are already CAR projections.

    planck_channels : iterable
        Planck frequencies.

    act_channels : iterable
        ACT frequencies.

    act_pas : dict
        Optional mapping {channel: pa}.

    include_planck_pol : bool
        Plot average of Q/U variance.

    savename : str
        Output filename.
    """

    if act_pas is None:
        act_pas = {}

    ##############################
    # Planck
    ##############################

    planck_freqs = []
    planck_var_I = []
    planck_var_P = []

    for ch in planck_channels:

        try:
            var_I, var_Q, var_U = planck_channel_avg_variance(
                channel=ch,
                dec_radius=dec_radius,
                ra_radius=ra_radius,
                res_arcmin=res_arcmin,
                rot=rot,
            )

        except FileNotFoundError as e:
            print(f"[Planck] Skipping {ch} GHz ({e})")
            continue

        planck_freqs.append(ch)
        planck_var_I.append(var_I)
        planck_var_P.append(np.nanmean([var_Q, var_U]))

    ##############################
    # ACT
    ##############################

    act_freqs = []
    act_var_I = []

    for ch in act_channels:

        try:
            var_I = act_channel_avg_variance(
                channel=ch,
                dec_radius=dec_radius,
                ra_radius=ra_radius,
                res_arcmin=res_arcmin,
                pa=act_pas.get(ch),
                rot=rot,
            )

        except FileNotFoundError as e:
            print(f"[ACT] Skipping {ch} GHz ({e})")
            continue

        act_freqs.append(ch)
        act_var_I.append(var_I)

    ##############################
    # Plot
    ##############################

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        planck_freqs,
        planck_var_I,
        "o-",
        color="tab:blue",
        label="Planck (I)",
    )

    if include_planck_pol:
        ax.plot(
            planck_freqs,
            planck_var_P,
            "o--",
            color="tab:blue",
            alpha=0.5,
            label="Planck (Q/U avg)",
        )

    ax.plot(
        act_freqs,
        act_var_I,
        "s-",
        color="tab:red",
        label="ACT (I)",
    )

    ax.set_yscale("log")

    ax.set_xlabel("Frequency channel (GHz)")
    ax.set_ylabel(r"Average noise variance ($\mu K^2_{\rm CMB}$)")
    ax.set_title(
        "Average Noise Variance After Projection to Common Geometry"
    )

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    os.makedirs(IMG_OUT_PATH, exist_ok=True)

    outpath = os.path.join(
        IMG_OUT_PATH,
        savename,
    )

    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {outpath}")