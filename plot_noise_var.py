import os
import matplotlib.pyplot as plt
from act_planck_noise import load_act_noise, load_planck_noise
from astropy.io import fits
import healpy as hp
import numpy as np
from pixell import enmap, reproject

IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

# Default channel sets — adjust to match whichever channels you actually have on disk
PLANCK_CHANNELS = [100, 143, 217, 353, 545, 857]
ACT_CHANNELS = [90, 150, 220]


def _avg_variance_from_ivar(ivar_np):
    """Mean of 1/ivar over pixels where ivar > 0. Returns np.nan if no valid pixels."""
    valid = ivar_np > 0
    if not valid.any():
        return np.nan
    return np.mean(1.0 / ivar_np[valid])


def planck_channel_avg_variance(channel):
    """Returns (var_I, var_Q, var_U) in uK^2_CMB for a given Planck channel.
    var_Q/var_U are np.nan if the ivar map doesn't include polarization."""
    ivar_full = load_planck_noise(channel)
    ivar_np = np.array(ivar_full)

    n_comp = ivar_np.shape[0] if ivar_np.ndim == 3 else 1
    if ivar_np.ndim == 2:
        ivar_np = ivar_np[np.newaxis, ...]

    var_I = _avg_variance_from_ivar(ivar_np[0])
    var_Q = _avg_variance_from_ivar(ivar_np[1]) if n_comp > 1 else np.nan
    var_U = _avg_variance_from_ivar(ivar_np[2]) if n_comp > 2 else np.nan

    return var_I, var_Q, var_U


def act_channel_avg_variance(channel, pa=None):
    """Returns var_I in uK^2_CMB for a given ACT channel (ACT ivar is I-only)."""
    ivar = load_act_noise(channel, pa)
    ivar_np = np.array(ivar)  # shape (ny, nx)
    return _avg_variance_from_ivar(ivar_np)


def plot_noise_variance_by_channel(
    planck_channels=PLANCK_CHANNELS,
    act_channels=ACT_CHANNELS,
    act_pas=None,
    include_planck_pol=True,
    savename="noise_variance_by_channel.png",
):
    """
    Plots average noise variance (uK^2_CMB) vs frequency channel (GHz)
    for Planck and ACT, using the native ivar maps.
    """
    act_pas = act_pas or {}

    # --- Planck ---
    planck_freqs, planck_var_I, planck_var_P = [], [], []
    for ch in planck_channels:
        try:
            var_I, var_Q, var_U = planck_channel_avg_variance(ch)
        except FileNotFoundError as e:
            print(f"[planck] skipping {ch} GHz, file not found: {e}")
            continue
        planck_freqs.append(ch)
        planck_var_I.append(var_I)
        planck_var_P.append(np.nanmean([var_Q, var_U]))

    # --- ACT ---
    act_freqs, act_var_I = [], []
    for ch in act_channels:
        try:
            var_I = act_channel_avg_variance(ch, pa=act_pas.get(ch))
        except FileNotFoundError as e:
            print(f"[act] skipping {ch} GHz, file not found: {e}")
            continue
        act_freqs.append(ch)
        act_var_I.append(var_I)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(planck_freqs, planck_var_I, "o-", color="tab:blue", label="Planck (I)")
    if include_planck_pol:
        ax.plot(planck_freqs, planck_var_P, "o--", color="tab:blue",
                 alpha=0.5, label="Planck (Q/U avg)")

    ax.plot(act_freqs, act_var_I, "s-", color="tab:red", label="ACT (I)")

    ax.set_yscale("log")
    ax.set_xlabel("Frequency channel (GHz)")
    ax.set_ylabel(r"Average noise variance ($\mu K^2_{CMB}$)")
    ax.set_title("Average Noise Variance by Frequency Channel")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()

    os.makedirs(IMG_OUT_PATH, exist_ok=True)
    outpath = os.path.join(IMG_OUT_PATH, savename)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {outpath}")