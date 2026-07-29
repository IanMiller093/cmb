"""
Plots the ACT 150 GHz (pa=6) bandpass overlaid with the dust SED curve,
visualizing the two functions that get multiplied/integrated together
to form one (channel, component) entry of the mixing matrix T.

Run this in your environment where bandpass.py, dust_sed_scaling, etc.
are importable and the data files are accessible.
"""

import numpy as np
import matplotlib.pyplot as plt

from bandpass import bandpass
from cmb_and_foreground import dust_sed_scaling  # adjust import path if this lives elsewhere

def plot_bandpass_sed_overlay():

    # ---- config for this slide ----
    TELESCOPE = "act"
    CHANNEL = 150
    PA = 6

    # typical d0 fixed dust params (single-component MBB, not pulled from a pixel)
    BETA_DUST = 1.54
    T_DUST = 20.0  # Kelvin
    NU_0_DUST = 353.0  # GHz, standard Planck dust reference freq for intensity

    IMG_OUT_PATH = "/data6/miller42/cmb_sim/image_outputs/"

    # ---- load real bandpass ----
    bp_freqs, bp_weights = bandpass(telescope=TELESCOPE, channel=CHANNEL, pa=PA)

    # normalize bandpass to peak 1 for plotting only (keeps y-axis shape clean)
    bp_weights_norm = bp_weights / bp_weights.max()

    # ---- compute dust SED over the same frequency range ----
    # dust_sed_scaling expects scalar or array nu; beta/T_dust here are scalars
    freq_range = np.linspace(bp_freqs.min(), bp_freqs.max(), 500)
    dust_sed = dust_sed_scaling(freq_range, NU_0_DUST, BETA_DUST, T_DUST)

    # normalize dust SED to 1 at band center for visual comparability
    band_center = bp_freqs[np.argmax(bp_weights)]
    dust_sed_at_center = dust_sed_scaling(band_center, NU_0_DUST, BETA_DUST, T_DUST)
    dust_sed_norm = dust_sed / dust_sed_at_center

    # ---- plot ----
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(bp_freqs, bp_weights_norm, color="#1f77b4", linewidth=2, label=f"ACT {CHANNEL} GHz bandpass (pa={PA})")
    ax1.fill_between(bp_freqs, bp_weights_norm, alpha=0.15, color="#1f77b4")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Bandpass response (normalized)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(freq_range, dust_sed_norm, color="#ff7f0e", linewidth=2, linestyle="--",
            label=f"Dust SED (β={BETA_DUST}, T={T_DUST}K)")
    ax2.set_ylabel("Dust SED (normalized to band center)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.title("One Element of T: Bandpass × Dust SED (ACT 150 GHz)")
    fig.tight_layout()
    plt.savefig(IMG_OUT_PATH + "bandpass_dust_sed_overlay.png", dpi=200)
    print("Saved plot.")