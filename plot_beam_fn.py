import numpy as np
import matplotlib.pyplot as plt
from act_planck_beam import load_planck_beam, load_act_beam 


def plot_beams(lmax=5000, planck_channels=(100, 143, 217), act_channels=(90, 150),
                act_pas=None, beam_type="jitter_cmb", split="coadd",
                logy=True, savepath="beams.png"):

    ell = np.arange(lmax + 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    for ch in planck_channels:
        bl = load_planck_beam(ch, lmax)[0]
        ax.plot(ell, bl, label=f"Planck {ch} GHz")

    for ch in act_channels:
        pa = None if act_pas is None else act_pas.get(ch)
        bl = load_act_beam(ch, lmax, pa, beam_type=beam_type, split=split)[0]
        ax.plot(ell, bl, linestyle="--", label=f"ACT {ch} GHz (pa={pa})")

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2)

    ax.set_xlim(0, lmax)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$b_\ell$")
    ax.set_title("Beam transfer functions")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print("yuh plot beam finished")