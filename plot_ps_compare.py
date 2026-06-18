# comparison power spectra plotting

import numpy as np
from pixell import enmap, curvedsky
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# color palettes: each polarization gets a base hue,
# and the two maps get a light vs dark shade of that hue.
_SPECTRA_COLORS = {
    #        map1 (lighter)    map2 (darker/saturated)
    "TT": ("#5B9BD5",         "#1A5C9A"),   # blues
    "EE": ("#F4A261",         "#C45E0A"),   # oranges
    "BB": ("#6DC972",         "#1F7A29"),   # greens
}
_SPECTRA_LABELS = ["TT", "EE", "BB"]


def _get_cl(imap, lmax=5000, fullsky=False):
    """
    Given a Pixell enmap of shape (3, ny, nx), apodize (unless fullsky),
    convert to alms, and return (l, cl, w2).
    cl has shape (3, lmax+1) — the diagonal [TT, EE, BB] auto-spectra.
    """
    tapered_map = imap.copy()
    w2 = 1.0

    if not fullsky:
        apod_width = max(5, imap[0].shape[1] // 8)
        taper_mask = enmap.apod(
            enmap.ones(imap[0].shape, imap[0].wcs), width=apod_width
        )
        w2 = np.sum(taper_mask.pixsizemap() * taper_mask ** 2) / (4 * np.pi)
        for i in range(3):
            tapered_map[i] = taper_mask * imap[i]

    alms = curvedsky.map2alm(tapered_map, lmax=lmax, spin=[0, 2])

    # cross-spectrum trick to get the 3x3 matrix, then pull the diagonal
    # alms[:,None,:] has shape (3,1,nalm); alms[None,:,:] has shape (1,3,nalm)
    cl_full = curvedsky.alm2cl(alms[:, None, :], alms[None, :, :])  # (3,3,lmax+1)
    cl_diag = np.array([cl_full[i, i] for i in range(3)])           # (3, lmax+1)

    l = np.arange(cl_diag.shape[1])
    return l, cl_diag / w2


def plot_ps_compare(
    imap1,
    imap2,
    name="/data6/miller42/cmb_sim/image_outputs/power_spectra_comparison",
    fullsky=False,
    map1_label="Map 1",
    map2_label="Map 2",
    lmax=5000,
    title="CMB Power Spectra Comparison"
):
    """
    Plot TT, EE, and BB power spectra of two Pixell maps on the same axes
    for side-by-side comparison.

    Parameters
    ----------
    imap1, imap2 : enmap.ndmap
        Pixell maps of shape (3, ny, nx) — Stokes [T, Q, U].
    name : str
        Output filename (without extension).
    lmax : int
        Maximum multipole for the alm transform.
    fullsky : bool
        If True, skip apodization (use for full-sky / mock maps).
    map1_label : str
        Legend label for the first map.
    map2_label : str
        Legend label for the second map.

    Returns
    -------
    None  — saves <name>.png to disk.
    """

    l1, cl1 = _get_cl(imap1, lmax=lmax, fullsky=fullsky)
    l2, cl2 = _get_cl(imap2, lmax=lmax, fullsky=fullsky)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.tick_params(colors="#CCCCDD", which="both")
    ax.xaxis.label.set_color("#CCCCDD")
    ax.yaxis.label.set_color("#CCCCDD")
    ax.title.set_color("#EEEEFF")

    for i, spec in enumerate(_SPECTRA_LABELS):
        c1, c2 = _SPECTRA_COLORS[spec]

        def dl(l, cl):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(l > 0, cl * l * (l + 1) / (2 * np.pi), np.nan)

        dl1 = dl(l1, cl1[i])
        dl2 = dl(l2, cl2[i])

        ax.semilogy(l1, dl1, color=c1, lw=1.6, alpha=0.90,
                    label=f"{spec}  {map1_label}")
        ax.semilogy(l2, dl2, color=c2, lw=1.6, alpha=0.90,
                    linestyle="--", label=f"{spec}  {map2_label}")

    ax.set_ylim(1e-6, 1e8)
    ax.set_xlim(left=2)
    ax.set_xlabel(r"Multipole $\ell$", fontsize=13)
    ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell\,/\,2\pi$", fontsize=13)
    ax.set_title(title, fontsize=14, pad=12)

    legend = ax.legend(
        ncol=2,
        fontsize=10,
        framealpha=0.25,
        facecolor="#1A1A2E",
        edgecolor="#444466",
        labelcolor="#DDDDEE",
        loc="upper right",
    )

    ax.yaxis.grid(True, which="both", linestyle=":", linewidth=0.5,
                  color="#333355", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(name + ".png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {name}.png")