import numpy as np
from pixell import enmap, reproject, utils, curvedsky, powspec, enplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def plot_2d_ps(imap, name="power_spectra_2d", fullsky=False):

    tapered_map = imap.copy()

    if not fullsky:
        apod_width = max(5, imap[0].shape[1] // 8)
        taper_mask = enmap.apod(
            enmap.ones(imap[0].shape, imap[0].wcs),
            width=apod_width
        )

        for i in range(3):
            tapered_map[i] = taper_mask * imap[i]

    harm = enmap.map2harm(tapered_map)
    cl = enmap.calc_ps2d(harm[:, None], harm[None, :])

    for i in range(3):
        plt.imshow(np.log(cl[i, i]), origin="lower")
        plt.colorbar(label="log(power)")
        plt.title(['TT', 'EE', 'BB'][i])

        plt.savefig(
            name + "_" + ['TT', 'EE', 'BB'][i] + ".png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()