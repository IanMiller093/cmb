# file for plotting power spectra

# imports
import numpy as np
from pixell import enmap, reproject, utils, curvedsky, powspec, enplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_ps(imap=None, alms=None, name="power_spectra", lmax=5000):
    '''
    given an imap xor alms, saves a png of the power spectrum.

    imap: if given an imap of shape (3, _, _), apodizes, calculates cl, l, and w2, and then plots the 
    power specturm

    alms: if given alms of shape (3, _), calculates cl and l and then plots the power spectrum

    if neither imap nor alms is inputted, prints error message.  if both imap and alms is inputted, 
    prints different error message.

    returns nothing, but saves plots to png
    '''
    if (imap is not None and alms is not None):
        print("one of either imap or alms must be None, can't input both")
    
    # imap given
    elif(imap is not None):
        # apodize
        tapered_map = imap.copy()
        apod_width = max(5, imap[0].shape[1] // 8)
        taper_mask = enmap.apod(enmap.ones(imap[0].shape, imap[0].wcs), width=apod_width)

        # w2 calculation taken DIRECTLY from spherical harmonics notebook this time.
        w2 = np.sum(taper_mask.pixsizemap() * taper_mask**2) / (4*np.pi)
        for i in range(3):
            tapered_map[i] = taper_mask * imap[i]

        calc_alms = curvedsky.map2alm(tapered_map, lmax=lmax, spin=[0, 2])
        cl = curvedsky.alm2cl(calc_alms) / w2
        l = np.arange(cl.shape[1])

        # plot
        for i in range(3):
            plt.semilogy(l, cl[i] * l *(l+1) / 2 / np.pi, label=['TT', 'EE', 'BB'][i])
        plt.ylim(1e-4, 1e4)
        plt.legend()
    
        plt.savefig(name + ".png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # alms given
    elif(alms is not None):
        cl = curvedsky.alm2cl(alms)
        l = np.arange(cl.shape[1])

        # plot
        for i in range(3):
            plt.semilogy(l, cl[i] * l *(l+1) / 2 / np.pi, label=['TT', 'EE', 'BB'][i])
        plt.ylim(1e-4, 1e4)
        plt.legend()

        plt.savefig(name + ".png", dpi=150, bbox_inches="tight")
        plt.close()

    else:
        print("one of either imap or alms must be inputted, can't leave both blank")