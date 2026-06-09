# imports
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import urllib.request
from pixell import enmap, reproject, utils, curvedsky, powspec, enplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from plot_power_spectrum import plot_ps

# frequency that dust is taken from
f = 150
shape, wcs = np.array([[-90, 180], [90, -180]]) * utils.degree

# map made solely of d0 dust
dust_sky = pysm3.Sky(nside=128, preset_strings=["d0"])
hp_dust_map = dust_sky.get_emission(f * u.GHz)
dust_map = reproject.healpix2map(hp_dust_map, shape, wcs)