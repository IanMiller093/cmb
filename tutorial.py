# largely following PySM tutorial from their documentation website with some small deviations
# comments are pretty much all ian's

# imports
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# nside: size is 12 * nside**2, where nside must be a power of 2
# preset_strings is a list of things to include in the model of the sky
# returns a simulated sky
sky = pysm3.Sky(nside=128, preset_strings=["d1", "s1"])

# astropy units (the u.GHz must be used)
# gets map at a specific frequency
# returns an 2D array of length (3, N_pixels) where the arrays are [I, Q, U] in that order
map_100GHz = sky.get_emission(100 * u.GHz)

# displays intensity map
#hp.mollview(map_100GHz[0], min=0, max=1e2, title="I map", unit=map_100GHz.unit)
#plt.show()

# below is actually including a simulated CMB
cmbsky = pysm3.Sky(nside=128, preset_strings=["d1", "s1", "c1"])
cmb_map_100GHz = cmbsky.get_emission(100 * u.GHz)

hp.mollview(cmb_map_100GHz[0], min=0, max=1e2, title="I map", unit=cmb_map_100GHz.unit)
plt.show()