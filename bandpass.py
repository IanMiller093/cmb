from astropy.io import fits
import h5py

def bandpass(telescope="planck", channel=100, pa=None):
    '''
    telescope: either "planck" or "act".  Default is "planck"

    channel: the frequency channel that the bandpass is to be taken from.  Default is 100

    pa: only applicable for act.  Some ACT channels have different come from different pas, or 
    "Polarization-sensitive Arrays" which have slightly different bandpass functions.  These PAs can be
    slightly different.  If None is inputted, then one is chosen from the 
    possible PAs of the given channel.  Default is None

    Outputs bandpass frequencies and corresponding weights to be inputted into PySM's Sky.get_emission
    function.
    '''

    assert telescope in ["planck", "act"]
    
    if telescope == "planck":
        hfis = [100, 143, 217, 353, 545, 857]
        lfis = [30, 44, 70]

        assert channel in lfis or channel in hfis

        if channel in hfis:
            hdul = fits.open("HFI_RIMO_R3.00.fits")
            bandpass = hdul["BANDPASS_F" + str(channel)].data
            # GHz conversion
            bp_freqs = bandpass["WAVENUMBER"] * 29.9792458
            bp_weights = bandpass["TRANSMISSION"]
            mask = bp_weights > 0.001  # or whatever threshold looks clean
            bp_freqs = bp_freqs[mask]
            bp_weights = bp_weights[mask]

        else:
            hdul = fits.open("LFI_RIMO_R3.31.fits")
            bandpass = hdul["BANDPASS_" + str(channel).zfill(3)].data
            bp_freqs = bandpass["WAVENUMBER"]
            bp_weights = bandpass["TRANSMISSION"]
            
        
        if channel in lfis:
            bp_weights = bp_weights / bp_freqs**2



    elif telescope == "act":
        file = h5py.File("AdvACT_Passbands.h5")
        
        channels_w_pas = [(150, 4), (220, 4), (90, 5), (150, 5), (90, 6), (150, 6), (30, 7), (40, 7)]
        default_pas = {
            30 : 7,
            40 : 7,
            90 : 6,
            150 : 6,
            220 : 4
        }

        if pa is not None:
            chan_pa_tuple = (channel, pa)
            
            assert chan_pa_tuple in channels_w_pas

        else:
            channels = [30, 40, 90, 150, 220]

            assert channel in channels
           
            pa = default_pas[channel]

        bandpass = file["PA" + str(pa) + "_f" + (("0" + str(channel)) if channel < 100 else str(channel))]
        bp_freqs = bandpass["frequencies"][:]
        bp_weights = bandpass["mean_band"][:]

        low, high = bandpass["integration_bounds"][:]

        mask = (bp_freqs >= low) & (bp_freqs <= high)
        bp_freqs = bp_freqs[mask]
        bp_weights = bp_weights[mask]

        bp_weights = bp_weights / bp_freqs**2

    return bp_freqs, bp_weights