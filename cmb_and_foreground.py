from pixell_cmb import make_cmb
from pysm_foreground import make_foreground

def make_cmb_and_foreground(dec_radius=90, ra_radius=180, ps_txt_filepath="ps.txt", seed=67, res=1, sky_f=150, foreground_components=["d0"], gaussian_noise=True, fwhm=1, beam=True):
    '''
    pretty much just a wrapper of make_cmb and make_foreground.  I'm too lazy to copy over the comments
    about params from the other files, and the fishies would get mad at me if I used Claude to grab the
    comments, so you will have to go over and read the comments from the other two files yourself :(

    returns make_cmb + make_foreground with the same dimensions
    '''
    cmb, _ = make_cmb(dec_radius=dec_radius, ra_radius=ra_radius, ps_txt_filepath=ps_txt_filepath, seed=seed, res=res, fwhm=fwhm, beam=beam)
    foreground = make_foreground(dec_radius=dec_radius, ra_radius=ra_radius, sky_f=sky_f, res=res, foreground_components=foreground_components, gaussian_noise=gaussian_noise, fwhm=fwhm, beam=beam)

    assert cmb.shape == foreground.shape

    return cmb + foreground