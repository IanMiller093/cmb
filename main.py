from pixell_cmb import make_cmb
from cmb_and_dust import make_foreground

cmb, _ = make_cmb(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5)
foreground = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], gaussian_noise=True)

print(cmb.shape)
print(foreground.shape)