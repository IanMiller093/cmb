from pixell_cmb import make_cmb
from pysm_foreground import make_foreground
from cmb_and_foreground import make_cmb_and_foreground
from plot_power_spectrum import plot_ps
from plot_rectangular_map import plot_rect_map

# get sim
full_b_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], gaussian_noise=True, beam=True, fwhm=5)
dust_b_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], gaussian_noise=True, fwhm=5, beam=True)
full_nb_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], gaussian_noise=True, beam=False, fwhm=5)
dust_nb_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], gaussian_noise=True, fwhm=5, beam=False)

plot_ps(imap=full_b_map, name="image_outputs/full_sim_beam_ps", lmax=min(full_b_map.shape[1:3]), fullsky=False)
plot_ps(imap=full_nb_map, name="image_outputs/full_sim_no_beam_ps", lmax=min(full_nb_map.shape[1:3]), fullsky=False)
plot_ps(imap=dust_b_map, name="image_outputs/dust_beam_ps", lmax=min(dust_b_map.shape[1:3]), fullsky=False)
plot_ps(imap=dust_nb_map, name="image_outputs/dust_no_beam_ps", lmax=min(dust_nb_map.shape[1:3]), fullsky=False)

plot_rect_map(imap=full_b_map, name="image_outputs/full_sim_beam_map")
plot_rect_map(imap=full_nb_map, name="image_outputs/full_sim_no_beam_map")
plot_rect_map(imap=dust_b_map, name="image_outputs/dust_beam_map")
plot_rect_map(imap=dust_nb_map, name="image_outputs/dust_no_beam_map")