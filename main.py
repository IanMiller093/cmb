from pixell_cmb import make_cmb
from pysm_foreground import make_foreground
from cmb_and_foreground import make_cmb_and_foreground
from pixell_noise import make_noise
from plot_power_spectrum import plot_ps
from plot_rectangular_map import plot_rect_map
from plot_ps_compare import plot_ps_compare

full_b_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], beam=True, fwhm=5)
full_nb_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], beam=False, fwhm=5)

plot_ps(imap=full_b_map, name="image_outputs/full_sim_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=full_nb_map, name="image_outputs/full_sim_no_beam_ps", lmax=2160, fullsky=False)

plot_rect_map(imap=full_b_map, name="image_outputs/full_sim_beam_map")
plot_rect_map(imap=full_nb_map, name="image_outputs/full_sim_no_beam_map")

cmb_b_map = make_cmb(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, fwhm=5, beam=True)
cmb_nb_map = make_cmb(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, fwhm=5, beam=False)
noise_b_map = make_noise(dec_radius=4, ra_radius=8, res=5, beam=True, fwhm=5)
noise_nb_map = make_noise(dec_radius=4, ra_radius=8, res=5, beam=False, fwhm=5)
dust_b_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], fwhm=5, beam=True)
dust_nb_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], fwhm=5, beam=False)

plot_rect_map(imap=cmb_b_map, name="image_outputs/cmb_beam_map")
plot_rect_map(imap=cmb_nb_map, name="image_outputs/cmb_no_beam_map")
plot_rect_map(imap=noise_b_map, name="image_outputs/noise_beam_map")
plot_rect_map(imap=noise_nb_map, name="image_outputs/noise_no_beam_map")
plot_rect_map(imap=dust_b_map, name="image_outputs/dust_beam_map")
plot_rect_map(imap=dust_nb_map, name="image_outputs/dust_no_beam_map")

plot_ps(imap=cmb_b_map, name="image_outputs/cmb_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=cmb_nb_map, name="image_outputs/cmb_no_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=noise_b_map, name="image_outputs/noise_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=noise_nb_map, name="image_outputs/noise_no_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=dust_b_map, name="image_outputs/dust_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=dust_nb_map, name="image_outputs/dust_no_beam_ps", lmax=2160, fullsky=False)

rotated_dust_b_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], fwhm=5, beam=True, rot=True)
rotated_dust_nb_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], fwhm=5, beam=False, rot=True)
rotated_full_b_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], beam=True, fwhm=5, rot=True)
rotated_full_nb_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], beam=False, fwhm=5, rot=True)

plot_rect_map(imap=rotated_dust_b_map, name="image_outputs/rotated_dust_beam_map")
plot_rect_map(imap=rotated_dust_nb_map, name="image_outputs/rotated_dust_no_beam_map")
plot_rect_map(imap=rotated_full_b_map, name="image_outputs/rotated_full_sim_beam_map")
plot_rect_map(imap=rotated_full_nb_map, name="image_outputs/rotated_full_sim_no_beam_map")

plot_ps(imap=rotated_dust_b_map, name="image_outputs/rotated_dust_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=rotated_dust_nb_map, name="image_outputs/rotated_dust_no_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=rotated_full_b_map, name="image_outputs/rotated_full_sim_beam_ps", lmax=2160, fullsky=False)
plot_ps(imap=rotated_full_nb_map, name="image_outputs/rotated_full_sim_no_beam_ps", lmax=2160, fullsky=False)

plot_ps_compare(imap1=full_b_map, imap2=full_nb_map, name="image_outputs/full_sim_ps_comparison", map1_label="beam", map2_label="no beam", lmax=2160, title="Full Sim Comparison")
plot_ps_compare(imap1=cmb_b_map, imap2=cmb_nb_map, name="image_outputs/cmb_ps_comparison", map1_label="beam", map2_label="no beam", lmax=2160, title="CMB Comparison")
plot_ps_compare(imap1=noise_b_map, imap2=noise_nb_map, name="image_outputs/noise_ps_comparison", map1_label="beam", map2_label="no beam", lmax=2160, title="Noise Comparison")
plot_ps_compare(imap1=dust_b_map, imap2=dust_nb_map, name="image_outputs/dust_ps_comparison", map1_label="beam", map2_label="no beam", lmax=2160, title="Dust Comparison")
plot_ps_compare(imap1=rotated_dust_b_map, imap2=rotated_dust_nb_map, name="image_outputs/rotated_dust_ps_comparison", map1_label="beam", map2_label="no beam", lmax=2160, title="Rotated Dust Comparison")
plot_ps_compare(imap1=rotated_full_b_map, imap2=rotated_full_nb_map, name="image_outputs/rotated_full_sim_ps_comparison", map1_label="beam", map2_label="no beam", lmax=2160, title="Rotated Full Sim Comparison")