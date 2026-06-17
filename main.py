from pixell_cmb import make_cmb
from pysm_foreground import make_foreground
from cmb_and_foreground import make_cmb_and_foreground
from pixell_noise import make_noise
from plot_power_spectrum import plot_ps
from plot_rectangular_map import plot_rect_map
from plot_ps_compare import plot_ps_compare
from plot_2d_power_spectrum import plot_2d_ps
from bandpass import bandpass
from concurrent.futures import ProcessPoolExecutor
from plot_bandpass_fn import plot_bandpass
from act_planck_beam import apply_beam
from plot_beam_fn import plot_beams

# full_b_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], beam=True, fwhm=5)
# full_nb_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], beam=False, fwhm=5)

# cmb_b_map = make_cmb(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, fwhm=5, beam=True)
# cmb_nb_map = make_cmb(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, fwhm=5, beam=False)
# noise_b_map = make_noise(dec_radius=4, ra_radius=8, res=5, beam=True, fwhm=5)
# noise_nb_map = make_noise(dec_radius=4, ra_radius=8, res=5, beam=False, fwhm=5)
# dust_b_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], fwhm=5, beam=True)
# dust_nb_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], fwhm=5, beam=False)

# rotated_dust_b_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], fwhm=5, beam=True, rot=True)
# rotated_dust_nb_map = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], fwhm=5, beam=False, rot=True)
# rotated_full_b_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], beam=True, fwhm=5, rot=True)
# rotated_full_nb_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], beam=False, fwhm=5, rot=True)

planck_full_sim_rotated_bb_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=1, sky_f=150, foreground_components=["d0"], gaussian_noise=True, beam=True, rot=True, bp=True, bb_telescope="planck", bb_channel=143, bb_pa=None, beam_type="jitter_cmb", beam_split="coadd")
act_full_sim_rotated_bb_map = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=1, sky_f=150, foreground_components=["d0"], gaussian_noise=True, beam=True, rot=True, bp=True, bb_telescope="act", bb_channel=150, bb_pa=None, beam_type="jitter_cmb", beam_split="coadd")

# plot_rect_map(planck_full_sim_rotated_bb_map, "image_outputs/beam_bp_planck_rotated_full_sim_map")
# plot_rect_map(act_full_sim_rotated_bb_map, "image_outputs/beam_bp_act_rotated_full_sim_map")

plot_ps_compare(
    planck_full_sim_rotated_bb_map,
    act_full_sim_rotated_bb_map,
    name="image_outputs/beam_bp_act_planck_ps_comparison",
    fullsky=False,
    map1_label="Planck",
    map2_label="ACT",
    lmax=5000,
    title="Planck vs ACT Power Spectra Comparison"
)

