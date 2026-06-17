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

# def make_and_plot_planck():
#     m = make_foreground(
#         dec_radius=4, ra_radius=8, sky_f=150, res=20,
#         foreground_components=["d0"], fwhm=20, beam=True,
#         rot=True, bp=True, bp_telescope="planck",
#         bp_channel=143, bp_pa=None
#     )
#     plot_rect_map(m, "image_outputs/planck_bp_rotated_dust_beam")
#     return m

# def make_and_plot_act():
#     m = make_foreground(
#         dec_radius=4, ra_radius=8, sky_f=150, res=20,
#         foreground_components=["d0"], fwhm=20, beam=True,
#         rot=True, bp=True, bp_telescope="act",
#         bp_channel=150, bp_pa=5
#     )
#     plot_rect_map(m, "image_outputs/act_bp_rotated_dust_beam")
#     return m

# if __name__ == "__main__":  # required for ProcessPoolExecutor on most OSes
#     with ProcessPoolExecutor(max_workers=2) as executor:
#         fut_planck = executor.submit(make_and_plot_planck)
#         fut_act    = executor.submit(make_and_plot_act)

#         planck_bp_foreground_map = fut_planck.result()  # blocks until done
#         act_bp_foreground_map    = fut_act.result()

