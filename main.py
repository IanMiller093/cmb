from pixell_cmb import make_cmb
from pysm_foreground import make_foreground
from cmb_and_foreground import make_cmb_and_foreground
from matplotlib import pyplot as plt
from plot_power_spectrum import plot_ps

# get sim
#imap = make_cmb_and_foreground(dec_radius=4, ra_radius=8, ps_txt_filepath="ps.txt", seed=67, res=5, sky_f=150, foreground_components=["d0"], gaussian_noise=True)
imap = make_foreground(dec_radius=4, ra_radius=8, sky_f=150, res=5, foreground_components=["d0"], gaussian_noise=True, fwhm=1)

# plot
stokes = ['I', 'Q', 'U']
for i in range(3):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(imap[i], origin='lower', cmap='RdBu_r')
    ax.set_title(f'Stokes {stokes[i]}')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'dust_{stokes[i]}_beam.png', dpi=150)
    plt.close()
    print(f"Saved dust_{stokes[i]}_beam.png")

plot_ps(imap=imap, name="dust_ps_beam", lmax=min(imap.shape[1:3]), fullsky=False)