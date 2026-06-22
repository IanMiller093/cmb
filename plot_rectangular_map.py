from matplotlib import pyplot as plt
import numpy as np

def plot_rect_map(imap, name):
    if imap.ndim == 2:
        imap = imap[np.newaxis]
    stokes = ['I', 'Q', 'U']
    for i in range(3):
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(imap[i], origin='lower', cmap='RdBu_r')
        ax.set_title(f'Stokes {stokes[i]}')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f'{name}_{stokes[i]}.png', dpi=150)
        plt.close()
        print(f"Saved {name}_{stokes[i]}.png")