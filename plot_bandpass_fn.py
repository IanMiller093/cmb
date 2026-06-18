import matplotlib.pyplot as plt
from bandpass import bandpass

def plot_bandpass(telescope="planck", channel=100, pa=None, ax=None, filename=None):
    """
    Plots the bandpass frequency response for a given telescope/channel.
    """
    bp_freqs, bp_weights = bandpass(telescope=telescope, channel=channel, pa=pa)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    label = f"{telescope.upper()} {channel} GHz"
    if telescope == "act" and pa is not None:
        label += f" PA{pa}"

    ax.plot(bp_freqs, bp_weights, label=label)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Transmission")
    ax.set_title(f"Bandpass: {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename is None:
        filename = f"/data6/miller42/cmb_sim/image_outputs/bandpass_{telescope}_{channel}.png"
        if telescope == "act" and pa is not None:
            filename = f"/data6/miller42/cmb_sim/image_outputs/bandpass_{telescope}_{channel}_pa{pa}.png"

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved to {filename}")