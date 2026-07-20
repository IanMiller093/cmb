import numpy as np
import matplotlib.pyplot as plt
import os


def make_test_prior(T, N, mode, scale_factor=None):
    """
    Build a constant mu0 and S array for verification testing, sized to match
    T's (N_comp, N_stokes, N_pix) amplitude convention.

    mode : 'tight' or 'loose'
        'tight'  -> S very small relative to the data term (prior dominates)
        'loose'  -> S very large relative to the data term (data dominates)
    scale_factor : float or None
        Multiplier applied on top of the automatic tight/loose scale.
        If None, uses 1e-4 for tight and 1e4 for loose.

    Returns
    -------
    mu0 : ndarray, shape (N_comp, N_stokes, N_pix)
    S   : ndarray, shape (N_comp, N_stokes, N_pix)
    """

    N_chan, N_comp, N_stokes, N_pix = T.shape

    # flag: mu0 = 0 for every component. this is deliberately "wrong" relative to
    # any nonzero truth map, so the tight-prior regime is visually obvious.
    # if your truth map happens to be ~0 in some region this won't look dramatic there,
    # so pick a pixel with real signal when you plot.
    mu0 = np.zeros((N_comp, N_stokes, N_pix))

    # estimate the natural scale of the data term T^T N^-1 T, per component,
    # averaged over pixels/stokes, so 'tight' and 'loose' are relative to your
    # actual data units instead of a blind guess like 1e-6 / 1e12
    Ninv = 1.0 / N  # (N_chan, N_stokes, N_pix)

    # sum_f T[f,c]^2 * Ninv[f], for each component c -- diagonal-only estimate,
    # ignores component cross-terms, just need an order-of-magnitude scale
    data_term_scale = np.zeros(N_comp)
    for c in range(N_comp):
        # (N_chan, N_stokes, N_pix)
        term = T[:, c, :, :]**2 * Ninv
        data_term_scale[c] = np.mean(term)

    if scale_factor is None:
        scale_factor = 1e-4 if mode == 'tight' else 1e4

    S = np.zeros((N_comp, N_stokes, N_pix))
    for c in range(N_comp):
        if mode == 'tight':
            # S small -> S^-1 large -> dominates over data_term_scale
            S[c, :, :] = data_term_scale[c] * scale_factor
        elif mode == 'loose':
            # S large -> S^-1 small -> negligible next to data_term_scale
            S[c, :, :] = data_term_scale[c] * scale_factor
        else:
            raise ValueError("mode must be 'tight' or 'loose'")

    return mu0, S


def analytic_posterior_mean_var(T, d, N, mu0, S, comp, stokes, pix):
    """
    Closed-form posterior mean and variance at a single pixel/component/stokes,
    with eta terms set to zero (i.e. this is the mean of the distribution
    posterior_sample draws from, not a sample itself).
    """

    T_p = T[:, :, stokes, pix]      # (N_chan, N_comp)
    N_p = N[:, stokes, pix]         # (N_chan,)
    d_p = d[:, stokes, pix]         # (N_chan,)
    S_p = S[:, stokes, pix]         # (N_comp,)
    mu0_p = mu0[:, stokes, pix]     # (N_comp,)

    Ninv_p = 1.0 / N_p
    Sinv_p = 1.0 / S_p

    lhs_p = T_p.T @ np.diag(Ninv_p) @ T_p + np.diag(Sinv_p)
    rhs_p = T_p.T @ (Ninv_p * d_p) + Sinv_p * mu0_p

    cov_p = np.linalg.inv(lhs_p)
    mean_p = cov_p @ rhs_p

    return mean_p[comp], cov_p[comp, comp]


def run_prior_verification(T, d, N, truth, comp, stokes, pix, ny, nx,
                            posterior_sample_fn, img_out_path, n_draws=3000):
    """
    Main verification function -- call this from main.py.

    Produces:
      (A) map triptych (truth / prior mean / sample) for tight and loose regimes
      (B) per-pixel histogram of samples vs analytic posterior, tight and loose

    Parameters
    ----------
    T, d, N : as used by posterior_sample_fn
    truth : ndarray, shape (N_comp, N_stokes, N_pix)
        Known ground truth used to generate d, for visual comparison only.
    comp, stokes, pix : int
        Which component / stokes / pixel to focus the histogram (B) on.
    ny, nx : int
        Map dimensions, for reshaping 1D pixel arrays back to 2D for imshow.
    posterior_sample_fn : callable
        Your posterior_sample function (passed in rather than imported, so this
        file doesn't need to know its module path).
    img_out_path : str
        Directory to save output figures into.
    n_draws : int
        Number of Monte Carlo draws for the histogram check.
    """

    mu0_tight, S_tight = make_test_prior(T, N, mode='tight')
    mu0_loose, S_loose = make_test_prior(T, N, mode='loose')

    x_tight = posterior_sample_fn(T, d, N, mu0_tight, S_tight)
    x_loose = posterior_sample_fn(T, d, N, mu0_loose, S_loose)

    # --- (A) map triptych ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(truth[comp, stokes].reshape(ny, nx))
    axes[0, 0].set_title('truth')
    axes[0, 1].imshow(mu0_tight[comp, stokes].reshape(ny, nx))
    axes[0, 1].set_title('prior mean')
    axes[0, 2].imshow(x_tight[comp, stokes].reshape(ny, nx))
    axes[0, 2].set_title('sample, tight prior')

    axes[1, 0].imshow(truth[comp, stokes].reshape(ny, nx))
    axes[1, 0].set_title('truth')
    axes[1, 1].imshow(mu0_loose[comp, stokes].reshape(ny, nx))
    axes[1, 1].set_title('prior mean')
    axes[1, 2].imshow(x_loose[comp, stokes].reshape(ny, nx))
    axes[1, 2].set_title('sample, loose prior')

    plt.tight_layout()
    plt.savefig(os.path.join(img_out_path, 'prior_verification_maps.png'))
    plt.close(fig)

    # --- (B) histogram vs analytic posterior, both regimes ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, mu0, S, label in [
        (axes[0], mu0_tight, S_tight, 'tight prior'),
        (axes[1], mu0_loose, S_loose, 'loose prior'),
    ]:
        samples = np.zeros(n_draws)
        for i in range(n_draws):
            x = posterior_sample_fn(T, d, N, mu0, S)
            samples[i] = x[comp, stokes, pix]

        mean_p, var_p = analytic_posterior_mean_var(T, d, N, mu0, S, comp, stokes, pix)

        xs = np.linspace(samples.min(), samples.max(), 200)
        gaussian = (1.0 / np.sqrt(2 * np.pi * var_p)) * np.exp(-(xs - mean_p)**2 / (2 * var_p))

        ax.hist(samples, bins=40, density=True, label='sampled')
        ax.plot(xs, gaussian, label='analytic')
        ax.axvline(mean_p, color='k', linestyle='--')
        ax.set_title(label)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(img_out_path, 'prior_verification_histograms.png'))
    plt.close(fig)