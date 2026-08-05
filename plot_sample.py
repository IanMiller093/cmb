import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2
from posterior_sampling import posterior_sample

def make_test_prior(T, N, mode, scale_factor=None):
    """
    Build a constant S array for verification testing, sized to match
    T's (N_comp, N_stokes, N_pix) amplitude convention.

    mode : 'tight' or 'loose'
        'tight'  -> S very small relative to the data term (prior dominates,
                    pulls the sample toward zero since the prior is zero-mean)
        'loose'  -> S very large relative to the data term (data dominates)
    scale_factor : float or None
        Multiplier applied on top of the automatic tight/loose scale.
        If None, uses 1e-4 for tight and 1e4 for loose.

    Returns
    -------
    S : ndarray, shape (N_comp, N_stokes, N_pix)
    """

    N_chan, N_comp, N_stokes, N_pix = T.shape

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

    S_inv = np.zeros((N_comp, N_stokes, N_pix))
    for c in range(N_comp):
        S_inv[c, :, :] = data_term_scale[c] * scale_factor

    return S_inv


def analytic_posterior_mean_var(T, d, N, S_inv, comp, stokes, pix):
    """
    Closed-form posterior mean and variance at a single pixel/component/stokes,
    with eta terms set to zero (i.e. this is the mean of the distribution
    posterior_sample draws from, not a sample itself).

    Zero-mean prior: rhs has no S^-1 mu0 term, just the data contribution.
    """

    T_p = T[:, :, stokes, pix]      # (N_chan, N_comp)
    N_p = N[:, stokes, pix]         # (N_chan,)
    d_p = d[:, stokes, pix]         # (N_chan,)
    Sinv_p = S_inv[:, stokes, pix]  # (N_comp,) -- already precision, no inversion needed

    Ninv_p = 1.0 / N_p

    lhs_p = T_p.T @ np.diag(Ninv_p) @ T_p + np.diag(Sinv_p)
    rhs_p = T_p.T @ (Ninv_p * d_p)

    cov_p = np.linalg.inv(lhs_p)
    mean_p = cov_p @ rhs_p

    return mean_p[comp], cov_p[comp, comp]


def run_prior_verification(T, d, N, truth, comp, stokes, pix, ny, nx,
                            posterior_sample_fn, img_out_path, histogram=False, n_draws=1000):
    """
    Main verification function -- call this from main.py.

    Produces:
      (A) map triptych (truth / zero map / sample) for tight and loose regimes.
          Zero-mean prior: "tight prior" means the sample gets pulled toward
          zero, not toward some template, so the middle column is a zero map.
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

    S_inv_tight = make_test_prior(T, N, mode='tight')
    S_inv_loose = make_test_prior(T, N, mode='loose')

    x_tight = posterior_sample_fn(T, d, N, S_tight)
    x_loose = posterior_sample_fn(T, d, N, S_inv_loose)

    # zero map, for the "prior mean" column -- the prior is zero-mean, so this
    # is what a tight prior should pull the sample toward
    zero_map = np.zeros((ny, nx))

    # --- (A) map triptych ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    im = axes[0, 0].imshow(truth[comp, stokes].reshape(ny, nx))
    axes[0, 0].set_title('truth')
    fig.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].imshow(zero_map)
    axes[0, 1].set_title('zero (prior mean)')
    fig.colorbar(im, ax=axes[0, 1])
    im = axes[0, 2].imshow(x_tight[comp, stokes].reshape(ny, nx))
    axes[0, 2].set_title('sample, tight prior')
    fig.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].imshow(truth[comp, stokes].reshape(ny, nx))
    axes[1, 0].set_title('truth')
    fig.colorbar(im, ax=axes[1, 0])
    im = axes[1, 1].imshow(zero_map)
    axes[1, 1].set_title('zero (prior mean)')
    fig.colorbar(im, ax=axes[1, 1])
    im = axes[1, 2].imshow(x_loose[comp, stokes].reshape(ny, nx))
    axes[1, 2].set_title('sample, loose prior')
    fig.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(os.path.join(img_out_path, 'prior_verification_maps.png'))
    plt.close(fig)

    # --- (B) histogram vs analytic posterior, both regimes ---
    if histogram:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, S, label in [
            (axes[0], S_inv_tight, 'tight prior'),
            (axes[1], S_inv_loose, 'loose prior'),
        ]:
            samples = np.zeros(n_draws)
            for i in range(n_draws):
                x = posterior_sample_fn(T, d, N, S)
                samples[i] = x[comp, stokes, pix]

            mean_p, var_p = analytic_posterior_mean_var(T, d, N, S, comp, stokes, pix)

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



def plot_component_separation(x_sample, truth, ny, nx, img_out_path,
                               comp_labels=None, stokes_labels=None,
                               show_residual=True, share_colorbar=False):
    """
    Plot truth vs sample (vs residual) for every component, separately for
    each Stokes parameter -- this is the actual "did component separation
    work" check, as opposed to run_prior_verification's single comp/stokes/pix
    prior sanity check.

    Produces one figure per Stokes parameter, saved as
    'component_separation_{stokes_label}.png', with one row per component and
    columns for truth / sample / (residual).

    Parameters
    ----------
    x_sample : ndarray, shape (N_comp, N_stokes, N_pix)
        Output of posterior_sample -- one Gibbs sample of the component
        amplitude maps.
    truth : ndarray, shape (N_comp, N_stokes, N_pix)
        Ground truth component maps, same layout as x_sample.
    ny, nx : int
        Map dimensions, for reshaping 1D pixel arrays back to 2D for imshow.
    img_out_path : str
        Directory to save output figures into.
    comp_labels : list of str or None
        Names for each component, e.g. ['cmb', 'dust']. Defaults to
        ['comp0', 'comp1', ...] if not given.
    stokes_labels : list of str or None
        Names for each Stokes parameter, e.g. ['I', 'Q', 'U']. Defaults to
        ['stokes0', 'stokes1', ...] if not given.
    show_residual : bool
        If True, add a third column with (sample - truth).
    share_colorbar : bool
        If True, all columns in each row (truth / sample / residual) share
        one colorbar, scaled to the truth (left) image's range. If False
        (default), each image gets its own independent colorbar.
    """

    N_comp, N_stokes, N_pix = x_sample.shape

    if comp_labels is None:
        comp_labels = [f'comp{c}' for c in range(N_comp)]
    if stokes_labels is None:
        stokes_labels = [f'stokes{s}' for s in range(N_stokes)]

    n_cols = 3 if show_residual else 2

    for s in range(N_stokes):
        fig, axes = plt.subplots(N_comp, n_cols, figsize=(4 * n_cols, 4 * N_comp), squeeze=False)

        for c in range(N_comp):
            truth_map = truth[c, s].reshape(ny, nx)
            sample_map = x_sample[c, s].reshape(ny, nx)

            if share_colorbar:
                vmin, vmax = truth_map.min(), truth_map.max()
            else:
                vmin = vmax = None  # let imshow auto-scale each independently

            im0 = axes[c, 0].imshow(truth_map, vmin=vmin, vmax=vmax)
            axes[c, 0].set_title(f'{comp_labels[c]} truth')
            fig.colorbar(im0, ax=axes[c, 0])

            im1 = axes[c, 1].imshow(sample_map, vmin=vmin, vmax=vmax)
            axes[c, 1].set_title(f'{comp_labels[c]} sample')
            if not share_colorbar:
                fig.colorbar(im1, ax=axes[c, 1])

            if show_residual:
                resid_map = sample_map - truth_map
                im2 = axes[c, 2].imshow(resid_map, vmin=vmin, vmax=vmax)
                axes[c, 2].set_title(f'{comp_labels[c]} sample - truth')
                if not share_colorbar:
                    fig.colorbar(im2, ax=axes[c, 2])

        fig.suptitle(f'Component separation, Stokes {stokes_labels[s]}')
        plt.tight_layout()
        plt.savefig(os.path.join(img_out_path, f'component_separation_{stokes_labels[s]}.png'), dpi=300)
        plt.close(fig)

def get_all_pixel_mean_var(T, d, N, S_inv):
    """

    Returns
    mean_all : ndarray, shape (N_comp, N_stokes, N_pix)
    var_all  : ndarray, shape (N_comp, N_stokes, N_pix)
        Diagonal of the posterior covariance at each pixel (i.e. the marginal
        variance of each component, ignoring cross-component covariance).
    """

    N_chan, N_comp, N_stokes, N_pix = T.shape

    Ninv = 1.0 / N          # (N_chan, N_stokes, N_pix)

    # A[f,c,s,p] = T[f,c,s,p] * Ninv[f,s,p]
    A = T * Ninv[:, None, :, :]

    # lhs[s,p,c,e] = sum_f T[f,c,s,p] * Ninv[f,s,p] * T[f,e,s,p]
    lhs = np.einsum('fcsp,fesp->spce', A, T)

    # add diag(Sinv) on top -- only touches the c==e entries
    diag_idx = np.arange(N_comp)
    lhs[:, :, diag_idx, diag_idx] += np.transpose(S_inv, (1, 2, 0))  # (N_stokes, N_pix, N_comp)

    # rhs[s,p,c] = sum_f T[f,c,s,p] * Ninv[f,s,p] * d[f,s,p]
    rhs = np.einsum('fcsp,fsp->spc', T, Ninv * d)

    # batched inversion over the leading (N_stokes, N_pix) dims
    cov = np.linalg.inv(lhs)                       # (N_stokes, N_pix, N_comp, N_comp)
    mean = np.einsum('spce,spe->spc', cov, rhs)     # (N_stokes, N_pix, N_comp)
    var = np.diagonal(cov, axis1=2, axis2=3)        # (N_stokes, N_pix, N_comp)

    # reshape to match the (N_comp, N_stokes, N_pix) amplitude convention used everywhere else
    mean_all = np.transpose(mean, (2, 0, 1))
    var_all = np.transpose(var, (2, 0, 1))

    return mean_all, var_all


def run_global_calibration_check(T, d, N, truth, ny, nx,
                                  posterior_sample_fn, img_out_path, n_draws=1000,
                                  comp_labels=None):
    """
    Parameters
    T, d, N : as used by posterior_sample_fn
    truth : ndarray, shape (N_comp, N_stokes, N_pix)
        Only truth.shape is used, to get N_comp/N_stokes/N_pix; kept for
        signature parity with run_prior_verification.
    ny, nx : int
        Unused directly here (kept for signature parity); map dimensions.
    posterior_sample_fn : callable
        Your posterior_sample function, passed in directly (not called).
    img_out_path : str
        Directory to save the output figure into.
    n_draws : int
        Number of Monte Carlo draws for the D histograms.
    comp_labels : list of str or None
        Labels for each component (e.g. ['cmb', 'dust']), used as row titles.
        Defaults to ['comp0', 'comp1', ...] if not given.
    """

    N_comp, N_stokes, N_pix = truth.shape
    N_total_per_comp = N_stokes * N_pix

    if comp_labels is None:
        comp_labels = [f'comp{c}' for c in range(N_comp)]

    S_inv_tight = make_test_prior(T, N, mode='tight')
    S_inv_loose = make_test_prior(T, N, mode='loose')

    fig, axes = plt.subplots(N_comp, 2, figsize=(10, 4 * N_comp), squeeze=False)

    for col, (S_inv, prior_label) in enumerate([
        (S_inv_tight, 'tight prior'),
        (S_inv_loose, 'loose prior'),
    ]):
        mean_all, var_all = get_all_pixel_mean_var(T, d, N, S)

        # D_samples_by_comp[c, i] = D for component c on draw i
        D_samples_by_comp = np.zeros((N_comp, n_draws))
        for i in range(n_draws):
            x = posterior_sample_fn(T, d, N, S_inv)
            z = (x - mean_all) / np.sqrt(var_all)
            D_samples_by_comp[:, i] = np.mean(z**2, axis=(1, 2)) 

        for c in range(N_comp):
            ax = axes[c, col]
            D_samples = D_samples_by_comp[c]

            # analytic density of D_c: sum(z_c^2) ~ chi2(N_total_per_comp),
            # D_c = sum(z_c^2)/N_total_per_comp, so via change of variables
            # pdf_D(x) = N_total_per_comp * chi2.pdf(N_total_per_comp * x, df=N_total_per_comp)
            xs = np.linspace(D_samples.min(), D_samples.max(), 200)
            analytic_pdf = N_total_per_comp * chi2.pdf(N_total_per_comp * xs, df=N_total_per_comp)

            ax.hist(D_samples, bins=40, density=True, label='sampled')
            ax.plot(xs, analytic_pdf, label='analytic (chi2/dof)')
            ax.axvline(1.0, color='k', linestyle='--', label='expected (D=1)')
            ax.set_title(f'{comp_labels[c]} -- {prior_label}')
            ax.set_xlabel(f'D = mean(z^2) over stokes/pix, {comp_labels[c]} only')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(img_out_path, 'global_calibration_check.png'))
    plt.close(fig)

def plot_sample_mean_map(T, d, N, S, ny, nx, img_out_path, fileheader, n_draws=1000):
    """
    returns nothing, saves 3 * 2 png images, one for each Stokes components and dust/cmb comp.
 
    For each frequency channel, takes the eta = 0 case, and plots that alongside the sample mean of like 1000 different samples, so two pictures for each saved png file.
    
    To get eta=0 case, take posterior_sample(T, d, N, S, zero_etas=True), for sample mean, take mean over 1000 different samples with zero_etas=False.  Plot both side by side with one color bar.
 
    """
 
    stokes_labels = ['I', 'Q', 'U']
 
    N_comp, N_stokes, N_pix = T.shape[1], T.shape[2], T.shape[3]
    comp_labels = [f'comp{c}' for c in range(N_comp)]
 
    # eta=0 case: single deterministic solve
    x_zero_eta = posterior_sample(T, d, N, S, zero_etas=True)
 
    # sample mean: average over n_draws stochastic solves
    x_sum = np.zeros((N_comp, N_stokes, N_pix))
    for _ in range(n_draws):
        x_sum += posterior_sample(T, d, N, S, zero_etas=False)
    x_mean = x_sum / n_draws
 
    for c in range(N_comp):
        for s in range(N_stokes):
            zero_eta_map = x_zero_eta[c, s, :].reshape(ny, nx)
            mean_map = x_mean[c, s, :].reshape(ny, nx)
 
            vmin = min(zero_eta_map.min(), mean_map.min())
            vmax = max(zero_eta_map.max(), mean_map.max())
 
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
 
            im0 = axes[0].imshow(zero_eta_map, origin='lower', vmin=vmin, vmax=vmax)
            axes[0].set_title(f'{comp_labels[c]} -- eta=0')
 
            im1 = axes[1].imshow(mean_map, origin='lower', vmin=vmin, vmax=vmax)
            axes[1].set_title(f'{comp_labels[c]} -- sample mean (n={n_draws})')
 
            fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
 
            fig.suptitle(f'Stokes {stokes_labels[s]}, {comp_labels[c]}')
 
            out_name = f'{fileheader}_stokes_{stokes_labels[s]}_chan{c}.png'
            plt.savefig(os.path.join(img_out_path, out_name))
            plt.close(fig)
 

def plot_covar_maps():
    """
    returns nothing, saves 3 * 3 png images, one for each Stokes component and each covariance (dust with dust variance, cmb with cmb variance, and cmb with dust).

    Each image should have two side by side pictures.

    Calculates true variance matrix/numpy tensor using F^-1 = S^-1 + T^T N^-1 T.  Get F from inverse (F^-1)

    For every pixel, stokes component, there should be a 2x2 matrix thingy containing CC, CD, DC, DD (CD = DC for symmetric matrix).

    Save one image containing a map of true CC values for each pixel with sample CC values, same for CD and DD.
    """

    pass