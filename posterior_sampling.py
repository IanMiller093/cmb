import numpy as np

def posterior_sample_1_pix(T, d, N):
    """
    Note to self:
    Prior: our initial expectation of the value of the pixel
    Likelihood: how likely we are to see the data given a certain "true" value of the pixel
    Posterior: our updated expectation of the value of the pixel after seeing the data
    
    Here, T and N can be thought of mathematically as diagonal matrices, although in the code they are
    stored as 1D vectors to approximate diagonal matrices, while d is a vector representing the value of 
    a pixel at multiple different channels.  The output is a sample from the distribution of x, which has
    values of the components (CMB, dust) of each pixel in every channel.
    """

    # Draw a random sample from ~ N(0, 1) for each entry of 
    eta1 = np.random.standard_normal(len(d))

    # for diagonal T, T.T = T
    numerator = np.sum(T * d / N) + np.sum(T * eta1 / np.sqrt(N))
    denominator = np.sum(T**2 / N)

    x_sample = numerator / denominator

    return x_sample

def posterior_sample(T, d, N, mu0, S):
    """
    Multi pixel, multi component generalization of posterior_sample_1_pix.

    Follows Appendix A.2 (BeyondPlanck), Eq. A.10, with the limit S^-1 -> 0 (no prior),
    diagonal N (no pixel-pixel or channel-channel noise correlations), and no beam.

    For each pixel p and each Stokes parameter s independently, solves the small
    (N_comp x N_comp) linear system:

        

    where, restricted to pixel p / Stokes s:
        T   : (N_chan, N_comp) mixing matrix (SED scaling per component per channel)
        N   : (N_chan,)        diagonal noise variance per channel
        d   : (N_chan,)        observed data across channels
        eta : (N_chan,)        iid N(0,1) draws (fluctuation term -> correct posterior variance,
                                  not just the posterior mean)
        x   : (N_comp,)        sampled component amplitudes at this pixel (the CMB and dust)
        S   : (N_comp,)        prior variance for each component (diagonal, no cross-component prior covariance)
        mu0 : (N_comp,)        prior mean for each component

    Parameters
    ----------
    T : ndarray, shape (N_chan, N_comp, N_stokes, N_pix)
        Mixing matrix. T[f, c, s, p] = contribution of component c to channel f's
        signal, at Stokes s, pixel p (CMB blackbody scaling, dust MBB scaling, etc.)
    d : ndarray, shape (N_chan, N_stokes, N_pix)
        Observed data map, per channel / Stokes / pixel.
    N : ndarray, shape (N_chan, N_stokes, N_pix)
        Diagonal noise covariance (variance), per channel / Stokes / pixel.
        NOTE: load_N gives you (3, ny, nx) per channel — stack over channels and
        flatten (ny, nx) -> N_pix before passing in here, to match d's layout.
    mu0 : ndarray, shape (N_comp, N_stokes, N_pix)
        Prior mean for each component amplitude.
    S : ndarray, shape (N_comp, N_stokes, N_pix)
        Prior variance (diagonal, no cross-component prior covariance).
        Use np.inf entries to recover the old no-prior behavior for that component.

    Returns
    -------
    x_sample : ndarray, shape (N_comp, N_stokes, N_pix)
        One Gibbs sample of the component amplitude maps, e.g. x_sample[0] = CMB map,
        x_sample[1] = dust map, over every pixel and Stokes parameter.
    """

    N_chan, N_comp, N_stokes, N_pix = T.shape

    # noise fluctuation term, same as before
    eta1 = np.random.standard_normal(size=(N_chan, N_stokes, N_pix))

    # prior fluctuation term -- one draw per component (not per channel)
    eta2 = np.random.standard_normal(size=(N_comp, N_stokes, N_pix))

    T_T = np.transpose(T, (2, 3, 0, 1))
    d_T = np.transpose(d, (1, 2, 0))
    N_T = np.transpose(N, (1, 2, 0))
    eta1_T = np.transpose(eta1, (1, 2, 0))

    # prior arrays, transposed the same way as mu0/S live per-component
    mu0_T = np.transpose(mu0, (1, 2, 0))
    S_T = np.transpose(S, (1, 2, 0))
    eta2_T = np.transpose(eta2, (1, 2, 0))

    Ninv = 1.0 / N_T
    Ninv_sqrt = np.sqrt(Ninv)

    # S can have np.inf entries (no prior on that component) -- 1/inf = 0 is fine,
    # sqrt(0) = 0 is fine, no nan risk here
    Sinv = 1.0 / S_T
    Sinv_sqrt = np.sqrt(Sinv)

    weighted_d = Ninv * d_T
    weighted_eta = Ninv_sqrt * eta1_T

    # data part of rhs, same as before
    rhs_data = np.einsum('spfc,spf->spc', T_T, weighted_d + weighted_eta)

    # prior part of rhs: S^-1 mu0 + S^-1/2 eta2
    rhs_prior = Sinv * mu0_T + Sinv_sqrt * eta2_T

    rhs = rhs_data + rhs_prior

    # data part of lhs, same as before
    lhs = np.einsum('spfc,spf,spfk->spck', T_T, Ninv, T_T)

    # add S^-1 onto the diagonal (c == k) of lhs, at every stokes/pixel
    comp_idx = np.arange(N_comp)
    lhs[..., comp_idx, comp_idx] += Sinv

    print("lhs:", lhs.shape)
    print("rhs:", rhs.shape)
    # used to be: x_T = np.linalg.solve(lhs, rhs)
    x_T = np.linalg.solve(lhs, rhs[..., None])[..., 0]
    x_sample = np.transpose(x_T, (2, 0, 1))

    return x_sample