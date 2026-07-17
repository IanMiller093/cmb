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

def posterior_sample(T, d, N):
    """
    Multi pixel, multi component generalization of posterior_sample_1_pix.

    Follows Appendix A.2 (BeyondPlanck), Eq. A.10, with the limit S^-1 -> 0 (no prior),
    diagonal N (no pixel-pixel or channel-channel noise correlations), and no beam.

    For each pixel p and each Stokes parameter s independently, solves the small
    (N_comp x N_comp) linear system:

        (T^T N^-1 T) x  =  T^T N^-1 d  +  T^T N^-1/2 eta

    where, restricted to pixel p / Stokes s:
        T   : (N_chan, N_comp) mixing matrix (SED scaling per component per channel)
        N   : (N_chan,)        diagonal noise variance per channel
        d   : (N_chan,)        observed data across channels
        eta : (N_chan,)        iid N(0,1) draws (fluctuation term -> correct posterior variance,
                                  not just the posterior mean)
        x   : (N_comp,)        sampled component amplitudes at this pixel (the CMB and dust)

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

    Returns
    -------
    x_sample : ndarray, shape (N_comp, N_stokes, N_pix)
        One Gibbs sample of the component amplitude maps, e.g. x_sample[0] = CMB map,
        x_sample[1] = dust map, over every pixel and Stokes parameter.
    """

    N_chan, N_comp, N_stokes, N_pix = T.shape

    # One standard normal draw per (channel, Stokes, pixel) -- this is the fluctuation
    # term that turns this into an actual *sample* from the posterior, not just its mean.
    eta1 = np.random.standard_normal(size=(N_chan, N_stokes, N_pix))

    # Move (N_stokes, N_pix) to the front as "batch" axes so we can use numpy's
    # batched linear algebra (np.linalg.solve broadcasts over leading dims).
    # (N_stokes, N_pix, N_chan, N_comp)
    T_T = np.transpose(T, (2, 3, 0, 1))
    # (N_stokes, N_pix, N_chan)
    d_T = np.transpose(d, (1, 2, 0))
    # (N_stokes, N_pix, N_chan)
    N_T = np.transpose(N, (1, 2, 0))
    # (N_stokes, N_pix, N_chan)
    eta1_T = np.transpose(eta1, (1, 2, 0))

    # Diagonal N^-1 and N^-1/2, per channel (broadcast against T_T below)
    # (N_stokes, N_pix, N_chan)
    Ninv = 1.0 / N_T
    # (N_stokes, N_pix, N_chan) 
    Ninv_sqrt = np.sqrt(Ninv)

    # --- RHS: T^T N^-1 d  +  T^T N^-1/2 eta1 ---
    # N^-1 d, per channel
    weighted_d = Ninv * d_T
    # N^-1/2 eta1, per channel  
    weighted_eta = Ninv_sqrt * eta1_T

    # Contract over the channel axis f: sum_f T[f,c] * (weighted_d + weighted_eta)[f]
    # (N_stokes, N_pix, N_comp)
    rhs = np.einsum('spfc,spf->spc', T_T, weighted_d + weighted_eta)

    # --- LHS: T^T N^-1 T, an (N_comp x N_comp) "curvature" matrix at every pixel ---
    # sum_f T[f,c] * N^-1[f] * T[f,k] -> matrix indexed by (c, k)
    # (N_stokes, N_pix, N_comp, N_comp)
    lhs = np.einsum('spfc,spf,spfk->spck', T_T, Ninv, T_T)

    # Solve the small N_comp x N_comp system simultaneously at every (Stokes, pixel)
    # (N_stokes, N_pix, N_comp)
    x_T = np.linalg.solve(lhs, rhs)

    # Back to (N_comp, N_stokes, N_pix) to match your existing array convention
    x_sample = np.transpose(x_T, (2, 0, 1))

    return x_sample