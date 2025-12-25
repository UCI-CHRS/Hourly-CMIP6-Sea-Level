
"""DiPMaC code adapted from the R-CoSMoS and PyCoSMoS packages, see:
    https://cran.r-project.org/web/packages/CoSMoS/vignettes/vignette.html
    https://github.com/FraCap90/PyCoSMoS/blob/main/Code
    This version uses Numba just-in-time compilation to speed up calculations.
"""

from collections.abc import Sequence
import numpy as np
import scipy
from numba import njit, prange
from aux import stats as s


@njit
def actf(rhox, b: float, c: float) -> float:
    """Provides tranformation for continuous distributions, based on two parameters.

    Args:
        rhox (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            marginal correlation value
        b (float):
            1st line parameter
        c (float):
            2nd line parameter

    Returns:
        float:
            the transformed ACS for the parent-Gaussian process
    """
    rhoz = ((1 + b * rhox) ** (1 - c) - 1)/((1 + b) ** (1 - c) - 1)
    return rhoz


@njit
def AR1(n: int, alpha: float, mean: float = 0, sd: float = 1) -> np.ndarray:
    """Stochastic simulation of n values with a first order autoregressive model

    Args:
        n (int):
            number of values to simulate
        alpha (float):
            autocorrelation of lag 1
        mean (float):
            mean of the gaussian noise to add
        sd (float):
            standard deviation of the gaussian noise to add

    Returns:
        np.ndarray:
            n values simulated from the AR1 model with parameter alpha.
    """
    emean = (1 - alpha) * mean  # Gaussian noise mean
    esd = np.sqrt(1 - alpha ** 2) * sd  # Gaussian noise sd
    val = np.empty(n) * np.nan
    val[0] = np.random.normal(loc=mean, scale=sd)  # values vector
    if n != 1:
        # Gaussian noise vector
        gn = np.random.normal(loc=emean, scale=esd, size=n)
        for i in range(1, n):  # AR
            val[i] = val[(i - 1)] * alpha + gn[i]
    return val


@njit
def ARp_params(
    acsvalue: float,
    actfpara,
    z_series,
    p: int
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Get parameters of a p-order autoregressive model.

    Args:
        acsvalue (float):
            autocorrelation of fine time series (x, not z)
        actfpara (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            ACTF curve fit parameters
        z_series (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            z values to use for the first p values of the model
        p (int):
            order of the AR model

    Returns:
        tuple[float, np.ndarray, np.ndarray, int]:
            float:
                esd (gaussian noise standard deviation)
            np.ndarray:
                val (vector of simulated values)
            np.ndarray:
                a_rev (solution to the Yule-Walker equations from the ARp model fit)
            int:
                p (order of the AR model, calculated if not supplied)
    """
    transacsvalue = actf(acsvalue, b=actfpara[0], c=actfpara[1])
    p = sum(transacsvalue > .01) - 1
    if z_series is not None:
        if p > len(z_series):
            p = len(z_series)
    P = np.empty((p, p)) * np.nan  # cov matrix generation
    for i in range(p):
        P[i, i:p] = transacsvalue[0:(p - i)]
        P[i, 0:i] = transacsvalue[i:0:-1]
    rho = transacsvalue[1:p+1]
    a = np.linalg.solve(P, rho)  # Yule-Walker
    esd = np.sqrt(1 - sum(rho*a))  # gaussian noise standard deviation
    if z_series is None:
        # values vector (first values are generated using AR1 to ensure ACS)
        val = AR1(n=p, alpha=rho[1])
    else:
        val = z_series[-p:]  # start with the last p values simulated
    a_rev = a[::-1]
    return esd, val, a_rev, p



@njit
def k_block_error(xk_hat: float, xk: float) -> float:
    """Calculate k-block error epsilon

    Args:
        xk_hat (float):
            averaged(or summed) simulated k-block
        xk (float):
            what we want xk.sim to average (or sum) to

    Returns:
        float:
            epsilon
    """
    epsilon = np.abs(xk - xk_hat)/np.abs(xk)
    return epsilon


@njit
def ARp_dipmac_numba(n, esd, val, a_rev, p, z_series=None):
    """
    Core AR(p) simulation in nopython mode.
    Only uses floats/arrays, no dicts or strings.
    """
    if z_series is None or len(z_series) == 0:
        prev_z = np.empty(0, dtype=np.float64)
    else:
        prev_z = z_series

    vals = np.empty(n + p, dtype=np.float64)
    vals[:p] = val
    gn = np.random.normal(0.0, esd, n + p)

    for i in range(p, n + p):
        acc = 0.0
        for j in range(p):
            acc += vals[i - p + j] * a_rev[j]
        vals[i] = acc + gn[i]

    # Convert to uniform [0,1] using normal CDF
    # Approximation to avoid calling erf on array (njit-friendly)
    # tanh approx for erf
    uval = 0.5 * (1.0 + np.tanh(vals[p:] * 0.7978845608028654))
    z_time_series = np.concatenate((prev_z, uval))

    return uval, z_time_series


@njit(parallel=True)
def ARp_dipmac_fast_batch(n_k_blocks, n, esd, val, a_rev, p, z_series=None, p0=0):
    """
    Batch version of ARp_dipmac_fast.
    Returns:
        uvals: (n_k_blocks, n) float64 array
        z_blocks_list: list of arrays (variable length per block)
    """
    uvals = np.empty((n_k_blocks, n), dtype=np.float64)
    z_blocks_list = [np.empty(0, dtype=np.float64) for _ in range(n_k_blocks)]

    for k in prange(n_k_blocks):
        # Call core Numba ARp function
        uval, z_out = ARp_dipmac_numba(
            n, esd, val, a_rev, p, z_series if z_series is not None else np.empty(0))

        # Adjust for p0
        if p0 != 0:
            u_adj = (uval - p0) / (1 - p0)
            for i in range(uval.shape[0]):
                if u_adj[i] < 0:
                    u_adj[i] = 0.0
            uval = u_adj

        uvals[k, :] = uval
        z_blocks_list[k] = z_out

    return uvals, z_blocks_list


def iterate_k_blocks(
    xk,
    n_k_blocks,
    marginal_family,
    marginal_params,
    acsvalue,
    actfpara,
    n_fine_per_coarse,
    z_values=float('nan'),
    p0=0,
):
    """
    Refactored version using ARp_dipmac_fast_batch.
    Returns best block (dict with 'x' and 'z').
    """
    # Determine previous z series
    z_series = None if (np.isnan(z_values).any()) else z_values
    # Compute AR(p) parameters once
    esd, val, a_rev, p = ARp_params(acsvalue, actfpara, z_series, p=1)
    # Generate blocks in batch
    uvals, z_blocks_list = ARp_dipmac_fast_batch(
        n_k_blocks, n_fine_per_coarse, esd, val, a_rev, p, z_series, p0
    )
    # Map uniform -> marginal
    x_blocks = np.empty_like(uvals)
    for k in range(n_k_blocks):
        x_blocks[k, :] = s.qdist(marginal_family, uvals[k, :], **marginal_params)
    # Compute mean errors and pick best block
    k_block_means = np.array([np.mean(x_blocks[k, :]) for k in range(n_k_blocks)])
    best_idx = np.argmin(np.abs(k_block_means - xk))
    best_x = x_blocks[best_idx, :]
    best_z = z_blocks_list[best_idx]
    return {"x": best_x, "z": best_z}
