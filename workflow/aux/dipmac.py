
"""DiPMaC code adapted from the R-CoSMoS and PyCoSMoS packages, see:
    https://cran.r-project.org/web/packages/CoSMoS/vignettes/vignette.html
    https://github.com/FraCap90/PyCoSMoS/blob/main/Code
"""
from numba import njit, prange
from numpy.polynomial.hermite import hermgauss
import scipy
import pandas as pd
import numpy as np
import csv
from math import erf, sqrt
from collections.abc import Sequence
import numba
from numba import njit
from . import stats as s
print("Numba threads:", numba.get_num_threads())

# ----------------------------------------------------------------------
# ACTF pnts based on CoSMoS fit (Marginal & ACS)
# Functions include:
# - ACTF functions (continuous and discrete):
#   used for Auto-correlation transformation fitting
# - ACTI: auto-correlation transformation integral (function supplied to
#   the double integral in ACTPnts
# - ACTPnts: get the (rho_x, rho_z) points
#   for a given marginal distribution
# - ACTF fitting to the (rho_x, rho_z) points
# ----------------------------------------------------------------------


@njit
def actf(rhox, b, c):
    """ACTF model: transform rhox -> rhoz
    Provides tranformation for continuous distributions, based on two parameters.

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
    alpha = 1.0 - c
    A = 1.0 + b * rhox
    B = 1.0 + b
    # safe denominator
    D = B**alpha - 1.0
    eps = 1e-12
    if np.abs(D) < eps:
        D += eps
    N = A**alpha - 1.0
    return N / D


def acti(x, y, dist: str, distarg: dict, rhoz: float, p0: float = 0):
    """ACTI - autocorrelation transformation integral function
    Expression supplied to double integral.

    Args:
        x (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            1st dimension to integrate over
        y (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            2nd dimension to integrate over
        dist (str):
            distribution name
        distarg (Sequence):
            a list of distribution arguments
        rhoz (float):
            Gaussian correlation
        p0 (float):
            probability of zero values

    Returns:
        float:
            the values of the function at x and y, but likely a Callable to
            pass to scipy.integrate if x and y are supplied as lambdas
    """
    # Guard against edge cases
    if not (-1.0 < rhoz < 1.0):
        raise ValueError("rhoz must be in (-1, 1)")
    # Φ(x) using erfc: Φ(x) = 0.5 * erfc(-x / √2)
    Phi_x = 0.5 * scipy.special.erfc(-x / np.sqrt(2.0))
    Phi_y = 0.5 * scipy.special.erfc(-y / np.sqrt(2.0))
    # Map to nonzero part of the marginal: u = (Φ - p0) / (1 - p0)
    denom = (1.0 - p0)
    if denom <= 0:
        raise ValueError("p0 must be < 1")
    u = (Phi_x - p0) / denom
    v = (Phi_y - p0) / denom
    # Clip to (0,1) to avoid ppf hitting ±inf at machine precision
    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    aa = s.qdist(dist, u, **distarg)   # vectorized ppf
    bb = s.qdist(dist, v, **distarg)
    # Correct Gaussian copula density term (NEGATIVE exponent)
    one_m_r2 = 1.0 - rhoz**2
    # exponent: -(x^2 - 2ρxy + y^2) / (2(1-ρ^2))
    expo = - (x*x - 2.0*rhoz*x*y + y*y) / (2.0 * one_m_r2)
    cc = np.exp(expo) / (2.0 * np.pi * np.sqrt(one_m_r2))
    return aa * bb * cc


def integrate_gauss_hermite(rhoz, p0, dist, distarg, n=40, eps=1e-12):
    """
    High-accuracy integral via 2D Gauss-Hermite quadrature.
    n ~ 20-60 is typical; 40 is a good default.
    """
    if not (-1.0 < rhoz < 1.0):
        raise ValueError("rhoz must be in (-1, 1)")
    if not (0.0 <= p0 < 1.0):
        raise ValueError("p0 must be in [0, 1)")
    # GH nodes/weights for ∫ e^{-t^2} g(t) dt
    t, w = hermgauss(n)             # shapes (n,), (n,)
    # For standard normal: x = √2 * t ; φ weight contributes factor 1/√π per dimension
    scale = np.sqrt(2.0)
    norm_factor = 1.0 / np.pi       # (1/√π)^2
    X = scale * t[:, None]          # shape (n,1)
    Z = scale * t[None, :]          # shape (1,n)
    Y = rhoz * X + np.sqrt(1.0 - rhoz**2) * Z   # (n,n)
    # Φ(x) using erfc
    Phi_X = 0.5 * scipy.special.erfc(-X / np.sqrt(2.0))
    Phi_Y = 0.5 * scipy.special.erfc(-Y / np.sqrt(2.0))
    denom = (1.0 - p0)
    u = (Phi_X - p0) / denom
    v = (Phi_Y - p0) / denom
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    # Vectorized quantiles (your qdist is array-safe)
    A = s.qdist(dist, u, **distarg)   # (n,n)
    B = s.qdist(dist, v, **distarg)   # (n,n)
    # Product rule weights
    W = (w[:, None] * w[None, :])     # (n,n)
    # Expectation under N(0,1)×N(0,1): (1/π) * Σ_i Σ_j w_i w_j f(√2 t_i, √2 t_j)
    return norm_factor * np.sum(W * (A * B))


def dblquad(rhoz, p0, dist, distarg):
    # fallback to slower but stable dblquad
    # Double integral upper and lower bounds
    min_ = -7.5 if p0 == 0 else -np.sqrt(2) * scipy.special.erfinv(2*p0)
    max_ = 7.5
    return scipy.integrate.dblquad(
        lambda x, y: acti(x=x, y=y, rhoz=rhoz, p0=p0,
                          dist=dist, distarg=distarg),
        min_, max_, min_, max_, epsrel=1.49e-5
    )[0]


def actpnts(
    margdist: str,
    margarg: Sequence,
    p0: float = 0,
    distbounds: Sequence[float] = [-np.inf, np.inf]
) -> pd.DataFrame:
    """AutoCorrelation Transformed Points
    Transforms a Gaussian process in order to match a target marginal lowers its
    autocorrelation values. The actpnts evaluates the corresponding autocorrelations
    for the given target marginal for a set of Gaussian correlations, i.e., it returns
    (rho_x , rho_z) points where rho_x and rho_z represent,
    respectively, the autocorrelations of the target and Gaussian process.

    Args:
        margdist (str):
            target marginal distribution
        margarg (Sequence):
            list of marginal distribution arguments
        p0 (float):
            probability zero
        distbounds (Sequence[float]):
            bounds of the marginal distribution

    Returns:
        pd.DataFrame:
            table of corresponding rho_x and rho_z values at rho_z values
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95
    """
    rho = pd.DataFrame({'rhox': 0,
                        'rhoz': np.concatenate([np.arange(0.1, 1., step=0.1), [0.95]])
                        })  # create data frame of marginal ACS values
    # Central moments
    dist = getattr(scipy.stats, margdist)
    m1 = dist.mean(**margarg)
    m2 = dist.var(**margarg)
    # Double integral
    temp = [
        integrate_gauss_hermite(rhoz, p0, margdist, margarg, n=40, eps=1e-12)
        for rhoz in rho['rhoz']
    ]
    # Convert to rho_x values
    vals = [(t - m1 ** 2) / m2 for t in temp]
    rho['rhox'] = vals
    # Fall back to dblquad if any rhox outside of (0, 0.1] range
    if any((rho['rhox'] > 1) | (rho['rhox'] < 0)):
        temp = [dblquad(rhoz, p0, margdist, margarg) for rhoz in rho['rhoz']]
        # Convert to rho_x values
        vals = [(t - m1 ** 2) / m2 for t in temp]
        rho['rhox'] = vals
    return rho


def jacobian_actf(rhox, b, c):
    """Analytic Jacobian of actf w.r.t b and c"""
    n = rhox.shape[0]
    J = np.zeros((n, 2))
    alpha = 1.0 - c
    B = 1.0 + b
    B_alpha = B**alpha
    D = B_alpha - 1.0
    eps = 1e-12
    if np.abs(D) < eps:
        D += eps
    B_alpha_m1 = B**(alpha - 1.0)
    for i in range(n):
        x = rhox[i]
        A = 1.0 + b * x
        A_alpha = A**alpha
        N = A_alpha - 1.0
        # df/db
        dN_db = alpha * x * A**(alpha - 1.0)
        dD_db = alpha * B_alpha_m1
        df_db = (dN_db * D - N * dD_db) / (D*D)
        # df/dc
        lnA = np.log(A) if A > 0 else 0.0
        lnB = np.log(B) if B > 0 else 0.0
        dN_dc = -A_alpha * lnA
        dD_dc = -B_alpha * lnB
        df_dc = (dN_dc * D - N * dD_dc) / (D*D)
        J[i, 0] = df_db
        J[i, 1] = df_dc
    return J


def fitactf(rhox, rhoz, max_iter=50, tol=1e-10,):
    """
    Robust Gauss–Newton fit for ACTF parameters (b, c)
    Multiprocessing safe.
    Fits the ACTF (Autocorrelation Transformation Function)
    to the estimated points (rho_x, rho_z)

    Args: 
        rhox (pd.DataFrame | pd.Series): 
            Original distribution rho
        rhoz (pd.DataFrame | pd.Series): 
            Parent gaussian rho
    Returns: 
        Sequence[float]: 
            the actf curve fit parameters
    """
    rhox = np.asarray(rhox, dtype=float)
    rhoz = np.asarray(rhoz, dtype=float)
    # ensure increasing order
    if rhox[0] > rhox[-1]:
        rhox = rhox[::-1]
        rhoz = rhoz[::-1]
    # initial guess near (but not at) identity
    # b, c = 1.0, 1.0
    b, c = 0.5, 0.9
    for _ in range(max_iter):
        f = actf(rhox, b, c)
        residual = rhoz - f
        if not np.all(np.isfinite(residual)):
            break
        J = jacobian_actf(rhox, b, c)
        if not np.all(np.isfinite(J)):
            break
        JTJ = J.T @ J
        JTr = J.T @ residual
        # damping to stabilize
        JTJ += np.eye(2) * 1e-6
        try:
            delta = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            break
        b += delta[0]
        c += delta[1]
        # clamp to reasonable ranges
        b = np.clip(b, 1e-6, 1e3)
        c = np.clip(c, 1e-6, 1e2)
        if np.all(np.abs(delta) < tol):
            break
    return b, c


# ----------------------------------------------------------------------
# Cache the number of kblocks needed to meet the error tolerance
# ----------------------------------------------------------------------

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


def bernoulli_fxk(Fxk_inv, k=24, N=1000):
    """Get the distribution of the daily means from the disaggregation 
    kernel.

    Args:
        Fxk_inv (Callable):
            Inverse of the CDF of the disaggregation kernel
        k (int): 
            Number of values in one k-block.
            Default is 24 (hourly to daily disaggregation)
        N (int): 
            Number of k-blocks to sample. Default is 1000

    Returns:
        Callable: 
            Interpolation function for the CDF of the daily means. 
    """
    F = np.random.random((N, k))
    xk_hat = Fxk_inv(F)
    xk_hat = np.mean(xk_hat, axis=1)
    xk_sorted = np.sort(xk_hat)
    Fxk = np.linspace(1/len(xk_sorted), 1, len(xk_sorted))
    return xk_sorted, Fxk


# @njit
# def compute_v_numba(xk_vals, Fxk_vals, p_e, P):
#     """Compute v-values quickly given precomputed xk and Fxk arrays."""
#     # Get min value for computing absolute (not relative) error near zero.
#     std_xk = np.std(xk_vals)
#     eps_min = p_e * std_xk
#     # V calc
#     res_v = np.empty_like(xk_vals)
#     for i, x in enumerate(xk_vals):
#         delta = max(p_e * abs(x), eps_min)
#         lower = x - delta
#         upper = x + delta
#         # Probability mass in tolerance band
#         u = (
#             np.interp(upper, xk_vals, Fxk_vals)
#             - np.interp(lower, xk_vals, Fxk_vals)
#         )
#         if u <= 0.0:
#             res_v[i] = np.inf
#         else:
#             res_v[i] = np.ceil(np.log(1.0 - P) / np.log(1.0 - u))
#     return res_v


@njit(inline="always")
def interp_numba(x, xp, fp):
    n = xp.size
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    i = np.searchsorted(xp, x) - 1
    x0 = xp[i]
    x1 = xp[i + 1]
    y0 = fp[i]
    y1 = fp[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@njit(parallel=True, fastmath=True)
def compute_v_numba(xk_vals, Fxk_vals, p_e, P):
    std_xk = np.std(xk_vals)
    eps_min = p_e * std_xk
    n = xk_vals.size
    res_v = np.empty(n)
    for i in prange(n):
        x = xk_vals[i]
        delta = max(p_e * abs(x), eps_min)
        lower = x - delta
        upper = x + delta
        u = (
            interp_numba(upper, xk_vals, Fxk_vals)
            - interp_numba(lower, xk_vals, Fxk_vals)
        )
        if u <= 0.0:
            res_v[i] = np.inf
        else:
            res_v[i] = np.ceil(np.log(1.0 - P) / np.log(1.0 - u))
    return res_v


# def number_of_k_blocks(
#     marginal_family: str,
#     marginal_params: dict,
#     p_e: float = 0.05,
#     P: float = 0.95,
#     k: int = 24,
#     N: int = 1000
#     # max_kblocks=5000
# ) -> int:
#     """Calculate the number of k-blocks needed to reach desired error p_e
#     for coarse value xk with confidence 100P%.
#     Default is <= 5% error with 95% certainty.

#     Args:
#         xk (float):
#             Current coarse-scale value
#         marginal_family (str):
#             Marginal distribution family
#         marginal_params (dict):
#             dict of marginal distribution parameters
#         p_e (float):
#             error tolerance
#         P (float):
#             confidence

#     Returns:
#         int:
#             v (number of k blocks required)
#     """
#     def Fx_disagg_kernel(F):
#         return s.qdist(marginal_family, F, **marginal_params)
#     xk_sorted, Fxk = bernoulli_fxk(Fx_disagg_kernel, k=k, N=N)
#     v = compute_v_numba(xk_sorted, Fxk, p_e, P)
#     return xk_sorted, Fxk, v

def number_of_k_blocks(
    marginal_family: str,
    marginal_params: dict,
    p_e: float,
    P: float,
    k: int = 24,
    N: int = 1000,
):
    def Fx_disagg_kernel(F):
        return s.qdist(marginal_family, F, **marginal_params)

    xk_sorted, Fxk = bernoulli_fxk(Fx_disagg_kernel, k=k, N=N)
    v = compute_v_numba(xk_sorted, Fxk, p_e, P)
    return xk_sorted, Fxk, v


def nkblocks(args):
    marginal_family, marg, p_e, P = args
    xk, Fxk, v = number_of_k_blocks(marginal_family, marg, p_e, P)
    return xk, Fxk, v, marg

# ----------------------------------------------------------------------
# DiPMaC run based on CoSMoS fit and corresponding ACTF points
# ----------------------------------------------------------------------


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


def ARp(
    marginal_family: str,
    marginal_params: dict[str, float],
    n: int,
    esd: float,
    val: np.ndarray,
    a_rev: np.ndarray,
    p: int,
    p0: float = 0
) -> dict[str, np.ndarray]:
    """p-order autoregressive model stochastic simulation.

    Args:
        marginal_family (str):
            marginal distribution family
        marginal_params (dict[str, float]):
            marginal distribution parameters
        n (int):
            number of fine-scale values to generate
        esd (float):
            gaussian noise standard deviation
        val (np.ndarray):
            vector of simulated values
        a_rev (np.ndarray):
            solution to the Yule-Walker equations from the ARp model fit
        p (int):
            order of the AR model
        p0 (float):
            probability of zero

    Returns:
        dict[str, np.ndarray]:
            contains x (current x values) and z (running gaussian series)
    """
    gn = np.random.normal(loc=0, scale=esd, size=p +
                          n)  # Gaussian noise generation
    vals = np.empty(n + p) * np.nan
    vals[:p] = val
    for i in range(p, n + p):  # (p + 1):(n + p)) { ## AR
        vals[i] = sum(vals[(i - p):i]*a_rev) + gn[i]
    # p0 + gaussian probabilities calculation
    uval = (scipy.stats.norm.cdf(vals[p:]) - p0)/(1 - p0)
    uval[uval < 0] = 0
    z_time_series = uval
    x_time_series = s.qdist(marginal_family, z_time_series, **marginal_params)
    return {
        'x': x_time_series,
        'z': z_time_series
    }


def ARp_dipmac(
    marginal_family: str,
    marginal_params: dict,
    n: int,
    esd: float,
    val: np.ndarray,
    a_rev: np.ndarray,
    p: int,
    z_series=None,
    p0: float = 0
) -> dict[str, np.ndarray]:
    """Modifies ARp function to be able to accept previous Z values
        as input, and to also output the running Z time series.

    Args:
        marginal_family (str):
            marginal distribution family
        marginal_params (dict):
            marginal distribution parameters
        n (int):
            number of fine-scale values to generate
        esd (float):
            gaussian noise standard deviation
        val (np.ndarray):
            vector of simulated values
        a_rev (np.ndarray):
            solution to the Yule-Walker equations from teh ARp model fit
        p (int):
            order of the AR model
        z_series (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            the Gaussian time series (before it is transformed back into the marginal distribution)
        p0 (float):
            probability of zero

    Returns:
        dict[str, np.ndarray]:
            list of x (current x values) and z (running gaussian series)
    """
    if z_series is None:  # for the first value of the coarse scale time series
        return ARp(marginal_family, marginal_params, n, esd, val, a_rev, p, p0)
    else:  # continued from the existing time series
        vals = np.empty(n + p) * np.nan
        vals[:p] = val
        # Gaussian noise generation
        gn = np.random.normal(loc=0, scale=esd, size=p + n)
        for i in range(p, n + p):
            vals[i] = sum(vals[(i - p):i]*a_rev) + gn[i]
        # p0 + gaussian probabilities calculation
        uval = (scipy.stats.norm.cdf(vals[p:]) - p0)/(1 - p0)
        uval[uval < 0] = 0
        z_time_series = np.concatenate([z_series, uval], axis=0)
        x_time_series = s.qdist(marginal_family, uval, **marginal_params)
        return {
            'x': x_time_series,
            'z': z_time_series
        }


def generateTS_dipmac(
    n: int,
    marginal_family: str,
    marginal_params: dict,
    acsvalue: float,
    actfpara,
    TSn: int = 1,
    z_series=float('nan'),
    p: int = float('nan'),
    p0: float = 0
) -> list[Sequence[float]]:
    """Generate TSn time series.
    Modifies the CoSMoS::generateTS R function to generate time series
    dependent on values generated from previous time steps
    (i.e., use ARp_dipmac instead of CoSMoS::ARp).
    Adapted from CoSMoS::generateTS function (and those referenced therein)

    Args:
        n (int):
            number of values to generate
        marginal_family (str):
            Marginal distribution family
        marginal_params (dict):
            Marginal distribution parameters
        acsvalue (float):
            autocorrelation of fine time series (x, not z)
        actfpara (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            ACTF curve fit parameters
        TSn (int):
            number of time series to generate
        z_series (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            the Gaussian time series (before it is transformed back into the marginal distribution)
        p (int): Order of the AR fit
        p0 (float): probability of zero values

    Returns:
         list[Sequence[float]]:
            list of TSn stochastically simulated sequences
    """
    esd, val, a_rev, p = ARp_params(acsvalue, actfpara, z_series, p)
    out = [ARp_dipmac(marginal_family, marginal_params, n, esd,
                      val, a_rev, p, z_series, p0) for _ in range(TSn)]
    return out


def iterate_k_blocks(
    xk: float,
    n_k_blocks: int,
    marginal_family: str,
    marginal_params: dict,
    acsvalue: float,
    actfpara,
    n_fine_per_coarse: int,
    z_values=float('nan')
) -> list[float]:
    """Create all the k-blocks needed for coarse value xk. Return the one with the
        smallest error.

    Args:
        xk (float):
            coarse scale value to disaggregate
        n_k_blocks (int):
            number of k blocks required to meet error tolerance.
        marginal_family (str):
            Marginal distribution family
        marginal_params (dict):
            Marginal distribution parameters
        acsvalue (float):
            autocorrelation of fine time series (x, not z)
        actfpara (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            ACTF curve fit parameters
        n_fine_per_coarse (int):
            number of fine scale values per coarse scale value
        z_values (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            preceding z_values (if not the beginning of the time series)

    Returns:
        list[float]:
            k-block (temporally downscaled series) with the smallest error
    """
    if any(np.isnan(z_values)):
        z_series = None
    else:
        z_series = z_values
    all_k_blocks = generateTS_dipmac(n=n_fine_per_coarse, marginal_family=marginal_family,
                                     marginal_params=marginal_params, acsvalue=acsvalue,
                                     actfpara=actfpara, TSn=n_k_blocks,
                                     z_series=z_series)
    k_block_means = [np.mean(x['x']) for x in all_k_blocks]
    all_errors = [k_block_error(ke, xk) for ke in k_block_means]
    best_kblock = [
        k_block for k_block, err in zip(all_k_blocks, all_errors)
        if err == min(all_errors)
    ]
    if len(best_kblock) < 1:
        return None
    else:
        return best_kblock[0]
