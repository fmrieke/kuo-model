"""
Biophysical cone phototransduction model (Rieke model).

Implements the full biophysical cascade and a linear approximation
for predicting cone photocurrent from light stimulus.

Biophysical cascade:
    Light -> rhodopsin (r) -> PDE (p) -> cGMP (g) -> current
    with calcium feedback through cyclase (s)

Stimulus must be provided in R*/cone/s. The model applies the
time step internally.

Reference parameters: peripheral primate cone (from LinearMovie.m).
"""

import numpy as np
from numba import njit


# Default parameters: peripheral primate cone (original MATLAB values)
DEFAULT_PARAMS = {
    'sigma': 22,        # rhodopsin activity decay rate (1/sec)
    'phi': 22,          # phosphodiesterase activity decay rate (1/sec)
    'eta': 2000,        # phosphodiesterase activation rate constant (1/sec)
    'k': 0.01,          # constant relating cGMP to current
    'h': 3,             # cooperativity for cGMP -> current
    'cdark': 1,         # dark calcium concentration
    'beta': 9,          # rate constant for calcium removal (1/sec)
    'hillcoef': 4,      # cooperativity for cyclase (Hill coefficient)
    'hillaffinity': 0.5,  # Hill affinity for cyclase
    'gamma': 10,        # rate of increase in opsin activity per R*/sec
    'gdark': 28,        # dark cGMP concentration
}

PARAMS_20260913 = {
    'sigma': 33.9,      # rhodopsin activity decay rate (1/sec)
    'phi': 33.9,        # phosphodiesterase activity decay rate (1/sec)
    'eta': 1344,        # phosphodiesterase activation rate constant (1/sec)
    'k': 0.01,          # constant relating cGMP to current
    'h': 3,             # cooperativity for cGMP -> current
    'cdark': 1,         # dark calcium concentration
    'beta': 16.7,       # rate constant for calcium removal (1/sec)
    'hillcoef': 4,      # cooperativity for cyclase (Hill coefficient)
    'hillaffinity': 6.6e-4,  # Hill affinity for cyclase
    'gamma': 10,        # rate of increase in opsin activity per R*/sec
}
# gdark and darkCurrent are derived
DEFAULT_PARAMS['darkCurrent'] = DEFAULT_PARAMS['k'] * DEFAULT_PARAMS['gdark']**DEFAULT_PARAMS['h'] / 2
PARAMS_20260913['darkCurrent'] = PARAMS_20260913['k'] * 35**PARAMS_20260913['h'] / 2
PARAMS_20260913['gdark'] = (2 * PARAMS_20260913['darkCurrent'] / PARAMS_20260913['k']) ** (1 / PARAMS_20260913['h'])

# Linear filter parameters (fitted at 5000 R*/s for 0.835ms time step)
# Legacy 3-param monophasic form
DEFAULT_LINEAR_PARAMS = {
    'ScFact': 1.2846,
    'TauR': 0.0195,
    'TauD': 0.0158,
}

# LN parametric linear filter parameters (fitted at 5000 R*/s, 1ms time step)
# Filter: (t/tauR)^n / (1+(t/tauR)^n) * exp(-t/tauD) * cos(2*pi*t/tauP + 2*pi*phi/360)
# Response = scale * conv(filter, stimulus * dt) - darkCurrent
DEFAULT_LN_LINEAR_PARAMS = {
    'scale': -150.57,
    'numFilt': 2.98,
    'tauR': 0.0146,
    'tauD': 0.0380,
    'tauP': 140.08,
    'phi': 269.8,
}


@njit(cache=True)
def _run_1d_numba(stm, dt, gdark, k, h, cdark, beta, eta, phi, sigma, gamma,
                  hillcoef, hillaffinity, darkCurrent, cur2ca, smax):
    n = len(stm)
    g = np.empty(n)
    s = np.empty(n)
    c = np.empty(n)
    p = np.empty(n)
    r = np.empty(n)
    cslow = np.empty(n)

    g[0] = gdark
    s[0] = gdark * eta / phi
    c[0] = cdark
    p[0] = eta / phi
    r[0] = 0.0
    cslow[0] = cdark

    for t in range(1, n):
        r[t] = r[t-1] + dt * (-sigma * r[t-1])
        r[t] += gamma * stm[t-1]
        p[t] = p[t-1] + dt * (r[t-1] + eta - phi * p[t-1])
        c[t] = c[t-1] + dt * (cur2ca * k * g[t-1]**h
                               / (1 + cslow[t-1] / cdark)
                               - beta * c[t-1])
        cslow[t] = cslow[t-1]
        s[t] = smax / (1 + (c[t] / hillaffinity) ** hillcoef)
        g[t] = g[t-1] + dt * (s[t-1] - p[t-1] * g[t-1])

    current = -k * g**h / (1 + cslow / cdark)
    return current


@njit(cache=True)
def _run_batch_numba(stim_scaled, dt, gdark, k, h, cdark, beta, eta, phi,
                     sigma, gamma, hillcoef, hillaffinity, darkCurrent,
                     cur2ca, smax):
    n_pixels, n_time = stim_scaled.shape
    out = np.empty((n_pixels, n_time))
    for i in range(n_pixels):
        out[i] = _run_1d_numba(stim_scaled[i], dt, gdark, k, h, cdark, beta,
                               eta, phi, sigma, gamma, hillcoef, hillaffinity,
                               darkCurrent, cur2ca, smax)
    return out


class ConeModel:
    """Biophysical cone phototransduction model.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    params : dict, optional
        Model parameters. Uses peripheral primate cone defaults if not provided.
    """

    def __init__(self, dt, params=None):
        self.dt = dt
        self.params = dict(DEFAULT_PARAMS)
        if params is not None:
            self.params.update(params)

        # Derived quantities
        p = self.params
        self.gdark = p['gdark']
        self.k = p['k']
        self.h = p['h']
        self.cdark = p['cdark']
        self.beta = p['beta']
        self.eta = p['eta']
        self.phi = p['phi']
        self.sigma = p['sigma']
        self.gamma = p['gamma']
        self.hillcoef = p['hillcoef']
        self.hillaffinity = p['hillaffinity']
        self.darkCurrent = p['darkCurrent']

        self.cur2ca = self.beta * self.cdark / self.darkCurrent
        self.smax = (self.eta / self.phi * self.gdark
                     * (1 + (self.cdark / self.hillaffinity) ** self.hillcoef))

    def predict(self, stim_rstar):
        """Run the full biophysical model.

        Parameters
        ----------
        stim_rstar : array-like
            Stimulus in R*/cone/s. Shape: (n_timepoints,) for a single
            trace, or (n_pixels, n_timepoints) for multiple pixels.

        Returns
        -------
        current : ndarray
            Photocurrent in pA (same shape as input). Negative values
            represent inward current (dark current).
        """
        stim_rstar = np.asarray(stim_rstar, dtype=float)
        args = (self.dt, self.gdark, self.k, self.h, self.cdark, self.beta,
                self.eta, self.phi, self.sigma, self.gamma, self.hillcoef,
                self.hillaffinity, self.darkCurrent, self.cur2ca, self.smax)

        if stim_rstar.ndim == 1:
            return _run_1d_numba(stim_rstar * self.dt, *args)
        elif stim_rstar.ndim == 2:
            return _run_batch_numba(stim_rstar * self.dt, *args)
        else:
            raise ValueError(f"stim_rstar must be 1D or 2D, got {stim_rstar.ndim}D")

    def _run_1d(self, stm):
        """Euler integration of the biophysical cascade for a single trace.

        Parameters
        ----------
        stm : ndarray
            Stimulus already scaled by dt (R*/cone/timestep).
        """
        n = len(stm)
        dt = self.dt

        g = np.empty(n)
        s = np.empty(n)
        c = np.empty(n)
        p = np.empty(n)
        r = np.empty(n)
        cslow = np.empty(n)

        # Initial conditions
        g[0] = self.gdark
        s[0] = self.gdark * self.eta / self.phi
        c[0] = self.cdark
        p[0] = self.eta / self.phi
        r[0] = 0.0
        cslow[0] = self.cdark

        for t in range(1, n):
            r[t] = r[t-1] + dt * (-self.sigma * r[t-1])
            r[t] += self.gamma * stm[t-1]
            p[t] = p[t-1] + dt * (r[t-1] + self.eta - self.phi * p[t-1])
            c[t] = c[t-1] + dt * (self.cur2ca * self.k * g[t-1]**self.h
                                   / (1 + cslow[t-1] / self.cdark)
                                   - self.beta * c[t-1])
            cslow[t] = cslow[t-1]  # betaSlow = 0
            s[t] = self.smax / (1 + (c[t] / self.hillaffinity) ** self.hillcoef)
            g[t] = g[t-1] + dt * (s[t-1] - p[t-1] * g[t-1])

        current = -self.k * g**self.h / (1 + cslow / self.cdark)
        return current

    def _steady_state_current(self, mean_rstar):
        """Get the adapted steady-state current at a given mean light level."""
        n_settle = int(1.0 / self.dt)  # 1 second to reach steady state
        stim_const = np.ones(n_settle) * mean_rstar
        resp = self._run_1d(stim_const * self.dt)
        return resp[-1]

    def predict_linear(self, stim_rstar, linear_params=None):
        """Run the linear cone model using the LN parametric filter.

        Parameters
        ----------
        stim_rstar : array-like
            Stimulus in R*/cone/s. Shape: (n_timepoints,) for a single
            trace, or (n_pixels, n_timepoints) for multiple pixels.
        linear_params : dict, optional
            Linear filter parameters (scale, numFilt, tauR, tauD, tauP, phi).
            Uses DEFAULT_LN_LINEAR_PARAMS if not provided.

        Returns
        -------
        current : ndarray
            Photocurrent in pA (same shape as input).
        """
        stim_rstar = np.asarray(stim_rstar, dtype=float)
        lp = linear_params or DEFAULT_LN_LINEAR_PARAMS

        is_1d = stim_rstar.ndim == 1
        if is_1d:
            stim_rstar = stim_rstar.reshape(1, -1)

        n = stim_rstar.shape[1]
        t = np.arange(1, n + 1) * self.dt

        # LN parametric filter
        rising = (t / lp['tauR'])**lp['numFilt'] / (1 + (t / lp['tauR'])**lp['numFilt'])
        decay = np.exp(-t / lp['tauD'])
        oscillation = np.cos(2 * np.pi * t / lp['tauP'] + 2 * np.pi * lp['phi'] / 360)
        filt = rising * decay * oscillation

        # Get adapted steady-state current at the mean light level
        mean_stim = np.mean(stim_rstar)
        steady_state = self._steady_state_current(mean_stim)

        # Compute filter response to constant mean (DC offset to remove)
        mean_dc = lp['scale'] * mean_stim * self.dt * np.sum(filt)

        filt_fft = np.fft.fft(filt)
        out = np.zeros_like(stim_rstar)
        for i in range(stim_rstar.shape[0]):
            stm = stim_rstar[i] * self.dt
            out[i] = lp['scale'] * np.real(np.fft.ifft(np.fft.fft(stm) * filt_fft)) - mean_dc + steady_state

        if is_1d:
            return out[0]
        return out
