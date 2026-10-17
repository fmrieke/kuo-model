"""2D extension of the Kuo bipolar network model (Kuo, Schwartz & Rieke 2016).

Same pipeline as the 1D model but on a 2D pixel grid:
    2D cone stimulus (H x W) -> bipolar array on 2D grid ->
      isotropic gap-junction spread -> Hill nonlinearity ->
      2D Gaussian RGC pool

Interfaces:
    process_frame(frame_2d, noise_vec)  ->  (rgc_scalar, bp_outputs, bp_currents)
    process_movie(movie_flat, n_noise_trials=5, seed=42)
        movie_flat : (n_pixels_2d, n_time)  where n_pixels_2d = H*W
        returns    : (rgc(t), bp_outputs(n_bipolars, n_time))
"""
import numpy as np


class BipolarNetworkModel2D:
    def __init__(self, n_pixels_y, n_pixels_x,
                 bipolar_spacing=4, bipolar_rf_sigma=2.0,
                 gj_space_constant=4.55, gj_gain=0.1,
                 rgc_rf_sigma=20.0,
                 xhalf=0.38, hill_n=2.52, noise_sd=0.14,
                 aperture_mask=None,
                 jitter=1.0, seed=1, rectify=True, nl_type=None,
                 gc_alpha=0.0, gc_tau=0.1,
                 softplus_beta=1.0):
        self.H = n_pixels_y
        self.W = n_pixels_x
        self.n_pixels_2d = self.H * self.W
        self.bipolar_spacing = bipolar_spacing
        self.bipolar_rf_sigma = bipolar_rf_sigma
        self.gj_lambda = gj_space_constant
        self.gj_gain = gj_gain
        self.rgc_rf_sigma = rgc_rf_sigma
        self.xhalf = xhalf
        self.hill_n = hill_n
        self.noise_sd = noise_sd
        # nl_type takes precedence over rectify (for backwards compat).
        # Supported: 'hill' (Kuo default), 'relu' (piecewise-linear
        # half-wave rectifier: max(0, c)), 'linear' (identity).
        if nl_type is None:
            nl_type = 'hill' if rectify else 'linear'
        assert nl_type in ('hill', 'relu', 'linear', 'softplus'), (
            f"nl_type must be one of 'hill','relu','linear','softplus', got {nl_type!r}")
        self.softplus_beta = softplus_beta
        self.nl_type = nl_type
        self.rectify = (nl_type != 'linear')
        self.gc_alpha = gc_alpha
        self.gc_tau = gc_tau

        if aperture_mask is None:
            self.aperture_mask = np.ones((self.H, self.W), dtype=bool)
        else:
            self.aperture_mask = aperture_mask.astype(bool)

        self._build_array(jitter, seed)
        self._build_rf_weights()
        self._build_gj_weights()
        self._build_rgc_weights()

    def _build_array(self, jitter, seed):
        # Bipolar centers on a 2D grid at `bipolar_spacing`
        margin = int(3 * self.bipolar_rf_sigma)
        ys = np.arange(-margin, self.H + margin, self.bipolar_spacing,
                       dtype=float)
        xs = np.arange(-margin, self.W + margin, self.bipolar_spacing,
                       dtype=float)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        centers = np.stack([yy.ravel(), xx.ravel()], axis=1)  # (N, 2)
        rng = np.random.RandomState(seed)
        centers += jitter * rng.randn(*centers.shape)
        # Keep only bipolars whose center is inside the aperture bounding box
        ci = centers[:, 0].round().astype(int)
        cj = centers[:, 1].round().astype(int)
        ok = ((ci >= 0) & (ci < self.H) &
              (cj >= 0) & (cj < self.W) &
              self.aperture_mask[np.clip(ci, 0, self.H - 1),
                                 np.clip(cj, 0, self.W - 1)])
        self.bp_centers = centers[ok]
        self.n_bipolars = self.bp_centers.shape[0]

    def _build_rf_weights(self):
        """Sparse 2D-Gaussian RF weights per bipolar; masked to aperture."""
        sigma = self.bipolar_rf_sigma
        radius = int(np.ceil(3 * sigma))
        H, W = self.H, self.W
        W_rf = np.zeros((self.n_bipolars, self.n_pixels_2d))
        for i, (cy, cx) in enumerate(self.bp_centers):
            y_lo, y_hi = max(0, int(cy - radius)), min(H, int(cy + radius) + 1)
            x_lo, x_hi = max(0, int(cx - radius)), min(W, int(cx + radius) + 1)
            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                       / (2 * sigma ** 2))
            g *= self.aperture_mask[y_lo:y_hi, x_lo:x_hi]
            s = g.sum()
            if s <= 0:
                continue
            g = g / s
            flat_idx = (yy * W + xx).ravel()
            W_rf[i, flat_idx] = g.ravel()
        self.bp_rf_weights = W_rf

    def _build_gj_weights(self):
        diff = self.bp_centers[:, None, :] - self.bp_centers[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        cutoff = 5 * self.gj_lambda
        w = np.exp(-dists / self.gj_lambda)
        w[dists > cutoff] = 0
        np.fill_diagonal(w, 0)
        self.gj_dist_weights = w
        self.gj_row_sums = w.sum(axis=1)

    def _build_rgc_weights(self):
        # 2D Gaussian centered on the aperture centroid
        ys, xs = np.where(self.aperture_mask)
        cy = ys.mean(); cx = xs.mean()
        dy = self.bp_centers[:, 0] - cy
        dx = self.bp_centers[:, 1] - cx
        w = np.exp(-(dy ** 2 + dx ** 2) / (2 * self.rgc_rf_sigma ** 2))
        if w.sum() > 0:
            w = w / w.sum()
        self.rgc_weights = w

    def hill_func(self, c):
        if self.nl_type == 'linear':
            return c
        if self.nl_type == 'relu':
            return np.maximum(c, 0.0)
        if self.nl_type == 'softplus':
            return np.log1p(np.exp(self.softplus_beta * c)) / self.softplus_beta
        out = np.zeros_like(c)
        pos = c > 0
        out[pos] = 1.0 / (1.0 + (self.xhalf / c[pos]) ** self.hill_n)
        return out

    def _gap_junction_spread(self, bp_currents):
        # Kuo et al. 2016 driving-force-dependent coupling:
        #   c_gj(i) = Σ_j g·(c_b(i) - c_b(j))·exp(-d/λ)
        #   c_total(i) = c_b(i) + c_gj(i)
        #            = (1 + g·Σw) · c_b(i)  -  g · Σ(w·c_b(j))
        spread = self.gj_dist_weights @ bp_currents
        if bp_currents.ndim == 1:
            self_term = (1 + self.gj_gain * self.gj_row_sums) * bp_currents
        else:
            self_term = ((1 + self.gj_gain * self.gj_row_sums)[:, None]
                         * bp_currents)
        return self_term - self.gj_gain * spread

    def _apply_nl_with_gc(self, bp_currents, dt):
        n_bp, n_time = bp_currents.shape
        decay = np.exp(-dt / self.gc_tau)
        scale = dt / self.gc_tau
        bp_out = np.zeros_like(bp_currents)
        h = np.zeros(n_bp)
        for t in range(n_time):
            raw = self.hill_func(bp_currents[:, t])
            gain = 1.0 / (1.0 + self.gc_alpha * h)
            bp_out[:, t] = raw * gain
            h = h * decay + bp_out[:, t] * scale
        return bp_out

    def process_movie(self, movie_flat, n_noise_trials=5, seed=42, dt=None):
        """movie_flat : (n_pixels_2d, n_time)."""
        n_time = movie_flat.shape[1]
        bp_contrasts_all = self.bp_rf_weights @ movie_flat  # (n_bp, n_time)
        rng = np.random.RandomState(seed)
        rgc_trials = np.zeros((n_noise_trials, n_time))
        bp_trials = np.zeros((n_noise_trials, self.n_bipolars, n_time))
        use_gc = self.gc_alpha > 0 and dt is not None
        for trial in range(n_noise_trials):
            noise = self.noise_sd * rng.randn(self.n_bipolars)[:, None]
            bp_contrasts = bp_contrasts_all + noise
            bp_currents = self._gap_junction_spread(bp_contrasts)
            if use_gc:
                bp_out = self._apply_nl_with_gc(bp_currents, dt)
            else:
                bp_out = self.hill_func(bp_currents)
            rgc = self.rgc_weights @ bp_out
            rgc_trials[trial] = rgc
            bp_trials[trial] = bp_out
        return rgc_trials.mean(axis=0), bp_trials.mean(axis=0)
