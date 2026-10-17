"""2D barcode stimulus generation with LinEquiv counterpart.

Generates a pair of movies with a circular aperture (diameter = 3*rgc_sigma):
    Original : vertical-stripe binary barcode (each stripe `bar_size` px wide)
               drifting rightward at `speed` px/frame.
    LinEquiv : each aperture pixel is set uniformly per frame to
               Σ_i w_i * Original_i(t) / Σ_i w_i, where w is a 2D Gaussian
               (sigma=rgc_sigma) restricted to the aperture.

Pixels outside the aperture in both movies are held at 0.5 * MAX_RSTAR
(mean gray). Intensity units are R*/cone/s.
"""
import numpy as np

DEFAULT_MAX_RSTAR = 30000.0


def make_vertical_stripe_barcode_pair(bar_size, speed, rgc_sigma, n_frames,
                                      seed=1, max_rstar=DEFAULT_MAX_RSTAR):
    """Return (original, lin_equiv, aperture_mask, rf_2d).

    Parameters
    ----------
    bar_size   : int   — stripe width in pixels
    speed      : float — drift speed in px/frame (rightward)
    rgc_sigma  : float — RGC RF sigma in pixels; frame is 3*sigma square
    n_frames   : int   — number of frames
    seed       : int   — RNG seed for the binary stripe sequence
    max_rstar  : float — intensity at pixel value 1.0 (R*/cone/s)

    Returns
    -------
    original : (H, W, n_frames) float32, R*/cone/s
    lin_eq   : (H, W, n_frames) float32, R*/cone/s
    aperture : (H, W) bool  — circular mask of radius 1.5*rgc_sigma
    rf_2d    : (H, W) float32 — 2D Gaussian RF, aperture-restricted, sums to 1
    """
    diam = int(np.ceil(3 * rgc_sigma))
    if diam % 2 == 1:
        diam += 1
    H = W = diam
    radius = 1.5 * rgc_sigma
    cy = cx = (diam - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    aperture = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

    rf_2d = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                   / (2 * rgc_sigma ** 2)).astype(np.float32)
    rf_2d *= aperture
    rf_2d /= rf_2d.sum()

    total_drift = int(np.ceil(speed * n_frames)) + W + bar_size
    n_tiles_x = int(np.ceil(total_drift / bar_size)) + 1
    rng = np.random.RandomState(seed)
    tiles_x = rng.randint(0, 2, size=n_tiles_x).astype(np.float32)
    row_1d = np.repeat(tiles_x, bar_size)
    strip = np.broadcast_to(row_1d[None, :], (H, row_1d.shape[0])).copy()

    orig = np.empty((H, W, n_frames), dtype=np.float32)
    lin_eq = np.empty((H, W, n_frames), dtype=np.float32)
    for t in range(n_frames):
        shift = int(np.floor(t * speed)) % (strip.shape[1] - W)
        seg = strip[:, shift:shift + W]
        orig[..., t] = seg
        lin_eq[..., t] = float((rf_2d * seg).sum())

    outside = ~aperture
    orig[outside, :] = 0.5
    lin_eq[outside, :] = 0.5
    return orig * max_rstar, lin_eq * max_rstar, aperture, rf_2d


def make_flashed_barcode_pair(bar_size, rgc_sigma, n_frames, fps,
                              flash_ms=250, seed=1,
                              max_rstar=DEFAULT_MAX_RSTAR):
    """Flashed barcode: new random pattern every flash_ms, held static.

    Returns same format as make_vertical_stripe_barcode_pair.
    """
    diam = int(np.ceil(3 * rgc_sigma))
    if diam % 2 == 1:
        diam += 1
    H = W = diam
    radius = 1.5 * rgc_sigma
    cy = cx = (diam - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    aperture = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

    rf_2d = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                   / (2 * rgc_sigma ** 2)).astype(np.float32)
    rf_2d *= aperture
    rf_2d /= rf_2d.sum()

    frames_per_flash = max(1, int(round(flash_ms * 1e-3 * fps)))
    n_flashes = int(np.ceil(n_frames / frames_per_flash))
    n_tiles_x = int(np.ceil(W / bar_size)) + 1
    rng = np.random.RandomState(seed)

    orig = np.empty((H, W, n_frames), dtype=np.float32)
    lin_eq = np.empty((H, W, n_frames), dtype=np.float32)
    for flash_i in range(n_flashes):
        tiles = rng.randint(0, 2, size=n_tiles_x).astype(np.float32)
        row_1d = np.repeat(tiles, bar_size)[:W]
        pattern = np.broadcast_to(row_1d[None, :], (H, W)).copy()
        le_val = float((rf_2d * pattern).sum())
        t0 = flash_i * frames_per_flash
        t1 = min(t0 + frames_per_flash, n_frames)
        for t in range(t0, t1):
            orig[..., t] = pattern
            lin_eq[..., t] = le_val

    outside = ~aperture
    orig[outside, :] = 0.5
    lin_eq[outside, :] = 0.5
    return orig * max_rstar, lin_eq * max_rstar, aperture, rf_2d
