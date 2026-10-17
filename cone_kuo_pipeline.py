"""Cone → 2D Kuo bipolar network pipeline (ON + OFF).

Given a 2D stimulus movie (R*/cone/s) and an aperture mask, this runs:
  1. Per-aperture-pixel biophysical cone model (batched, chunked)
  2. Baseline subtraction: subtract the steady-state photocurrent that a cone
     OUTSIDE the stimulus region (constant mean light) would produce.
     Bipolars see raw pA deviations from that baseline. NO normalization.
  3. 2D Kuo bipolar network (Hill on +input for ON, on -input for OFF),
     with configurable rectify flag and gap-junction gain.
  4. RGC output (2D Gaussian pool over aperture bipolars), averaged over
     `n_noise_trials` bipolar-noise realizations.

Note: Kuo's Hill nonlinearity uses xhalf=0.38 which is calibrated for
normalized contrast. When rectify=True and inputs are raw pA, the Hill
will typically saturate; adjust xhalf via `kuo_kw={'xhalf': ...}` accordingly.
Setting rectify=False bypasses the Hill and is the safe default when using
raw-pA baseline-subtracted inputs.
"""
import numpy as np

from src.cone_model import ConeModel
from src.bipolar_network_model_2d import BipolarNetworkModel2D


DEFAULT_DT = 1e-3
DEFAULT_SETTLE_MS = 500
DEFAULT_KUO_KW = dict(bipolar_spacing=4, bipolar_rf_sigma=2.0,
                      gj_space_constant=4.55,
                      xhalf=0.38, hill_n=2.52, noise_sd=0.14)


def run_cone_kuo_pipeline(stim_r, fps, aperture,
                          dt=DEFAULT_DT, settle_ms=DEFAULT_SETTLE_MS,
                          rectify=True, nl_type=None, gj_gain=0.1,
                          rgc_rf_sigma=None,
                          gc_alpha=0.0, gc_tau=0.1,
                          rgc_rectify=False,
                          softplus_beta=1.0,
                          kuo_kw=None,
                          cone_params=None,
                          n_noise_trials=5, pixel_chunk=4000,
                          on_seed=42, off_seed=43):
    """Run cone -> Kuo (ON + OFF) on a stimulus.

    Parameters
    ----------
    stim_r : (H, W, n_frames) float — intensity in R*/cone/s
    fps    : float — movie frame rate
    aperture : (H, W) bool — cone-model + bipolar aperture mask
    rectify : bool — if False, Hill nonlinearity is bypassed
    gj_gain : float — gap-junction gain (0 = no coupling)
    rgc_rf_sigma : optional override for the RGC RF sigma; defaults to W/3.
    kuo_kw : dict overrides for the model kwargs (see DEFAULT_KUO_KW)
    cone_params : dict, optional — cone model parameters (default: DEFAULT_PARAMS)
    n_noise_trials : bipolar-noise trials to average
    pixel_chunk : per-batch pixel count for the cone model

    Returns
    -------
    dict with keys:
      cone_resp    (H*W, n_samp)   per-pixel cone photocurrent (pA);
                                    zero outside the aperture
      on_bp, off_bp  (n_bipolars, n_samp) Hill outputs
      on_rgc, off_rgc (n_samp,)   RGC scalar time-courses
      on_model, off_model  BipolarNetworkModel2D instances
      baseline_pA : float — steady-state cone current at mean light, subtracted
                            from every cone before bipolar pooling
      ap_flat    : (H*W,) bool — flattened aperture mask
      n_ap       : int — number of aperture pixels
    """
    H, W, n_frames = stim_r.shape
    dur_s = n_frames / fps
    mean_rstar = float(stim_r.mean())
    ap_flat = aperture.ravel()
    n_ap = int(ap_flat.sum())

    # Upsample to model dt (nearest-neighbor hold-per-frame)
    n_samp = int(round(dur_s / dt))
    idx = np.clip((np.arange(n_samp) * fps * dt).astype(np.int64),
                  0, n_frames - 1)
    stim_1k = stim_r[..., idx]                                 # (H, W, n_samp)
    stim_flat = stim_1k.reshape(H * W, n_samp)[ap_flat]        # (n_ap, n_samp)

    n_settle = int(round(settle_ms * 1e-3 / dt))
    settle = np.full((n_ap, n_settle), mean_rstar, dtype=np.float32)
    stim_px = np.concatenate([settle, stim_flat], axis=1)

    cone = ConeModel(dt=dt, params=cone_params)
    resp = np.empty_like(stim_px)
    for i in range(0, n_ap, pixel_chunk):
        j = min(i + pixel_chunk, n_ap)
        resp[i:j] = cone.predict(stim_px[i:j])
    cone_resp_ap = resp[:, n_settle:]                          # (n_ap, n_samp)

    # Baseline = steady-state photocurrent for a cone OUTSIDE the aperture
    # (constant mean-light stimulus). Subtract this scalar from every cone.
    # No further normalization: bipolars receive raw pA deviations.
    baseline_pA = float(cone._steady_state_current(mean_rstar))
    cone_input_ap = cone_resp_ap - baseline_pA

    cone_input_flat = np.zeros((H * W, n_samp), dtype=np.float32)
    cone_input_flat[ap_flat] = cone_input_ap
    cone_resp_flat = np.zeros((H * W, n_samp), dtype=np.float32)
    cone_resp_flat[ap_flat] = cone_resp_ap

    kw = dict(DEFAULT_KUO_KW); kw.update(kuo_kw or {})
    if rgc_rf_sigma is None:
        rgc_rf_sigma = W / 3.0

    on_model = BipolarNetworkModel2D(
        n_pixels_y=H, n_pixels_x=W, rgc_rf_sigma=rgc_rf_sigma,
        aperture_mask=aperture, rectify=rectify, nl_type=nl_type,
        gj_gain=gj_gain, gc_alpha=gc_alpha, gc_tau=gc_tau,
        softplus_beta=softplus_beta, **kw)
    on_rgc, on_bp = on_model.process_movie(
        cone_input_flat, n_noise_trials=n_noise_trials, seed=on_seed, dt=dt)

    off_model = BipolarNetworkModel2D(
        n_pixels_y=H, n_pixels_x=W, rgc_rf_sigma=rgc_rf_sigma,
        aperture_mask=aperture, rectify=rectify, nl_type=nl_type,
        gj_gain=gj_gain, gc_alpha=gc_alpha, gc_tau=gc_tau,
        softplus_beta=softplus_beta, **kw)
    off_rgc, off_bp = off_model.process_movie(
        -cone_input_flat, n_noise_trials=n_noise_trials, seed=off_seed, dt=dt)

    if rgc_rectify:
        on_rgc = np.maximum(on_rgc, 0.0)
        off_rgc = np.maximum(off_rgc, 0.0)

    return dict(cone_resp=cone_resp_flat, on_bp=on_bp, off_bp=off_bp,
                on_rgc=on_rgc, off_rgc=off_rgc,
                on_model=on_model, off_model=off_model,
                baseline_pA=baseline_pA, ap_flat=ap_flat, n_ap=n_ap)
