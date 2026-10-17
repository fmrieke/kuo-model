"""Runner: 2D barcode -> cone -> Kuo bipolar (ON+OFF) -> RGC pipeline.

Only CLI, orchestration, and plotting live here. The reusable core is in:
  src/barcode_stim.py           — stimulus generation
  src/cone_kuo_pipeline.py      — cone + Kuo pipeline (ON + OFF)
  src/cone_model.py             — biophysical cone model
  src/bipolar_network_model_2d.py — 2D Kuo bipolar network
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/rieke/Library/CloudStorage/Dropbox/claude')
from src.barcode_stim import (make_vertical_stripe_barcode_pair,
                              make_flashed_barcode_pair, DEFAULT_MAX_RSTAR)
from src.cone_kuo_pipeline import run_cone_kuo_pipeline, DEFAULT_DT

OUT_DIR = '/Users/rieke/Dropbox/claude'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bar_size', type=int, default=10)
    ap.add_argument('--speed', type=float, default=2.0)
    ap.add_argument('--rgc_sigma', type=float, default=20.0)
    ap.add_argument('--n_frames', type=int, default=600)
    ap.add_argument('--fps', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--flashed', action='store_true',
                    help='Flashed barcode (new pattern each --flash_ms)')
    ap.add_argument('--flash_ms', type=float, default=250,
                    help='Flash duration in ms (default 250)')
    ap.add_argument('--no_rectify', action='store_true',
                    help='Disable rectification (linear bipolars)')
    ap.add_argument('--nl_type', choices=['hill', 'relu', 'linear', 'softplus'],
                    default=None,
                    help="Bipolar nonlinearity. 'hill' (Kuo default), "
                         "'relu' (hard rectification), 'softplus' "
                         "(log(1+exp(beta*x))/beta), 'linear' (identity). "
                         "Overrides --no_rectify.")
    ap.add_argument('--softplus_beta', type=float, default=1.0,
                    help='Softplus sharpness (higher = closer to relu)')
    ap.add_argument('--gj_gain', type=float, default=0.1,
                    help='Gap-junction gain (0 = no gap junctions)')
    ap.add_argument('--gc_alpha', type=float, default=0.0,
                    help='Gain control strength (0 = off)')
    ap.add_argument('--gc_tau', type=float, default=0.1,
                    help='Gain control time constant in seconds')
    ap.add_argument('--rgc_rectify', action='store_true',
                    help='Rectify RGC output after spatial summation')
    ap.add_argument('--out_tag', default=None)
    args = ap.parse_args()

    rectify = not args.no_rectify
    nl_type = args.nl_type or ('hill' if rectify else 'linear')
    stim_mode = 'flashed' if args.flashed else 'moving'
    print(f"[2D] bar={args.bar_size} px  mode={stim_mode}  "
          f"RGC σ={args.rgc_sigma} px  seed={args.seed}")
    if args.flashed:
        print(f"flash_ms = {args.flash_ms:.0f}")
    else:
        print(f"speed = {args.speed} px/frame")
    print(f"nl_type = {nl_type}   GJ gain = {args.gj_gain}"
          f"   GC alpha = {args.gc_alpha}  tau = {args.gc_tau*1000:.0f}ms")

    if args.flashed:
        orig_r, lin_r, aperture, rf_2d = make_flashed_barcode_pair(
            args.bar_size, args.rgc_sigma, args.n_frames, args.fps,
            flash_ms=args.flash_ms, seed=args.seed)
    else:
        orig_r, lin_r, aperture, rf_2d = make_vertical_stripe_barcode_pair(
            args.bar_size, args.speed, args.rgc_sigma, args.n_frames, args.seed)
    H, W, _ = orig_r.shape
    print(f"Frame = {H}x{W}, aperture radius = {1.5*args.rgc_sigma:.1f} px "
          f"({int(aperture.sum())} pixels), duration "
          f"{args.n_frames/args.fps:.2f} s")

    def _run(stim, label):
        r = run_cone_kuo_pipeline(stim, args.fps, aperture,
                                  nl_type=nl_type, gj_gain=args.gj_gain,
                                  gc_alpha=args.gc_alpha, gc_tau=args.gc_tau,
                                  rgc_rectify=args.rgc_rectify,
                                  softplus_beta=args.softplus_beta)
        print(f"  [{label}] {r['n_ap']} cones  baseline={r['baseline_pA']:.3f} pA"
              f"  ON RGC {r['on_rgc'].mean():+.4f}±{r['on_rgc'].std():.4f}"
              f"  OFF RGC {r['off_rgc'].mean():+.4f}±{r['off_rgc'].std():.4f}")
        return r

    print("Running Original...");  r_orig = _run(orig_r, 'Original')
    print("Running LinEquiv...");  r_lin  = _run(lin_r,  'LinEquiv')

    if args.out_tag is None:
        suffix = f'_{nl_type}' if nl_type != 'hill' else ''
        suffix += ('_nogj' if args.gj_gain == 0 else '')
        if args.gc_alpha > 0:
            suffix += f'_gc{args.gc_alpha:g}'
        if args.rgc_rectify:
            suffix += '_rgcrect'
        if args.flashed:
            motion = f'flash{args.flash_ms:g}ms'
        else:
            motion = f'sp{args.speed:g}'
        args.out_tag = (f'kuo2d_barcode_bar{args.bar_size}_{motion}'
                        f'_sig{args.rgc_sigma:g}_seed{args.seed}{suffix}')
    out_png = os.path.join(OUT_DIR, args.out_tag + '.png')
    _plot(args, orig_r, lin_r, aperture, r_orig, r_lin, out_png)
    print(f"Saved {out_png}")
    print()
    print(f"{'Movie':<12}{'ON RGC mean':>13}{'ON RGC std':>12}"
          f"{'OFF RGC mean':>14}{'OFF RGC std':>13}")
    for lbl, r in [('LinEquiv', r_lin), ('Original', r_orig)]:
        print(f"  {lbl:<10}{r['on_rgc'].mean():+12.4f}{r['on_rgc'].std():12.4f}"
              f"{r['off_rgc'].mean():+14.4f}{r['off_rgc'].std():13.4f}")


def _plot(args, orig_r, lin_r, aperture, r_orig, r_lin, out_png):
    H, W, _ = orig_r.shape
    t = np.arange(r_orig['cone_resp'].shape[1]) * DEFAULT_DT
    movies = [('LinEquiv',  'C0', r_lin),
              ('Original',  'C3', r_orig)]

    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15,
                         'axes.labelsize': 14, 'xtick.labelsize': 12,
                         'ytick.labelsize': 12, 'legend.fontsize': 12})

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1.6])

    mid = args.n_frames // 2
    for col, (movie, label) in enumerate([(orig_r, 'Original'),
                                           (lin_r,  'LinEquiv')]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(movie[..., mid], cmap='gray', vmin=0, vmax=DEFAULT_MAX_RSTAR,
                  interpolation='nearest')
        ax.contour(aperture.astype(int), levels=[0.5], colors='r', linewidths=1)
        ax.set_title(f'{label}  |  frame {mid}')
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :])
    for lbl, col, r in movies:
        m = r['cone_resp'][r['ap_flat']].mean(axis=0)
        ax.plot(t, m, col, lw=1.5, label=f'{lbl} mean ({m.mean():+.2f} pA)')
    ax.set_ylabel('mean cone current (pA)')
    ax.set_title(f'Mean cone across {r_orig["n_ap"]} aperture cones'
                 f'  |  bar={args.bar_size}px, sp={args.speed} px/fr, '
                 f'σ={args.rgc_sigma} px')
    ax.legend(); ax.grid(alpha=0.3)

    y_all = np.concatenate([r['on_rgc'] for _, _, r in movies] +
                            [r['off_rgc'] for _, _, r in movies])
    ylim = (y_all.min() - 0.02, y_all.max() + 0.02)
    axL = fig.add_subplot(gs[2, 0])
    axC = fig.add_subplot(gs[2, 1], sharey=axL)
    for lbl, col, r in movies:
        axL.plot(t, r['on_rgc'], col, lw=1.2,
                 label=f'{lbl}  mean {r["on_rgc"].mean():+.4f}')
        axC.plot(t, r['off_rgc'], col, lw=1.2,
                 label=f'{lbl}  mean {r["off_rgc"].mean():+.4f}')
    axL.set_title('ON RGC'); axC.set_title('OFF RGC')
    axL.set_ylabel('RGC output')
    for ax in (axL, axC):
        ax.set_xlabel('time (s)'); ax.set_ylim(ylim)
        ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
