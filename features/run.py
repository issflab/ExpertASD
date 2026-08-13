#!/usr/bin/env python3
"""Convenience runner: extract every enabled feature, then plot each one.

  python -m features.run --config config/features.yaml
  python -m features.run --config config/features.yaml --out /data/feat/eval --no-plot

This is pure orchestration -- it calls `run_extraction` then `plot_umap`, the
same functions `features.extract` and `features.visualize` expose, and adds no
logic of its own. `extract` stays extraction-only and `visualize` stays
plotting-only; this file just chains them for the common "do both" case.

Plots land beside the embeddings as `<feature>_umap.png`. A feature whose run
produced no embeddings (everything failed or was too short) is skipped rather
than erroring.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import ConfigError
from .extract import _progress_callback, build_parser, load_config
from .pipeline import embeddings_path, output_specs, plot_path, run_extraction

logger = logging.getLogger("features.run")


def _build_parser():
    parser = build_parser()  # inherit every extract override flag verbatim
    parser.prog = "python -m features.run"
    parser.description = "Extract enabled features, then write a UMAP plot for each."
    parser.add_argument(
        "--no-plot", action="store_true", help="extract only; skip the UMAP plots"
    )
    parser.add_argument(
        "--force-extract", action="store_true",
        help="re-extract every file, ignoring resume (same as --no-resume)",
    )
    parser.add_argument(
        "--force-plot", action="store_true",
        help="redraw each UMAP plot even if the PNG already exists",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_config(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.force_extract:
        cfg.resume = False  # re-extract everything, don't skip existing outputs

    # -- extract (identical to features.extract) --------------------------
    progress, bar = _progress_callback()
    try:
        result = run_extraction(cfg, progress=progress)
    finally:
        if bar is not None:
            bar.close()

    print(
        f"\nprocessed={result.n_processed} skipped={result.n_skipped} "
        f"too_short={result.n_too_short} failed={result.n_failed}"
    )
    for artifact in result.artifact_manifest():
        print(f"  {artifact['kind']:<11} {artifact['path']}  ({artifact['media_type']})")

    if args.no_plot:
        return 1 if result.n_processed == 0 else 0

    # -- plot each feature that actually produced embeddings --------------
    from .visualize import plot_umap  # imports plotly lazily

    print("\nplots:")
    for spec in output_specs(cfg):
        npz = embeddings_path(cfg, spec)
        if not npz.exists():
            logger.warning("no embeddings for '%s'; skipping its plot", spec.stem)
            continue
        out_png = plot_path(cfg, spec)
        if out_png.exists() and not args.force_plot:
            # Incremental by default: keep existing figures. Use --force-plot to
            # redraw (e.g. after changing umap/plot params in the config).
            print(f"  {'exists':<11} {out_png}  (use --force-plot to redraw)")
            continue
        try:
            for artifact in plot_umap(npz, out_png, cfg.umap, cfg.plot):
                print(f"  {artifact.kind:<11} {artifact.path}  ({artifact.media_type})")
        except Exception as exc:
            # A plotting failure must not discard already-written embeddings.
            logger.error("failed to plot '%s': %s: %s", spec.stem, type(exc).__name__, exc)

    return 1 if result.n_processed == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
