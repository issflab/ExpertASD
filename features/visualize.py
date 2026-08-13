#!/usr/bin/env python3
"""UMAP projection of an embeddings `.npz`, rendered with plotly.

  python -m features.visualize --features features_out/ssl.npz --out ssl_umap.png

Writes an interactive `.html` (hover shows filename + label) and, when kaleido is
available, a static `.png` beside it. Colour contract, relied on when skimming
many plots side by side:
  real  -> always blue (plot.real_color)
  fake  -> never blue; one colour per system when the protocol names systems,
           otherwise a single contrasting colour (plot.fake_color)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import ConfigError, ExtractionConfig, PlotConfig, UmapConfig
from .pipeline import HTML_MEDIA_TYPE, PNG_MEDIA_TYPE, Artifact

logger = logging.getLogger("features.visualize")

#: Qualitative palette for fake systems. Every blue is deliberately absent so a
#: fake can never be mistaken for a real point.
FAKE_PALETTE: List[str] = [
    "#ff7f0e",  # orange
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#bcbd22",  # olive
    "#ff9896",  # salmon
    "#c5b0d5",  # lilac
    "#c49c94",  # taupe
    "#f7b6d2",  # rose
    "#dbdb8d",  # sand
    "#98df8a",  # mint
    "#ffbb78",  # apricot
    "#7f7f7f",  # grey
]


def _fake_color(index: int) -> str:
    return FAKE_PALETTE[index % len(FAKE_PALETTE)]


def load_embeddings(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"embeddings file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [k for k in ("embeddings", "filenames", "labels", "systems") if k not in data]
        if missing:
            raise ValueError(f"{npz_path} is missing array(s): {', '.join(missing)}")
        return (
            data["embeddings"],
            data["labels"].astype(str),
            data["systems"].astype(str),
            data["filenames"].astype(str),
        )


def fit_umap(embeddings: np.ndarray, cfg: UmapConfig) -> np.ndarray:
    import umap

    if len(embeddings) <= cfg.n_neighbors:
        # UMAP errors out when n_neighbors >= n_samples; shrink rather than fail
        # so smoke runs on a handful of files still produce a plot.
        n_neighbors = max(2, len(embeddings) - 1)
        logger.warning(
            "only %d samples; reducing umap.n_neighbors %d -> %d",
            len(embeddings), cfg.n_neighbors, n_neighbors,
        )
    else:
        n_neighbors = cfg.n_neighbors

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        n_components=cfg.n_components,
        random_state=cfg.random_state,
    )
    return reducer.fit_transform(embeddings)


def plot_umap(
    npz_path: Path,
    out_path: Path,
    umap_cfg: Optional[UmapConfig] = None,
    plot_cfg: Optional[PlotConfig] = None,
    title: Optional[str] = None,
) -> List[Artifact]:
    """Project the embeddings and write an interactive HTML (+ PNG). Returns the
    artifacts written (PNG first when available, then HTML)."""
    import pandas as pd
    import plotly.graph_objects as go

    umap_cfg = umap_cfg or UmapConfig()
    plot_cfg = plot_cfg or PlotConfig()

    npz_path, out_path = Path(npz_path), Path(out_path)
    embeddings, labels, systems, filenames = load_embeddings(npz_path)
    logger.info("projecting %d x %d embeddings from %s", *embeddings.shape, npz_path)
    coords = fit_umap(embeddings, umap_cfg)

    is_real = labels == "real"
    fake_systems = sorted({s for s, fake in zip(systems, ~is_real) if fake and s})

    # Ordered groups: real first so fakes render on top of the denser real cloud.
    groups: List[Tuple[str, str, np.ndarray]] = [("real", plot_cfg.real_color, is_real)]
    if fake_systems:
        for index, system in enumerate(fake_systems):
            groups.append((system, _fake_color(index), (~is_real) & (systems == system)))
        unlabelled = (~is_real) & np.isin(systems, fake_systems, invert=True)
        if unlabelled.any():
            groups.append(
                ("fake, no system", _fake_color(len(fake_systems)), unlabelled)
            )
    else:
        groups.append(("fake", plot_cfg.fake_color, ~is_real))

    basenames = np.array([os.path.basename(f) for f in filenames])

    fig = go.Figure()
    for name, colour, mask in groups:
        mask = np.asarray(mask)
        if not mask.any():
            continue
        fig.add_trace(go.Scattergl(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=f"{name} ({int(mask.sum())})",
            marker=dict(color=colour, size=plot_cfg.point_size, opacity=plot_cfg.alpha),
            customdata=np.stack([basenames[mask], labels[mask]], axis=1),
            hovertemplate="%{customdata[0]}<br>label=%{customdata[1]}<extra></extra>",
        ))

    ttl = title or (
        f"UMAP — {npz_path.stem}  (n={len(embeddings)}, "
        f"real={int(is_real.sum())}, fake={int((~is_real).sum())})"
    )
    fig.update_layout(
        template="plotly_white",
        title=dict(text=ttl, font=dict(size=18)),
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend=dict(title_text="Labels", x=1.02, y=1.0, xanchor="left", yanchor="top"),
        margin=dict(r=220),
        width=int(plot_cfg.figsize[0] * 96),
        height=int(plot_cfg.figsize[1] * 96),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_path.with_suffix(".html")
    fig.write_html(str(html_path))
    logger.info("wrote %s", html_path)
    artifacts = [Artifact("plot", html_path, HTML_MEDIA_TYPE)]

    # PNG needs kaleido; treat its absence as non-fatal so the HTML still lands.
    try:
        fig.write_image(str(out_path), scale=max(1.0, plot_cfg.dpi / 96.0))
        logger.info("wrote %s", out_path)
        artifacts.insert(0, Artifact("plot", out_path, PNG_MEDIA_TYPE))
    except Exception as exc:  # kaleido missing or export failure
        logger.warning("could not write PNG (%s); HTML written at %s", exc, html_path)

    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m features.visualize",
        description="UMAP-project an embeddings .npz with plotly; blue for real, other colours for fakes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", required=True, help="path to a <feature>.npz")
    parser.add_argument("--out", required=True, help="output PNG path (an .html is written beside it)")
    parser.add_argument("--config", help="features YAML, for the umap/plot blocks")
    parser.add_argument("--title", help="override the plot title")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    umap_cfg, plot_cfg = UmapConfig(), PlotConfig()
    if args.config:
        try:
            cfg = ExtractionConfig.from_yaml(args.config)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        umap_cfg, plot_cfg = cfg.umap, cfg.plot

    for artifact in plot_umap(Path(args.features), Path(args.out), umap_cfg, plot_cfg, args.title):
        print(f"{artifact.kind:<11} {artifact.path}  ({artifact.media_type})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
