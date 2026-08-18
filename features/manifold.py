#!/usr/bin/env python3
"""Frozen reference manifold: project queries, score them in the original space.

A UMAP fitted with the query in its training data is not evidence. Refitting
changes the whole map, so two runs of the same case produce different pictures,
and 2D distance carries no calibrated meaning at all -- UMAP preserves local
neighbourhoods, not global geometry. Worse, in SSL and speaker embeddings the
dominant axis of variation is usually channel (microphone, codec, session), so a
clean visual separation is very often separating YouTube from telephone rather
than genuine from synthetic.

So this module splits the two jobs the plot was implicitly doing:

  the picture   a projection fitted ONCE over the reference corpus and frozen.
                Queries go through `transform`, never `fit_transform`, so the
                map is identical across runs and across cases.

  the evidence  distances computed in the ORIGINAL high-dimensional space and
                reported as percentiles against held-out genuine clips of the
                same speaker. "This clip is further from the genuine manifold
                than 99.2% of known-genuine clips of this speaker" is a claim a
                report can carry. "It looks far from the blue cluster" is not.

Two guards make the output honest rather than merely confident:

  support       a query further from EVERY reference (genuine and fake alike)
                than any reference is from its own neighbours sits outside the
                corpus entirely. Its position is extrapolation. The correct
                output is `inconclusive`, not a score.

  trust         the agreement between the query's neighbours in high-dimensional
                space and its neighbours in 2D. When it is low, the plot is
                lying about where the clip sits, and the UI must say so.

Calibration is honest by construction: the genuine references are split, the
manifold is fitted on one part, and the held-out part is scored exactly like a
query. Nothing is ever scored against a reference set that contains it.

CLI:

  python -m features.manifold fit \
      --features feat/ssl_xls_r_300m_L7.npz --out feat/manifold_L7.joblib

  python -m features.manifold score \
      --manifold feat/manifold_L7.joblib --features feat/query_L7.npz \
      --json out/scores.json --plot out/query_umap.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ConfigError, ExtractionConfig, ManifoldConfig, PlotConfig, UmapConfig
from .pipeline import META_COMPAT_KEYS, load_stream_meta

logger = logging.getLogger("features.manifold")

MANIFOLD_MEDIA_TYPE = "application/octet-stream"
JSON_MEDIA_TYPE = "application/json"

#: Bumped whenever the persisted structure or the meaning of a score changes,
#: so an old .joblib fails loudly instead of being scored under new semantics.
MANIFOLD_FORMAT_VERSION = 1

REAL = "real"


class ManifoldError(RuntimeError):
    """Raised when a manifold cannot be fitted, loaded, or applied to a query."""


# -- distances --------------------------------------------------------------


def _prepare(X: np.ndarray, metric: str) -> np.ndarray:
    """Cast to float64 and, for cosine, L2-normalise so distance is 1 - dot.

    Rows of zero norm would divide by zero; they are left as zeros, which puts
    them at distance 1 from everything -- far, which is the honest reading of a
    degenerate embedding.
    """
    Z = np.asarray(X, dtype=np.float64)
    if Z.ndim != 2:
        raise ManifoldError(f"expected a 2-D embeddings array, got shape {Z.shape}")
    if metric == "cosine":
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = np.divide(Z, norms, out=np.zeros_like(Z), where=norms > 0)
    return Z


def _distances(query: np.ndarray, reference: np.ndarray, metric: str) -> np.ndarray:
    """Full [n_query, n_reference] distance matrix.

    Brute force on purpose. Reference corpora here are hundreds to a few
    thousand vectors, where an exact matrix costs milliseconds; an approximate
    index would add a dependency, a build step, and a source of run-to-run
    variation to a number that goes into a forensic report.
    """
    if metric == "cosine":
        return np.clip(1.0 - query @ reference.T, 0.0, 2.0)
    # euclidean, via the expansion -- same result, one BLAS call
    sq = (
        np.sum(query ** 2, axis=1)[:, None]
        + np.sum(reference ** 2, axis=1)[None, :]
        - 2.0 * (query @ reference.T)
    )
    return np.sqrt(np.maximum(sq, 0.0))


def _mean_knn(dist: np.ndarray, k: int, exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean distance to the `k` nearest references, per query row.

    `exclude[i]` is a reference index to drop for query row i (or -1 for none).
    Leave-one-out matters when scoring rows that are themselves in the reference
    set: without it every reference scores ~0 against itself and the calibration
    distribution collapses to nonsense.
    """
    if dist.shape[1] == 0:
        return np.full(dist.shape[0], np.nan)
    d = dist
    if exclude is not None:
        d = dist.copy()
        rows = np.nonzero(exclude >= 0)[0]
        d[rows, exclude[rows]] = np.inf
    kk = min(k, int(np.isfinite(d).sum(axis=1).min()))
    if kk < 1:
        return np.full(d.shape[0], np.nan)
    idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    return np.take_along_axis(d, idx, axis=1).mean(axis=1)


def _percentile_of(value: float, sample: np.ndarray) -> float:
    """Fraction of `sample` at or below `value`, as a percentage in [0, 100].

    Reported for a distance, so 99.0 reads as "further from the genuine manifold
    than 99% of known-genuine clips of this speaker".
    """
    if sample.size == 0 or not np.isfinite(value):
        return float("nan")
    return 100.0 * float(np.count_nonzero(sample <= value)) / float(sample.size)


# -- results ----------------------------------------------------------------


@dataclass
class QueryScore:
    """Every number this module is willing to stand behind for one clip.

    Deliberately NOT a probability of being fake. Calibrating a probability
    needs a fusion step over the generic detector, the speaker-specific model
    and this stream; here the honest unit is a distance and its percentile
    against known-genuine audio of the same speaker.
    """

    filename: str
    verdict: str                  # atypical | typical | inconclusive
    knn_genuine: float            # mean distance to the k nearest genuine references
    knn_genuine_pct: float        # its percentile among held-out genuine clips
    knn_fake: float               # mean distance to the k nearest generated fakes
    margin: float                 # (knn_genuine - knn_fake) / (sum); >0 leans fake
    mahalanobis: float            # distance from the genuine centroid, PCA subspace
    mahalanobis_pct: float
    residual: float               # energy outside the genuine PCA subspace
    residual_pct: float
    out_of_support: bool          # further from everything than the corpus reaches
    projection_trust: float       # high-dim vs 2D neighbourhood COMPOSITION agreement, [0, 1]
    neighbor_overlap: float       # same-clips overlap; diagnostic only, see _projection_trust
    trust_ok: bool
    coords: List[float] = field(default_factory=list)  # frozen-projection position
    notes: List[str] = field(default_factory=list)     # human-readable caveats

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -- the manifold -----------------------------------------------------------


@dataclass
class ReferenceManifold:
    """A fitted projection plus everything needed to score a query against it.

    Persist with `save`, reload with `load`. The reducer, the reference vectors
    and the calibration samples travel together on purpose: a projection without
    the corpus it was fitted on cannot be audited, and a score without the
    calibration sample it was compared against cannot be interpreted.
    """

    reducer: Any                    # fitted umap.UMAP, frozen
    cfg: ManifoldConfig
    umap_cfg: UmapConfig
    reference: np.ndarray           # prepared (normalised) reference vectors
    reference_coords: np.ndarray    # their 2D positions under `reducer`
    labels: np.ndarray
    systems: np.ndarray
    filenames: np.ndarray
    genuine_mask: np.ndarray
    pca: Any                        # fitted PCA (whiten=True) or None
    cal_knn_genuine: np.ndarray     # held-out genuine scores, the calibration sample
    cal_mahalanobis: np.ndarray
    cal_residual: np.ndarray
    holdout_filenames: np.ndarray
    corpus_radius: float            # largest gap between a reference and its neighbour
    support_radius: float           # beyond this a query is extrapolation, not a score
    holdout_radius: float = 0.0     # furthest a held-out genuine clip actually landed
    source_meta: Dict[str, Any] = field(default_factory=dict)
    format_version: int = MANIFOLD_FORMAT_VERSION

    # -- fitting -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: Sequence[str],
        systems: Sequence[str],
        filenames: Sequence[str],
        cfg: Optional[ManifoldConfig] = None,
        umap_cfg: Optional[UmapConfig] = None,
        source_meta: Optional[Dict[str, Any]] = None,
    ) -> "ReferenceManifold":
        cfg = cfg or ManifoldConfig()
        umap_cfg = umap_cfg or UmapConfig()

        labels = np.asarray(labels, dtype=str)
        systems = np.asarray(systems, dtype=str)
        filenames = np.asarray(filenames, dtype=str)
        is_genuine = labels == REAL
        n_genuine = int(is_genuine.sum())
        if n_genuine < cfg.min_genuine:
            raise ManifoldError(
                f"only {n_genuine} genuine reference clip(s); manifold.min_genuine is "
                f"{cfg.min_genuine}. A manifold fitted on this little genuine audio "
                "cannot support a percentile, so the speaker-specific path should "
                "abstain rather than emit a weak score."
            )

        # The held-out genuine clips are excluded from BOTH the projection fit and
        # the reference set they are scored against, so their scores are drawn
        # from the same process a real query goes through. Anything less and the
        # percentiles are optimistic by construction.
        rng = np.random.default_rng(cfg.holdout_seed)
        genuine_idx = np.nonzero(is_genuine)[0]
        n_holdout = max(1, int(round(cfg.holdout_fraction * n_genuine)))
        n_holdout = min(n_holdout, n_genuine - 1)
        holdout_idx = np.sort(rng.choice(genuine_idx, size=n_holdout, replace=False))
        keep = np.ones(len(labels), dtype=bool)
        keep[holdout_idx] = False

        prepared = _prepare(embeddings, cfg.metric)
        ref = prepared[keep]
        ref_labels, ref_systems, ref_files = labels[keep], systems[keep], filenames[keep]
        ref_genuine = ref_labels == REAL
        logger.info(
            "fitting on %d reference vector(s) (%d genuine, %d fake); %d genuine held out",
            len(ref), int(ref_genuine.sum()), int((~ref_genuine).sum()), n_holdout,
        )
        if not (~ref_genuine).any():
            logger.warning(
                "no fake references: knn_fake and margin will be nan. Generate fakes "
                "for this speaker to get a two-sided score."
            )

        reducer = _fit_reducer(ref, umap_cfg)
        ref_coords = np.asarray(reducer.embedding_, dtype=np.float64)

        pca = _fit_pca(prepared[keep & is_genuine], cfg.pca_components)

        # How far apart the corpus itself is: the largest gap any reference has
        # to its own nearest neighbour. Half of the support radius; the other
        # half comes from the held-out genuine clips below.
        ref_dist = _distances(ref, ref, cfg.metric)
        np.fill_diagonal(ref_dist, np.inf)
        corpus_radius = float(np.max(ref_dist.min(axis=1)))

        manifold = cls(
            reducer=reducer,
            cfg=cfg,
            umap_cfg=umap_cfg,
            reference=ref,
            reference_coords=ref_coords,
            labels=ref_labels,
            systems=ref_systems,
            filenames=ref_files,
            genuine_mask=ref_genuine,
            pca=pca,
            cal_knn_genuine=np.array([]),
            cal_mahalanobis=np.array([]),
            cal_residual=np.array([]),
            holdout_filenames=filenames[holdout_idx],
            corpus_radius=corpus_radius,
            support_radius=corpus_radius,
            source_meta=dict(source_meta or {}),
        )

        raw = manifold._raw_scores(prepared[holdout_idx])
        manifold.cal_knn_genuine = np.sort(raw["knn_genuine"])
        manifold.cal_mahalanobis = np.sort(raw["mahalanobis"])
        manifold.cal_residual = np.sort(raw["residual"])
        # Out of support means "further from every reference than anything we have
        # legitimately observed" -- observed both as the corpus's own internal
        # spacing and as how far a genuine clip of this speaker actually lands
        # when it was not in the reference set. Taking the larger of the two keeps
        # abstention rare enough to mean something when it fires.
        manifold.holdout_radius = float(np.max(raw["nearest"]))
        manifold.support_radius = max(corpus_radius, manifold.holdout_radius)
        logger.info(
            "calibration on %d held-out genuine clip(s): knn_genuine median %.4f, 95th %.4f",
            n_holdout,
            float(np.median(manifold.cal_knn_genuine)),
            float(np.percentile(manifold.cal_knn_genuine, 95)),
        )
        return manifold

    # -- scoring -----------------------------------------------------------

    def _raw_scores(self, prepared: np.ndarray) -> Dict[str, np.ndarray]:
        """Uncalibrated distances for already-prepared query vectors."""
        cfg = self.cfg
        dist = _distances(prepared, self.reference, cfg.metric)
        knn_genuine = _mean_knn(dist[:, self.genuine_mask], cfg.k)
        knn_fake = (
            _mean_knn(dist[:, ~self.genuine_mask], cfg.k)
            if (~self.genuine_mask).any()
            else np.full(len(prepared), np.nan)
        )
        total = knn_genuine + knn_fake
        with np.errstate(invalid="ignore", divide="ignore"):
            margin = np.where(total > 0, (knn_genuine - knn_fake) / total, np.nan)

        if self.pca is not None:
            projected = self.pca.transform(prepared)
            mahalanobis = np.linalg.norm(projected, axis=1)
            # Energy the genuine subspace cannot explain. A synthesiser artefact
            # confined to a low-variance direction is invisible to `mahalanobis`
            # (which lives inside the top components) but shows up here.
            reconstructed = self.pca.inverse_transform(projected)
            residual = np.linalg.norm(prepared - reconstructed, axis=1)
        else:
            mahalanobis = np.full(len(prepared), np.nan)
            residual = np.full(len(prepared), np.nan)

        return {
            "knn_genuine": knn_genuine,
            "knn_fake": knn_fake,
            "margin": margin,
            "mahalanobis": mahalanobis,
            "residual": residual,
            "nearest": dist.min(axis=1),
        }

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """2D positions under the FROZEN projection. Never refits."""
        prepared = _prepare(embeddings, self.cfg.metric)
        self._check_dim(prepared.shape[1])
        return np.asarray(self.reducer.transform(prepared), dtype=np.float64)

    def score(
        self,
        embeddings: np.ndarray,
        filenames: Sequence[str],
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[QueryScore]:
        """Score and project query clips against this frozen manifold."""
        prepared = _prepare(embeddings, self.cfg.metric)
        self._check_dim(prepared.shape[1])
        meta_notes = self._meta_notes(meta)

        raw = self._raw_scores(prepared)
        coords = np.asarray(self.reducer.transform(prepared), dtype=np.float64)
        trust, overlap = self._projection_trust(prepared, coords)

        results: List[QueryScore] = []
        for i, name in enumerate(filenames):
            notes = list(meta_notes)
            pct = _percentile_of(raw["knn_genuine"][i], self.cal_knn_genuine)
            out_of_support = bool(raw["nearest"][i] > self.support_radius)
            trust_ok = bool(trust[i] >= self.cfg.trust_threshold)

            if out_of_support:
                verdict = "inconclusive"
                notes.append(
                    f"outside the reference corpus (nearest reference at {raw['nearest'][i]:.4f}, "
                    f"corpus support radius {self.support_radius:.4f}): the position on the plot "
                    "is extrapolation, and neither the percentile nor the margin is meaningful. "
                    "Most often this is a channel mismatch, not a synthesiser -- compare the "
                    "recording conditions before reading anything into it."
                )
            elif pct >= 95.0:
                verdict = "atypical"
            else:
                verdict = "typical"

            if not trust_ok:
                notes.append(
                    f"2D placement unreliable (neighbourhood composition agreement "
                    f"{trust[i]:.2f} < {self.cfg.trust_threshold:.2f}): the projection puts this "
                    "clip among a different mix of reference clips than it is actually nearest "
                    "to. Read the scores, not the plot."
                )
            if not np.isfinite(raw["knn_fake"][i]):
                notes.append("no fake references in this manifold; margin unavailable.")

            results.append(QueryScore(
                filename=str(name),
                verdict=verdict,
                knn_genuine=float(raw["knn_genuine"][i]),
                knn_genuine_pct=float(pct),
                knn_fake=float(raw["knn_fake"][i]),
                margin=float(raw["margin"][i]),
                mahalanobis=float(raw["mahalanobis"][i]),
                mahalanobis_pct=_percentile_of(raw["mahalanobis"][i], self.cal_mahalanobis),
                residual=float(raw["residual"][i]),
                residual_pct=_percentile_of(raw["residual"][i], self.cal_residual),
                out_of_support=out_of_support,
                projection_trust=float(trust[i]),
                neighbor_overlap=float(overlap[i]),
                trust_ok=trust_ok,
                coords=[float(coords[i, 0]), float(coords[i, 1])],
                notes=notes,
            ))
        return results

    def _projection_trust(
        self, prepared: np.ndarray, coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-query reliability of the 2D placement: (composition, overlap).

        Both compare each query's `trust_k` nearest references in the original
        space with its nearest references in 2D, but they answer different
        questions, and only the first bears on what a human reads off the plot.

        composition  agreement between the two neighbourhoods' GROUP MIX -- one
                     minus the total-variation distance over {genuine, each fake
                     system}. This is the claim the picture actually makes ("this
                     clip sits among the blue points"), so it is what gates the
                     warning. 1.0 means the plot places the query among the same
                     kinds of clip it is genuinely nearest to.

        overlap      fraction of the two neighbourhoods that are the SAME CLIPS.
                     Diagnostic only, never a flag: inside a dense cluster of
                     near-identical genuine audio, exactly which neighbours rank
                     top-k is arbitrary in both spaces, so this runs low even for
                     placements that are entirely correct. Low overlap with high
                     composition is the common and benign case -- the group is
                     right, the fine position is not worth reading.
        """
        n = len(prepared)
        k = min(self.cfg.trust_k, len(self.reference))
        if k < 1:
            return np.full(n, np.nan), np.full(n, np.nan)

        hi = np.argpartition(
            _distances(prepared, self.reference, self.cfg.metric), k - 1, axis=1
        )[:, :k]
        lo = np.argpartition(
            _distances(coords, self.reference_coords, "euclidean"), k - 1, axis=1
        )[:, :k]

        # Group each reference into genuine or its named synthesiser, so the
        # composition check is over the categories the plot's colours encode.
        groups = np.where(self.genuine_mask, REAL, np.where(self.systems != "", self.systems, "fake"))
        codes, encoded = np.unique(groups, return_inverse=True)

        composition = np.empty(n)
        overlap = np.empty(n)
        for i in range(n):
            p = np.bincount(encoded[hi[i]], minlength=len(codes)) / float(k)
            q = np.bincount(encoded[lo[i]], minlength=len(codes)) / float(k)
            composition[i] = 1.0 - 0.5 * float(np.abs(p - q).sum())
            overlap[i] = len(set(hi[i].tolist()) & set(lo[i].tolist())) / float(k)
        return composition, overlap

    # -- compatibility -----------------------------------------------------

    def _check_dim(self, dim: int) -> None:
        if dim != self.reference.shape[1]:
            raise ManifoldError(
                f"query embeddings are {dim}-dimensional but this manifold was fitted on "
                f"{self.reference.shape[1]}-dimensional vectors. They came from different "
                "feature streams; extract the query with the same config."
            )

    def _meta_notes(self, meta: Optional[Dict[str, Any]]) -> List[str]:
        """Compare query provenance against the manifold's, one note per mismatch.

        A warning rather than an error: older `.npz` files carry no meta at all,
        and refusing to score them would be worse than saying so out loud. But
        every mismatch lands in the report, because a query extracted from a
        different layer or a different front-end produces distances that look
        entirely normal and mean nothing.
        """
        if not meta or not self.source_meta:
            return ["provenance unverified: one side of this comparison has no stream metadata."]
        notes = []
        for key in META_COMPAT_KEYS:
            mine, theirs = self.source_meta.get(key), meta.get(key)
            if mine != theirs:
                notes.append(
                    f"provenance mismatch on '{key}': manifold={mine!r} query={theirs!r}. "
                    "The distances below are not comparable."
                )
        return notes

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(self, tmp, compress=3)
        tmp.replace(path)
        logger.info("wrote %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ReferenceManifold":
        import joblib

        path = Path(path)
        if not path.exists():
            raise ManifoldError(f"manifold not found: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise ManifoldError(f"{path} does not hold a ReferenceManifold")
        if obj.format_version != MANIFOLD_FORMAT_VERSION:
            raise ManifoldError(
                f"{path} was written in manifold format v{obj.format_version}, this build "
                f"reads v{MANIFOLD_FORMAT_VERSION}. Refit it -- scores across format "
                "versions are not comparable."
            )
        return obj

    def summary(self) -> Dict[str, Any]:
        """Everything a report needs to state what the query was compared against."""
        return {
            "format_version": self.format_version,
            "n_reference": int(len(self.reference)),
            "n_genuine_reference": int(self.genuine_mask.sum()),
            "n_fake_reference": int((~self.genuine_mask).sum()),
            "fake_systems": sorted({s for s, f in zip(self.systems, ~self.genuine_mask) if f and s}),
            "n_calibration": int(len(self.cal_knn_genuine)),
            "dim": int(self.reference.shape[1]),
            "metric": self.cfg.metric,
            "k": self.cfg.k,
            "support_radius": self.support_radius,
            "corpus_radius": self.corpus_radius,
            "holdout_radius": self.holdout_radius,
            "pca_components": int(self.pca.n_components_) if self.pca is not None else 0,
            "calibration_quantiles": {
                q: float(np.percentile(self.cal_knn_genuine, q)) if len(self.cal_knn_genuine) else None
                for q in (50, 90, 95, 99)
            },
            "source_meta": self.source_meta,
        }


def _fit_reducer(reference: np.ndarray, umap_cfg: UmapConfig):
    """Fit the UMAP that every future query will be projected through.

    `random_state` is always set: a projection that moves between runs cannot
    back a report, and the loss of parallelism is irrelevant at this corpus size.
    """
    import umap

    n_neighbors = umap_cfg.n_neighbors
    if len(reference) <= n_neighbors:
        n_neighbors = max(2, len(reference) - 1)
        logger.warning(
            "only %d reference vectors; reducing umap.n_neighbors %d -> %d",
            len(reference), umap_cfg.n_neighbors, n_neighbors,
        )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=umap_cfg.min_dist,
        metric=umap_cfg.metric,
        n_components=umap_cfg.n_components,
        random_state=umap_cfg.random_state,
    )
    reducer.fit(reference)
    return reducer


def _fit_pca(genuine: np.ndarray, n_components: int):
    """Whitened PCA over the genuine references only.

    Whitening is what makes `||transform(x)||` a Mahalanobis distance from the
    genuine centroid under the genuine covariance -- computed in a subspace
    small enough to be estimable, which a full covariance in 1024 dimensions
    from a few hundred clips is not.
    """
    if n_components <= 0:
        return None
    from sklearn.decomposition import PCA

    k = min(n_components, len(genuine) - 1, genuine.shape[1])
    if k < 1:
        logger.warning("too few genuine references for PCA; Mahalanobis/residual disabled")
        return None
    if k < n_components:
        logger.info("PCA components reduced %d -> %d by the genuine sample size", n_components, k)
    return PCA(n_components=k, whiten=True, random_state=0).fit(genuine)


# -- CLI --------------------------------------------------------------------


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    from .visualize import load_embeddings

    path = Path(path)
    embeddings, labels, systems, filenames = load_embeddings(path)
    return embeddings, labels, systems, filenames, load_stream_meta(path)


def _configs(config_path: Optional[str]) -> Tuple[ManifoldConfig, UmapConfig, PlotConfig]:
    if not config_path:
        return ManifoldConfig(), UmapConfig(), PlotConfig()
    cfg = ExtractionConfig.from_yaml(config_path)
    return cfg.manifold, cfg.umap, cfg.plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m features.manifold",
        description="Fit a frozen reference manifold, or score query clips against one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    sub = parser.add_subparsers(dest="command", required=True)

    fit_p = sub.add_parser("fit", help="fit and persist a manifold from a reference .npz")
    fit_p.add_argument("--features", required=True, help="reference embeddings .npz")
    fit_p.add_argument("--out", required=True, help="path to write the .joblib manifold")
    fit_p.add_argument("--config", help="features YAML, for the manifold/umap blocks")

    score_p = sub.add_parser("score", help="score query clips against a fitted manifold")
    score_p.add_argument("--manifold", required=True, help="a .joblib written by `fit`")
    score_p.add_argument("--features", required=True, help="query embeddings .npz")
    score_p.add_argument("--json", help="write the full score report here")
    score_p.add_argument("--plot", help="write a PNG of the frozen map with the query overlaid")
    return parser


def _cmd_fit(args) -> int:
    manifold_cfg, umap_cfg, _ = _configs(args.config)
    embeddings, labels, systems, filenames, meta = _load_npz(Path(args.features))
    manifold = ReferenceManifold.fit(
        embeddings, labels, systems, filenames, manifold_cfg, umap_cfg, meta
    )
    manifold.save(Path(args.out))
    print(json.dumps(manifold.summary(), indent=2, default=str))
    return 0


def _cmd_score(args) -> int:
    manifold = ReferenceManifold.load(Path(args.manifold))
    embeddings, labels, systems, filenames, meta = _load_npz(Path(args.features))
    scores = manifold.score(embeddings, filenames, meta)

    report = {
        "manifold": {"path": str(Path(args.manifold).resolve()), **manifold.summary()},
        "query": {"path": str(Path(args.features).resolve()), "meta": meta, "n": len(scores)},
        "scores": [s.as_dict() for s in scores],
    }
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"json        {out}  ({JSON_MEDIA_TYPE})")

    for s in scores:
        print(
            f"{s.verdict:<12} {Path(s.filename).name:<40} "
            f"knn_genuine={s.knn_genuine:.4f} (p{s.knn_genuine_pct:.1f})  "
            f"margin={s.margin:+.3f}  trust={s.projection_trust:.2f}"
        )
        for note in s.notes:
            print(f"             ! {note}")

    if args.plot:
        from .visualize import plot_manifold

        _, _, plot_cfg = _configs(getattr(args, "config", None))
        for artifact in plot_manifold(manifold, Path(args.plot), scores, plot_cfg):
            print(f"{artifact.kind:<11} {artifact.path}  ({artifact.media_type})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return _cmd_fit(args) if args.command == "fit" else _cmd_score(args)
    except (ManifoldError, ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
