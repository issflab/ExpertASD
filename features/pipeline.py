"""The one entrypoint: `run_extraction(cfg, progress) -> ExtractionResult`.

Everything else in this package is either config, I/O, or plotting. A future
`worker-features` container is a thin wrapper that builds an ExtractionConfig
from a request payload, passes a progress callback that writes job status, and
returns `result.artifact_manifest()` to the gateway -- which is why this module
prints nothing, reads no environment for its parameters, and holds no global
state.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import ExtractionConfig
from .extractors import build_extractor
from .protocol import ProtocolRecord, load_protocol, resolve_audio

logger = logging.getLogger(__name__)

#: progress(done, total, filename)
ProgressCallback = Callable[[int, int, str], None]

NPZ_MEDIA_TYPE = "application/x-npz"
CSV_MEDIA_TYPE = "text/csv"
PNG_MEDIA_TYPE = "image/png"
HTML_MEDIA_TYPE = "text/html"


@dataclass
class Artifact:
    """One output file. Mirrors the Artifact envelope the gateway will adopt."""

    kind: str  # embeddings | manifest | plot
    path: Path
    media_type: str

    def as_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "path": str(self.path), "media_type": self.media_type}


@dataclass
class FileFailure:
    filename: str
    error: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ExtractionResult:
    artifacts: List[Artifact] = field(default_factory=list)
    n_processed: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    n_too_short: int = 0
    failures: List[FileFailure] = field(default_factory=list)
    too_short: List[FileFailure] = field(default_factory=list)

    def artifact_manifest(self) -> List[Dict[str, str]]:
        return [a.as_dict() for a in self.artifacts]

    def paths(self, kind: str) -> List[Path]:
        return [a.path for a in self.artifacts if a.kind == kind]


# -- audio -----------------------------------------------------------------


class AudioTooShort(ValueError):
    """A clip below `audio.min_duration` -- an intentional skip, not a failure."""


def _audio_duration(path: Path) -> float:
    """Clip duration in seconds from the file header only (no decode)."""
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def load_waveform(
    path: Path,
    target_sr: int,
    max_duration: Optional[float],
    min_duration: Optional[float] = None,
    pad_mode: str = "none",
) -> torch.Tensor:
    """Read `path` as a mono float32 tensor at `target_sr`, length-adjusted.

    Raises AudioTooShort when the clip is shorter than `min_duration`, so a
    degenerate near-empty file (which would otherwise crash a downstream conv)
    is reported as an explicit, counted skip with a readable reason.

    Length handling against `max_duration`:
      pad_mode 'none'  crop only -- long clips truncated, short clips left as-is.
      pad_mode 'tile'  fixed length -- long clips truncated, short clips repeated
                       (tiled) to fill exactly max_duration. Matches the
                       reference pipeline's fixed-window behaviour.
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [channel, sample]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.reshape(-1)
    if wav.numel() == 0:
        raise ValueError("audio file is empty after decoding")
    # Checked before any length adjustment: tiling/cropping would mask the real
    # clip length, and the guard must measure the audio as recorded.
    duration = wav.numel() / target_sr
    if min_duration is not None and duration < min_duration:
        raise AudioTooShort(f"too short: {duration:.3f}s < {min_duration:.3f}s (min_duration)")

    if max_duration is not None:
        target_len = int(max_duration * target_sr)
        if pad_mode == "tile" and wav.numel() < target_len:
            repeats = target_len // wav.numel() + 1
            wav = wav.repeat(repeats)[:target_len]
        else:
            wav = wav[:target_len]
    return wav


# -- output layout ---------------------------------------------------------
#
# The unit of output is a *stream*, not a feature type. Names carry the model
# (and, for SSL, the layer) so switching either writes a new file instead of
# overwriting, and -- via `embeddings_path` -- resume keys on the same, so a run
# never appends one model/layer's vectors onto another's. Crucially, each SSL
# layer is its own stream: `ssl.layers: [6, 7]` writes ssl_<model>_L6.npz AND
# ssl_<model>_L7.npz, each single-layer. Layout: <feature>_<tag>.npz/.csv and
# umap_<tag>.png.


def model_tag(model: str) -> str:
    """Filename-safe tag for a model id: its basename, unsafe chars hyphenated.

    e.g. 'speechbrain/spkrec-ecapa-voxceleb' -> 'spkrec-ecapa-voxceleb',
         'xls_r_300m' -> 'xls_r_300m', 'pyannote/embedding' -> 'embedding'.
    """
    basename = str(model).rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip("-") or "model"


def layer_tag(layer: int) -> str:
    """`7` -> 'L7', `-1` -> 'L-1'."""
    return f"L{layer}"


@dataclass(frozen=True)
class OutputSpec:
    """One output stream: an embeddings file + its plot.

    SSL yields one spec per configured layer (each single-layer); speaker and
    pyannote yield exactly one. `layer` is the SSL layer this stream holds, as
    written in the config (so -1 stays -1), and None for non-SSL streams.
    """

    feature: str          # ssl | speaker | pyannote
    tag: str              # model tag, plus _L<layer> for ssl
    layer: Optional[int]  # the ssl layer this stream holds; None otherwise

    @property
    def stem(self) -> str:
        return f"{self.feature}_{self.tag}"


def output_specs(cfg: ExtractionConfig) -> List[OutputSpec]:
    """Expand enabled features into one OutputSpec per file to be written."""
    specs: List[OutputSpec] = []
    for feature in cfg.features:
        base = model_tag(cfg.extractor_config(feature).model)
        if feature == "ssl":
            for layer in cfg.ssl.layers:
                specs.append(OutputSpec("ssl", f"{base}_{layer_tag(layer)}", layer))
        else:
            specs.append(OutputSpec(feature, base, None))
    return specs


def embeddings_path(cfg: ExtractionConfig, spec: OutputSpec) -> Path:
    return cfg.output_root / f"{spec.stem}.npz"


def manifest_path(cfg: ExtractionConfig, spec: OutputSpec) -> Path:
    return cfg.output_root / f"{spec.stem}.csv"


def plot_path(cfg: ExtractionConfig, spec: OutputSpec) -> Path:
    return cfg.output_root / f"umap_{spec.tag}.png"


def stream_meta(cfg: ExtractionConfig, spec: OutputSpec, dim: int, n: int) -> Dict[str, Any]:
    """Provenance for one embeddings file: what produced these vectors.

    Stored inside the `.npz` so a manifold fitted on one stream can refuse a
    query extracted with a different model, layer, pooling, or audio front-end
    -- a mismatch that is otherwise invisible (the arrays still have the same
    shape) and would silently produce meaningless distances.
    """
    extractor = cfg.extractor_config(spec.feature)
    return {
        "feature": spec.feature,
        "model": str(extractor.model),
        "layer": spec.layer,
        "pooling": cfg.ssl.pooling if spec.feature == "ssl" else None,
        "sample_rate": cfg.audio.sample_rate,
        "max_duration": cfg.audio.max_duration,
        "min_duration": cfg.audio.min_duration,
        "pad_mode": cfg.audio.pad_mode,
        "dim": int(dim),
        "n": int(n),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


#: Meta keys that must agree between a manifold's reference stream and a query
#: stream. `n`/`written_at` legitimately differ; the rest change the geometry.
META_COMPAT_KEYS = (
    "feature", "model", "layer", "pooling",
    "sample_rate", "max_duration", "pad_mode", "dim",
)


def load_stream_meta(npz_path: Path) -> Dict[str, Any]:
    """Read the provenance dict from an embeddings `.npz`; {} for older files."""
    with np.load(npz_path, allow_pickle=False) as data:
        if "meta" not in data.files:
            return {}
        try:
            return json.loads(str(data["meta"].item()))
        except (ValueError, TypeError):
            logger.warning("unreadable meta in %s; treating as absent", npz_path)
            return {}


def _load_existing(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception as exc:  # a truncated npz from a killed run
        logger.warning("ignoring unreadable existing output %s (%s)", path, exc)
        return None


def _write_outputs(
    cfg: ExtractionConfig,
    spec: OutputSpec,
    embeddings: np.ndarray,
    records: Sequence[ProtocolRecord],
) -> Tuple[Artifact, Artifact]:
    """Write `<stem>.npz` + `<stem>.csv` atomically-ish (tmp then replace)."""
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    filenames = np.array([r.filename for r in records], dtype=object)
    labels = np.array([r.label for r in records], dtype=object)
    systems = np.array([r.system if r.system else "" for r in records], dtype=object)

    npz = embeddings_path(cfg, spec)
    tmp = npz.with_suffix(".npz.tmp")
    # Write through a handle: np.savez appends '.npz' to any path that lacks it,
    # which would defeat the tmp-then-rename.
    with tmp.open("wb") as fh:
        np.savez(
            fh,
            embeddings=embeddings.astype(np.float32, copy=False),
            filenames=filenames.astype(str),
            labels=labels.astype(str),
            systems=systems.astype(str),
            meta=np.array(json.dumps(
                stream_meta(cfg, spec, embeddings.shape[1], embeddings.shape[0])
            )),
        )
    tmp.replace(npz)

    csv_path = manifest_path(cfg, spec)
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with tmp_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "label", "system", "speaker"])
        for record in records:
            writer.writerow([record.filename, record.label, record.system or "", record.speaker or ""])
    tmp_csv.replace(csv_path)

    logger.info("wrote %s (%d x %d) and %s", npz, *embeddings.shape, csv_path)
    return (
        Artifact("embeddings", npz, NPZ_MEDIA_TYPE),
        Artifact("manifest", csv_path, CSV_MEDIA_TYPE),
    )


# -- device -----------------------------------------------------------------

CUDA_INIT_ATTEMPTS = 6
CUDA_INIT_DELAY_SEC = 5.0


def _prime_cuda(device: str) -> None:
    """Establish a CUDA context before loading models, retrying transient init.

    On shared nodes, CUDA context creation intermittently fails with 'CUDA
    driver initialization failed' even when the GPU is healthy and idle
    (nvidia-smi fine, cuInit ok) -- it recovers within seconds. torch can retry
    in-process, so a few spaced attempts land in a good window instead of
    aborting a long run on a momentary glitch. A no-op for CPU runs.
    """
    if not str(device).startswith("cuda"):
        return
    last: Optional[Exception] = None
    for attempt in range(1, CUDA_INIT_ATTEMPTS + 1):
        try:
            torch.zeros(1, device=device).cpu()  # force real alloc + context
            if attempt > 1:
                logger.info("CUDA ready on attempt %d/%d", attempt, CUDA_INIT_ATTEMPTS)
            return
        except Exception as exc:
            last = exc
            first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            logger.warning(
                "CUDA init attempt %d/%d failed (%s)%s",
                attempt, CUDA_INIT_ATTEMPTS, first_line[:100],
                f"; retrying in {CUDA_INIT_DELAY_SEC:.0f}s" if attempt < CUDA_INIT_ATTEMPTS else "",
            )
            if attempt < CUDA_INIT_ATTEMPTS:
                time.sleep(CUDA_INIT_DELAY_SEC)
    raise RuntimeError(
        f"CUDA could not initialize after {CUDA_INIT_ATTEMPTS} attempts on {device}: "
        f"{type(last).__name__}: {last}. The GPU/driver appears transiently unavailable "
        "on this node -- retry the command, or set device: cpu in the config."
    ) from last


# -- the entrypoint --------------------------------------------------------


def run_extraction(
    cfg: ExtractionConfig,
    progress: Optional[ProgressCallback] = None,
) -> ExtractionResult:
    """Extract every enabled feature type in one pass over the audio.

    `progress(done, total, filename)` is called after each file when supplied;
    it is the only channel this function uses to report incremental state, so a
    worker can turn it straight into job status.
    """
    result = ExtractionResult()

    records = load_protocol(cfg)
    found, missing = resolve_audio(records, cfg.audio_root, cfg.audio.extensions)
    for record in missing:
        result.failures.append(
            FileFailure(record.filename, f"audio file not found under {cfg.audio_root}")
        )

    features = list(cfg.features)
    specs = output_specs(cfg)
    # Each SSL layer is its own stream; group them so one forward pass fills all.
    ssl_specs = [s for s in specs if s.feature == "ssl"]
    nonssl_spec = {s.feature: s for s in specs if s.feature != "ssl"}

    existing = {s.stem: _load_existing(embeddings_path(cfg, s)) for s in specs} if cfg.resume else {}

    # A file is only skippable if every output stream already has it.
    done_per_output = {
        s.stem: set(existing[s.stem]["filenames"].tolist()) if existing.get(s.stem) is not None else set()
        for s in specs
    }
    already = set.intersection(*done_per_output.values()) if done_per_output else set()
    pending = [r for r in found if r.filename not in already]
    result.n_skipped = len(found) - len(pending)
    if result.n_skipped:
        logger.info("resume: skipping %d file(s) already present in all outputs", result.n_skipped)

    # Pre-filter pending files below min_duration BEFORE deciding we need the
    # models. Degenerate clips can never be extracted, so without this they stay
    # pending forever -- forcing a model/GPU load on every run (even a re-plot)
    # just to fail on them again. Duration comes from the header (no decode).
    if cfg.audio.min_duration is not None and pending:
        keep: List[ProtocolRecord] = []
        for record in pending:
            try:
                duration = _audio_duration(record.path)
            except Exception:
                keep.append(record)  # unreadable header -> let the loop report it
                continue
            if duration < cfg.audio.min_duration:
                result.too_short.append(FileFailure(
                    record.filename,
                    f"too short: {duration:.3f}s < {cfg.audio.min_duration:.3f}s (min_duration)",
                ))
            else:
                keep.append(record)
        if len(keep) != len(pending):
            logger.info(
                "min_duration: skipping %d pending clip(s) shorter than %.3fs",
                len(pending) - len(keep), cfg.audio.min_duration,
            )
            pending = keep

    # Only touch the device/models when there is something to extract. A
    # resume-only run (every file already present) needs no GPU at all, so it
    # must not fail just because the node's CUDA is unavailable -- this is what
    # lets re-plotting proceed while the GPU is down.
    extractors: Dict[str, object] = {}
    if pending:
        device = cfg.resolve_device()
        logger.info(
            "extracting %s (%d stream(s)) for %d file(s) on %s",
            ", ".join(features), len(specs), len(pending), device,
        )
        # Establish the CUDA context up front, tolerating transient driver-init
        # failures, so model loading doesn't abort the run on a momentary glitch.
        _prime_cuda(device)
        # Built and loaded once per call, never per file: model loading dominates
        # runtime, and a model that cannot load at all must fail the run now rather
        # than raise the same ImportError once per file across a 10k-file protocol.
        extractors = {f: build_extractor(f, cfg.extractor_config(f), device) for f in features}
        for name, extractor in extractors.items():
            try:
                extractor.warmup()
            except Exception as exc:
                raise RuntimeError(
                    f"could not load the '{name}' model: {type(exc).__name__}: {exc}"
                ) from exc
    else:
        logger.info("all %d file(s) already extracted; skipping model load", len(found))

    # Per stream, because streams can be at different completion levels: adding
    # a layer makes files pending for the new stream while the others already
    # hold them, and those must not be re-appended (which would duplicate rows).
    collected: Dict[str, List[np.ndarray]] = {s.stem: [] for s in specs}
    kept_per_output: Dict[str, List[ProtocolRecord]] = {s.stem: [] for s in specs}
    kept: List[ProtocolRecord] = []  # any stream advanced -> counts as processed
    total = len(pending)
    for index, record in enumerate(pending, start=1):
        try:
            waveform = load_waveform(
                record.path, cfg.audio.sample_rate, cfg.audio.max_duration,
                cfg.audio.min_duration, cfg.audio.pad_mode,
            )
            vectors: Dict[str, np.ndarray] = {}
            if ssl_specs:
                # One forward pass, then split into per-layer streams.
                per_layer = extractors["ssl"].extract_layers(waveform, cfg.audio.sample_rate)
                for spec in ssl_specs:
                    vectors[spec.stem] = np.asarray(
                        per_layer[spec.layer], dtype=np.float32
                    ).reshape(-1)
            for feature, spec in nonssl_spec.items():
                vectors[spec.stem] = np.asarray(
                    extractors[feature].extract(waveform, cfg.audio.sample_rate),
                    dtype=np.float32,
                ).reshape(-1)
        except AudioTooShort as exc:
            # Deliberate, configured drop -- not an error. Kept out of `failures`
            # so a clean run over messy data still reports failed=0.
            logger.info("skipping %s (%s)", record.filename, exc)
            result.too_short.append(FileFailure(record.filename, str(exc)))
        except Exception as exc:
            # One unreadable file must not end a 10k-file run.
            logger.warning("failed on %s: %s: %s", record.filename, type(exc).__name__, exc)
            result.failures.append(FileFailure(record.filename, f"{type(exc).__name__}: {exc}"))
        else:
            advanced = False
            for stem, vector in vectors.items():
                if record.filename in done_per_output[stem]:
                    continue  # this stream already has the file; don't duplicate
                collected[stem].append(vector)
                kept_per_output[stem].append(record)
                advanced = True
            if advanced:
                kept.append(record)
        if progress is not None:
            progress(index, total, record.filename)

    result.n_processed = len(kept)
    result.n_failed = len(result.failures)
    result.n_too_short = len(result.too_short)

    for spec in specs:
        prior = existing.get(spec.stem)
        new = np.stack(collected[spec.stem]) if collected[spec.stem] else None
        merged_records = list(kept_per_output[spec.stem])
        if prior is not None and len(prior["filenames"]):
            prior_records = [
                ProtocolRecord(
                    filename=str(fn),
                    label=str(lb),
                    system=str(sy) or None,
                )
                for fn, lb, sy in zip(prior["filenames"], prior["labels"], prior["systems"])
            ]
            merged_records = prior_records + merged_records
            new = prior["embeddings"] if new is None else np.concatenate(
                [prior["embeddings"], new], axis=0
            )
        if new is None or not len(merged_records):
            logger.warning("no embeddings produced for '%s'; nothing written", spec.stem)
            continue
        result.artifacts.extend(_write_outputs(cfg, spec, new, merged_records))

    logger.info(
        "done: processed=%d skipped=%d too_short=%d failed=%d",
        result.n_processed, result.n_skipped, result.n_too_short, result.n_failed,
    )
    return result
