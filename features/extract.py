#!/usr/bin/env python3
"""CLI shell over `run_extraction` -- argument parsing, logging and progress only.

  python -m features.extract --config config/features.yaml
  python -m features.extract --config config/features.yaml \
      --protocol /data/protocols/eval.txt --audio-dir /data/eval --out /data/feat/eval

CLI flags override the corresponding config values.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import FEATURE_TYPES, ConfigError, ExtractionConfig
from .pipeline import run_extraction

logger = logging.getLogger("features.extract")


# Curated s3prl upstreams for `ssl.model` in the config. These download cleanly
# from the Hub; the full set is `S3PRLUpstream.available_names()` (~196 names),
# but the *_local/*_url/*_custom variants need a checkpoint you supply.
SSL_MODEL_EPILOG = """\
SSL upstreams for `ssl.model` in the config (set it there, not on the CLI):

  WavLM            wavlm_base   wavlm_base_plus   wavlm_large
  wav2vec2 / XLS-R wav2vec2_base_960   wav2vec2_large_960   wav2vec2_large_ll60k
                   xls_r_300m   xls_r_1b   xls_r_2b
  HuBERT           hubert_base   hubert_large_ll60k

List every upstream available in this env (~196):
  python -c "from s3prl.nn import S3PRLUpstream; print('\\n'.join(S3PRLUpstream.available_names()))"

Note: *_local / *_url / *_custom names are templates needing a checkpoint you
supply. wavlm_large and xls_r_300m are 1024-dim; changing model does NOT resume
onto another model's .npz -- point --out at a fresh dir when you switch.
"""


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show argument defaults AND keep the epilog's hand-formatting intact."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m features.extract",
        description="Extract SSL / speaker / pyannote embeddings over a protocol file.",
        epilog=SSL_MODEL_EPILOG,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--config", required=True, help="path to a features YAML config")
    parser.add_argument("--protocol", help="override protocol.path")
    parser.add_argument("--audio-dir", help="override audio_dir")
    parser.add_argument("--out", dest="output_dir", help="override output_dir")
    parser.add_argument(
        "--features", nargs="+", choices=FEATURE_TYPES, help="override the feature list"
    )
    parser.add_argument("--device", help="override device (auto|cpu|cuda|cuda:N)")
    parser.add_argument(
        "--no-resume", action="store_true", help="re-extract even if outputs already exist"
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def load_config(args: argparse.Namespace) -> ExtractionConfig:
    """Apply CLI overrides on the raw mapping, so validation still sees them."""
    import yaml
    from pathlib import Path

    path = Path(args.config)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}

    if args.protocol:
        data.setdefault("protocol", {})
        data["protocol"] = {**(data.get("protocol") or {}), "path": args.protocol}
    if args.audio_dir:
        data["audio_dir"] = args.audio_dir
    if args.output_dir:
        data["output_dir"] = args.output_dir
    if args.features:
        data["features"] = list(args.features)
    if args.device:
        data["device"] = args.device
    if args.no_resume:
        data["resume"] = False
    return ExtractionConfig.from_dict(data)


def _progress_callback():
    """tqdm when available, a periodic log line otherwise."""
    try:
        from tqdm import tqdm
    except ImportError:
        def log_progress(done: int, total: int, filename: str) -> None:
            if done % 100 == 0 or done == total:
                logger.info("progress %d/%d (%s)", done, total, filename)

        return log_progress, None

    bar = tqdm(unit="file", dynamic_ncols=True)

    def update(done: int, total: int, filename: str) -> None:
        if bar.total != total:
            bar.total = total
            bar.refresh()
        bar.n = done
        bar.set_postfix_str(filename[-40:], refresh=False)
        bar.refresh()

    return update, bar


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_config(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

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
    print("artifacts:")
    for artifact in result.artifact_manifest():
        print(f"  {artifact['kind']:<11} {artifact['path']}  ({artifact['media_type']})")
    if result.too_short:
        print(f"\ntoo short ({len(result.too_short)}), first 10:")
        for skip in result.too_short[:10]:
            print(f"  {skip.filename}: {skip.error}")
    if result.failures:
        print(f"\nfailures ({len(result.failures)}), first 10:")
        for failure in result.failures[:10]:
            print(f"  {failure.filename}: {failure.error}")
    return 1 if result.n_processed == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
