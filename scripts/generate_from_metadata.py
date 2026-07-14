#!/usr/bin/env python3
"""Generate synthetic speech driven by a metadata file, with the reference voice
sampled from a pool directory (reference and text are decoupled).

Iterates rows of a metadata CSV (filename, Transcript). For each row the
Transcript is the target text; the reference/voice clip is chosen from
--reference-dir (a subdir of data_fixtures/smoke/, mounted into the workers as
/data/fixtures). Outputs are named after the text row (via the gateway `label`),
so you can tell which line each file speaks.

Reference selection:
  default            one random reference clip for the whole batch (seeded)
  --random-per-row   a fresh random reference per row
  --reference NAME   a specific reference clip for all rows

Row selection:
  --filenames a.wav,b.wav   only these rows (matched on the metadata filename col)
  (omitted)                 every row in the metadata

--reference-metadata: required for systems whose registry entry sets
requires_reference_text (currently cosyvoice2, maskgct, fish-speech) — the
gateway returns 400 Bad Request without it. Optional but recommended for
xtts/f5-tts (f5-tts otherwise auto-transcribes via Whisper, and its output
pacing is derived from the ref audio/text pair, so an accurate transcript
matters more there than elsewhere). Not needed for tortoise-tts/metavoice-1b/
styletts2. See the "Reference-audio length constraints" table in
docs/resource-requirements.md, or `GET /v1/systems` (requires_reference_text)
for the live/current answer for any system.

Examples:
  # Short-term: 5 specific rows' texts, random 40s reference from trump_long, MetaVoice
  python3 scripts/generate_from_metadata.py \
      --reference-dir trump_long --system metavoice-1b \
      --metadata /data/Famous_Figures/demo_data/Donald_Trump_metadata.csv \
      --filenames Donald_Trump_104.wav,Donald_Trump_185.wav,Donald_Trump_335.wav,Donald_Trump_671.wav,Donald_Trump_766.wav

  # Long-term: every row in a metadata file
  python3 scripts/generate_from_metadata.py \
      --reference-dir trump_long --system metavoice-1b --metadata my_lines.csv

  # with limit: generate for n rows in the metadata
  python3 scripts/generate_from_metadata.py  \
      --reference-dir trump_long --system metavoice-1b \
      --metadata /data/Famous_Figures/demo_data/Donald_Trump_metadata.csv --limit 5

  # Tortoise: reference clip can be any length, no reference_text needed
  python3 scripts/generate_from_metadata.py \
      --reference-dir trump --system tortoise-tts \
      --metadata /data/Famous_Figures/demo_data/Donald_Trump_metadata.csv --limit 5

  # CosyVoice2: reference must be <=30s AND needs the reference clip's own
  # transcript (reference_text), so use a short-clip pool and pass
  # --reference-metadata mapping each reference clip to its transcript.
  python3 scripts/generate_from_metadata.py \
      --reference-dir trump --system cosyvoice2 \
      --metadata /data/Famous_Figures/demo_data/Donald_Trump_metadata.csv \
      --reference-metadata /data/Famous_Figures/demo_data/Donald_Trump_metadata.csv --limit 5
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import _params  # scripts/_params.py (same directory)

GATEWAY = "http://localhost:8000"
FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "data_fixtures" / "smoke"
CONTAINER_MOUNT = "/data/fixtures"
POLL_TIMEOUT_SEC = 600


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        GATEWAY + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get(path: str) -> dict:
    with urllib.request.urlopen(GATEWAY + path, timeout=30) as resp:
        return json.loads(resp.read())


def load_rows(csv_path: Path) -> list[tuple[str, str]]:
    """Return [(filename, transcript), ...] preserving CSV order."""
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = list(row.keys())
            fn = row.get("filename", row[keys[0]]).strip()
            txt = (row.get("Transcript", row.get("transcript", row[keys[1]])) or "").strip()
            out.append((fn, txt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", required=True,
                    help="subdir under data_fixtures/smoke/ to sample reference clips from")
    ap.add_argument("--metadata", required=True, help="CSV with filename,Transcript rows")
    ap.add_argument("--system", required=True, help="tts_system id (see /v1/systems)")
    ap.add_argument("--filenames", help="comma-separated subset of metadata filenames to generate")
    ap.add_argument("--reference", help="use this specific reference clip for all rows")
    ap.add_argument("--random-per-row", action="store_true",
                    help="pick a fresh random reference per row (default: one for the batch)")
    ap.add_argument("--reference-metadata",
                    help="CSV mapping reference-clip filename -> transcript "
                         "(needed as reference_text for systems like cosyvoice2)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reference sampling")
    ap.add_argument("--limit", type=int, help="process at most N rows")
    ap.add_argument("--requested-by", default="generate_from_metadata")
    # Per-system generation params come from a client-side config keyed by
    # --system (config/client_params.yaml), not per-system CLI flags. Use
    # --param KEY=VALUE for a one-off override of a single key.
    ap.add_argument("--params-config", default=str(_params.DEFAULT_CONFIG),
                    help="YAML of per-system client params (default: config/client_params.yaml)")
    ap.add_argument("--param", action="append", metavar="KEY=VALUE",
                    help="one-off param override, repeatable (e.g. --param guidance_scale=1.5)")
    args = ap.parse_args()

    knob_params = _params.resolve_params(Path(args.params_config), args.system, args.param)

    ref_dir = FIXTURES_ROOT / args.reference_dir
    if not ref_dir.is_dir():
        print(f"ERROR: {ref_dir} does not exist. Put reference clips under "
              f"data_fixtures/smoke/{args.reference_dir}/ first.", file=sys.stderr)
        return 2
    pool = sorted(p.name for p in ref_dir.iterdir() if p.suffix.lower() == ".wav")
    if not pool:
        print(f"ERROR: no .wav reference clips in {ref_dir}", file=sys.stderr)
        return 2

    if args.reference and args.reference not in pool:
        print(f"ERROR: --reference {args.reference} not in {ref_dir}", file=sys.stderr)
        return 2

    rows = load_rows(Path(args.metadata))
    if args.filenames:
        wanted = [s.strip() for s in args.filenames.split(",") if s.strip()]
        by_name = {fn: txt for fn, txt in rows}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            print(f"ERROR: these --filenames are not in the metadata: {missing}", file=sys.stderr)
            return 2
        rows = [(w, by_name[w]) for w in wanted]
    if args.limit:
        rows = rows[: args.limit]

    ref_transcripts = {}
    if args.reference_metadata:
        ref_transcripts = {fn: txt for fn, txt in load_rows(Path(args.reference_metadata))}

    rng = random.Random(args.seed)
    batch_ref = args.reference or (None if args.random_per_row else rng.choice(pool))
    if batch_ref and not args.random_per_row:
        print(f"Using reference for the whole batch: {batch_ref}")

    results = []
    for fn, text in rows:
        if not text:
            results.append((fn, False, "empty transcript in metadata"))
            continue
        ref = batch_ref if batch_ref else rng.choice(pool)
        label = Path(fn).stem  # name the output after the text row
        payload = {
            "tts_system": args.system,
            "text": text,
            "reference_audio_url": f"file://{CONTAINER_MOUNT}/{args.reference_dir}/{ref}",
            "label": label,
            "params": knob_params,
            "requested_by": args.requested_by,
        }
        if ref in ref_transcripts:
            payload["reference_text"] = ref_transcripts[ref]

        try:
            job_id = post("/v1/synthesize", payload)["job_id"]
        except Exception as exc:
            results.append((fn, False, f"submit failed: {exc}"))
            continue

        print(f"[{fn}] ref={ref} job={job_id}  text={text[:55]!r}...")
        deadline = time.monotonic() + POLL_TIMEOUT_SEC
        job = {"status": "queued"}
        while time.monotonic() < deadline:
            job = get(f"/v1/jobs/{job_id}")
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(5)

        if job["status"] == "succeeded":
            r = job["result"]
            results.append((fn, True, f"ref={ref} -> {r['duration_sec']}s {r['audio_url']}"))
        else:
            results.append((fn, False, f"ref={ref} status={job['status']}: {job.get('error')}"))

    print("\n=== Results ===")
    ok = True
    for fn, passed, detail in results:
        print(f"[{'OK ' if passed else 'ERR'}] {fn}: {detail}")
        ok = ok and passed
    print(f"\n{sum(1 for _, p, _ in results if p)}/{len(results)} succeeded")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
