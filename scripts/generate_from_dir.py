#!/usr/bin/env python3
"""Generate synthetic speech for every reference clip in a fixtures directory.

For each .wav under a subdirectory of data_fixtures/smoke/ (which the workers
see mounted at /data/fixtures/), submit a synthesis job that clones the clip's
voice. Target text comes from a metadata CSV (filename -> transcript) when
given, otherwise a fixed --text is used for all clips.

The directory MUST live under data_fixtures/smoke/ because that is the only
host path bind-mounted into the workers (as /data/fixtures). Put/copy your
reference clips there first.

--reference-text-from-metadata: required for systems whose registry entry
sets requires_reference_text (currently cosyvoice2, maskgct, fish-speech) —
the gateway returns 400 Bad Request without it. Since this script clones each
clip saying its own transcript, the metadata transcript doubles as both the
target text and the reference_text, so pass this flag rather than supplying a
separate reference-metadata file (contrast generate_from_metadata.py, where
reference and target text come from different sources). Optional but
recommended for xtts/f5-tts; not needed for tortoise-tts/metavoice-1b/
styletts2. See the "Reference-audio length constraints" table in
docs/resource-requirements.md, or `GET /v1/systems` (requires_reference_text)
for the live/current answer for any system.

Examples:
  # Clone each Trump clip saying its own transcript, via Tortoise:
  python3 scripts/generate_from_dir.py --dir trump --system tortoise-tts \
      --metadata /data/Famous_Figures/demo_data/Donald_Trump_metadata.csv

  # Fixed text for every clip, first 3 only:
  python3 scripts/generate_from_dir.py --dir trump --system tortoise-tts \
      --text "This is a synthetic voice sample." --limit 3

  # CosyVoice2/MaskGCT/Fish-Speech require reference_text: pass
  # --reference-text-from-metadata so the transcript is sent as both the
  # reference transcript and the target text:
  python3 scripts/generate_from_dir.py --dir trump --system cosyvoice2 \
      --metadata /data/Famous_Figures/demo_data/Donald_Trump_metadata.csv \
      --reference-text-from-metadata --limit 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import _params  # scripts/_params.py (same directory)

GATEWAY = "http://localhost:8000"
FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "data_fixtures" / "smoke"
CONTAINER_MOUNT = "/data/fixtures"  # ./data_fixtures/smoke -> /data/fixtures
POLL_TIMEOUT_SEC = 600


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        GATEWAY + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get(path: str) -> dict:
    with urllib.request.urlopen(GATEWAY + path, timeout=30) as resp:
        return json.loads(resp.read())


def load_transcripts(csv_path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # accept either (filename, Transcript) or first two columns
        for row in reader:
            keys = list(row.keys())
            fn = row.get("filename", row[keys[0]])
            txt = row.get("Transcript", row.get("transcript", row[keys[1]]))
            m[fn.strip()] = (txt or "").strip()
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="subdirectory under data_fixtures/smoke/ holding the .wav clips")
    ap.add_argument("--system", required=True, help="tts_system id (see /v1/systems)")
    ap.add_argument("--metadata", help="CSV mapping clip filename -> target transcript")
    ap.add_argument("--text", help="fixed target text for all clips (if no --metadata)")
    ap.add_argument("--reference-text-from-metadata", action="store_true",
                    help="also send the transcript as reference_text "
                         "(required by systems like cosyvoice2)")
    ap.add_argument("--limit", type=int, help="process at most N clips")
    ap.add_argument("--requested-by", default="generate_from_dir")
    # Per-system generation params come from a client-side config keyed by
    # --system (config/client_params.yaml), not per-system CLI flags. Use
    # --param KEY=VALUE for a one-off override of a single key.
    ap.add_argument("--params-config", default=str(_params.DEFAULT_CONFIG),
                    help="YAML of per-system client params (default: config/client_params.yaml)")
    ap.add_argument("--param", action="append", metavar="KEY=VALUE",
                    help="one-off param override, repeatable (e.g. --param guidance_scale=1.5)")
    args = ap.parse_args()

    knob_params = _params.resolve_params(Path(args.params_config), args.system, args.param)

    local_dir = FIXTURES_ROOT / args.dir
    if not local_dir.is_dir():
        print(f"ERROR: {local_dir} does not exist. Copy your clips under "
              f"data_fixtures/smoke/{args.dir}/ first.", file=sys.stderr)
        return 2

    clips = sorted(p for p in local_dir.iterdir() if p.suffix.lower() == ".wav")
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print(f"ERROR: no .wav files in {local_dir}", file=sys.stderr)
        return 2

    transcripts = load_transcripts(Path(args.metadata)) if args.metadata else {}
    if not transcripts and not args.text:
        print("ERROR: provide --metadata (per-clip transcripts) or --text (fixed).",
              file=sys.stderr)
        return 2

    results = []
    for clip in clips:
        text = transcripts.get(clip.name, args.text)
        if not text:
            results.append((clip.name, False, "no transcript in metadata and no --text"))
            continue
        payload = {
            "tts_system": args.system,
            "text": text,
            "reference_audio_url": f"file://{CONTAINER_MOUNT}/{args.dir}/{clip.name}",
            "params": knob_params,
            "requested_by": args.requested_by,
        }
        if args.reference_text_from_metadata:
            payload["reference_text"] = text

        try:
            job_id = post("/v1/synthesize", payload)["job_id"]
        except Exception as exc:
            results.append((clip.name, False, f"submit failed: {exc}"))
            continue

        print(f"[{clip.name}] job={job_id}  text={text[:60]!r}...")
        deadline = time.monotonic() + POLL_TIMEOUT_SEC
        job = {"status": "queued"}
        while time.monotonic() < deadline:
            job = get(f"/v1/jobs/{job_id}")
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(5)

        if job["status"] == "succeeded":
            r = job["result"]
            results.append((clip.name, True,
                            f"{r['duration_sec']}s @ {r['sample_rate']}Hz -> {r['audio_url']}"))
        else:
            results.append((clip.name, False, f"status={job['status']}: {job.get('error')}"))

    print("\n=== Results ===")
    ok = True
    for name, passed, detail in results:
        print(f"[{'OK ' if passed else 'ERR'}] {name}: {detail}")
        ok = ok and passed
    print(f"\n{sum(1 for _, p, _ in results if p)}/{len(results)} succeeded")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
