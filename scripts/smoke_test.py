#!/usr/bin/env python3
"""End-to-end smoke test against a running gateway.

For each pilot system: submit a fixed sentence + reference clip, poll to a
terminal state, and assert a non-empty WAV plus complete metadata (including
the license block) come back. Run individually and, with --concurrent, all at
once to confirm the three GPUs work in parallel.

Usage:
  python3 scripts/smoke_test.py                  # sequential, all systems
  python3 scripts/smoke_test.py --concurrent     # fire all at once
  python3 scripts/smoke_test.py --system cosyvoice2
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

GATEWAY = "http://localhost:8000"
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data_fixtures" / "smoke"
REF_WAV = FIXTURE_DIR / "reference_female_en.wav"
REF_TXT = FIXTURE_DIR / "reference_female_en.txt"
POLL_TIMEOUT_SEC = 600
TEXT = "The quick brown fox jumps over the lazy dog near the riverbank at dawn."


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        GATEWAY + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(GATEWAY + path, timeout=30) as resp:
        return json.loads(resp.read())


def run_one(system: str) -> tuple[str, bool, str]:
    payload = {
        "tts_system": system,
        "text": TEXT,
        "reference_audio_url": f"file:///data/fixtures/{REF_WAV.name}",
        "requested_by": "smoke-test",
    }
    if REF_TXT.exists():
        payload["reference_text"] = REF_TXT.read_text().strip()

    try:
        accepted = _post("/v1/synthesize", payload)
    except Exception as exc:
        return system, False, f"submit failed: {exc}"

    job_id = accepted["job_id"]
    deadline = time.monotonic() + POLL_TIMEOUT_SEC
    status = "queued"
    while time.monotonic() < deadline:
        job = _get(f"/v1/jobs/{job_id}")
        status = job["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(3)

    if status != "succeeded":
        err = job.get("error") if "job" in dir() else "timeout"
        return system, False, f"status={status}: {err}"

    result = job.get("result") or {}
    meta = result.get("metadata") or {}
    if result.get("duration_sec", 0) <= 0.5:
        return system, False, f"suspicious duration {result.get('duration_sec')}"
    if not meta.get("license", {}).get("code_license"):
        return system, False, "metadata missing license block"

    # Fetch the audio bytes and check the file is non-empty.
    audio_url = result["audio_url"]
    with urllib.request.urlopen(GATEWAY + audio_url, timeout=30) as resp:
        nbytes = len(resp.read())
    if nbytes < 1000:
        return system, False, f"audio too small ({nbytes} bytes)"

    return system, True, (
        f"{result['duration_sec']}s @ {result['sample_rate']}Hz, "
        f"{nbytes} bytes, latency {meta.get('generation', {}).get('latency_sec')}s"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrent", action="store_true")
    ap.add_argument("--system", help="run a single system only")
    args = ap.parse_args()

    systems = [args.system] if args.system else list(_get("/v1/systems")["systems"].keys())

    print(f"Testing systems: {systems}")
    results = []
    if args.concurrent and not args.system:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(systems)) as ex:
            results = list(ex.map(run_one, systems))
    else:
        results = [run_one(s) for s in systems]

    print("\n--- Results ---")
    ok = True
    for system, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {system}: {detail}")
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
