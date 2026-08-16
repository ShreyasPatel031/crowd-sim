#!/usr/bin/env python3
"""Submit / poll a Vertex supervised-tuning job for OPeRA next-action prediction.

    python scripts/submit_vertex_sft.py submit --epochs 1
    python scripts/submit_vertex_sft.py status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/vertex/tuning_job.json"

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
LOCATION = os.environ.get("VERTEX_TUNING_LOCATION", "us-central1")
BUCKET = os.environ.get("OPERA_SFT_BUCKET", "gs://opera-sft-project-amer-scs-sandbox")
BASE_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

ADAPTER_SIZES = {
    1: "ADAPTER_SIZE_ONE",
    2: "ADAPTER_SIZE_TWO",
    4: "ADAPTER_SIZE_FOUR",
    8: "ADAPTER_SIZE_EIGHT",
    16: "ADAPTER_SIZE_SIXTEEN",
}


def _token() -> str:
    return subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _call(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        return json.load(urllib.request.urlopen(request))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{exc.code} {exc.reason}\n{exc.read().decode()[:2000]}") from exc


def submit(args: argparse.Namespace) -> None:
    body = {
        "baseModel": BASE_MODEL,
        "tunedModelDisplayName": args.name,
        "supervisedTuningSpec": {
            "trainingDatasetUri": f"{BUCKET}/data/train_sft.jsonl",
            "validationDatasetUri": f"{BUCKET}/data/val_sft.jsonl",
            "hyperParameters": {
                "epochCount": str(args.epochs),
                "adapterSize": ADAPTER_SIZES[args.adapter_size],
                "learningRateMultiplier": args.lr_multiplier,
            },
        },
    }
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/tuningJobs"
    print(json.dumps(body, indent=2))
    job = _call(url, body)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(job, indent=2))
    print(f"\nsubmitted: {job['name']}")
    print(f"state:     {job.get('state')}")
    print(f"saved to   {STATE}")


def status(args: argparse.Namespace) -> None:
    name = args.job or json.loads(STATE.read_text())["name"]
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/{name}"
    while True:
        job = _call(url)
        state = job.get("state", "?")
        started = job.get("startTime", "")
        print(f"{time.strftime('%H:%M:%S')}  {state}  started={started}")
        if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}:
            if state == "JOB_STATE_SUCCEEDED":
                tuned = job.get("tunedModel", {})
                print(f"\n  model:    {tuned.get('model')}")
                print(f"  endpoint: {tuned.get('endpoint')}")
            if job.get("error"):
                print(f"\n  error: {json.dumps(job['error'], indent=2)}")
            STATE.write_text(json.dumps(job, indent=2))
            break
        if not args.watch:
            break
        time.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--epochs", type=int, default=1)
    p_submit.add_argument("--adapter-size", type=int, default=4, choices=sorted(ADAPTER_SIZES))
    p_submit.add_argument("--lr-multiplier", type=float, default=1.0)
    p_submit.add_argument("--name", default="opera-flash-sft-e1")
    p_submit.set_defaults(func=submit)

    p_status = sub.add_parser("status")
    p_status.add_argument("--job", default=None)
    p_status.add_argument("--watch", action="store_true")
    p_status.add_argument("--interval", type=int, default=120)
    p_status.set_defaults(func=status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
