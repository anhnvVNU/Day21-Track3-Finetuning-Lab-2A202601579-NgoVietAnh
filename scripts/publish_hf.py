#!/usr/bin/env python3
"""Publish the completed LoRA adapter and evidence to Hugging Face Hub.

The access token is read from HF_TOKEN (including from the ignored .env through
labkit). It is never accepted as a command-line argument, which keeps it out of
shell history and process listings.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import labkit  # noqa: F401  # loads the nearest .env without overriding real env vars

DEFAULT_REPO = "mob2711/lab21-2A202601579-qwen35-triage-vi"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repo (omit this for rubric bonus B5).",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is missing. Put it in the ignored .env or Colab Secrets.", file=sys.stderr)
        return 2

    adapter = ROOT / "adapters" / "correct"
    results = ROOT / "results"
    report = ROOT / "submission" / "REPORT.md"
    required = [
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        results / "verdict.json",
        results / "runs.csv",
        report,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print("Cannot publish; missing: " + ", ".join(missing), file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(folder_path=adapter, repo_id=args.repo_id, repo_type="model")
    api.upload_folder(
        folder_path=results,
        path_in_repo="results",
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=[".gitkeep"],
    )
    api.upload_file(
        path_or_fileobj=report,
        path_in_repo="REPORT.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Published: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
