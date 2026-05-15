from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command_output(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pashu Saathi Kaggle launch preflight.")
    parser.add_argument("--package-dir", type=Path, default=ROOT / "kaggle_packages" / "sft_smoke_package")
    parser.add_argument("--require-kaggle-config", action="store_true")
    args = parser.parse_args()

    kaggle_config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(ROOT.parent / ".kaggle_2")))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_dir": str(args.package_dir),
        "package_exists": args.package_dir.exists(),
        "kaggle_config_dir": str(kaggle_config_dir),
        "kaggle_config_exists": (kaggle_config_dir / "kaggle.json").exists(),
        "kaggle_cli": shutil.which("kaggle") or "",
        "nvidia_smi": command_output(["nvidia-smi"]) if shutil.which("nvidia-smi") else "nvidia-smi not found",
        "errors": [],
        "warnings": [],
    }
    if args.require_kaggle_config and not report["kaggle_config_exists"]:
        report["errors"].append("isolated Kaggle config not found")
    if args.package_dir.exists():
        forbidden = sorted(path.name for path in args.package_dir.glob("*") if path.name in {"final_eval.jsonl", "eval_rubric.jsonl", "adversarial_audit_prompts.jsonl"})
        if forbidden:
            report["errors"].append(f"training package contains eval-only files: {forbidden}")
    else:
        report["warnings"].append("package directory does not exist yet")
    report["pass"] = not report["errors"]
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
