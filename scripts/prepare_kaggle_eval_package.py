from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.eval_readiness import run_eval_readiness


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_required(src_dir: Path, out_dir: Path, names: list[str]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name in names:
        src = src_dir / name
        if not src.exists():
            raise SystemExit(f"BLOCKED: missing required eval artifact {src}")
        dst = out_dir / name
        shutil.copy2(src, dst)
        checksums[f"{name}_sha256"] = sha256_file(dst)
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare eval-only Kaggle input package for Pashu Saathi.")
    parser.add_argument("--workspace-dir", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "kaggle_packages" / "eval_package")
    args = parser.parse_args()

    decision = run_eval_readiness(args.workspace_dir)
    eval_dir = args.workspace_dir / "data" / "processed" / "pilot_eval_package"
    if decision.get("rubric_pass") is not True or decision.get("contamination_guard_pass") is not True:
        raise SystemExit("BLOCKED: eval rubric or contamination guard failed.")

    checksums = copy_required(
        eval_dir,
        args.out_dir,
        [
            "eval_rubric.jsonl",
            "eval_rubric_validation_report.json",
            "contamination_guard_report.json",
            "eval_manifest.json",
            "adversarial_audit_prompts.jsonl",
        ],
    )
    forbidden = {"train.jsonl", "dev.jsonl", "sft_train.jsonl", "sft_dev.jsonl"}
    leaked = sorted(path.name for path in args.out_dir.glob("*.jsonl") if path.name in forbidden)
    if leaked:
        raise SystemExit(f"BLOCKED: eval package contains train/dev files: {leaked}")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_type": "eval_only",
        "status": "EVAL_ONLY_DO_NOT_TRAIN",
        "source_eval_dir": str(eval_dir),
        "checksums": checksums,
        "readiness_decision": decision,
        "mock_allowed": False,
    }
    (args.out_dir / "kaggle_eval_package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Pashu Saathi Eval Package",
                "id": "nehak76044/pashu-saathi-eval-package",
                "licenses": [{"name": "CC0-1.0"}],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
