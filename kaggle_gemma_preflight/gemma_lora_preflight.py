from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    data_dir = Path(os.environ.get("PASHU_SAATHI_DATA_DIR", "/kaggle/input/pashu-saathi-rural-livestock-sft-approved"))
    out_dir = Path(os.environ.get("PASHU_SAATHI_OUT_DIR", "/kaggle/working/pashu_saathi_preflight"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        failure = {"stage": "preflight", "error": "missing manifest.json", "data_dir": str(data_dir)}
        (out_dir / "preflight_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise SystemExit(json.dumps(failure))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project") != "pashu_saathi":
        raise SystemExit("wrong dataset manifest: expected project=pashu_saathi")
    report = {
        "stage": "preflight",
        "status": "ok",
        "quality_inference": "Preflight validates plumbing only, not model quality.",
        "resolved_paths": {"data_dir": str(data_dir), "out_dir": str(out_dir)},
        "required_files": ["train.jsonl", "dev.jsonl", "eval_prompts.jsonl", "eval_rubric.jsonl", "training_config.json"],
    }
    (out_dir / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
