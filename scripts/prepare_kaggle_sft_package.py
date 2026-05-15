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

from pashu_saathi_dataset.validators import validate_dataset


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


FORBIDDEN_PACKAGE_FIELD_PREFIXES = ("review", "eval", "prediction", "rubric", "adversarial", "label")


def sanitize_training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "EVAL_ONLY_DO_NOT_TRAIN"
        and not any(key.lower().startswith(prefix) for prefix in FORBIDDEN_PACKAGE_FIELD_PREFIXES)
    }


def assert_no_eval_leakage(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> None:
    leaked = []
    for split, rows in [("train", train), ("dev", dev)]:
        for row in rows:
            if row.get("EVAL_ONLY_DO_NOT_TRAIN") is True:
                leaked.append(f"{split}:{row.get('row_id')}:eval_only_flag")
            if row.get("parent_seed_split") == "final_eval_seed":
                leaked.append(f"{split}:{row.get('row_id')}:final_eval_lineage")
            if row.get("split") == "final_eval":
                leaked.append(f"{split}:{row.get('row_id')}:final_eval_split")
            bad_fields = [key for key in row if any(key.lower().startswith(prefix) for prefix in FORBIDDEN_PACKAGE_FIELD_PREFIXES)]
            if bad_fields:
                leaked.append(f"{split}:{row.get('row_id')}:forbidden_fields:{bad_fields[:5]}")
    if leaked:
        raise SystemExit(f"BLOCKED: SFT package contains eval leakage: {leaked[:10]}")


def require_ready_for_sft_planning(workspace_dir: Path) -> dict[str, Any]:
    decision_path = workspace_dir / "data" / "processed" / "pilot_eval_package" / "sft_planning_readiness_decision.json"
    if not decision_path.exists():
        raise SystemExit(f"BLOCKED: missing SFT planning readiness decision: {decision_path}")
    decision = read_json(decision_path)
    if decision.get("decision") != "ready_for_sft_planning":
        raise SystemExit(f"BLOCKED: readiness decision is not ready_for_sft_planning: {decision.get('decision')}")
    if decision.get("sft_allowed") is True:
        raise SystemExit("BLOCKED: readiness decision must not set sft_allowed true.")
    return decision


def build_config(mode: str, train_count: int, dev_count: int) -> dict[str, Any]:
    smoke = mode == "smoke"
    max_length = 512 if smoke else 768
    return {
        "seed": 76044,
        "package_mode": mode,
        "train_rows": train_count,
        "dev_rows": dev_count,
        "max_seq_length": max_length,
        "max_length": max_length,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4 if smoke else 8,
        "learning_rate": 2e-4 if smoke else 1.5e-4,
        "num_train_epochs": 1 if smoke else 2,
        "max_steps": 20 if smoke else 0,
        "eval_steps": 5 if smoke else 25,
        "save_steps": 10 if smoke else 25,
        "save_total_limit": 2,
        "lora_r": 8 if smoke else 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": "language_model_self_attention_only",
        "gradient_checkpointing": "unsloth",
        "load_in_4bit": True,
        "dtype": None,
        "fp16": True,
        "bf16": False,
        "dataset_num_proc": 1,
        "packing": False,
        "train_on_responses_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gated Kaggle SFT package for Pashu Saathi.")
    parser.add_argument("--workspace-dir", type=Path, default=ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "processed" / "full_expansion")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "kaggle_packages" / "sft_smoke_package")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--smoke-train-rows", type=int, default=24)
    parser.add_argument("--smoke-dev-rows", type=int, default=8)
    args = parser.parse_args()

    readiness = require_ready_for_sft_planning(args.workspace_dir)
    result = validate_dataset(args.dataset_dir, phase="full_expansion_candidate")
    if not result["valid"]:
        raise SystemExit(f"BLOCKED: full expansion validation failed: {result['errors']}")
    review_decision = read_json(args.dataset_dir / "expansion_review_decision.json")
    if review_decision.get("decision") != "approved_for_pilot_eval_only":
        raise SystemExit(f"BLOCKED: expansion review decision is not clean: {review_decision.get('decision')}")

    train = read_jsonl(args.dataset_dir / "sft_train.jsonl")
    dev = read_jsonl(args.dataset_dir / "sft_dev.jsonl")
    if args.mode == "smoke":
        train = train[: args.smoke_train_rows]
        dev = dev[: args.smoke_dev_rows]
    train = [sanitize_training_row(row) for row in train]
    dev = [sanitize_training_row(row) for row in dev]
    assert_no_eval_leakage(train, dev)

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    write_jsonl(args.out_dir / "sft_train.jsonl", train)
    write_jsonl(args.out_dir / "sft_dev.jsonl", dev)
    config = build_config(args.mode, len(train), len(dev))
    (args.out_dir / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = {
        "source_sft_train_sha256": sha256_file(args.dataset_dir / "sft_train.jsonl"),
        "source_sft_dev_sha256": sha256_file(args.dataset_dir / "sft_dev.jsonl"),
        "package_sft_train_sha256": sha256_file(args.out_dir / "sft_train.jsonl"),
        "package_sft_dev_sha256": sha256_file(args.out_dir / "sft_dev.jsonl"),
        "training_config_sha256": sha256_file(args.out_dir / "training_config.json"),
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_type": "sft_training",
        "package_mode": args.mode,
        "status": "SMOKE_ONLY_NOT_AUTHORITATIVE" if args.mode == "smoke" else "FULL_SFT_PACKAGE_GATED",
        "source_dataset_dir": str(args.dataset_dir),
        "row_counts": {"sft_train": len(train), "sft_dev": len(dev), "final_eval": 0},
        "checksums": checksums,
        "readiness_decision_summary": {
            "decision": readiness.get("decision"),
            "baseline_score_pass": readiness.get("baseline_score_pass"),
            "rubric_pass": readiness.get("rubric_pass"),
            "contamination_guard_pass": readiness.get("contamination_guard_pass"),
            "checksums": {
                "seed_cases_sha256": readiness.get("checksums", {}).get("seed_cases_sha256"),
                "source_claims_sha256": readiness.get("checksums", {}).get("source_claims_sha256"),
            },
        },
        "expansion_review_summary": {
            "decision": review_decision.get("decision"),
            "blocking_failure_count": review_decision.get("blocking_failure_count"),
            "reviewer_id": review_decision.get("reviewer_id"),
        },
        "final_eval_included": False,
        "adversarial_included": False,
        "sft_allowed": False,
        "full_sft_launch_blocked_until_param_review": args.mode == "full",
    }
    (args.out_dir / "sft_package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode == "full":
        review_request = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "pending_review",
            "required_decision": "approved_for_full_sft",
            "required_reviewer_roles": ["runtime", "dataset_safety", "hyperparameter"],
            "training_config": config,
            "training_config_sha256": checksums["training_config_sha256"],
            "sft_package_manifest_sha256": sha256_file(args.out_dir / "sft_package_manifest.json"),
            "package_checksums": {
                "package_sft_train_sha256": checksums["package_sft_train_sha256"],
                "package_sft_dev_sha256": checksums["package_sft_dev_sha256"],
            },
            "source_readiness_hashes": manifest["readiness_decision_summary"]["checksums"],
            "notes": "Full SFT launch is blocked until sft_param_review_decision.json approves these exact hashes.",
        }
        (args.out_dir / "sft_param_review_request.json").write_text(
            json.dumps(review_request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (args.out_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": f"Pashu Saathi SFT {args.mode.title()} Package",
                "id": f"nehak76044/pashu-saathi-sft-{args.mode}-package",
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
