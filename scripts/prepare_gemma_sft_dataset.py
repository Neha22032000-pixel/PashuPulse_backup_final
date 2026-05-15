from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.validators import validate_dataset


DATASET_NAME = "pashu-saathi-rural-livestock-sft-approved"


def assert_approved_for_sft(dataset_dir: Path) -> dict:
    result = validate_dataset(dataset_dir, update_report=False, require_approved_state="APPROVED_FOR_SFT", phase="sft_candidate")
    if not result["valid"]:
        raise SystemExit(f"BLOCKED: dataset is not approved for SFT export: {result['errors']}")
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sft_allowed") is not True:
        raise SystemExit("BLOCKED: manifest sft_allowed is not true.")
    if manifest.get("row_counts") == {"sft_train": 0, "sft_dev": 0, "final_eval": 0}:
        raise SystemExit("BLOCKED: seed-only bank has zero approved expanded rows.")
    return manifest


def build_export(dataset_dir: Path, out_dir: Path) -> dict:
    manifest = assert_approved_for_sft(dataset_dir)
    raise SystemExit(
        "BLOCKED: export writer is intentionally disabled in the seed-rebuild pass. "
        "Implement expansion/export only after APPROVED_FOR_EXPANSION and APPROVED_FOR_SFT policies exist. "
        f"Current manifest: {manifest.get('dataset_name')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Pashu Saathi Gemma SFT export bundle.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "processed" / "starter"))
    parser.add_argument("--out-dir", default=str(ROOT / "exports" / DATASET_NAME))
    parser.add_argument(
        "--allow-unapproved-smoke",
        action="store_true",
        help="Deprecated: ignored. Seed-rebuild artifacts cannot be exported for SFT.",
    )
    args = parser.parse_args()
    build_export(Path(args.dataset_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
