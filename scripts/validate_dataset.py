from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.validators import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Pashu Saathi dataset gates.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--phase", choices=["seed_bank", "expansion_candidate", "full_expansion_candidate", "sft_candidate"], default="seed_bank")
    parser.add_argument("--no-update-report", action="store_true")
    args = parser.parse_args()
    result = validate_dataset(Path(args.dataset_dir), update_report=not args.no_update_report, phase=args.phase)
    print(json.dumps(result["report"], indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
