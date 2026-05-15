from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.generator import build_dataset, build_full_expansion_dataset, build_pilot_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Pashu Saathi seed-card or expansion dataset.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "starter"))
    parser.add_argument("--phase", choices=["seed_bank", "expansion_candidate", "full_expansion_candidate"], default="seed_bank")
    args = parser.parse_args()
    if args.phase == "full_expansion_candidate":
        manifest = build_full_expansion_dataset(Path(args.out_dir))
    elif args.phase == "expansion_candidate":
        manifest = build_pilot_dataset(Path(args.out_dir))
    else:
        manifest = build_dataset(Path(args.out_dir))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
