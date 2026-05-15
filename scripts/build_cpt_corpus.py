from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.cpt_corpus import build_cpt_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the broad PashuPulse CPT/DAPT corpus candidate.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "cpt_corpus"))
    args = parser.parse_args()
    manifest = build_cpt_corpus(Path(args.out_dir))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

