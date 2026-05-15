from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.cpt_research import build_cpt_research_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PashuPulse CPT/DAPT research-only artifacts.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "cpt_research"))
    parser.add_argument("--library-dir", default=str(ROOT / "data" / "sources" / "offline_library_v1"))
    args = parser.parse_args()
    manifest = build_cpt_research_artifacts(Path(args.out_dir), Path(args.library_dir))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
