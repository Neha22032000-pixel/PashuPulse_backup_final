from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.offline_library import build_offline_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build the PashuPulse offline source library.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "sources" / "offline_library_v1"))
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true", help="Build manifests without downloading; useful for dry checks.")
    args = parser.parse_args()
    manifest = build_offline_library(Path(args.out_dir), max_sources=args.max_sources, skip_download=args.skip_download)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
