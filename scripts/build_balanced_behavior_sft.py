from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.post_rag_behavior import build_balanced_behavior_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the balanced post-RAG behavior SFT candidate package.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "kaggle_packages" / "2k_curated_behavior_sft")
    parser.add_argument("--target-rows", type=int, default=2000)
    parser.add_argument("--cleaned-sft-dir", type=Path, default=ROOT / "kaggle_packages" / "sft_cleaned_candidate")
    parser.add_argument("--retrieval-cards", type=Path, default=ROOT / "data" / "processed" / "retrieval_cards" / "retrieval_cards.jsonl")
    args = parser.parse_args()
    manifest = build_balanced_behavior_sft(args.out_dir, args.cleaned_sft_dir, args.retrieval_cards, args.target_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
