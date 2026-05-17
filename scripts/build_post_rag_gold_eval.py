from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pashu_saathi_dataset.post_rag_behavior import build_gold_eval_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the post-RAG gold eval v2 candidate package.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "gold_eval_v2")
    parser.add_argument("--target-rows", type=int, default=500)
    parser.add_argument("--final-eval", type=Path, default=ROOT / "data" / "processed" / "full_expansion" / "final_eval.jsonl")
    parser.add_argument("--adversarial", type=Path, default=ROOT / "kaggle_packages" / "eval_package" / "adversarial_audit_prompts.jsonl")
    parser.add_argument("--retrieval-cards", type=Path, default=ROOT / "data" / "processed" / "retrieval_cards" / "retrieval_cards.jsonl")
    args = parser.parse_args()
    manifest = build_gold_eval_v2(args.out_dir, args.final_eval, args.adversarial, args.retrieval_cards, args.target_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
