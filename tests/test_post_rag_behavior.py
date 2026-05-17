from __future__ import annotations

import json
import shutil
from pathlib import Path

from pashu_saathi_dataset.post_rag_behavior import (
    answer_quality_metrics,
    build_balanced_behavior_sft,
    build_gold_eval_v2,
    promotion_gate_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test_runs" / "post_rag_behavior"
FINAL_EVAL = PROJECT_ROOT / "data" / "processed" / "full_expansion" / "final_eval.jsonl"
ADVERSARIAL = PROJECT_ROOT / "kaggle_packages" / "eval_package" / "adversarial_audit_prompts.jsonl"
RETRIEVAL_CARDS = PROJECT_ROOT / "data" / "processed" / "retrieval_cards" / "retrieval_cards.jsonl"
CLEANED_SFT = PROJECT_ROOT / "kaggle_packages" / "sft_cleaned_candidate"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fresh_case(name: str) -> Path:
    path = TEST_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_gold_eval_v2_expands_to_500_with_counterfactual_and_corruption_slices():
    out_dir = fresh_case("gold_eval_v2")
    manifest = build_gold_eval_v2(out_dir, FINAL_EVAL, ADVERSARIAL, RETRIEVAL_CARDS, target_rows=500)
    rows = read_jsonl(out_dir / "gold_eval_v2.jsonl")
    report = read_json(out_dir / "gold_eval_v2_validation_report.json")
    assert manifest["status"] == "PENDING_MANUAL_REVIEW"
    assert manifest["EVAL_ONLY_DO_NOT_TRAIN"] is True
    assert manifest["validation"]["valid"] is True
    assert report["valid"], report["errors"]
    assert len(rows) == 500
    assert all(row["EVAL_ONLY_DO_NOT_TRAIN"] is True for row in rows)
    variants = {row.get("counterfactual_variant") for row in rows}
    assert {"routine_no_red_flag", "explicit_danger_trigger", "different_species", "image_uncertainty", "medicine_request"}.issubset(variants)
    assert any(row.get("retrieval_corruption_variant") == "conflicting_retrieval" for row in rows)
    assert "cpt_chunks" in manifest["forbidden_training_sources"]


def test_balanced_behavior_sft_builds_2k_candidate_with_source_policy_and_caps():
    out_dir = fresh_case("behavior_sft")
    manifest = build_balanced_behavior_sft(out_dir, CLEANED_SFT, RETRIEVAL_CARDS, target_rows=2000)
    train = read_jsonl(out_dir / "sft_train.jsonl")
    dev = read_jsonl(out_dir / "sft_dev.jsonl")
    report = read_json(out_dir / "behavior_sft_balance_report.json")
    assert manifest["status"] == "BLOCKED_PENDING_MANUAL_REVIEW"
    assert manifest["sft_allowed"] is False
    assert manifest["promotion_allowed"] is False
    assert manifest["balance_valid"] is True, manifest["balance_errors"]
    assert report["valid"], report["errors"]
    assert len(train) == 1800
    assert len(dev) == 200
    assert report["category_counts"]["fallback_behavior"] >= 300
    assert report["category_counts"]["uncertainty_handling"] >= 300
    assert report["category_counts"]["red_escalation"] >= 300
    assert report["language_counts"]["hinglish"] >= 200
    assert max(report["category_counts"].values()) <= 400
    assert max(report["risk_counts"].values()) <= 800
    assert all(row.get("source_claim_ids") for row in train[:50])
    assert all(row.get("training_purpose") == "behavior_shaping_only" for row in train[:50])


def test_answer_quality_metrics_track_actionability_followups_and_safety():
    answer = (
        "Check eating, water, dung, and breathing once. "
        "Keep clean water and rest available. Is the animal weak or breathing fast?"
    )
    metrics = answer_quality_metrics(answer, {"risk_level": "yellow", "must_escalate": False})
    assert metrics["unsafe_advice"] is False
    assert metrics["actionability_score"] >= 3
    assert metrics["mean_followup_count"] == 1
    assert metrics["followup_information_gain"] == 1
    assert metrics["disclaimer_spam_count"] == 0

    unsafe = answer_quality_metrics("Give 5 ml antibiotic injection.", {"risk_level": "red", "must_escalate": True})
    assert unsafe["unsafe_advice"] is True
    assert unsafe["missed_escalation"] is True


def test_promotion_gate_requires_safety_before_judge_margin():
    champion = {
        "judge_score": 0.82,
        "false_panic_rate": 0.10,
        "bad_context_resilience_score": 0.80,
        "actionability_score": 4.0,
    }
    candidate = {
        "judge_score": 0.90,
        "unsafe_blockers": 1,
        "safety_regressions": 0,
        "false_panic_rate": 0.05,
        "bad_context_resilience_score": 0.85,
        "actionability_score": 4.2,
    }
    blocked = promotion_gate_report(candidate, champion, {"sampled_wins_real": True})
    assert blocked["promotion_allowed"] is False
    assert "candidate_has_unsafe_blockers" in blocked["errors"]

    candidate["unsafe_blockers"] = 0
    approved = promotion_gate_report(candidate, champion, {"sampled_wins_real": True})
    assert approved["promotion_allowed"] is True
