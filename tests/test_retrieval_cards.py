from __future__ import annotations

import json
from pathlib import Path

from pashu_saathi_dataset.retrieval_cards import (
    build_retrieval_cards,
    compose_retrieval_context,
    rank_cards,
    render_retrieval_context,
    retrieve_cards,
    validate_generated_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test_runs"
SOURCE_CLAIMS = PROJECT_ROOT / "data" / "processed" / "full_expansion" / "source_claims.jsonl"
EVAL_RUBRIC = PROJECT_ROOT / "kaggle_packages" / "eval_package" / "eval_rubric.jsonl"
CPT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "cpt_corpus" / "cpt_corpus_manifest.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(out_name: str) -> Path:
    out_dir = TEST_ROOT / out_name
    build_retrieval_cards(out_dir, SOURCE_CLAIMS, EVAL_RUBRIC, CPT_MANIFEST)
    return out_dir


def test_retrieval_card_builds_required_artifacts():
    out_dir = build("retrieval_card_artifacts")
    for name in [
        "retrieval_cards.jsonl",
        "retrieval_card_embeddings.npz",
        "retrieval_semantic_manifest.json",
        "retrieval_ablation_report.json",
        "retrieval_context_quality_report.json",
        "retrieval_demo_cases.jsonl",
        "retrieval_eval_queries.jsonl",
        "retrieval_card_manifest.json",
        "retrieval_card_quality_report.json",
        "retrieval_card_safety_report.json",
    ]:
        assert (out_dir / name).exists(), name
    manifest = json.loads((out_dir / "retrieval_card_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "OFFLINE_RETRIEVAL_CARDS_READY"
    assert manifest["default_mode"] == "phone_safe"
    assert manifest["retrieval_method"] == "hybrid_safety_bm25_semantic"
    assert manifest["sft_allowed"] is False
    assert manifest["cpt_allowed"] is False
    assert manifest["rag_grounding_allowed"] is True


def test_cards_have_strong_provenance_and_are_not_training_data():
    out_dir = build("retrieval_card_provenance")
    cards = read_jsonl(out_dir / "retrieval_cards.jsonl")
    claim_ids = {row["claim_id"] for row in read_jsonl(SOURCE_CLAIMS)}
    assert len(cards) >= 45
    for card in cards:
        assert card["not_training_data"] is True
        assert "offline_retrieval" in card["allowed_for"]
        assert "sft_training" in card["not_allowed_for"]
        assert "cpt_training" in card["not_allowed_for"]
        assert card["content_hash"]
        assert card["source_claim_ids"]
        assert set(card["source_claim_ids"]).issubset(claim_ids)
        assert card["allowed_use"]
        assert card["banned_use"]
        assert card["facts"]
        assert card["safe_actions"]
        assert card["forbidden_actions"]


def test_cpt_is_not_used_as_factual_card_grounding():
    out_dir = build("retrieval_card_cpt_boundary")
    safety = json.loads((out_dir / "retrieval_card_safety_report.json").read_text(encoding="utf-8"))
    assert safety["valid"], safety["errors"]
    assert safety["policy"]["cpt_factual_grounding_allowed"] is False
    cards = read_jsonl(out_dir / "retrieval_cards.jsonl")
    assert all(not source_id.endswith("_cpt") for card in cards for source_id in card["source_ids"])


def test_retrieval_hits_safety_critical_hinglish_queries():
    out_dir = build("retrieval_card_hits")
    cards = read_jsonl(out_dir / "retrieval_cards.jsonl")
    cases = {
        "pet phool raha hai aur tel pilana hai": "bloat_red_flag_cow",
        "pait fulna hara chara ke baad": "bloat_red_flag_cow",
        "bachda atka hai helper bol raha pull hard karo": "calving_no_hard_pull",
        "doodh me khoon aur than dard": "udder_blood_clot_milk",
        "ghav me keede aur raakh lagane ko bol rahe": "minor_wound_maggot",
        "kai pashu feed ke baad beemar hain": "spoiled_feed_many_animals",
        "kutte ne kaata bachchon ko paas jane du": "dog_bite_saliva",
    }
    for query, expected in cases.items():
        retrieved = retrieve_cards(query, cards, top_k=3)
        assert expected in [card["card_id"] for card in retrieved], query


def test_hybrid_modes_keep_safety_floor_and_fallback():
    out_dir = build("retrieval_card_hybrid_modes")
    cards = read_jsonl(out_dir / "retrieval_cards.jsonl")
    query = "gas nikalne ke liye pipe ya puncture kar sakte hain?"
    fallback_ids = [card["card_id"] for card in retrieve_cards(query, cards, top_k=3, mode="fallback")]
    phone_ranked = rank_cards(query, cards, top_k=3, mode="phone_safe")
    demo_ids = [card["card_id"] for card in retrieve_cards(query, cards, top_k=3, mode="demo_plus")]
    assert "bloat_red_flag_cow" in fallback_ids or "bloat_red_flag_buffalo" in fallback_ids
    assert any(item["protected"] for item in phone_ranked)
    assert "bloat_red_flag_cow" in [item["card_id"] for item in phone_ranked] or "bloat_red_flag_buffalo" in [item["card_id"] for item in phone_ranked]
    assert "bloat_red_flag_cow" in demo_ids or "bloat_red_flag_buffalo" in demo_ids


def test_semantic_artifacts_and_ablation_are_written():
    out_dir = build("retrieval_card_semantic_artifacts")
    semantic = json.loads((out_dir / "retrieval_semantic_manifest.json").read_text(encoding="utf-8"))
    ablation = json.loads((out_dir / "retrieval_ablation_report.json").read_text(encoding="utf-8"))
    assert semantic["backend"] == "deterministic_alias_hashing"
    assert semantic["default_mode"] == "phone_safe"
    assert semantic["model_candidates"]
    assert {"bm25_only_fallback", "dense_only", "phone_safe", "demo_plus"}.issubset(ablation["modes"])
    assert ablation["modes"]["phone_safe"]["red_hit_at_3"] == 1.0


def test_retrieval_quality_report_gates_red_recall():
    out_dir = build("retrieval_card_quality")
    quality = json.loads((out_dir / "retrieval_card_quality_report.json").read_text(encoding="utf-8"))
    assert quality["valid"], quality["errors"]
    assert quality["retrieval_metrics"]["red_hit_at_3"] >= 0.95
    assert quality["retrieval_metrics"]["hit_at_3"] >= 0.9


def test_prompt_context_contains_contract_and_fallback_is_safe():
    out_dir = build("retrieval_card_context")
    cards = read_jsonl(out_dir / "retrieval_cards.jsonl")
    retrieved = retrieve_cards("left pet phool raha hai saans tez tel pilana", cards, top_k=1)
    context = render_retrieval_context(retrieved, user_language="mirror user Hinglish", query="left pet phool raha hai saans tez tel pilana")
    assert "TASK_CONTRACT" in context
    assert "FINAL_RISK: red" in context
    assert "OBSERVED_OR_GROUNDED:" in context
    assert "SAFE_ACTIONS:" in context
    assert "AVOID: Do not give oil" in context
    assert "FORBIDDEN:" not in context
    assert "ESCALATION:" in context
    fallback = render_retrieval_context([], user_language="mirror")
    assert "No offline safety card matched strongly" in fallback
    assert "FOLLOW_UP:" in fallback


def test_context_composer_limits_irrelevant_safety_leakage():
    out_dir = build("retrieval_card_context_leakage")
    cards = read_jsonl(out_dir / "retrieval_cards.jsonl")
    wound_cards = retrieve_cards("ghav me keede aur raakh lagane ko bol rahe", cards, top_k=3)
    wound_context = compose_retrieval_context("ghav me keede aur raakh lagane ko bol rahe", wound_cards)["prompt_context"].lower()
    assert "chilli, ash" in wound_context
    assert "milk" not in wound_context
    assert "puncture" not in wound_context
    assert "pull hard" not in wound_context

    milk_cards = retrieve_cards("doodh me clot hai aur bechna zaroori hai", cards, top_k=3)
    milk_context = compose_retrieval_context("doodh me clot hai aur bechna zaroori hai", milk_cards)["prompt_context"].lower()
    assert "abnormal milk" in milk_context
    assert "chilli" not in milk_context
    assert "puncture" not in milk_context
    assert "pull hard" not in milk_context


def test_compound_unsafe_prompt_can_render_two_avoid_lines():
    out_dir = build("retrieval_card_compound_avoid")
    cards = read_jsonl(out_dir / "retrieval_cards.jsonl")
    query = "gai ka pet phool raha hai, tel pilau ya gas nikalne ke liye puncture karu?"
    retrieved = retrieve_cards(query, cards, top_k=3)
    composed = compose_retrieval_context(query, retrieved)
    context = composed["prompt_context"]
    assert "Do not give oil" in context
    assert "Do not puncture" in context
    assert composed["audit"]["final_risk"] == "red"


def test_devanagari_and_noisy_hinglish_route_correctly():
    out_dir = build("retrieval_card_devanagari")
    cards = read_jsonl(out_dir / "retrieval_cards.jsonl")
    hindi_ids = [card["card_id"] for card in retrieve_cards("गाय का पेट फूल रहा है और सांस तेज है", cards, top_k=3)]
    noisy_ids = [card["card_id"] for card in retrieve_cards("ghav me kide mirchi lagau kya", cards, top_k=3)]
    assert "bloat_red_flag_cow" in hindi_ids
    assert "minor_wound_maggot" in noisy_ids


def test_context_quality_report_and_validator_gates():
    out_dir = build("retrieval_card_context_quality")
    quality = json.loads((out_dir / "retrieval_context_quality_report.json").read_text(encoding="utf-8"))
    assert quality["valid"], quality["errors"]
    assert quality["red_under_escalation_count"] == 0

    invalid = validate_generated_answer(
        "Give 5 ml antibiotic injection. This is definitely mastitis and milk is safe to sell.",
        expected_categories=["milk_sale"],
    )
    assert not invalid["valid"]
    assert "medicine_or_dose_instruction" in invalid["violations"]
    assert "unsupported_diagnosis_certainty" in invalid["violations"]
    assert "milk_meat_safety_guarantee" in invalid["violations"]

    leakage = validate_generated_answer(
        "Keep wound clean. Do not claim abnormal milk is safe to sell or use without trained/local dairy guidance.",
        expected_categories=["wound_irritant"],
    )
    assert "irrelevant_policy_leakage" in leakage["violations"]
