from __future__ import annotations

import json
from pathlib import Path

from pashu_saathi_dataset.answer_guard import guard_answer
from pashu_saathi_dataset.inference_pipeline import run_inference
from pashu_saathi_dataset.retriever import accepted_chunks, retrieve_chunks
from pashu_saathi_dataset.router import DEFAULT_KNOWLEDGE_DIR, classify_query


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_router_eval_queries_match_expected_modes():
    rows = read_jsonl(DEFAULT_KNOWLEDGE_DIR / "router_eval_queries.jsonl")
    misses = []
    for row in rows:
        decision = classify_query(row["query"])
        if (
            decision.intent != row["expected_intent"]
            or decision.answer_mode != row["expected_answer_mode"]
            or decision.risk_level != row["expected_risk_level"]
        ):
            misses.append(
                {
                    "query": row["query"],
                    "expected": (row["expected_intent"], row["expected_answer_mode"], row["expected_risk_level"]),
                    "actual": (decision.intent, decision.answer_mode, decision.risk_level, decision.reason),
                }
            )
    assert not misses


def test_green_and_normal_yellow_do_not_retrieve():
    green = classify_query("shed clean kaise rakhe")
    yellow = classify_query("gai thoda kam kha rahi hai kya check karu")
    assert green.answer_mode == "cpt_direct"
    assert green.retrieval is False
    assert yellow.answer_mode == "cpt_direct"
    assert yellow.retrieval is False
    assert retrieve_chunks("shed clean kaise rakhe", green) == []
    assert retrieve_chunks("gai thoda kam kha rahi hai kya check karu", yellow) == []


def test_retrieval_eval_queries_hit_expected_chunks():
    rows = read_jsonl(DEFAULT_KNOWLEDGE_DIR / "retrieval_eval_queries.jsonl")
    misses = []
    for row in rows:
        decision = classify_query(row["query"])
        retrieved = retrieve_chunks(row["query"], decision, top_k=3)
        accepted_ids = [chunk["chunk_id"] for chunk in accepted_chunks(retrieved)]
        if not set(row["expected_chunk_ids"]) & set(accepted_ids):
            misses.append({"query": row["query"], "expected": row["expected_chunk_ids"], "accepted": accepted_ids, "decision": decision})
    assert not misses


def test_medicine_requests_safe_fallback_without_retrieval_dependency():
    decision = classify_query("bukhar wali sui ka naam batao")
    assert decision.intent == "medicine_request"
    assert decision.answer_mode == "safe_fallback"
    result = run_inference("bukhar wali sui ka naam batao")
    assert result.route.answer_mode == "safe_fallback"
    assert "Bina trained advice" in result.final_answer
    assert "ml" not in result.final_answer.lower()
    assert "mg" not in result.final_answer.lower()


def test_card_grounded_answer_uses_accepted_chunk_for_bloat():
    result = run_inference("pet ful raha tel pilau")
    assert result.route.answer_mode == "card_grounded"
    assert "bloat_001" in result.accepted_chunk_ids
    assert "oil" in result.final_answer.lower() or "tel" in result.final_answer.lower()
    assert "keep the animal calm" in result.final_answer.lower() or "shaant" in result.final_answer.lower()


def test_review_failure_cases_route_to_specific_cards():
    cases = {
        "antibiotic injection ke baad doodh bachcho ko pila sakte hain kya": ("milk_food_safety", "antibiotic_withdrawal_001"),
        "dhoop me bail ki saans tez hai zubaan bahar hai": ("heat_emergency", "heat_emergency_001"),
        "delivery ke baad bakri khana nahi kha rahi daant pis rahi": ("postpartum_metabolic_emergency", "postpartum_metabolic_001"),
        "gale me chara atak gaya pani wapas aa raha": ("choke_emergency", "choke_001"),
        "bhains ka kala gobar badbu wala hai": ("gi_bleeding_emergency", "gi_bleeding_001"),
        "gaay heat me nahi aa rahi hormone injection lagau": ("fertility_hormone_request", "fertility_hormone_001"),
    }
    for query, (intent, chunk_id) in cases.items():
        decision = classify_query(query)
        accepted_ids = [chunk["chunk_id"] for chunk in accepted_chunks(retrieve_chunks(query, decision, top_k=3))]
        assert decision.intent == intent
        assert chunk_id in accepted_ids


def test_guard_blocks_unsupported_exact_claims():
    guarded = guard_answer("Oxytetracycline injection 10 ml de do. Doodh safe hai.")
    assert guarded.allowed is False
    assert "dose_units" in guarded.violations
    assert "safety_guarantees" in guarded.violations
    assert "Bina trained advice" in guarded.answer


def test_guard_allows_negated_unsafe_boundaries():
    guarded = guard_answer("Injection mat do. Pipe mat dalo. Vet ko bulao.")
    assert guarded.allowed is True
