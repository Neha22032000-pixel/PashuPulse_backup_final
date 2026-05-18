from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pashu_saathi_dataset.router import DEFAULT_KNOWLEDGE_DIR, RouteDecision, load_json, normalize_text


@dataclass(frozen=True)
class RetrievalResult:
    chunk: dict[str, Any]
    final_score: float
    bm25_score: float
    embedding_score: float
    intent_match: float
    species_match: float
    risk_boost: float
    accepted: bool
    reject_reason: str


def load_chunks(knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR) -> list[dict[str, Any]]:
    path = knowledge_dir / "knowledge_chunks.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieve_chunks(
    query: str,
    route: RouteDecision,
    knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
    top_k: int = 3,
) -> list[RetrievalResult]:
    policy = load_json(knowledge_dir / "router_policy.yaml")
    chunks = load_chunks(knowledge_dir)
    if not route.retrieval and route.answer_mode != "safe_fallback":
        return []

    query_terms = _tokens(route.retrieval_query or query)
    docs = [_document_terms(chunk) for chunk in chunks]
    doc_freq = Counter(term for doc in docs for term in set(doc))
    avg_len = sum(len(doc) for doc in docs) / max(len(docs), 1)
    raw_bm25 = [
        _bm25_score(query_terms, doc, doc_freq, len(docs), avg_len) for doc in docs
    ]
    max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
    weights = policy["retrieval_gate"]["scoring"]

    results = []
    for chunk, doc_terms, bm25 in zip(chunks, docs, raw_bm25):
        bm25_norm = bm25 / max(max_bm25, 1e-9)
        embedding = _semantic_score(query, chunk)
        intent_match = _intent_match(route.intent, chunk)
        species_match = _species_match(query, chunk)
        risk_boost = 1.0 if chunk.get("risk_level") == route.risk_level == "red" else 0.0
        final_score = (
            weights["bm25_weight"] * bm25_norm
            + weights["embedding_weight"] * embedding
            + weights["intent_match_weight"] * intent_match
            + weights["species_match_weight"] * species_match
            + weights["risk_boost_weight"] * risk_boost
        )
        accepted, reject_reason = _accept_result(final_score, route, chunk, policy)
        if final_score > 0:
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    final_score=round(final_score, 4),
                    bm25_score=round(bm25_norm, 4),
                    embedding_score=round(embedding, 4),
                    intent_match=intent_match,
                    species_match=species_match,
                    risk_boost=risk_boost,
                    accepted=accepted,
                    reject_reason=reject_reason,
                )
            )
    results.sort(key=lambda item: (-item.accepted, -item.final_score, item.chunk["chunk_id"]))
    return results[:top_k]


def accepted_chunks(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    return [result.chunk for result in results if result.accepted]


def render_evidence_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return (
            "No accepted offline evidence chunk. Use safe fallback only: do not diagnose, "
            "prescribe medicine, give dose, inject, cut, puncture, force-feed, or guarantee milk/meat safety."
        )
    blocks = []
    for chunk in chunks:
        do_now = chunk.get("do_now", [])
        do_not = chunk.get("do_not", chunk.get("forbidden_actions", []))
        red_flags = chunk.get("call_help_if", [])
        blocks.append(
            "\n".join(
                [
                    f"CHUNK_ID: {chunk['chunk_id']}",
                    f"RISK: {chunk['risk_level']}",
                    f"TOPICS: {', '.join(chunk.get('topics', []))}",
                    f"INTENTS: {', '.join(chunk.get('semantic_intents', []))}",
                    f"TEXT: {chunk['text']}",
                    f"DO_NOW: {', '.join(do_now)}",
                    f"DO_NOT: {', '.join(do_not)}",
                    f"WHY: {chunk.get('why', '')}",
                    f"CALL_HELP_IF: {', '.join(red_flags)}",
                    f"FORBIDDEN: {', '.join(chunk.get('forbidden_actions', []))}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _accept_result(final_score: float, route: RouteDecision, chunk: dict[str, Any], policy: dict[str, Any]) -> tuple[bool, str]:
    rules = policy["retrieval_gate"]["minimum_rules"]
    if rules["require_allowed_use"] not in chunk.get("allowed_use", []):
        return False, "allowed_use_missing"
    if rules["require_intent_or_topic_match"] and not _compatible(route.intent, chunk):
        return False, "intent_topic_mismatch"
    if rules["reject_generic_style_chunks_for_high_risk"] and route.risk_level == "red":
        if chunk.get("risk_level") == "green" or "routine_care" in chunk.get("semantic_intents", []):
            return False, "generic_chunk_for_high_risk"
    threshold_key = route.threshold_key or ("red" if route.risk_level == "red" else "yellow_boundary")
    threshold = policy["retrieval_gate"]["thresholds"].get(threshold_key, 1.0)
    if final_score < threshold:
        return False, f"below_threshold:{threshold}"
    return True, ""


def _compatible(intent: str, chunk: dict[str, Any]) -> bool:
    intents = set(chunk.get("semantic_intents", []))
    topics = set(chunk.get("topics", []))
    if intent in intents or intent in topics:
        return True
    if intent == "unsafe_remedy" and "unsafe_remedy" in intents:
        return True
    if intent == "milk_food_safety" and ("meat_safety" in topics or "milk_udder" in topics):
        return True
    return False


def _document_terms(chunk: dict[str, Any]) -> list[str]:
    fields = [
        chunk.get("chunk_id", "").replace("_", " "),
        chunk.get("title", ""),
        " ".join(chunk.get("species", [])),
        " ".join(chunk.get("topics", [])),
        " ".join(chunk.get("aliases", [])),
        " ".join(chunk.get("semantic_intents", [])),
        chunk.get("text", ""),
        " ".join(chunk.get("do_now", [])),
        " ".join(chunk.get("do_not", [])),
        chunk.get("why", ""),
        " ".join(chunk.get("call_help_if", [])),
        " ".join(chunk.get("forbidden_actions", [])),
    ]
    return _tokens(" ".join(fields))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\u0900-\u097f]+", normalize_text(text))


def _bm25_score(query_terms: list[str], doc_terms: list[str], doc_freq: Counter[str], doc_count: int, avg_len: float) -> float:
    counts = Counter(doc_terms)
    score = 0.0
    k1 = 1.2
    b = 0.75
    doc_len = max(len(doc_terms), 1)
    for term in query_terms:
        if term not in counts:
            continue
        idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
        tf = counts[term]
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avg_len, 1)))
    return score


def _semantic_score(query: str, chunk: dict[str, Any]) -> float:
    query_grams = _char_ngrams(normalize_text(query))
    text = " ".join([chunk.get("title", ""), " ".join(chunk.get("aliases", [])), chunk.get("text", "")])
    chunk_grams = _char_ngrams(normalize_text(text))
    return _cosine(query_grams, chunk_grams)


def _char_ngrams(text: str, n: int = 3) -> dict[str, int]:
    compact = f"  {text}  "
    grams: dict[str, int] = {}
    for index in range(max(len(compact) - n + 1, 0)):
        gram = compact[index : index + n]
        grams[gram] = grams.get(gram, 0) + 1
    return grams


def _cosine(left: dict[str, int], right: dict[str, int]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return dot / max(left_norm * right_norm, 1e-9)


def _intent_match(intent: str, chunk: dict[str, Any]) -> float:
    return 1.0 if _compatible(intent, chunk) else 0.0


def _species_match(query: str, chunk: dict[str, Any]) -> float:
    species_aliases = {
        "cow": ["cow", "gai"],
        "buffalo": ["buffalo", "bhains"],
        "calf": ["calf", "bachda", "bachdi"],
        "goat": ["goat", "bakri", "bakra"],
        "sheep": ["sheep", "bhed"],
        "ox": ["ox", "bail"],
    }
    normalized = normalize_text(query)
    mentioned = {
        species
        for species, aliases in species_aliases.items()
        if any(alias in normalized for alias in aliases)
    }
    if not mentioned:
        return 0.0
    chunk_species = set(chunk.get("species", []))
    if "livestock" in chunk_species:
        return 0.5
    return 1.0 if mentioned & chunk_species else 0.0
