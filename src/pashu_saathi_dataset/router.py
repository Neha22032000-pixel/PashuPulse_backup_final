from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIR = PACKAGE_ROOT / "data" / "offline_knowledge" / "pashupulse_v1"


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    risk_level: str
    needs_evidence: bool
    answer_mode: str
    retrieval: bool
    threshold_key: str | None
    retrieval_query: str
    reason: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_query(
    query: str,
    knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
    policy: dict[str, Any] | None = None,
    prototypes: list[dict[str, Any]] | None = None,
) -> RouteDecision:
    policy = policy or load_json(knowledge_dir / "router_policy.yaml")
    prototypes = prototypes or read_jsonl(knowledge_dir / "intent_prototypes.jsonl")
    normalized = normalize_text(query)

    hard = _hard_trigger_match(normalized, policy)
    if hard:
        intent, trigger = hard
        return _decision_from_intent(
            intent,
            policy,
            query,
            reason=f"hard_trigger:{trigger}",
        )

    semantic = _semantic_intent_match(normalized, prototypes)
    if semantic and semantic[1] >= 0.32:
        intent, score = semantic
        return _decision_from_intent(
            intent,
            policy,
            query,
            reason=f"semantic_prototype:{score:.3f}",
        )

    if _looks_like_routine(normalized):
        return _decision_from_intent("routine_care", policy, query, reason="routine_default")
    return _decision_from_intent("symptom_triage", policy, query, reason="triage_default")


def _decision_from_intent(intent: str, policy: dict[str, Any], query: str, reason: str) -> RouteDecision:
    route = policy["intent_routes"].get(intent, policy["intent_routes"]["symptom_triage"])
    return RouteDecision(
        intent=intent if intent in policy["intent_routes"] else "symptom_triage",
        risk_level=route["risk_level"],
        needs_evidence=bool(route["needs_evidence"]),
        answer_mode=route["answer_mode"],
        retrieval=bool(route["retrieval"]),
        threshold_key=route.get("threshold_key"),
        retrieval_query=_retrieval_query(query, intent),
        reason=reason,
    )


def _hard_trigger_match(normalized: str, policy: dict[str, Any]) -> tuple[str, str] | None:
    special_cases = [
        ("outbreak_public_health", ["kai", "multiple", "several", "5 janwaro", "ek sath"], ["mar", "beemar", "bukhar", "naak", "khansi"]),
        ("bite_zoonotic", ["kutta", "dog", "saanp", "saap", "snake", "kaat", "bite", "rabies"], []),
        ("wound_emergency", ["navel", "nabhi", "peep", "pus"], ["calf", "bachda", "bachdi", "sujan"]),
        ("breathing_collapse", ["gale", "neck", "saans me awaaz", "saans mein awaaz"], ["sujan", "swelling", "awaaz"]),
        ("fertility_hormone_request", ["heat", "garam", "hormone", "desi dawa"], ["nahi aa", "injection", "delivery"]),
        ("milk_food_safety", ["antibiotic", "injection", "dawa"], ["doodh", "milk"]),
        ("milk_food_safety", ["doodh", "milk"], ["bech", "clot", "smell", "badbu", "watery", "namkeen", "safe", "pila"]),
        ("gi_bleeding_emergency", ["kala gobar", "gobar kala", "black dung", "internal bleeding"], []),
        ("heat_emergency", ["dhoop", "garmi", "heat", "kaan garam", "zubaan bahar"], ["saans", "panting", "gir", "not drinking"]),
    ]
    for intent, required_any, context_any in special_cases:
        if intent not in policy["hard_triggers"]:
            continue
        if any(term in normalized for term in required_any) and (
            not context_any or any(term in normalized for term in context_any)
        ):
            return intent, f"special_case:{intent}"

    # Specific danger/medicine/newborn/postpartum intents must win before broad
    # words like "doodh", otherwise the router retrieves milk cards for unrelated
    # painkiller, fertility, calf, and postpartum questions.
    priority = [
        "bloat_emergency",
        "calving_emergency",
        "postpartum_metabolic_emergency",
        "calf_emergency",
        "poisoning_emergency",
        "choke_emergency",
        "heat_emergency",
        "gi_bleeding_emergency",
        "fertility_hormone_request",
        "medicine_request",
        "urinary_emergency",
        "neuro_emergency",
        "bite_zoonotic",
        "wound_emergency",
        "carcass_public_health",
        "outbreak_public_health",
        "milk_food_safety",
    ]
    ordered_intents = [intent for intent in priority if intent in policy["hard_triggers"]]
    ordered_intents.extend(key for key in policy["hard_triggers"] if key not in ordered_intents)
    for intent in ordered_intents:
        for trigger in policy["hard_triggers"][intent]:
            if normalize_text(trigger) in normalized:
                return intent, trigger
    return None


def _semantic_intent_match(normalized: str, prototypes: list[dict[str, Any]]) -> tuple[str, float] | None:
    query_grams = _char_ngrams(normalized)
    if not query_grams:
        return None
    best: tuple[str, float] | None = None
    for row in prototypes:
        proto_grams = _char_ngrams(normalize_text(row["text"]))
        score = _cosine(query_grams, proto_grams)
        if best is None or score > best[1]:
            best = (row["intent"], score)
    return best


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


def _looks_like_routine(normalized: str) -> bool:
    routine_terms = [
        "clean",
        "safai",
        "shed",
        "paani break",
        "water break",
        "routine",
        "warm",
        "garam rakhe",
        "milking",
        "chara kaise",
    ]
    return any(term in normalized for term in routine_terms)


def _retrieval_query(query: str, intent: str) -> str:
    return f"{query} {intent.replace('_', ' ')}".strip()
