from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pashu_saathi_dataset.answer_guard import GuardResult, guard_answer
from pashu_saathi_dataset.retriever import RetrievalResult, accepted_chunks, render_evidence_context, retrieve_chunks
from pashu_saathi_dataset.router import DEFAULT_KNOWLEDGE_DIR, RouteDecision, classify_query


DEFAULT_CPT_ADAPTER_PATH = (
    Path(__file__).resolve().parents[3]
    / "kaggle-output"
    / "pashupulse-gemma-raw-cpt"
    / "output"
    / "pashu_pulse_raw_cpt_v1"
    / "adapter_final"
)


@dataclass(frozen=True)
class InferenceResult:
    query: str
    route: RouteDecision
    retrieval_results: list[RetrievalResult]
    accepted_chunk_ids: list[str]
    prompt: str
    draft_answer: str
    guard: GuardResult
    final_answer: str
    cpt_adapter_path: str


Generator = Callable[[str], str]


def run_inference(
    query: str,
    generator: Generator | None = None,
    knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
    cpt_adapter_path: Path = DEFAULT_CPT_ADAPTER_PATH,
) -> InferenceResult:
    route = classify_query(query, knowledge_dir)
    retrieval_results = retrieve_chunks(query, route, knowledge_dir)
    chunks = accepted_chunks(retrieval_results)
    if route.answer_mode == "card_grounded" and not chunks:
        route = RouteDecision(
            intent=route.intent,
            risk_level=route.risk_level,
            needs_evidence=route.needs_evidence,
            answer_mode="safe_fallback",
            retrieval=route.retrieval,
            threshold_key=route.threshold_key,
            retrieval_query=route.retrieval_query,
            reason=f"{route.reason};no_accepted_evidence",
        )
    prompt = build_prompt(query, route, chunks)
    draft = generator(prompt) if generator else heuristic_generate(query, route, chunks)
    guard = guard_answer(draft, chunks, knowledge_dir)
    return InferenceResult(
        query=query,
        route=route,
        retrieval_results=retrieval_results,
        accepted_chunk_ids=[chunk["chunk_id"] for chunk in chunks],
        prompt=prompt,
        draft_answer=draft,
        guard=guard,
        final_answer=guard.answer,
        cpt_adapter_path=str(cpt_adapter_path),
    )


def build_prompt(query: str, route: RouteDecision, chunks: list[dict]) -> str:
    style = (
        "Mirror the user's language/script. Keep the answer short. "
        "Use bullets for urgent/checklist answers. Do not diagnose with certainty, "
        "give medicine, dose, injection, procedure, or milk/meat safety guarantee."
    )
    if route.answer_mode == "cpt_direct":
        return f"{style}\n\nMODE: cpt_direct\nUSER: {query}\nANSWER:"
    if route.answer_mode == "safe_fallback":
        context = render_evidence_context(chunks) if chunks else ""
        return (
            f"{style}\n\nMODE: safe_fallback\n"
            "Answer the unsafe request directly first: say not to give the medicine, injection, dose, or procedure without trained advice. "
            "If evidence is provided, use its DO_NOW/DO_NOT/CALL_HELP_IF fields for safe holding steps. "
            "Do not give exact medicine, dose, withdrawal period, or procedure.\n\n"
            f"{context}\n\nUSER: {query}\nANSWER:"
        )
    return (
        f"{style}\n\nMODE: card_grounded\n"
        "Use the accepted offline evidence below as safety-critical boundaries and action guidance.\n"
        "Answer the user's exact question first. Then give useful farmer steps in this order: "
        "1) do not do unsafe action, 2) do-now low-risk holding steps, 3) why it matters briefly, "
        "4) when/who to call. You may add only low-risk supportive steps that do not conflict with the evidence. "
        "Do not add medicine, dose, procedure, diagnosis certainty, or exact milk/meat safety claims unless the evidence says so.\n\n"
        f"{render_evidence_context(chunks)}\n\nUSER: {query}\nANSWER:"
    )


def heuristic_generate(query: str, route: RouteDecision, chunks: list[dict]) -> str:
    # Testable offline stand-in for the real CPT adapter call. Production code can pass
    # a generator that loads Gemma + adapter_final and generates from build_prompt().
    if route.answer_mode == "safe_fallback":
        if chunks:
            chunk = chunks[0]
            do_now = chunk.get("do_now", ["Animal ko shaant rakho"])[:2]
            do_not = chunk.get("do_not", chunk.get("forbidden_actions", []))[:2]
            call_help = chunk.get("call_help_if", ["red flags ho"])[:2]
            return (
                f"- {_format_do_not(do_not)}\n"
                f"- {', '.join(do_now)}.\n"
                f"- {chunk.get('why', 'Bina trained advice ke dawa ya procedure unsafe ho sakta hai').rstrip('.')}.\n"
                f"- {', '.join(call_help)} ho toh trained para-vet/doctor ko jaldi bulao."
            )
        return (
            "Bina trained advice ke dawa, sui, dose, ya procedure mat do. "
            "Animal ko shaant rakho aur khana, paani, saans, bukhar, aur behavior check karo. "
            "Agar saans tez hai, animal gir raha hai, khana band hai, ya bahut sust hai toh para-vet/doctor ko jaldi bulao."
        )
    if route.answer_mode == "cpt_direct":
        return (
            "Paani, chara, gobar, chalna, saans, doodh aur behavior check karo. "
            "Agar khana bilkul band ho, bukhar ho, saans tez ho, ya animal bahut sust ho toh para-vet se baat karo."
        )
    if chunks:
        chunk = chunks[0]
        do_now = chunk.get("do_now", ["Animal ko shaant rakho"])[:2]
        do_not = chunk.get("do_not", chunk.get("forbidden_actions", []))[:2]
        call_help = chunk.get("call_help_if", ["red flags ho"])[:2]
        return (
            f"- {_format_do_not(do_not)}\n"
            f"- {', '.join(do_now)}.\n"
            f"- {chunk.get('why', chunk['text'].split('.')[0].strip()).rstrip('.')}.\n"
            f"- {', '.join(call_help)} ho toh trained para-vet/doctor ko jaldi bulao."
        )
    return (
        "Mere offline documents is exact baat ko support nahi karte. "
        "Bina trained advice ke dawa, dose, ya procedure mat do. Red flags ho toh para-vet/doctor ko bulao."
    )


def _format_do_not(items: list[str]) -> str:
    if not items:
        return "Unsafe home treatment mat karo."
    text = ", ".join(items)
    lowered = text.lower()
    if lowered.startswith("do not") or " do not " in lowered:
        return text.rstrip(".") + "."
    return text.rstrip(".") + " mat karo."
