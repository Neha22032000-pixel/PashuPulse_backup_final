"""Validate PashuPulse source-document bundle.

This is intentionally lightweight: it checks that the manifest, source briefs,
and structured claim catalog stay usable for later SFT/RAG work without
requiring a full ingestion pipeline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs" / "source_documents" / "pashupulse_sft_v2"
MANIFEST = BUNDLE / "source_manifest.json"
CLAIMS = BUNDLE / "source_claims.jsonl"
DOCUMENTS = BUNDLE / "documents"

REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "title",
    "canonical_url",
    "publisher",
    "publisher_type",
    "document_type",
    "country_or_region",
    "language",
    "species",
    "topics",
    "retrieved_at",
    "brief_file",
    "license_status",
    "license_note",
    "allowed_use",
    "risk_level",
    "review_status",
}

REQUIRED_CLAIM_FIELDS = {
    "source_id",
    "claim_id",
    "claim",
    "species",
    "condition_or_topic",
    "allowed_for_sft",
    "safety_label",
    "evidence_location",
    "notes",
}

ALLOWED_SAFETY_LABELS = {
    "safe_general_info",
    "triage_or_referral",
    "requires_veterinarian",
    "diagnosis_claim",
    "drug_dose_or_route",
    "withdrawal_period",
    "procedure_instruction",
    "product_or_commercial_bias",
    "region_specific_regulation",
    "do_not_train_as_answer",
}

RISKY_PATTERNS = [
    r"\b\d+\s*(ml|mg|iu)\b",
    r"\binject(?:ion)?\s+\d",
    r"\bwithdrawal\s+period\s*[:=]",
    r"\broute\s*[:=]\s*(im|iv|sc|subcutaneous|intramuscular)",
]

MOJIBAKE_PATTERNS = ["�", "Ã", "Â"]


def load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_claims() -> list[dict]:
    claims: list[dict] = []
    with CLAIMS.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                claims.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"source_claims.jsonl line {line_no} is invalid JSON: {exc}") from exc
    return claims


def has_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def validate_manifest(errors: list[str]) -> dict[str, dict]:
    manifest = load_manifest()
    sources = manifest.get("sources", [])
    seen: set[str] = set()
    by_id: dict[str, dict] = {}

    if not sources:
        errors.append("manifest has no sources")
        return by_id

    for source in sources:
        sid = source.get("source_id")
        if not sid:
            errors.append("manifest source missing source_id")
            continue
        if sid in seen:
            errors.append(f"duplicate source_id in manifest: {sid}")
        seen.add(sid)
        by_id[sid] = source

        missing = sorted(REQUIRED_SOURCE_FIELDS - set(source))
        if missing:
            errors.append(f"{sid}: missing manifest fields {missing}")

        brief_file = source.get("brief_file")
        if brief_file and not (BUNDLE / brief_file).exists():
            errors.append(f"{sid}: brief_file does not exist: {brief_file}")

        if not source.get("license_status"):
            errors.append(f"{sid}: license_status is empty")
        if not source.get("allowed_use"):
            errors.append(f"{sid}: allowed_use is empty")

    return by_id


def validate_claims(errors: list[str], sources_by_id: dict[str, dict]) -> None:
    claims = load_claims()
    seen_claims: set[str] = set()
    claim_sources: set[str] = set()

    if not claims:
        errors.append("source_claims.jsonl has no claims")
        return

    for claim in claims:
        cid = claim.get("claim_id", "<missing>")
        sid = claim.get("source_id")
        missing = sorted(REQUIRED_CLAIM_FIELDS - set(claim))
        if missing:
            errors.append(f"{cid}: missing claim fields {missing}")
        if cid in seen_claims:
            errors.append(f"duplicate claim_id: {cid}")
        seen_claims.add(cid)
        if sid not in sources_by_id:
            errors.append(f"{cid}: source_id not in manifest: {sid}")
        if claim.get("safety_label") not in ALLOWED_SAFETY_LABELS:
            errors.append(f"{cid}: unknown safety_label {claim.get('safety_label')!r}")
        if not isinstance(claim.get("allowed_for_sft"), bool):
            errors.append(f"{cid}: allowed_for_sft must be boolean")
        claim_sources.add(sid)

    missing_claim_sources = sorted(set(sources_by_id) - claim_sources)
    if missing_claim_sources:
        errors.append(f"sources missing structured claims: {missing_claim_sources}")


def validate_briefs(errors: list[str], sources_by_id: dict[str, dict]) -> None:
    for sid, source in sources_by_id.items():
        brief_path = BUNDLE / source["brief_file"]
        if not brief_path.exists():
            continue
        text = brief_path.read_text(encoding="utf-8")
        if len(text.strip()) < 200:
            errors.append(f"{sid}: brief is too short")
        if not has_any(text, ["# Short Summary", "## What This Source Is Useful For"]):
            errors.append(f"{sid}: brief missing summary/usefulness section")
        if not has_any(text, ["# Safety Boundaries", "## SFT Safety Boundaries", "# Safety Boundaries And Exclusions"]):
            errors.append(f"{sid}: brief missing safety boundary section")
        if not has_any(text, ["# QA Notes", "License/usage note", "license_status"]):
            errors.append(f"{sid}: brief missing QA/license signal")
        for pattern in MOJIBAKE_PATTERNS:
            if pattern in text:
                errors.append(f"{sid}: possible mojibake marker {pattern!r}")
        for pattern in RISKY_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                errors.append(f"{sid}: possible risky procedural/dose text matching {pattern!r}")


def main() -> int:
    errors: list[str] = []
    if not MANIFEST.exists():
        print(f"Missing manifest: {MANIFEST}")
        return 1
    if not CLAIMS.exists():
        print(f"Missing claims file: {CLAIMS}")
        return 1
    if not DOCUMENTS.exists():
        print(f"Missing documents dir: {DOCUMENTS}")
        return 1

    sources_by_id = validate_manifest(errors)
    validate_claims(errors, sources_by_id)
    validate_briefs(errors, sources_by_id)

    if errors:
        print("PashuPulse source document validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"PashuPulse source document validation passed: {len(sources_by_id)} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
