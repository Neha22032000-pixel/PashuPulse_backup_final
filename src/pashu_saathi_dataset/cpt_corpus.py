from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


RETRIEVED_DATE = "2026-05-15"
CPT_CORPUS_VERSION = "cpt-corpus-v1"

ACCEPT = "accept"
QUARANTINE = "quarantine"
REJECT = "reject"

RISKY_SPAN_PATTERNS = [
    re.compile(r"\b\d+(\.\d+)?\s*(mg|ml|g|iu)\s*/\s*kg\b", re.IGNORECASE),
    re.compile(r"\b(inject|injection|intramuscular|subcutaneous|iv|i/v|s/c)\b", re.IGNORECASE),
    re.compile(r"\b(dose|dosage|route|withdrawal period|antibiotic|oxytocin|xylazine|diclofenac)\b", re.IGNORECASE),
    re.compile(r"\b(cut|puncture|lance|incision|stomach tube|trocar)\b", re.IGNORECASE),
]

HARD_REJECT_PATTERNS = [
    re.compile(r"\b(miracle cure|guaranteed cure|100% cure|secret remedy)\b", re.IGNORECASE),
    re.compile(r"\b(kerosene|tobacco water|acid|caustic|engine oil)\b", re.IGNORECASE),
    re.compile(r"\b(anti[- ]?vaccine|vaccines are useless|avoid veterinarian)\b", re.IGNORECASE),
    re.compile(r"\b(buy now|affiliate|promo code|limited offer)\b", re.IGNORECASE),
]

LOW_QUALITY_PATTERNS = [
    re.compile(r"(.)\1{12,}"),
    re.compile(r"\b(lorem ipsum|click here|subscribe now)\b", re.IGNORECASE),
]

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(home|about us|privacy policy|terms of use|cookie policy|share this|print page)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(copyright|all rights reserved).*$", re.IGNORECASE),
]


@dataclass(frozen=True)
class CptSource:
    source_id: str
    title: str
    publisher: str
    country: str
    url: str
    source_tier: str
    topics: list[str]
    species: list[str]
    language: str
    license_note: str
    raw_text: str
    notes: str = ""
    state: str = ""
    state_reason: str = ""
    removed_risky_spans: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "publisher": self.publisher,
            "country": self.country,
            "url": self.url,
            "retrieved_date": RETRIEVED_DATE,
            "source_tier": self.source_tier,
            "topics": self.topics,
            "species": self.species,
            "language": self.language,
            "license_note": self.license_note,
            "notes": self.notes,
        }


def _catalog() -> list[CptSource]:
    return [
        CptSource(
            "dahd_lhdcp_cpt",
            "Livestock Health and Disease Control Programme",
            "Department of Animal Husbandry and Dairying, Government of India",
            "India",
            "https://www.dahd.gov.in/schemes-programmes/lh-dc",
            "official_india",
            ["animal health infrastructure", "disease control", "extension"],
            ["cattle", "buffalo"],
            "English",
            "Government public information page; verify reuse before redistribution.",
            "The programme covers veterinary infrastructure, disease surveillance, mobile veterinary units, and call centre support. It is useful for learning Indian animal-health system language and escalation context.",
        ),
        CptSource(
            "dahd_nadcp_cpt",
            "National Animal Disease Control Programme",
            "Department of Animal Husbandry and Dairying, Government of India",
            "India",
            "https://www.dahd.gov.in/schemes/programmes/nadcp",
            "official_india",
            ["FMD", "brucellosis", "movement control", "vaccination policy"],
            ["cattle", "buffalo"],
            "English",
            "Government public information page; verify reuse before redistribution.",
            "NADCP describes FMD and brucellosis control, outbreak investigation, and regulation of animal movement. It is high-value India anchor text for reportable disease vocabulary.",
        ),
        CptSource(
            "nddb_good_dairy_husbandry_cpt",
            "Handbook of Good Dairy Husbandry Practices",
            "National Dairy Development Board",
            "India",
            "https://www.nddb.coop/sites/default/files/pdfs/Handbook-of-Good-Dairy-Husbandry-Practices.pdf",
            "official_extension_india",
            ["dairy husbandry", "feeding", "housing", "health", "milk hygiene"],
            ["cattle", "buffalo", "calf"],
            "English",
            "NDDB public PDF; verify reuse before redistribution.",
            "This handbook covers dairy animal management, housing, feeding, health practices, calf care, and clean milk production. It anchors Indian dairy extension language for CPT.",
        ),
        CptSource(
            "nddb_water_cpt",
            "Importance of Drinking Water for Dairy Animals",
            "National Dairy Development Board",
            "India",
            "https://www.nddb.coop/farmer/animal-nutrition/importance-of-drinking-water-for-dairy-animals",
            "official_extension_india",
            ["water", "heat stress", "routine care"],
            ["cattle", "buffalo"],
            "English",
            "NDDB public farmer page; verify reuse before redistribution.",
            "Dairy animals need regular access to clean drinking water. Water support is especially important in summer and for milk production, making this useful routine-care CPT text.",
        ),
        CptSource(
            "nddb_calf_nutrition_cpt",
            "Calf Nutrition",
            "National Dairy Development Board",
            "India",
            "https://www.nddb.coop/node/511",
            "official_extension_india",
            ["calf care", "feeding", "growth"],
            ["calf", "cattle", "buffalo"],
            "English",
            "NDDB public farmer page; verify reuse before redistribution.",
            "Young calves should be reared carefully with attention to feeding, clean practices, growth, and health. This gives farmer-facing calf-care vocabulary.",
        ),
        CptSource(
            "nddb_clean_milk_cpt",
            "Clean Milk Production Awareness",
            "National Dairy Development Board",
            "India",
            "https://www.nddb.coop/services/qa/capacity-building",
            "official_extension_india",
            ["milk hygiene", "clean milking", "quality"],
            ["cattle", "buffalo"],
            "English",
            "NDDB public service page; verify reuse before redistribution.",
            "Clean milk production depends on hygiene, clean utensils, and careful handling at village level. It is useful for milk hygiene language without certifying sale safety.",
        ),
        CptSource(
            "nddb_bloat_cpt",
            "Bloat",
            "National Dairy Development Board",
            "India",
            "https://www.nddb.coop/farmer/animal-health/disease/others/bloat",
            "official_extension_india",
            ["bloat", "emergency signs", "feeding prevention"],
            ["cattle", "buffalo"],
            "English",
            "NDDB public farmer page; verify reuse before redistribution.",
            "Bloat involves gas accumulation in the rumen and visible belly swelling. Prevention context includes careful feeding and avoiding risky grazing patterns. Do not include invasive treatment instructions in CPT chunks.",
        ),
        CptSource(
            "nddb_aflatoxicosis_cpt",
            "Aflatoxicosis",
            "National Dairy Development Board",
            "India",
            "https://www.nddb.coop/farmer/animal-health/disease/fungal/Aflatoxicosis",
            "official_extension_india",
            ["moldy feed", "feed safety", "poisoning risk"],
            ["cattle", "buffalo"],
            "English",
            "NDDB public farmer page; verify reuse before redistribution.",
            "Mold on damp feed can produce toxins and reduce animal health and production. This text supports feed-storage and spoiled-feed vocabulary for CPT.",
        ),
        CptSource(
            "nddb_fly_control_cpt",
            "Fly Control",
            "National Dairy Development Board",
            "India",
            "https://www.nddb.coop/farmer/animal-health/general/fly-control",
            "official_extension_india",
            ["shed hygiene", "fly control", "manure management"],
            ["cattle", "buffalo"],
            "English",
            "NDDB public farmer page; verify reuse before redistribution.",
            "Regular disposal of manure and urine and avoiding stagnant drainage around cattle sheds helps reduce flies and maintain cleaner animal surroundings.",
        ),
        CptSource(
            "tnau_calf_management_cpt",
            "Calf Management",
            "Tamil Nadu Agricultural University Agritech Portal",
            "India",
            "https://agritech.tnau.ac.in/expert_system/cattlebuffalo/Calf%20management.html",
            "official_extension_india",
            ["calving", "calf care", "bedding"],
            ["cattle", "buffalo", "calf"],
            "English",
            "University extension page; verify reuse before redistribution.",
            "Calving and calf management text covers clean calving areas, bedding, newborn care, and basic management around birth. It is useful for Indian extension terminology.",
        ),
        CptSource(
            "fao_dairy_housing_cpt",
            "Small-Scale Dairy Farming Manual: Dairy Cattle and Buffalo Housing",
            "Food and Agriculture Organization",
            "International",
            "https://www.fao.org/4/t1265e/t1270e05.htm",
            "international_extension",
            ["housing", "manure", "feeders", "waterers", "welfare"],
            ["cattle", "buffalo", "calf"],
            "English",
            "FAO public manual; check FAO reuse terms before redistribution.",
            "Clean housing improves cow and calf health and milk quality. The manual covers manure handling, clean feeders and drinkers, insect control, safe floors, ventilation, and environmental comfort.",
        ),
        CptSource(
            "fao_dairy_feeding_cpt",
            "Small-Scale Dairy Farming Manual: Feeding Dairy Cattle and Buffalo",
            "Food and Agriculture Organization",
            "International",
            "https://www.fao.org/4/t1265e/t1275e01.htm",
            "international_extension",
            ["feeding", "nutrition", "roughage", "concentrates", "buffalo"],
            ["cattle", "buffalo"],
            "English",
            "FAO public manual; check FAO reuse terms before redistribution.",
            "Dairy cattle and buffalo need nutrients from roughages, concentrates, minerals, vitamins, and water. The manual explains digestion, feed types, feed value, and productivity context.",
        ),
        CptSource(
            "fao_dairy_disease_cpt",
            "Small-Scale Dairy Farming Manual: Conditions Affecting Dairy Cattle and Buffalo",
            "Food and Agriculture Organization",
            "International",
            "https://www.fao.org/4/t1265e/t1285e01.htm",
            "international_extension",
            ["disease concepts", "symptoms", "records", "production loss"],
            ["cattle", "buffalo"],
            "English",
            "FAO public manual; check FAO reuse terms before redistribution.",
            "Disease is described as a change that can lower production or cause death. The manual introduces common condition categories and disease observation language without needing raw treatment recipes.",
        ),
        CptSource(
            "fao_dairy_parasites_cpt",
            "Small-Scale Dairy Farming Manual: Parasites in Dairy Cattle and Buffalo",
            "Food and Agriculture Organization",
            "International",
            "https://www.fao.org/4/t1265e/t1285e06.htm",
            "international_extension",
            ["parasites", "ticks", "worms", "production loss"],
            ["cattle", "buffalo", "calf"],
            "English",
            "FAO public manual; check FAO reuse terms before redistribution.",
            "Parasites can reduce production and affect young animals strongly. This source is useful for broad parasite vocabulary and observation language, not farmer dosing instructions.",
        ),
        CptSource(
            "fao_idf_good_dairy_practice_cpt",
            "Guide to Good Dairy Farming Practice",
            "FAO and International Dairy Federation",
            "International",
            "https://www.fao.org/4/ba0027e/ba0027e.pdf",
            "international_extension",
            ["animal health", "milk hygiene", "nutrition", "welfare", "environment"],
            ["dairy animals", "cattle", "buffalo"],
            "English",
            "FAO/IDF publication; check reuse terms before redistribution.",
            "Good dairy farming practice covers animal health, milk hygiene, nutrition, welfare, environment, and socio-economic management in practical farmer-oriented language.",
        ),
        CptSource(
            "woah_animal_welfare_cpt",
            "Animal Welfare and Dairy Cattle Production Systems",
            "World Organisation for Animal Health",
            "International",
            "https://www.woah.org/en/what-we-do/standards/codes-and-manuals/terrestrial-code-online-access/",
            "international_standard",
            ["welfare", "dairy systems", "handling", "housing"],
            ["cattle", "buffalo"],
            "English",
            "WOAH standards page; verify reuse before redistribution.",
            "WOAH standards provide high-level vocabulary for animal welfare, handling, housing, and production systems. Country-specific legal language should remain metadata, not farmer advice.",
        ),
        CptSource(
            "msd_vet_manual_cattle_cpt",
            "MSD Veterinary Manual: Cattle Health Topics",
            "MSD Veterinary Manual",
            "International",
            "https://www.msdvetmanual.com/",
            "reputable_veterinary_reference",
            ["disease descriptions", "clinical signs", "diagnostics"],
            ["cattle", "buffalo"],
            "English",
            "Reference text; verify license and keep only non-procedural snippets for CPT.",
            "Veterinary reference material can improve disease terminology and clinical-sign vocabulary. Sections with drug doses, routes, surgery, and procedures must be quarantined or stripped.",
        ),
        CptSource(
            "penn_state_dairy_extension_cpt",
            "Dairy Extension Articles",
            "Penn State Extension",
            "United States",
            "https://extension.psu.edu/animals-and-livestock/dairy",
            "university_extension",
            ["dairy management", "nutrition", "calf care", "housing"],
            ["cattle", "calf"],
            "English",
            "University extension content; verify reuse before redistribution.",
            "Dairy extension articles cover practical herd management, calf care, nutrition, housing, recordkeeping, and farm workflows. Highly local regulations should be labeled, not used as India advice.",
        ),
        CptSource(
            "umn_dairy_extension_cpt",
            "Dairy Extension Resources",
            "University of Minnesota Extension",
            "United States",
            "https://extension.umn.edu/dairy",
            "university_extension",
            ["dairy management", "calf care", "health", "feeding"],
            ["cattle", "calf"],
            "English",
            "University extension content; verify reuse before redistribution.",
            "University dairy extension resources provide broad management, feeding, housing, calf, and health vocabulary that can help CPT domain familiarity.",
        ),
        CptSource(
            "vikaspedia_dairy_context_cpt",
            "Dairy Animal Management Context",
            "Vikaspedia",
            "India",
            "https://vikaspedia.in/agriculture/livestock",
            "secondary_context",
            ["livestock management", "farmer language", "India context"],
            ["cattle", "buffalo", "calf"],
            "English/Hindi",
            "Secondary public information portal; verify reuse before redistribution.",
            "Useful as farmer-facing context and language exposure. Do not use as sole factual grounding for safety-critical SFT/RAG claims.",
        ),
        CptSource(
            "unsafe_blog_mustard_oil_bloat",
            "Home Cure for Cattle Bloat with Mustard Oil",
            "Unknown livestock tips blog",
            "Unknown",
            "https://example.invalid/cattle-bloat-home-cure",
            "unsafe_blog",
            ["bloat", "home remedies"],
            ["cattle", "buffalo"],
            "English",
            "Reject fixture.",
            "Guaranteed cure: pour mustard oil or kerosene into the animal mouth and puncture the swelling if it does not improve. Avoid veterinarian costs.",
        ),
        CptSource(
            "commercial_tonic_spam",
            "Best Herbal Tonic for More Milk",
            "Commercial supplement seller",
            "Unknown",
            "https://example.invalid/milk-tonic-buy-now",
            "commercial",
            ["milk production", "supplement"],
            ["cattle", "buffalo"],
            "English",
            "Reject fixture.",
            "Buy now with promo code. This miracle tonic gives 100% cure for mastitis and increases milk fast. Subscribe now for limited offer.",
        ),
        CptSource(
            "quarantine_dose_table_fixture",
            "Dairy Cattle Treatment Table",
            "Veterinary training notes fixture",
            "International",
            "https://example.invalid/dose-table",
            "training_notes",
            ["treatment", "medicine"],
            ["cattle"],
            "English",
            "Quarantine fixture.",
            "Mastitis may require veterinary treatment and supportive hygiene. Dose table: antibiotic 10 mg/kg intramuscular injection every 24 hours. Withdrawal period varies by product.",
        ),
    ]


def classify_source(source: CptSource) -> CptSource:
    text = source.raw_text
    if any(pattern.search(text) for pattern in HARD_REJECT_PATTERNS):
        return _replace_state(source, REJECT, "hard_reject_pattern")
    if any(pattern.search(text) for pattern in LOW_QUALITY_PATTERNS):
        return _replace_state(source, REJECT, "low_quality_text")
    risky = _extract_risky_spans(text)
    if risky:
        return _replace_state(source, QUARANTINE, "actionable_clinical_or_procedure_span", risky)
    if source.source_tier in {"unsafe_blog", "commercial"}:
        return _replace_state(source, REJECT, "untrusted_source_tier")
    return _replace_state(source, ACCEPT, "clean_cpt_source")


def _replace_state(source: CptSource, state: str, reason: str, removed: list[str] | None = None) -> CptSource:
    return CptSource(
        **{
            **source.__dict__,
            "state": state,
            "state_reason": reason,
            "removed_risky_spans": removed or [],
        }
    )


def _extract_risky_spans(text: str) -> list[str]:
    spans: list[str] = []
    for sentence in _sentences(text):
        if any(pattern.search(sentence) for pattern in RISKY_SPAN_PATTERNS):
            spans.append(sentence)
    return spans


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def clean_text_for_cpt(source: CptSource) -> tuple[str, list[str]]:
    removed: list[str] = []
    lines = []
    for line in source.raw_text.splitlines():
        stripped = line.strip()
        if not stripped or any(pattern.search(stripped) for pattern in BOILERPLATE_PATTERNS):
            continue
        lines.append(stripped)
    text = " ".join(lines)
    safe_sentences = []
    for sentence in _sentences(text):
        if any(pattern.search(sentence) for pattern in RISKY_SPAN_PATTERNS):
            removed.append(sentence)
            continue
        if any(pattern.search(sentence) for pattern in HARD_REJECT_PATTERNS):
            removed.append(sentence)
            continue
        safe_sentences.append(sentence)
    cleaned = re.sub(r"\s+", " ", " ".join(safe_sentences)).strip()
    return cleaned, removed


def build_cpt_corpus(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    classified = [classify_source(source) for source in _catalog()]

    candidate_rows = [_source_row(source) for source in classified]
    accepted_rows = [_source_row(source) for source in classified if source.state == ACCEPT]
    quarantined_rows = [_source_row(source) for source in classified if source.state == QUARANTINE]
    rejected_rows = [_source_row(source) for source in classified if source.state == REJECT]

    chunks = []
    filter_events = []
    for source in classified:
        if source.state == REJECT:
            continue
        cleaned, removed = clean_text_for_cpt(source)
        if not cleaned:
            continue
        if source.state == QUARANTINE and removed:
            filter_events.append(
                {
                    "source_id": source.source_id,
                    "removed_span_count": len(removed),
                    "removed_spans": removed,
                }
            )
        chunk = {
            "chunk_id": f"{source.source_id}_chunk_001",
            "source_id": source.source_id,
            "text": cleaned,
            "token_estimate": max(1, len(cleaned.split())),
            "state": source.state,
            "allowed_for": ["cpt_dapt"],
            "not_allowed_for": ["sft_factual_grounding"] if source.state == QUARANTINE or source.source_tier in {"secondary_context", "international_extension", "international_standard", "reputable_veterinary_reference", "university_extension"} else [],
            "metadata": source.metadata(),
            "content_hash": _sha256_text(cleaned),
        }
        chunks.append(chunk)

    _write_jsonl(out_dir / "candidate_cpt_sources.jsonl", candidate_rows)
    _write_jsonl(out_dir / "accepted_cpt_sources.jsonl", accepted_rows)
    _write_jsonl(out_dir / "quarantined_cpt_sources.jsonl", quarantined_rows)
    _write_jsonl(out_dir / "rejected_cpt_sources.jsonl", rejected_rows)
    _write_jsonl(out_dir / "cpt_clean_chunks.jsonl", chunks)

    quality_report = _quality_report(classified, chunks)
    safety_report = _safety_report(classified, filter_events)
    _write_json(out_dir / "cpt_source_quality_report.json", quality_report)
    _write_json(out_dir / "cpt_safety_filter_report.json", safety_report)
    manifest = _manifest(out_dir, classified, chunks)
    _write_json(out_dir / "cpt_corpus_manifest.json", manifest)
    return manifest


def _source_row(source: CptSource) -> dict[str, Any]:
    row = source.metadata()
    row.update(
        {
            "state": source.state,
            "state_reason": source.state_reason,
            "removed_risky_spans": source.removed_risky_spans,
            "raw_text_hash": _sha256_text(source.raw_text),
            "text_preview": source.raw_text[:240],
        }
    )
    return row


def _quality_report(sources: list[CptSource], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    states = Counter(source.state for source in sources)
    tiers = Counter(source.source_tier for source in sources)
    countries = Counter(source.country for source in sources)
    chunk_tokens = sum(chunk["token_estimate"] for chunk in chunks)
    errors = []
    if states[ACCEPT] < 12:
        errors.append("expected_at_least_12_accepted_sources")
    if not any(source.country == "India" and source.state == ACCEPT for source in sources):
        errors.append("missing_india_anchor_sources")
    if not any(source.country == "International" and source.state == ACCEPT for source in sources):
        errors.append("missing_international_sources")
    return {
        "valid": not errors,
        "errors": errors,
        "source_state_counts": dict(states),
        "source_tier_counts": dict(tiers),
        "country_counts": dict(countries),
        "chunk_count": len(chunks),
        "estimated_tokens": chunk_tokens,
    }


def _safety_report(sources: list[CptSource], filter_events: list[dict[str, Any]]) -> dict[str, Any]:
    rejected_reasons = Counter(source.state_reason for source in sources if source.state == REJECT)
    quarantined_reasons = Counter(source.state_reason for source in sources if source.state == QUARANTINE)
    errors = []
    for source in sources:
        if source.source_tier in {"unsafe_blog", "commercial"} and source.state != REJECT:
            errors.append(f"{source.source_id} should be rejected")
    return {
        "valid": not errors,
        "errors": errors,
        "hard_filter_policy": {
            "reject": ["miracle cures", "unsafe folk remedies", "anti-vaccine claims", "commercial spam", "bad OCR"],
            "quarantine_or_strip": ["raw doses", "injection routes", "procedure steps", "withdrawal tables", "pesticide instructions"],
            "allow": ["broad husbandry", "nutrition", "housing", "symptom descriptions", "veterinarian-supervised medicine mentions"],
        },
        "rejected_reason_counts": dict(rejected_reasons),
        "quarantined_reason_counts": dict(quarantined_reasons),
        "risky_span_filter_events": filter_events,
    }


def _manifest(out_dir: Path, sources: list[CptSource], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    states = Counter(source.state for source in sources)
    artifacts = [
        "candidate_cpt_sources.jsonl",
        "accepted_cpt_sources.jsonl",
        "quarantined_cpt_sources.jsonl",
        "rejected_cpt_sources.jsonl",
        "cpt_clean_chunks.jsonl",
        "cpt_source_quality_report.json",
        "cpt_safety_filter_report.json",
    ]
    artifact_hashes = {}
    for artifact in artifacts:
        path = out_dir / artifact
        if path.exists():
            artifact_hashes[artifact] = _sha256_bytes(path.read_bytes())
    return {
        "corpus_version": CPT_CORPUS_VERSION,
        "created_date": date.today().isoformat(),
        "status": "CPT_CORPUS_CANDIDATE",
        "sft_allowed": False,
        "rag_grounding_allowed": False,
        "source_count": len(sources),
        "source_state_counts": dict(states),
        "chunk_count": len(chunks),
        "estimated_tokens": sum(chunk["token_estimate"] for chunk in chunks),
        "artifact_hashes": artifact_hashes,
        "notes": [
            "Broad CPT corpus candidate; not a factual SFT/RAG source registry.",
            "Quarantined chunks may be used for CPT only after risky spans are stripped.",
            "Safety-critical farmer advice must continue to use stricter SFT/RAG grounding.",
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
