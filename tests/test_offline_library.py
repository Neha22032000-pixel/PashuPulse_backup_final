from __future__ import annotations

import re
from pathlib import Path

from pashu_saathi_dataset.offline_library import (
    ACCEPTED,
    ACCEPTED_STRIPPED,
    OfflineSourceSpec,
    _coverage_report,
    _risk_flags,
    _status_for_source,
    clean_source_text,
    source_catalog,
)


def test_offline_catalog_is_coverage_oriented():
    specs = source_catalog()
    assert len(specs) >= 50
    species = {item for spec in specs for item in spec.species}
    topics = {item for spec in specs for item in spec.topics}
    for required in ["cow", "buffalo", "ox/bullock", "calf"]:
        assert required in species
    for required in ["feeding", "water", "shed hygiene", "milk hygiene", "calf care", "bloat", "parasites", "working ox care"]:
        assert required in topics


def test_cleaning_strips_dose_injection_and_unsafe_remedy_spans():
    text = (
        "Clean shed and water are useful. "
        "Give antibiotic 10 mg/kg intramuscular injection every day. "
        "A miracle cure is kerosene for bloat. "
        "Observe appetite and dung."
    )
    clean, removed = clean_source_text(text)
    assert "Clean shed" in clean
    assert "Observe appetite" in clean
    assert "10 mg/kg" not in clean
    assert "intramuscular injection" not in clean
    assert "kerosene" not in clean
    assert len(removed) == 2


def test_status_accepts_stripped_high_trust_source():
    spec = OfflineSourceSpec(
        source_id="fixture",
        title="Fixture",
        publisher="University",
        url="https://example.org",
        country="International",
        source_tier="university_extension",
        language="en",
        species=["cow"],
        topics=["feeding"],
        geo_relevance=["International"],
        license_or_terms="fixture",
    )
    extracted = "Cattle feeding and water management are important. " * 60 + "Dose: antibiotic 10 mg/kg."
    clean, removed = clean_source_text(extracted)
    flags = _risk_flags(spec, extracted, clean, removed)
    status = _status_for_source(spec, extracted, clean, flags, removed)
    assert status == ACCEPTED_STRIPPED


def test_coverage_report_detects_required_species_and_topics():
    quality_rows = []
    for spec in source_catalog():
        quality_rows.append(
            {
                "status": ACCEPTED,
                "species": spec.species,
                "topics": spec.topics,
                "language": spec.language,
                "source_tier": spec.source_tier,
                "country": spec.country,
                "publisher": spec.publisher,
                "clean_tokens": 100,
            }
        )
    report = _coverage_report(quality_rows, [])
    assert not report["missing_species"]
    assert not report["missing_topics"]
    assert report["accepted_source_count"] >= 25


def test_hindi_mojibake_gate_flags_missing_devanagari():
    spec = OfflineSourceSpec(
        source_id="hindi_fixture",
        title="Hindi Fixture",
        publisher="Fixture",
        url="https://example.org",
        country="India",
        source_tier="official_extension_india",
        language="hi",
        species=["cow"],
        topics=["feeding"],
        geo_relevance=["India"],
        license_or_terms="fixture",
    )
    extracted = "This claims to be Hindi cattle feeding material but has no Devanagari. " * 20
    clean, removed = clean_source_text(extracted)
    flags = _risk_flags(spec, extracted, clean, removed)
    assert "hindi_source_without_devanagari_detected" in flags
