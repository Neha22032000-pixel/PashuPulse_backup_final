from __future__ import annotations

import json
from pathlib import Path

from pashu_saathi_dataset.cpt_corpus import build_cpt_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test_runs"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cpt_corpus_builds_required_artifacts():
    out_dir = TEST_ROOT / "cpt_corpus"
    manifest = build_cpt_corpus(out_dir)
    assert manifest["status"] == "CPT_CORPUS_CANDIDATE"
    assert manifest["sft_allowed"] is False
    assert manifest["rag_grounding_allowed"] is False
    for name in [
        "candidate_cpt_sources.jsonl",
        "accepted_cpt_sources.jsonl",
        "quarantined_cpt_sources.jsonl",
        "rejected_cpt_sources.jsonl",
        "cpt_clean_chunks.jsonl",
        "cpt_corpus_manifest.json",
        "cpt_source_quality_report.json",
        "cpt_safety_filter_report.json",
    ]:
        assert (out_dir / name).exists(), name


def test_cpt_source_states_keep_volume_but_reject_bad_sources():
    out_dir = TEST_ROOT / "cpt_states"
    build_cpt_corpus(out_dir)
    accepted = read_jsonl(out_dir / "accepted_cpt_sources.jsonl")
    quarantined = read_jsonl(out_dir / "quarantined_cpt_sources.jsonl")
    rejected = read_jsonl(out_dir / "rejected_cpt_sources.jsonl")
    assert len(accepted) >= 12
    assert any(row["country"] == "India" for row in accepted)
    assert any(row["country"] == "International" for row in accepted)
    assert any(row["source_id"] == "quarantine_dose_table_fixture" for row in quarantined)
    assert any(row["source_id"] == "unsafe_blog_mustard_oil_bloat" for row in rejected)
    assert any(row["source_id"] == "commercial_tonic_spam" for row in rejected)


def test_clean_chunks_strip_actionable_medicine_and_procedure_spans():
    out_dir = TEST_ROOT / "cpt_cleaning"
    build_cpt_corpus(out_dir)
    chunks = read_jsonl(out_dir / "cpt_clean_chunks.jsonl")
    all_text = "\n".join(chunk["text"].lower() for chunk in chunks)
    assert "10 mg/kg" not in all_text
    assert "intramuscular injection" not in all_text
    assert "mustard oil or kerosene" not in all_text
    assert "puncture the swelling" not in all_text
    assert "miracle tonic" not in all_text


def test_quarantined_cpt_chunks_cannot_be_used_for_sft_grounding():
    out_dir = TEST_ROOT / "cpt_sft_boundary"
    build_cpt_corpus(out_dir)
    chunks = read_jsonl(out_dir / "cpt_clean_chunks.jsonl")
    quarantined = [chunk for chunk in chunks if chunk["state"] == "quarantine"]
    assert quarantined
    assert all("sft_factual_grounding" in chunk["not_allowed_for"] for chunk in quarantined)


def test_cpt_reports_are_gate_valid():
    out_dir = TEST_ROOT / "cpt_reports"
    build_cpt_corpus(out_dir)
    quality = json.loads((out_dir / "cpt_source_quality_report.json").read_text(encoding="utf-8"))
    safety = json.loads((out_dir / "cpt_safety_filter_report.json").read_text(encoding="utf-8"))
    assert quality["valid"], quality["errors"]
    assert safety["valid"], safety["errors"]
    assert safety["rejected_reason_counts"]["hard_reject_pattern"] >= 2

