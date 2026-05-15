from __future__ import annotations

import json
from pathlib import Path

from pashu_saathi_dataset.cpt_research import REQUIRED_SPECIES, REQUIRED_TOPICS, build_cpt_research_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test_runs"
LIBRARY_DIR = PROJECT_ROOT / "data" / "sources" / "offline_library_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cpt_research_artifacts_build_without_training_approval():
    out_dir = TEST_ROOT / "cpt_research"
    manifest = build_cpt_research_artifacts(out_dir, LIBRARY_DIR)
    assert manifest["status"] == "CPT_DAPT_RESEARCH_ARTIFACTS_READY"
    assert manifest["training_allowed"] is False
    assert manifest["sft_allowed"] is False
    for name in [
        "cpt_corpus_audit_report.json",
        "cpt_split_manifest.json",
        "cpt_question_bank_seed.jsonl",
        "cpt_research_decision_template.json",
        "cpt_experiment_matrix.json",
        "cpt_research_manifest.json",
    ]:
        assert (out_dir / name).exists(), name


def test_split_manifest_is_source_isolated():
    out_dir = TEST_ROOT / "cpt_research_split"
    build_cpt_research_artifacts(out_dir, LIBRARY_DIR)
    split_manifest = json.loads((out_dir / "cpt_split_manifest.json").read_text(encoding="utf-8"))
    splits = split_manifest["splits"]
    train = set(splits["train"]["source_ids"])
    dev = set(splits["dev"]["source_ids"])
    test = set(splits["test"]["source_ids"])
    assert train
    assert dev
    assert test
    assert not train & dev
    assert not train & test
    assert not dev & test


def test_audit_preserves_required_coverage_and_flags_imbalance():
    out_dir = TEST_ROOT / "cpt_research_audit"
    build_cpt_research_artifacts(out_dir, LIBRARY_DIR)
    audit = json.loads((out_dir / "cpt_corpus_audit_report.json").read_text(encoding="utf-8"))
    assert audit["source_library"]["accepted_source_count"] >= 60
    assert audit["source_library"]["chunk_count"] >= 3000
    assert all(audit["required_species_present"][species] for species in REQUIRED_SPECIES)
    assert all(audit["required_topics_present"][topic] for topic in REQUIRED_TOPICS)
    assert audit["imbalance_flags"]


def test_question_bank_is_eval_only_and_uses_test_sources():
    out_dir = TEST_ROOT / "cpt_research_questions"
    build_cpt_research_artifacts(out_dir, LIBRARY_DIR)
    questions = read_jsonl(out_dir / "cpt_question_bank_seed.jsonl")
    split_manifest = json.loads((out_dir / "cpt_split_manifest.json").read_text(encoding="utf-8"))
    test_sources = set(split_manifest["splits"]["test"]["source_ids"])
    assert len(questions) >= 20
    assert {row["probe_type"] for row in questions} >= {"short_qa", "cloze", "mcq", "entailment", "contradiction"}
    assert all(row["eval_only"] is True for row in questions)
    assert all(row["source_id"] in test_sources for row in questions)
    assert all("medicine dose" in row["forbidden_unsafe_extrapolations"] for row in questions)


def test_experiment_matrix_blocks_param_and_launch_decisions():
    out_dir = TEST_ROOT / "cpt_research_matrix"
    build_cpt_research_artifacts(out_dir, LIBRARY_DIR)
    matrix = json.loads((out_dir / "cpt_experiment_matrix.json").read_text(encoding="utf-8"))
    decision = json.loads((out_dir / "cpt_research_decision_template.json").read_text(encoding="utf-8"))
    assert matrix["status"] == "RESEARCH_DESIGN_ONLY"
    assert matrix["training_params_defined"] is False
    assert "Kaggle launch" in matrix["blocked_until_research_complete"]
    assert "continue_to_cpt_pilot_design" in decision["allowed_decisions"]
    assert decision["status"] == "TEMPLATE_NOT_DECIDED"
