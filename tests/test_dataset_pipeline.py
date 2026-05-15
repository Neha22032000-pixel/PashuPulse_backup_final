from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pashu_saathi_dataset.generator import build_dataset, build_full_expansion_dataset, build_pilot_dataset
from pashu_saathi_dataset.eval_readiness import run_eval_readiness
from pashu_saathi_dataset.readiness import run_readiness_loop
from pashu_saathi_dataset.review import run_expansion_review
from pashu_saathi_dataset.validators import APPROVAL_ROLES, REPORT_FILES, row_content_hash, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test_runs"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_seed_bank_v4_has_300_seeds_zero_rows_and_reports():
    out_dir = TEST_ROOT / "v4_counts"
    manifest = build_dataset(out_dir)
    result = validate_dataset(out_dir)
    assert result["valid"], result["errors"]
    assert manifest["seed_bank_version"] == "seed-bank-v4"
    assert manifest["seed_count"] == 300
    assert manifest["seed_family_count"] == 100
    assert manifest["split_counts"] == {"dev_seed": 45, "final_eval_seed": 45, "train_seed": 210}
    assert manifest["row_counts"] == {"sft_train": 0, "sft_dev": 0, "final_eval": 0}
    for name in REPORT_FILES:
        assert (out_dir / name).exists(), name


def test_field_level_claim_roles_are_enforced():
    out_dir = TEST_ROOT / "v4_claim_roles"
    build_dataset(out_dir)
    seeds = read_jsonl(out_dir / "seed_cases.jsonl")
    claims = {row["claim_id"]: row for row in read_jsonl(out_dir / "source_claims.jsonl")}
    for seed in seeds:
        assert seed["care_claim_ids"]
        assert seed["escalation_claim_ids"]
        assert seed["policy_claim_ids"]
        for item in seed["safe_supportive_steps"] + seed["must_ask_checks"]:
            assert any(claims[claim_id]["claim_role"] == "care" for claim_id in item["claim_ids"])
            assert not all(claims[claim_id]["claim_role"] in {"policy", "escalation", "context"} for claim_id in item["claim_ids"])


def test_policy_only_escalation_only_and_mastitis_index_cannot_support_positive_care():
    out_dir = TEST_ROOT / "v4_bad_claim_roles"
    build_dataset(out_dir)
    seeds_path = out_dir / "seed_cases.jsonl"
    seeds = read_jsonl(seeds_path)
    seeds[0]["care_claim_ids"] = ["C_DAHD_MVU_ACCESS"]
    seeds[0]["safe_supportive_steps"][0]["claim_ids"] = ["C_DAHD_MVU_ACCESS"]
    seeds_path.write_text("\n".join(json.dumps(seed, ensure_ascii=False, sort_keys=True) for seed in seeds) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir)
    assert not result["valid"]
    assert any("positive care" in error or "topic/action" in error or "lacks compatible care claim" in error for error in result["errors"])


def test_split_stratification_scalar_counts_and_behavior_clusters():
    out_dir = TEST_ROOT / "v4_splits"
    build_dataset(out_dir)
    report = json.loads((out_dir / "split_stratification_report.json").read_text(encoding="utf-8"))
    coarse = json.loads((out_dir / "coarse_cluster_leakage_report.json").read_text(encoding="utf-8"))
    assert report["errors"] == []
    assert coarse["errors"] == []
    for split, count in report["split_counts"].items():
        assert sum(report["split_scalars"][split].values()) <= count * 4
    seeds = read_jsonl(out_dir / "seed_cases.jsonl")
    clusters: dict[str, set[str]] = {}
    for seed in seeds:
        clusters.setdefault(seed["behavior_cluster_id"], set()).add(seed["intended_split"])
    assert all(len(splits) == 1 for splits in clusters.values())


def test_family_variants_have_meaningful_axes_and_contract_differences():
    out_dir = TEST_ROOT / "v4_variant_diversity"
    build_dataset(out_dir)
    report = json.loads((out_dir / "variant_diversity_report.json").read_text(encoding="utf-8"))
    delta = json.loads((out_dir / "variant_contract_delta_report.json").read_text(encoding="utf-8"))
    assert report["errors"] == []
    assert delta["errors"] == []
    seeds = read_jsonl(out_dir / "seed_cases.jsonl")
    family = [seed for seed in seeds if seed["family_key"] == seeds[0]["family_key"]]
    assert len({seed["variant_axes"]["duration"] for seed in family}) == 3
    assert len({seed["variant_axes"]["resource_constraint"] for seed in family}) == 3


def test_prompt_quality_and_template_caps():
    out_dir = TEST_ROOT / "v4_language"
    build_dataset(out_dir)
    language_report = json.loads((out_dir / "language_quality_report.json").read_text(encoding="utf-8"))
    template_report = json.loads((out_dir / "template_similarity_report.json").read_text(encoding="utf-8"))
    prompt_contract = json.loads((out_dir / "prompt_contract_alignment_report.json").read_text(encoding="utf-8"))
    repetition = json.loads((out_dir / "repetition_caps_report.json").read_text(encoding="utf-8"))
    assert language_report["errors"] == []
    assert template_report["errors"] == []
    assert prompt_contract["errors"] == []
    assert repetition["errors"] == []
    prompts = "\n".join(seed["farmer_prompt"] for seed in read_jsonl(out_dir / "seed_cases.jsonl"))
    assert "Meri gai mein clean water" not in prompts
    assert "My buffalo has summer water breaks" not in prompts


def test_seed_bank_phase_rejects_any_nonempty_rows():
    out_dir = TEST_ROOT / "v4_seed_phase_rows"
    build_dataset(out_dir)
    (out_dir / "sft_train.jsonl").write_text(json.dumps({"row_id": "bad"}) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir, phase="seed_bank")
    assert not result["valid"]
    assert any("non-empty expanded rows" in error for error in result["errors"])


def test_approval_requires_external_files_and_stale_checksum_rejected():
    out_dir = TEST_ROOT / "v4_approval"
    if (out_dir / "approvals").exists():
        shutil.rmtree(out_dir / "approvals")
    build_dataset(out_dir)
    manifest_path = out_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["approval_state"] = "APPROVED_FOR_SEED_ONLY"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = validate_dataset(out_dir, require_approved_state="APPROVED_FOR_SEED_ONLY")
    assert not result["valid"]
    assert any("approval files" in error for error in result["errors"])
    approvals = out_dir / "approvals"
    approvals.mkdir(exist_ok=True)
    for role in APPROVAL_ROLES:
        (approvals / f"{role}_approval.json").write_text(
            json.dumps(
                {
                    "role": role,
                    "reviewer_id": f"{role}-1",
                    "reviewer_name": role.title(),
                    "timestamp": "2026-05-09T00:00:00Z",
                    "decision": "approved",
                    "notes": "test approval",
                    "role_rubric_result": {"passed": True},
                    "reviewed_artifact_hashes": manifest["checksums"],
                    "reviewed_report_hashes": {},
                    "unresolved_risks": [],
                    "validator_config_sha256": "stale",
                    "approval_bundle_sha256": "stale",
                }
            ),
            encoding="utf-8",
        )
    stale = validate_dataset(out_dir, require_approved_state="APPROVED_FOR_SEED_ONLY")
    assert not stale["valid"]
    assert any("stale approval checksum" in error or "stale validator config hash" in error for error in stale["errors"])


def test_approval_rejects_bad_decision_duplicate_reviewer_and_timestamp():
    out_dir = TEST_ROOT / "v4_bad_approval_files"
    build_dataset(out_dir)
    manifest_path = out_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["approval_state"] = "APPROVED_FOR_SEED_ONLY"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    approvals = out_dir / "approvals"
    approvals.mkdir(exist_ok=True)
    for role in APPROVAL_ROLES:
        (approvals / f"{role}_approval.json").write_text(
            json.dumps(
                {
                    "role": role,
                    "reviewer_id": "same-reviewer",
                    "reviewer_name": "Reviewer",
                    "timestamp": "not-a-date",
                    "decision": "rejected" if role == "source" else "approved",
                    "notes": "test approval",
                    "role_rubric_result": {"passed": role != "source"},
                    "reviewed_artifact_hashes": manifest["checksums"],
                    "reviewed_report_hashes": {},
                    "unresolved_risks": [],
                    "validator_config_sha256": "stale",
                    "approval_bundle_sha256": "stale",
                }
            ),
            encoding="utf-8",
        )
    result = validate_dataset(out_dir, require_approved_state="APPROVED_FOR_SEED_ONLY")
    assert not result["valid"]
    assert any("decision must be approved" in error for error in result["errors"])
    assert any("distinct reviewer IDs" in error for error in result["errors"])
    assert any("timestamp must be ISO" in error for error in result["errors"])


def test_export_hard_fails_even_with_smoke_flag():
    out_dir = TEST_ROOT / "v4_export_blocked"
    export_dir = TEST_ROOT / "export_should_not_exist"
    build_dataset(out_dir)
    for extra_args in ([], ["--allow-unapproved-smoke"]):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "prepare_gemma_sft_dataset.py"),
                "--dataset-dir",
                str(out_dir),
                "--out-dir",
                str(export_dir),
                *extra_args,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "BLOCKED" in (result.stderr + result.stdout)


def test_bad_case_gate_and_stale_exports_are_reported():
    out_dir = TEST_ROOT / "v4_bad_cases"
    build_dataset(out_dir)
    bad_case_report = json.loads((out_dir / "bad_case_gate_report.json").read_text(encoding="utf-8"))
    inventory_report = json.loads((out_dir / "artifact_inventory_report.json").read_text(encoding="utf-8"))
    assert bad_case_report["errors"] == []
    assert "bad_shop_injection" in bad_case_report["case_ids"]
    assert isinstance(inventory_report["stale_exports"], list)


def test_live_stale_export_jsonl_fails_until_physically_quarantined():
    out_dir = TEST_ROOT / "v4_live_stale_export"
    live = PROJECT_ROOT / "exports" / "live_stale_test"
    if live.exists():
        shutil.rmtree(live)
    quarantined = PROJECT_ROOT / "exports" / "DO_NOT_USE" / "live_stale_test"
    if quarantined.exists():
        shutil.rmtree(quarantined)
    build_dataset(out_dir)
    live.mkdir(parents=True)
    (live / "train.jsonl").write_text(json.dumps({"row_id": "bad"}) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir)
    assert not result["valid"]
    assert any("live stale export" in error for error in result["errors"])
    shutil.move(str(live), str(quarantined))
    clean = validate_dataset(out_dir)
    assert clean["valid"], clean["errors"]


def test_prompt_contract_source_and_species_gates_catch_mutations():
    out_dir = TEST_ROOT / "v4_mutation_gates"
    build_dataset(out_dir)
    seeds_path = out_dir / "seed_cases.jsonl"
    seeds = read_jsonl(seeds_path)
    seeds[0]["farmer_prompt"] = "My cow is calving. Is this bloat?"
    seeds[1]["care_claim_ids"] = ["C_FAO_EMERGENCY_TOPICS"]
    seeds[1]["safe_supportive_steps"][0]["claim_ids"] = ["C_FAO_EMERGENCY_TOPICS"]
    milk_seed = next(seed for seed in seeds if "milk safety" in seed["tags"])
    milk_seed["species"] = "ox"
    seeds_path.write_text("\n".join(json.dumps(seed, ensure_ascii=False, sort_keys=True) for seed in seeds) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir)
    assert not result["valid"]
    assert any("prompt" in error for error in result["errors"])
    assert any("topic/action" in error or "fallback" in error for error in result["errors"])
    assert any("milk safety seed" in error for error in result["errors"])


def test_pilot_expansion_has_300_rows_with_required_provenance():
    out_dir = TEST_ROOT / "pilot_counts"
    manifest = build_pilot_dataset(out_dir)
    result = validate_dataset(out_dir, phase="expansion_candidate")
    assert result["valid"], result["errors"]
    assert manifest["dataset_name"] == "pashu-saathi-pilot-expansion-v1"
    assert manifest["approval_state"] == "BLOCKED_PENDING_PILOT_APPROVAL"
    assert manifest["row_counts"] == {"sft_train": 210, "sft_dev": 45, "final_eval": 45}
    assert manifest["sft_allowed"] is False
    for name in [
        "expansion_provenance_report.json",
        "row_contract_fidelity_report.json",
        "answer_safety_drift_report.json",
        "escalation_calibration_report.json",
        "expansion_split_leakage_report.json",
        "expansion_pattern_collapse_report.json",
        "hinglish_naturalness_report.json",
        "pilot_review_queue.csv",
        "pilot_approval_audit_report.json",
        "scale_readiness_report.json",
    ]:
        assert (out_dir / name).exists(), name
    row = read_jsonl(out_dir / "sft_train.jsonl")[0]
    assert row["content_hash"] == row_content_hash(row)
    assert row["parent_seed_split"] == "train_seed"
    assert row["parent_seed_cases_sha256"] == manifest["checksums"]["seed_cases_sha256"]
    assert row["parent_source_claims_sha256"] == manifest["checksums"]["source_claims_sha256"]


def test_seed_bank_phase_rejects_pilot_rows_but_expansion_phase_allows_them():
    out_dir = TEST_ROOT / "pilot_phase_gate"
    build_pilot_dataset(out_dir)
    seed_phase = validate_dataset(out_dir, phase="seed_bank")
    assert not seed_phase["valid"]
    assert any("seed_bank phase rejects" in error for error in seed_phase["errors"])
    expansion_phase = validate_dataset(out_dir, phase="expansion_candidate")
    assert expansion_phase["valid"], expansion_phase["errors"]


def test_pilot_parent_split_and_content_hash_are_enforced():
    out_dir = TEST_ROOT / "pilot_bad_provenance"
    build_pilot_dataset(out_dir)
    train_path = out_dir / "sft_train.jsonl"
    rows = read_jsonl(train_path)
    rows[0]["parent_seed_split"] = "dev_seed"
    rows[1]["messages"][1]["content"] += " changed"
    train_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir, phase="expansion_candidate")
    assert not result["valid"]
    assert any("cross-split expansion lineage" in error for error in result["errors"])
    assert any("content hash mismatch" in error for error in result["errors"])


def test_pilot_contract_fidelity_rejects_new_claims_and_shape_drift():
    out_dir = TEST_ROOT / "pilot_bad_contract"
    build_pilot_dataset(out_dir)
    train_path = out_dir / "sft_train.jsonl"
    rows = read_jsonl(train_path)
    rows[0]["care_claim_ids"] = ["C_DAHD_MVU_ACCESS"]
    rows[0]["answer_sections"][0]["claim_ids"] = ["C_DAHD_MVU_ACCESS"]
    rows[1]["answer_shape"] = "made_up_shape"
    for row in rows[:2]:
        row["content_hash"] = row_content_hash(row)
    train_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir, phase="expansion_candidate")
    assert not result["valid"]
    assert any("claim inheritance mismatch" in error for error in result["errors"])
    assert any("positive answer section lacks care claim" in error for error in result["errors"])
    assert any("risk/answer_shape drift" in error for error in result["errors"])


def test_pilot_safety_drift_mutations_are_blocked():
    out_dir = TEST_ROOT / "pilot_safety_drift"
    build_pilot_dataset(out_dir)
    train_path = out_dir / "sft_train.jsonl"
    rows = read_jsonl(train_path)
    bad_phrases = [
        " Give antibiotic injection 5 ml twice daily.",
        " Put turmeric and chilli inside the wound.",
        " Give mustard oil drench for bloat.",
        " Pull hard during calving.",
        " Force-feed the weak calf.",
        " Sell abnormal milk if money is needed.",
        " This is definitely FMD disease.",
    ]
    for row, phrase in zip(rows, bad_phrases):
        row["messages"][1]["content"] += phrase
        row["answer_sections"].append({"section_type": "safe_step", "text": phrase, "claim_ids": row["care_claim_ids"]})
        row["content_hash"] = row_content_hash(row)
    train_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir, phase="expansion_candidate")
    assert not result["valid"]
    assert any("unsafe advice drift" in error for error in result["errors"])


def test_pilot_leakage_and_escalation_calibration_are_blocked():
    out_dir = TEST_ROOT / "pilot_leakage_calibration"
    build_pilot_dataset(out_dir)
    train_path = out_dir / "sft_train.jsonl"
    dev_path = out_dir / "sft_dev.jsonl"
    final_path = out_dir / "final_eval.jsonl"
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    final = read_jsonl(final_path)
    dev[0]["messages"][0]["content"] = train[0]["messages"][0]["content"]
    dev[0]["content_hash"] = row_content_hash(dev[0])
    red = next(row for row in final if row["risk_level"] == "red")
    red["messages"][1]["content"] = "Keep calm and watch."
    red["answer_sections"] = [{"section_type": "safe_step", "text": "Keep calm and watch.", "claim_ids": red["care_claim_ids"]}]
    red["content_hash"] = row_content_hash(red)
    dev_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in dev) + "\n", encoding="utf-8")
    final_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in final) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir, phase="expansion_candidate")
    assert not result["valid"]
    assert any("cross-split prompt leakage" in error for error in result["errors"])
    assert any("red row lacks trained-help escalation" in error for error in result["errors"])


def test_full_expansion_has_1500_rows_with_five_variants_per_seed():
    out_dir = TEST_ROOT / "full_counts"
    manifest = build_full_expansion_dataset(out_dir)
    result = validate_dataset(out_dir, phase="full_expansion_candidate")
    assert result["valid"], result["errors"]
    assert manifest["dataset_name"] == "pashu-saathi-full-expansion-v1"
    assert manifest["approval_state"] == "BLOCKED_PENDING_EXPANSION_REVIEW"
    assert manifest["row_counts"] == {"sft_train": 1050, "sft_dev": 225, "final_eval": 225}
    assert manifest["sft_allowed"] is False
    assert manifest["expansion_allowed"] is False

    rows = read_jsonl(out_dir / "sft_train.jsonl") + read_jsonl(out_dir / "sft_dev.jsonl") + read_jsonl(out_dir / "final_eval.jsonl")
    assert len(rows) == 1500
    assert Counter(row["parent_seed_id"] for row in rows).most_common(1)[0][1] == 5
    assert set(Counter(row["parent_seed_id"] for row in rows).values()) == {5}
    assert {row["expansion_variant_index"] for row in rows} == {0, 1, 2, 3, 4}
    for row in rows:
        assert row["content_hash"] == row_content_hash(row)
        assert row["generator_version"] == "full-expansion-v1"
        assert row["generation_config"]["rows_per_seed"] == 5
        assert row["parent_seed_cases_sha256"] == manifest["checksums"]["seed_cases_sha256"]
        assert row["parent_source_claims_sha256"] == manifest["checksums"]["source_claims_sha256"]


def test_full_expansion_phase_gates_block_pilot_seed_and_sft_paths():
    out_dir = TEST_ROOT / "full_phase_gate"
    export_dir = TEST_ROOT / "full_export_should_not_exist"
    build_full_expansion_dataset(out_dir)

    seed_phase = validate_dataset(out_dir, phase="seed_bank")
    assert not seed_phase["valid"]
    assert any("seed_bank phase rejects" in error for error in seed_phase["errors"])

    pilot_phase = validate_dataset(out_dir, phase="expansion_candidate")
    assert not pilot_phase["valid"]
    assert any("300-row pilot" in error for error in pilot_phase["errors"])

    full_phase = validate_dataset(out_dir, phase="full_expansion_candidate")
    assert full_phase["valid"], full_phase["errors"]

    for extra_args in ([], ["--allow-unapproved-smoke"]):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "prepare_gemma_sft_dataset.py"),
                "--dataset-dir",
                str(out_dir),
                "--out-dir",
                str(export_dir),
                *extra_args,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "BLOCKED" in (result.stderr + result.stdout)


def test_full_expansion_same_seed_variants_are_unique_and_distribution_is_capped():
    out_dir = TEST_ROOT / "full_diversity"
    build_full_expansion_dataset(out_dir)
    rows = read_jsonl(out_dir / "sft_train.jsonl") + read_jsonl(out_dir / "sft_dev.jsonl") + read_jsonl(out_dir / "final_eval.jsonl")
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[row["parent_seed_id"]].append(row)
    for seed_rows in by_seed.values():
        assert len({row["messages"][0]["content"] for row in seed_rows}) == 5
        assert len({row["answer_template_id"] for row in seed_rows}) == 5
        assert len({row["messages"][1]["content"].split(".")[0] for row in seed_rows}) == 5
        assert len({row["messages"][1]["content"].split(".")[-2].strip() for row in seed_rows}) == 5

    report = json.loads((out_dir / "expansion_pattern_collapse_report.json").read_text(encoding="utf-8"))
    assert report["errors"] == []
    assert report["top_structures"][0][1] / report["row_count"] <= 0.18
    assert report["top_openings"][0][1] / report["row_count"] <= 0.12
    assert report["top_closings"][0][1] / report["row_count"] <= 0.12


def test_full_expansion_mutation_gates_block_unsafe_drift_and_split_leakage():
    out_dir = TEST_ROOT / "full_mutation_gates"
    build_full_expansion_dataset(out_dir)
    train_path = out_dir / "sft_train.jsonl"
    dev_path = out_dir / "sft_dev.jsonl"
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    bad_phrases = [
        " Give antibiotic injection 5 ml twice daily.",
        " Give painkiller tablets from the local shop.",
        " Use dewormer dose by guessing weight.",
        " Give mustard oil drench for bloat.",
        " Put turmeric, chilli, and ash inside the wound.",
        " Pull the calf hard if calving is slow.",
        " Force-feed the weak calf.",
        " Sell abnormal milk if money is needed.",
        " Open the carcass to check what happened.",
        " Take the animal to market even with mouth and hoof sores.",
        " The photo confirms this disease.",
    ]
    for row, phrase in zip(train, bad_phrases):
        row["messages"][1]["content"] += phrase
        row["answer_sections"].append({"section_type": "safe_step", "text": phrase, "claim_ids": row["care_claim_ids"]})
        row["content_hash"] = row_content_hash(row)
    dev[0]["parent_seed_split"] = "train_seed"
    dev[0]["content_hash"] = row_content_hash(dev[0])
    train_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in train) + "\n", encoding="utf-8")
    dev_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in dev) + "\n", encoding="utf-8")
    result = validate_dataset(out_dir, phase="full_expansion_candidate")
    assert not result["valid"]
    assert any("unsafe advice drift" in error for error in result["errors"])
    assert any("cross-split expansion lineage" in error for error in result["errors"])


def test_hardened_review_workflow_creates_checksum_bound_artifacts_and_keeps_sft_blocked():
    out_dir = TEST_ROOT / "full_review_workflow"
    build_full_expansion_dataset(out_dir)
    decision = run_expansion_review(out_dir, reviewer_id="reviewer-1", reviewer_name="Review One")

    for name in [
        "expansion_manifest.json",
        "review_sample_manifest.json",
        "claim_span_review.jsonl",
        "safety_review_report.json",
        "source_fidelity_review_report.json",
        "language_usefulness_review_report.json",
        "diversity_review_report.json",
        "adversarial_audit_prompts.jsonl",
        "expansion_review_decision.json",
    ]:
        assert (out_dir / name).exists(), name

    sample = json.loads((out_dir / "review_sample_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert sample["checksums"]["sft_train_sha256"] == manifest["checksums"]["sft_train_sha256"]
    assert sample["checksums"]["source_claims_sha256"] == manifest["checksums"]["source_claims_sha256"]
    assert sample["review_required_rows"] == 1500
    assert sample["sample_meta"]["review_random_seed"] == 76044
    assert len({item["parent_seed_id"] for item in sample["reviewed_row_ids"]}) == 300

    spans = read_jsonl(out_dir / "claim_span_review.jsonl")
    assert spans
    assert {"allowed_use", "banned_use", "evidence_span_ids", "source_claim_snapshot_hashes"} <= set(spans[0])

    probes = read_jsonl(out_dir / "adversarial_audit_prompts.jsonl")
    assert probes
    assert {"must_refuse", "must_escalate", "must_not_include", "source_seed_ids", "blocking_if_failed"} <= set(probes[0])

    assert decision["decision"] in {"repair_required", "approved_for_pilot_eval_only"}
    assert decision["sft_allowed"] is False
    unchanged = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert unchanged["approval_state"] == "BLOCKED_PENDING_EXPANSION_REVIEW"
    assert unchanged["sft_allowed"] is False


def test_hardened_review_forces_repair_on_missing_reviewer_and_unsafe_mutations():
    out_dir = TEST_ROOT / "full_review_mutation"
    build_full_expansion_dataset(out_dir)
    train_path = out_dir / "sft_train.jsonl"
    rows = read_jsonl(train_path)
    rows[0]["messages"][1]["content"] += " Give antibiotic injection 5 ml now."
    rows[0]["answer_sections"].append({"section_type": "safe_step", "text": "Give antibiotic injection 5 ml now.", "claim_ids": rows[0]["care_claim_ids"]})
    rows[0]["content_hash"] = row_content_hash(rows[0])
    rows[1]["answer_sections"][0]["claim_ids"] = rows[1]["policy_claim_ids"]
    rows[1]["content_hash"] = row_content_hash(rows[1])
    train_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    decision = run_expansion_review(out_dir)
    assert decision["decision"] == "repair_required"
    assert "positive decision requires named reviewer_id" in decision["metadata_errors"][0]

    safety = json.loads((out_dir / "safety_review_report.json").read_text(encoding="utf-8"))
    source = json.loads((out_dir / "source_fidelity_review_report.json").read_text(encoding="utf-8"))
    assert any(item["category"] == "medicine_dose_injection" for item in safety["failures"])
    assert any(item["category"] == "unsupported_positive_care" for item in source["failures"])


def test_readiness_loop_repairs_known_blockers_and_stops_on_real_backend_blocker():
    out_dir = TEST_ROOT / "readiness_loop_workspace"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    result = run_readiness_loop(out_dir, max_iterations=3, reviewer_id="loop-reviewer", reviewer_name="Loop Reviewer")

    assert result["status"] == "blocked_by_eval_runtime_unavailable"
    assert result["ready_for_sft_planning"] is False
    assert result["sft_allowed"] is False

    iteration_dir = Path(result["final_iteration_dir"])
    assert (iteration_dir / "iteration_manifest.json").exists()
    assert (iteration_dir / "repair_manifest.jsonl").exists()
    assert (iteration_dir / "repair_diff_report.json").exists()
    assert (iteration_dir / "provenance_diff_report.json").exists()
    assert (iteration_dir / "blocker_trend_report.json").exists()
    assert (iteration_dir / "known_regression_report.json").exists()
    assert (iteration_dir / "iteration_decision.json").exists()

    decision = json.loads((iteration_dir / "iteration_decision.json").read_text(encoding="utf-8"))
    regression = json.loads((iteration_dir / "known_regression_report.json").read_text(encoding="utf-8"))
    review = json.loads((out_dir / "data" / "processed" / "full_expansion" / "expansion_review_decision.json").read_text(encoding="utf-8"))
    eval_manifest = json.loads((out_dir / "data" / "processed" / "pilot_eval_package" / "eval_manifest.json").read_text(encoding="utf-8"))
    contamination = json.loads((out_dir / "data" / "processed" / "pilot_eval_package" / "contamination_report.json").read_text(encoding="utf-8"))
    sft_decision = json.loads((out_dir / "data" / "processed" / "pilot_eval_package" / "sft_planning_readiness_decision.json").read_text(encoding="utf-8"))
    rubric_report = json.loads((out_dir / "data" / "processed" / "pilot_eval_package" / "eval_rubric_validation_report.json").read_text(encoding="utf-8"))
    runtime_blocker = json.loads((out_dir / "data" / "processed" / "pilot_eval_package" / "runtime_blocker_report.json").read_text(encoding="utf-8"))

    assert decision["status"] == "approved_for_pilot_eval_only"
    assert review["decision"] == "approved_for_pilot_eval_only"
    assert review["blocking_failure_count"] == 0
    assert regression["pass"] is True
    assert eval_manifest["status"] == "EVAL_ONLY_DO_NOT_TRAIN"
    assert eval_manifest["eval_prompt_count"] == 233
    assert eval_manifest["mock_eval_allowed"] is False
    assert eval_manifest["base_gemma_runnable"] is False
    assert contamination["train_rows_included"] == 0
    assert contamination["dev_rows_included"] == 0
    assert rubric_report["pass"] is True
    assert rubric_report["row_count"] == 233
    assert sft_decision["decision"] == "blocked_by_eval_runtime_unavailable"
    assert sft_decision["sft_allowed"] is False
    assert "reportable" in runtime_blocker["backend_result"]["error"]


def test_eval_readiness_rejects_mock_or_missing_reportable_predictions():
    out_dir = TEST_ROOT / "eval_readiness_runtime"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    build_full_expansion_dataset(out_dir / "data" / "processed" / "full_expansion")
    run_expansion_review(out_dir / "data" / "processed" / "full_expansion", reviewer_id="eval-reviewer", reviewer_name="Eval Reviewer")
    from pashu_saathi_dataset.readiness import build_eval_package

    build_eval_package(out_dir / "data" / "processed" / "full_expansion", out_dir / "data" / "processed" / "pilot_eval_package")
    decision = run_eval_readiness(out_dir)

    assert decision["decision"] == "blocked_by_eval_runtime_unavailable"
    assert decision["rubric_pass"] is True
    assert decision["contamination_guard_pass"] is True
    score = json.loads((out_dir / "data" / "processed" / "pilot_eval_package" / "baseline_score_report.json").read_text(encoding="utf-8"))
    rubric_report = json.loads((out_dir / "data" / "processed" / "pilot_eval_package" / "eval_rubric_validation_report.json").read_text(encoding="utf-8"))
    assert rubric_report["row_count"] == 233
    assert any(item["type"] == "no_predictions" for item in score["hard_failures"])


def test_kaggle_scripts_are_reportable_and_metadata_is_pashu_scoped():
    eval_script = (PROJECT_ROOT / "kaggle_gemma_eval" / "gemma_lora_eval.py").read_text(encoding="utf-8")
    sft_script = (PROJECT_ROOT / "kaggle_gemma_sft" / "gemma_lora_sft.py").read_text(encoding="utf-8")
    eval_meta = json.loads((PROJECT_ROOT / "kaggle_gemma_eval" / "kernel-metadata.json").read_text(encoding="utf-8"))
    sft_meta = json.loads((PROJECT_ROOT / "kaggle_gemma_sft" / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert "--allow-mock" not in eval_script
    assert "backend\": \"mock" not in eval_script
    assert "reportable" in eval_script
    assert "completed_steps" in eval_script and "remaining_steps" in eval_script
    assert "train_eval_curve.jsonl" in sft_script
    assert "eval_batches_left" in sft_script
    assert "steps_left" in sft_script
    assert "FastLanguageModel.from_pretrained" in sft_script
    assert "train_on_responses_only" in sft_script
    assert "unsloth==2026.5.2" in sft_script
    assert "language_attention_lora_targets" in sft_script
    assert "vision_tower" in sft_script and "audio_tower" in sft_script
    assert eval_meta["id"].startswith("nehak76044/pashu-saathi")
    assert sft_meta["id"].startswith("nehak76044/pashu-saathi")
    assert eval_meta["code_file"] == "gemma_lora_eval.py"
    assert sft_meta["code_file"] == "gemma_lora_sft.py"
    assert eval_meta["enable_gpu"] is True
    assert sft_meta["enable_gpu"] is True


def test_kaggle_eval_package_contains_eval_only_artifacts():
    out_dir = TEST_ROOT / "kaggle_eval_package_workspace"
    package_dir = TEST_ROOT / "kaggle_eval_package"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    out_dir.mkdir(parents=True)
    build_full_expansion_dataset(out_dir / "data" / "processed" / "full_expansion")
    run_expansion_review(out_dir / "data" / "processed" / "full_expansion", reviewer_id="pkg-reviewer", reviewer_name="Package Reviewer")
    from pashu_saathi_dataset.readiness import build_eval_package

    build_eval_package(out_dir / "data" / "processed" / "full_expansion", out_dir / "data" / "processed" / "pilot_eval_package")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_kaggle_eval_package.py"),
            "--workspace-dir",
            str(out_dir),
            "--out-dir",
            str(package_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (package_dir / "eval_rubric.jsonl").exists()
    assert not (package_dir / "sft_train.jsonl").exists()
    manifest = json.loads((package_dir / "kaggle_eval_package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "EVAL_ONLY_DO_NOT_TRAIN"
    assert manifest["mock_allowed"] is False


def test_kaggle_sft_package_requires_ready_decision_and_rejects_eval_leakage():
    out_dir = TEST_ROOT / "kaggle_sft_package_workspace"
    package_dir = TEST_ROOT / "kaggle_sft_package"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    out_dir.mkdir(parents=True)
    full_dir = out_dir / "data" / "processed" / "full_expansion"
    eval_dir = out_dir / "data" / "processed" / "pilot_eval_package"
    build_full_expansion_dataset(full_dir)
    run_expansion_review(full_dir, reviewer_id="sft-pkg-reviewer", reviewer_name="SFT Package Reviewer")
    eval_dir.mkdir(parents=True)
    (eval_dir / "sft_planning_readiness_decision.json").write_text(
        json.dumps({"decision": "ready_for_sft_planning", "sft_allowed": False}, indent=2),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_kaggle_sft_package.py"),
            "--workspace-dir",
            str(out_dir),
            "--dataset-dir",
            str(full_dir),
            "--out-dir",
            str(package_dir),
            "--mode",
            "smoke",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (package_dir / "sft_train.jsonl").exists()
    assert (package_dir / "sft_dev.jsonl").exists()
    assert not (package_dir / "final_eval.jsonl").exists()
    assert not (package_dir / "eval_rubric.jsonl").exists()
    manifest = json.loads((package_dir / "sft_package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_mode"] == "smoke"
    assert manifest["row_counts"] == {"sft_train": 24, "sft_dev": 8, "final_eval": 0}
    assert manifest["final_eval_included"] is False
    assert manifest["adversarial_included"] is False
    assert manifest["sft_allowed"] is False

    leaked_train = read_jsonl(package_dir / "sft_train.jsonl")
    leaked_train[0]["EVAL_ONLY_DO_NOT_TRAIN"] = True
    (package_dir / "sft_train.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in leaked_train) + "\n", encoding="utf-8")
    smoke = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "kaggle_gemma_sft" / "gemma_lora_sft_smoke.py"),
        ],
        cwd=str(PROJECT_ROOT / "kaggle_gemma_sft"),
        env={**os.environ, "PASHU_SAATHI_DATA_DIR": str(package_dir), "PASHU_SAATHI_OUT_DIR": str(TEST_ROOT / "kaggle_sft_smoke_failure")},
        capture_output=True,
        text=True,
    )
    assert smoke.returncode != 0
    assert "eval-only" in (smoke.stderr + smoke.stdout)


def test_full_sft_package_is_blocked_until_param_review():
    out_dir = TEST_ROOT / "kaggle_sft_full_review_workspace"
    package_dir = TEST_ROOT / "kaggle_sft_full_package"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    out_dir.mkdir(parents=True)
    full_dir = out_dir / "data" / "processed" / "full_expansion"
    eval_dir = out_dir / "data" / "processed" / "pilot_eval_package"
    build_full_expansion_dataset(full_dir)
    run_expansion_review(full_dir, reviewer_id="full-sft-reviewer", reviewer_name="Full SFT Reviewer")
    eval_dir.mkdir(parents=True)
    (eval_dir / "sft_planning_readiness_decision.json").write_text(
        json.dumps({"decision": "ready_for_sft_planning", "sft_allowed": False}, indent=2),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_kaggle_sft_package.py"),
            "--workspace-dir",
            str(out_dir),
            "--dataset-dir",
            str(full_dir),
            "--out-dir",
            str(package_dir),
            "--mode",
            "full",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads((package_dir / "sft_package_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((package_dir / "training_config.json").read_text(encoding="utf-8"))
    assert manifest["package_mode"] == "full"
    assert manifest["sft_allowed"] is False
    assert manifest["full_sft_launch_blocked_until_param_review"] is True
    assert config["num_train_epochs"] == 2
    assert config["eval_steps"] == 25
    assert (package_dir / "sft_param_review_request.json").exists()

    full_run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "kaggle_gemma_sft" / "gemma_lora_sft.py"),
        ],
        cwd=str(PROJECT_ROOT / "kaggle_gemma_sft"),
        env={**os.environ, "PASHU_SAATHI_DATA_DIR": str(package_dir), "PASHU_SAATHI_TRAIN_MODE": "full", "PASHU_SAATHI_OUT_DIR": str(TEST_ROOT / "kaggle_sft_full_blocked")},
        capture_output=True,
        text=True,
    )
    assert full_run.returncode != 0
    assert "cleaned_candidate" in (full_run.stderr + full_run.stdout)


def test_cleaned_sft_package_removes_language_residue_and_preserves_lineage():
    out_dir = TEST_ROOT / "sft_cleaned_candidate"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_cleaned_sft_package.py"),
            "--source-dir",
            str(PROJECT_ROOT / "kaggle_packages" / "sft_full_package"),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads((out_dir / "cleaning_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "sft_package_manifest.json").read_text(encoding="utf-8"))
    assert report["clean_package_passed"] is True
    assert report["error_counts"] == {}
    assert manifest["package_mode"] == "cleaned_candidate"
    assert manifest["status"] == "BLOCKED_PENDING_CLEAN_DATA_REVIEW"
    assert manifest["sft_allowed"] is False
    assert manifest["promotion_allowed"] is False

    train = read_jsonl(out_dir / "sft_train.jsonl")
    dev = read_jsonl(out_dir / "sft_dev.jsonl")
    assert len(train) == 1050
    assert len(dev) == 225
    residue = ["mudda", "field note", "close note", "review card", "water na drinking"]
    for row in train + dev:
        prompt = row["messages"][0]["content"].lower()
        answer = row["messages"][1]["content"].lower()
        assert not any(term in prompt or term in answer for term in residue), row["row_id"]
        assert row["messages"][0]["content"] == row["user_prompt"]
        assert row["messages"][1]["content"] == row["assistant_response"]
        assert row["parent_seed_split"] != "final_eval_seed"
        assert row.get("EVAL_ONLY_DO_NOT_TRAIN") is not True
        assert row["cleaning_state"] == "auto_approved"


def test_cleaned_sft_variants_are_diverse_and_old_packages_blocked():
    out_dir = TEST_ROOT / "sft_cleaned_candidate_diversity"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_cleaned_sft_package.py"),
            "--source-dir",
            str(PROJECT_ROOT / "kaggle_packages" / "sft_full_package"),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    diversity = json.loads((out_dir / "diversity_cleanup_report.json").read_text(encoding="utf-8"))
    assert diversity["valid"] is True
    assert diversity["dominant_opening_share"] < 0.25

    rows = read_jsonl(out_dir / "sft_train.jsonl") + read_jsonl(out_dir / "sft_dev.jsonl")
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[row["parent_seed_id"]].append(row)
    for seed_rows in by_seed.values():
        assert len({row["assistant_response"] for row in seed_rows}) == len(seed_rows)
        assert len({row["user_prompt"] for row in seed_rows}) == len(seed_rows)

    full_run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "kaggle_gemma_sft" / "gemma_lora_sft.py"),
        ],
        cwd=str(PROJECT_ROOT / "kaggle_gemma_sft"),
        env={**os.environ, "PASHU_SAATHI_DATA_DIR": str(out_dir), "PASHU_SAATHI_TRAIN_MODE": "full", "PASHU_SAATHI_OUT_DIR": str(TEST_ROOT / "cleaned_full_blocked_until_review")},
        capture_output=True,
        text=True,
    )
    assert full_run.returncode != 0
    assert "sft_param_review_decision.json" in (full_run.stderr + full_run.stdout)
