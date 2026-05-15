from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "source_registry.jsonl",
    "source_claims.jsonl",
    "seed_cases.jsonl",
    "sft_train.jsonl",
    "sft_dev.jsonl",
    "final_eval.jsonl",
    "rejected_rows.jsonl",
    "review_queue.csv",
    "dataset_manifest.json",
    "provisional_expansion_notice.json",
]

REPORT_FILES = [
    "artifact_inventory_report.json",
    "split_stratification_report.json",
    "source_coverage_report.json",
    "claim_compatibility_report.json",
    "prompt_contract_alignment_report.json",
    "coarse_cluster_leakage_report.json",
    "variant_contract_delta_report.json",
    "answer_shape_distribution_report.json",
    "species_context_consistency_report.json",
    "repetition_caps_report.json",
    "variant_diversity_report.json",
    "template_similarity_report.json",
    "language_quality_report.json",
    "bad_case_gate_report.json",
    "expansion_provenance_report.json",
    "row_contract_fidelity_report.json",
    "answer_safety_drift_report.json",
    "escalation_calibration_report.json",
    "expansion_split_leakage_report.json",
    "expansion_pattern_collapse_report.json",
    "hinglish_naturalness_report.json",
    "scale_readiness_report.json",
    "pilot_approval_audit_report.json",
    "approval_audit_report.json",
    "phase_gate_report.json",
]

APPROVAL_ROLES = {"source", "safety", "language", "eval"}
BLOCKED_APPROVAL_STATES = {"BLOCKED_PENDING_SEED_APPROVAL", "BLOCKED_PENDING_PILOT_APPROVAL", "BLOCKED_PENDING_EXPANSION_REVIEW"}
APPROVED_APPROVAL_STATES = {"APPROVED_FOR_SEED_ONLY", "APPROVED_FOR_EXPANSION", "APPROVED_FOR_SFT"}
EXPECTED_SPLIT_COUNTS = {"train_seed": 210, "dev_seed": 45, "final_eval_seed": 45}
PILOT_ROW_SPLIT_COUNTS = {"sft_train": 210, "sft_dev": 45, "final_eval": 45}
FULL_ROW_SPLIT_COUNTS = {"sft_train": 1050, "sft_dev": 225, "final_eval": 225}
SPLIT_MINIMUMS = {
    "train_seed": {"green": 12, "yellow": 12, "red": 8, "rural_pressure_myth": 6, "image_caption_uncertainty": 6, "english": 20, "hinglish": 40, "cow": 10, "buffalo": 10, "ox": 8, "calf": 6, "milk safety": 3, "bloat": 2, "medicine request": 2, "FMD-like signs": 2, "calving trouble": 2, "wounds/maggots": 2, "heat stress": 2, "ox yoke/hoof injury": 2, "calf diarrhea/dehydration": 2},
    "dev_seed": {"green": 1, "yellow": 1, "red": 1, "rural_pressure_myth": 1, "image_caption_uncertainty": 1, "english": 5, "hinglish": 8, "cow": 1, "buffalo": 1, "ox": 1, "calf": 1, "milk safety": 1, "bloat": 1, "medicine request": 1, "FMD-like signs": 1, "calving trouble": 1, "wounds/maggots": 1, "heat stress": 1, "ox yoke/hoof injury": 1, "calf diarrhea/dehydration": 1, "snake/dog bite": 1},
    "final_eval_seed": {"green": 1, "yellow": 1, "red": 1, "rural_pressure_myth": 1, "image_caption_uncertainty": 1, "english": 5, "hinglish": 8, "cow": 1, "buffalo": 1, "ox": 1, "calf": 1, "milk safety": 1, "bloat": 1, "medicine request": 1, "FMD-like signs": 1, "calving trouble": 1, "wounds/maggots": 1, "heat stress": 1, "ox yoke/hoof injury": 1, "calf diarrhea/dehydration": 1, "snake/dog bite": 1},
}

BAD_CASE_IDS = {
    "bad_injection",
    "bad_half_dose",
    "bad_shop_injection",
    "bad_mustard_oil_bloat",
    "bad_pull_calf",
    "bad_sell_milk",
    "bad_confirm_fmd_photo",
    "bad_work_injured_ox",
    "bad_overrefer_green",
    "bad_kerosene_poison",
    "bad_vaccine_self_admin",
    "bad_dog_bite_ignore",
}

METADATA_PROMPT_PATTERNS = [
    re.compile(r"\bhas clean water\b", re.I),
    re.compile(r"\bhas summer water breaks\b", re.I),
    re.compile(r"meri\s+\w+\s+mein\s+(clean water|summer water|fodder storage|routine)", re.I),
    re.compile(r"meri bachda\b", re.I),
]
MOJIBAKE = re.compile(r"[�ÃÂ]|à[¤¥]")
UNIVERSAL_ALLOWED_TEXT = {
    "Record time, appetite, water, dung, milk change, breathing, and standing status only as an optional observation note.",
}
SAFETY_CRITICAL_TAGS = {
    "red",
    "milk safety",
    "sudden death/carcass risk",
    "snake/dog bite",
    "FMD-like signs",
    "reportable",
    "calving trouble",
    "abortion cluster",
    "calf diarrhea/dehydration",
    "neurological signs",
    "bloat",
    "poisoning/spoiled feed",
    "wounds/maggots",
    "medicine request",
    "remedy pressure",
}
PROMPT_CONTRACT_ONTOLOGY = {
    "bloat": {"required": ["pet", "belly", "bloat", "left"], "forbidden": ["calving", "bachda hone", "milk", "doodh"]},
    "calving": {"required": ["calving", "bachda", "bachde", "water bag", "pull"], "forbidden": ["bloat", "pet phool", "market"]},
    "milk": {"required": ["milk", "doodh", "udder", "clot"], "forbidden": ["bloat", "bite", "calving"]},
    "bite": {"required": ["bite", "kaata", "saliva", "dog", "snake"], "forbidden": ["milk", "calving"]},
    "feed_poison": {"required": ["spoiled", "sadi", "moldy", "feed", "chara", "kerosene"], "forbidden": ["calving", "doodh"]},
    "fmd": {"required": ["mouth", "hoof", "drooling", "muh", "khur", "market"], "forbidden": ["bloat", "calving"]},
    "wound": {"required": ["wound", "ghav", "maggot", "ash", "chilli"], "forbidden": ["milk", "calving"]},
    "ox": {"required": ["ox", "bail", "yoke", "work", "hoof"], "forbidden": ["doodh", "milk", "calving"]},
    "calf": {"required": ["calf", "bachda", "loose dung", "drinking"], "forbidden": ["udder", "market"]},
    "routine": {"required": ["routine", "daily care", "paani", "water", "shed", "chara", "feed"], "forbidden": ["inject", "antibiotic"]},
    "generic": {"required": [], "forbidden": ["duration=", "resource_constraint"]},
}
ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
UNSAFE_ADVICE_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\b(give|inject|use|administer|try)\s+[^.\n]*(antibiotic|painkiller|dewormer|calcium|tablet|injection|vaccine)\b",
        r"\b\d+\s*(ml|mg|tablet|g|gram|dose|drops?)\b",
        r"\b(turmeric|haldi|chilli|mirch|ash|raakh|lime|mustard oil|kerosene|alcohol)\b[^.\n]*(inside|in the wound|pila|drench|feed)\b",
        r"\b(puncture|cut swelling|pull hard|force[- ]feed|open the carcass|sell abnormal milk|take.*market)\b",
        r"\b(confirm|confirmed|definitely|pakka)\b[^.\n]*(disease|fmd|mastitis|rabies|bloat|poisoning)\b",
        r"\b(call|phone|whatsapp)\s*\+?\d[\d -]{6,}\b",
    ]
]
SAFE_NEGATION = re.compile(r"\b(do not|don't|avoid|never|cannot|can't|should not|without trained|without a vet|refuse|do not say)\b", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checksum_rows(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validator_config_sha256() -> str:
    payload = json.dumps(
        {
            "report_files": REPORT_FILES,
            "approval_roles": sorted(APPROVAL_ROLES),
            "blocked_states": sorted(BLOCKED_APPROVAL_STATES),
            "approved_states": sorted(APPROVED_APPROVAL_STATES),
            "split_minimums": SPLIT_MINIMUMS,
            "pilot_row_split_counts": PILOT_ROW_SPLIT_COUNTS,
            "full_row_split_counts": FULL_ROW_SPLIT_COUNTS,
            "safety_critical_tags": sorted(SAFETY_CRITICAL_TAGS),
            "prompt_contract_ontology": PROMPT_CONTRACT_ONTOLOGY,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_bundle_sha256(claims: list[dict[str, Any]], seeds: list[dict[str, Any]], train: list[dict[str, Any]] | None = None, dev: list[dict[str, Any]] | None = None, final_eval: list[dict[str, Any]] | None = None) -> str:
    parts = [
        checksum_rows(claims),
        checksum_rows(seeds),
        checksum_rows(train or []),
        checksum_rows(dev or []),
        checksum_rows(final_eval or []),
        validator_config_sha256(),
    ]
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


def row_content_hash(row: dict[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "content_hash"}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def validate_dataset(
    dataset_dir: Path,
    update_report: bool = True,
    require_approved_state: str | None = None,
    phase: str = "seed_bank",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing_files = [name for name in REQUIRED_FILES if not (dataset_dir / name).exists()]
    if missing_files:
        errors.extend(f"missing required file: {name}" for name in missing_files)
        return {"valid": False, "errors": errors, "warnings": warnings, "report": {}}

    registry = read_jsonl(dataset_dir / "source_registry.jsonl")
    claims = read_jsonl(dataset_dir / "source_claims.jsonl")
    seeds = read_jsonl(dataset_dir / "seed_cases.jsonl")
    train = read_jsonl(dataset_dir / "sft_train.jsonl")
    dev = read_jsonl(dataset_dir / "sft_dev.jsonl")
    final_eval = read_jsonl(dataset_dir / "final_eval.jsonl")
    rejected = read_jsonl(dataset_dir / "rejected_rows.jsonl")
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    claim_by_id = {claim["claim_id"]: claim for claim in claims}

    reports = build_reports(dataset_dir, registry, claims, seeds, train, dev, final_eval, rejected, manifest, phase)
    errors.extend(validate_sources(registry, claims))
    errors.extend(validate_seed_structure(seeds, claim_by_id))
    errors.extend(reports["split_stratification_report.json"]["errors"])
    errors.extend(reports["claim_compatibility_report.json"]["errors"])
    for report_name in [
        "prompt_contract_alignment_report.json",
        "coarse_cluster_leakage_report.json",
        "variant_contract_delta_report.json",
        "answer_shape_distribution_report.json",
        "species_context_consistency_report.json",
        "repetition_caps_report.json",
        "variant_diversity_report.json",
        "template_similarity_report.json",
    ]:
        errors.extend(reports[report_name]["errors"])
    errors.extend(reports["language_quality_report.json"]["errors"])
    errors.extend(reports["bad_case_gate_report.json"]["errors"])
    errors.extend(reports["artifact_inventory_report.json"]["errors"])
    for report_name in [
        "expansion_provenance_report.json",
        "row_contract_fidelity_report.json",
        "answer_safety_drift_report.json",
        "escalation_calibration_report.json",
        "expansion_split_leakage_report.json",
        "expansion_pattern_collapse_report.json",
        "hinglish_naturalness_report.json",
        "scale_readiness_report.json",
        "pilot_approval_audit_report.json",
    ]:
        errors.extend(reports[report_name]["errors"])
    errors.extend(validate_approval_state(manifest, claims, seeds, reports, require_approved_state))
    errors.extend(validate_phase(phase, train, dev, final_eval, manifest, reports, require_approved_state))
    warnings.extend(reports["source_coverage_report.json"]["warnings"])
    warnings.extend(reports["artifact_inventory_report.json"]["warnings"])

    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "seed_count": len(seeds),
        "row_counts": {"sft_train": len(train), "sft_dev": len(dev), "final_eval": len(final_eval)},
        "risk_counts": dict(Counter(seed.get("risk_level") for seed in seeds)),
        "language_counts": dict(Counter(seed.get("language_style") for seed in seeds)),
        "approval_state": manifest.get("approval_state"),
        "phase": phase,
        "expansion_allowed": manifest.get("expansion_allowed"),
        "sft_allowed": manifest.get("sft_allowed"),
    }
    reports["phase_gate_report.json"].update({"valid": not errors, "errors": errors, "warnings": warnings})
    if update_report:
        for report_name, payload in reports.items():
            (dataset_dir / report_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (dataset_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "report": report}


def validate_sources(registry: list[dict[str, Any]], claims: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    source_ids = {row.get("source_id") for row in registry}
    claim_ids = set()
    required = {"claim_id", "source_id", "url", "publisher", "authority_level", "section_locator", "evidence_excerpt", "evidence_span_ids", "retrieved_date", "snapshot_hash", "claim_type", "allowed_use", "banned_use", "source_priority", "claim_role", "supports_topics", "supports_actions"}
    for claim in claims:
        missing = required - set(claim)
        if missing:
            errors.append(f"source claim missing fields {sorted(missing)}: {claim.get('claim_id')}")
        if claim.get("source_id") not in source_ids:
            errors.append(f"source claim references unknown source: {claim.get('claim_id')} {claim.get('source_id')}")
        if claim.get("claim_id") in claim_ids:
            errors.append(f"duplicate source claim id: {claim.get('claim_id')}")
        claim_ids.add(claim.get("claim_id"))
    return errors


def validate_seed_structure(seeds: list[dict[str, Any]], claim_by_id: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    required = {"seed_id", "seed_bank_version", "intended_split", "seed_family", "family_key", "family_bucket", "behavior_cluster_id", "coarse_behavior_cluster_id", "variant_axes", "answer_shape", "topic_tags", "action_tags", "prompt_topic", "care_claim_ids", "escalation_claim_ids", "policy_claim_ids", "safe_supportive_steps", "must_ask_checks", "red_flags", "must_not_say", "tags"}
    seen = set()
    for seed in seeds:
        missing = required - set(seed)
        if missing:
            errors.append(f"seed missing fields {sorted(missing)}: {seed.get('seed_id')}")
            continue
        if seed["seed_id"] in seen:
            errors.append(f"duplicate seed id: {seed['seed_id']}")
        seen.add(seed["seed_id"])
        all_claims = set(seed.get("care_claim_ids", []) + seed.get("escalation_claim_ids", []) + seed.get("policy_claim_ids", []))
        unknown = sorted(all_claims - set(claim_by_id))
        if unknown:
            errors.append(f"seed references unknown claims {unknown}: {seed['seed_id']}")
        errors.extend(validate_field_claims(seed, claim_by_id))
        if seed.get("channel") == "image_caption" and "C_POLICY_IMAGE_UNCERTAINTY" not in seed.get("policy_claim_ids", []):
            errors.append(f"image-caption seed lacks image uncertainty policy: {seed['seed_id']}")
    return errors


def validate_field_claims(seed: dict[str, Any], claim_by_id: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    topics = set(seed.get("topic_tags", [])) | set(seed.get("tags", []))
    actions = set(seed.get("action_tags", []))
    for section in ["safe_supportive_steps", "must_ask_checks"]:
        for item in seed.get(section, []):
            if not item.get("claim_ids") or not item.get("evidence_span_ids"):
                errors.append(f"{section} missing claim grounding: {seed['seed_id']}")
            if not any(claim_by_id.get(claim_id, {}).get("claim_role") == "care" for claim_id in item.get("claim_ids", [])):
                errors.append(f"{section} lacks compatible care claim: {seed['seed_id']} {item.get('text')}")
            if all(claim_by_id.get(claim_id, {}).get("claim_role") in {"policy", "escalation", "context"} for claim_id in item.get("claim_ids", [])):
                errors.append(f"{section} uses only policy/escalation/context claims for positive care: {seed['seed_id']}")
            if not any(claim_matches_seed(claim_by_id.get(claim_id, {}), topics, actions) for claim_id in item.get("claim_ids", [])):
                errors.append(f"{section} lacks topic/action-compatible claim: {seed['seed_id']} {item.get('text')}")
    for item in seed.get("red_flags", []):
        if not item.get("claim_ids"):
            errors.append(f"red_flag missing claim ids: {seed['seed_id']}")
    for item in seed.get("must_not_say", []):
        if not item.get("claim_ids") or not item.get("harm_rationale") or not item.get("safer_alternative"):
            errors.append(f"must_not_say incomplete: {seed['seed_id']}")
    return errors


def claim_matches_seed(claim: dict[str, Any], topics: set[str], actions: set[str]) -> bool:
    if claim.get("claim_role") != "care":
        return False
    supported_topics = set(claim.get("supports_topics", []))
    supported_actions = set(claim.get("supports_actions", []))
    if not supported_topics or not supported_actions:
        return False
    topic_ok = bool(supported_topics & topics) or "supportive_first_aid" in supported_topics
    action_ok = bool(supported_actions & actions) or "comfort_support" in supported_actions
    return topic_ok and action_ok


def build_reports(dataset_dir: Path, registry: list[dict[str, Any]], claims: list[dict[str, Any]], seeds: list[dict[str, Any]], train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]], rejected: list[dict[str, Any]], manifest: dict[str, Any], phase: str) -> dict[str, dict[str, Any]]:
    claim_by_id = {claim["claim_id"]: claim for claim in claims}
    return {
        "artifact_inventory_report.json": artifact_inventory_report(dataset_dir, train, dev, final_eval),
        "split_stratification_report.json": split_stratification_report(seeds),
        "source_coverage_report.json": source_coverage_report(seeds, claim_by_id),
        "claim_compatibility_report.json": claim_compatibility_report(seeds, claim_by_id, manifest),
        "prompt_contract_alignment_report.json": prompt_contract_alignment_report(seeds),
        "coarse_cluster_leakage_report.json": coarse_cluster_leakage_report(seeds),
        "variant_contract_delta_report.json": variant_contract_delta_report(seeds),
        "answer_shape_distribution_report.json": answer_shape_distribution_report(seeds),
        "species_context_consistency_report.json": species_context_consistency_report(seeds),
        "repetition_caps_report.json": repetition_caps_report(seeds),
        "variant_diversity_report.json": variant_diversity_report(seeds),
        "template_similarity_report.json": template_similarity_report(seeds),
        "language_quality_report.json": language_quality_report(seeds),
        "bad_case_gate_report.json": bad_case_gate_report(seeds, rejected),
        "expansion_provenance_report.json": expansion_provenance_report(seeds, claims, train, dev, final_eval),
        "row_contract_fidelity_report.json": row_contract_fidelity_report(seeds, claim_by_id, train, dev, final_eval),
        "answer_safety_drift_report.json": answer_safety_drift_report(train, dev, final_eval),
        "escalation_calibration_report.json": escalation_calibration_report(train, dev, final_eval),
        "expansion_split_leakage_report.json": expansion_split_leakage_report(train, dev, final_eval),
        "expansion_pattern_collapse_report.json": expansion_pattern_collapse_report(train, dev, final_eval),
        "hinglish_naturalness_report.json": hinglish_naturalness_report(train, dev, final_eval),
        "scale_readiness_report.json": scale_readiness_report(train, dev, final_eval, manifest),
        "pilot_approval_audit_report.json": pilot_approval_audit_report(dataset_dir, claims, seeds, train, dev, final_eval, manifest),
        "approval_audit_report.json": approval_audit_report(dataset_dir, claims, seeds, manifest),
        "phase_gate_report.json": {"phase": phase, "requested_state": manifest.get("approval_state")},
    }


def artifact_inventory_report(dataset_dir: Path, train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> dict[str, Any]:
    root = project_root_for(dataset_dir)
    stale_exports = []
    export_root = root / "exports"
    if export_root.exists():
        for path in export_root.rglob("*.jsonl"):
            quarantined = any(part.lower() in {"do_not_use", "quarantine", "quarantined"} for part in path.parts)
            if not quarantined:
                stale_exports.append(str(path.relative_to(root)))
    errors = [f"live stale export artifact not quarantined: {path}" for path in stale_exports]
    return {"errors": errors, "warnings": [], "stale_exports": stale_exports, "row_counts": {"sft_train": len(train), "sft_dev": len(dev), "final_eval": len(final_eval)}}


def project_root_for(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return path.parents[2]


def split_stratification_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    split_counts = Counter(seed.get("intended_split") for seed in seeds)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        errors.append(f"split counts must be {EXPECTED_SPLIT_COUNTS}, found {dict(split_counts)}")
    split_tags: dict[str, Counter[str]] = defaultdict(Counter)
    split_scalars: dict[str, Counter[str]] = defaultdict(Counter)
    clusters: dict[str, set[str]] = defaultdict(set)
    for seed in seeds:
        split = seed["intended_split"]
        clusters[seed["behavior_cluster_id"]].add(split)
        for tag in seed.get("tags", []):
            split_tags[split][tag] += 1
        split_scalars[split][seed.get("risk_level", "")] += 1
        split_scalars[split][seed.get("family_bucket", "")] += 1
        split_scalars[split][seed.get("species", "")] += 1
        split_scalars[split][seed.get("language_style", "")] += 1
    for split, minimums in SPLIT_MINIMUMS.items():
        for tag, minimum in minimums.items():
            found = max(split_tags[split].get(tag, 0), split_scalars[split].get(tag, 0))
            if found < minimum:
                errors.append(f"{split} missing minimum {tag}: {found} < {minimum}")
        scalar_total = sum(split_scalars[split].values())
        if scalar_total > split_counts.get(split, 0) * 4:
            errors.append(f"{split} scalar counts exceed expected scalar dimensions")
    leaks = {cluster: sorted(splits) for cluster, splits in clusters.items() if len(splits) > 1}
    if leaks:
        errors.append(f"behavior cluster leakage across splits: {leaks}")
    return {
        "errors": errors,
        "warnings": [],
        "split_counts": dict(split_counts),
        "split_tags": {split: dict(counter) for split, counter in split_tags.items()},
        "split_scalars": {split: dict(counter) for split, counter in split_scalars.items()},
    }


def source_coverage_report(seeds: list[dict[str, Any]], claim_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    usage = Counter(claim_id for seed in seeds for claim_id in seed.get("care_claim_ids", []) + seed.get("escalation_claim_ids", []) + seed.get("policy_claim_ids", []))
    field_usage = Counter()
    warnings = []
    for seed in seeds:
        for item in seed.get("safe_supportive_steps", []) + seed.get("must_ask_checks", []) + seed.get("red_flags", []) + seed.get("must_not_say", []):
            for claim_id in item.get("claim_ids", []):
                field_usage[claim_id] += 1
        if any(claim_by_id.get(claim_id, {}).get("approval_blocker") for claim_id in seed.get("care_claim_ids", [])) and seed["approval_state"] if "approval_state" in seed else False:
            warnings.append(f"fallback source in seed: {seed['seed_id']}")
    return {"errors": [], "warnings": warnings, "claim_usage": dict(usage), "field_usage": dict(field_usage)}


def claim_compatibility_report(seeds: list[dict[str, Any]], claim_by_id: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    for seed in seeds:
        tags = set(seed.get("tags", []))
        topics = set(seed.get("topic_tags", [])) | tags
        actions = set(seed.get("action_tags", []))
        care_roles = [claim_by_id.get(claim_id, {}).get("claim_role") for claim_id in seed.get("care_claim_ids", [])]
        if not any(role == "care" for role in care_roles):
            errors.append(f"seed lacks care claim support: {seed['seed_id']}")
        if any(role in {"policy", "escalation", "context"} for role in care_roles) and not any(role == "care" for role in care_roles):
            errors.append(f"seed care claims are only policy/escalation/context: {seed['seed_id']}")
        if not any(claim_matches_seed(claim_by_id.get(claim_id, {}), topics, actions) for claim_id in seed.get("care_claim_ids", [])):
            errors.append(f"seed lacks topic/action compatible positive care claim: {seed['seed_id']}")
        if "milk safety" in seed.get("tags", []) and seed.get("care_claim_ids") == ["C_NDDB_MASTITIS_TOPIC"]:
            errors.append(f"milk safety supported only by mastitis index: {seed['seed_id']}")
        is_critical = seed.get("risk_level") == "red" or bool(tags & SAFETY_CRITICAL_TAGS)
        unsupported_policy = [
            claim_id
            for claim_id in seed.get("care_claim_ids", [])
            if claim_by_id.get(claim_id, {}).get("claim_role") in {"policy", "escalation", "context"}
        ]
        if unsupported_policy:
            errors.append(f"positive care uses non-care claims: {seed['seed_id']} {unsupported_policy}")
        fallback_claims = [
            claim_id
            for claim_id in seed.get("care_claim_ids", [])
            if claim_by_id.get(claim_id, {}).get("approval_blocker")
        ]
        if is_critical and fallback_claims:
            blockers.append(f"fallback source requires family-specific source-review override: {seed['seed_id']} {fallback_claims}")
            errors.append(f"safety-critical seed uses fallback-blocked care claim: {seed['seed_id']} {fallback_claims}")
    return {"errors": errors, "warnings": blockers, "approval_blockers": blockers}


def variant_diversity_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        by_family[seed["family_key"]].append(seed)
    for family, rows in by_family.items():
        if len(rows) != 3:
            errors.append(f"family must have exactly 3 variants: {family}")
            continue
        axis_names = list(rows[0].get("variant_axes", {}))
        differing = [axis for axis in axis_names if len({row["variant_axes"].get(axis) for row in rows}) > 1]
        if len([axis for axis in differing if axis not in {"species", "language"}]) < 2:
            errors.append(f"family variants lack two meaningful axis differences: {family}")
        contracts = {
            json.dumps(
                {
                    "checks": [item["text"] for item in row["must_ask_checks"]],
                    "steps": [item["text"] for item in row["safe_supportive_steps"]],
                    "red": [item["text"] for item in row["red_flags"]],
                    "no": [item["text"] for item in row["must_not_say"]],
                },
                sort_keys=True,
            )
            for row in rows
        }
        if len(contracts) == 1:
            errors.append(f"family variants have identical contracts: {family}")
    return {"errors": errors, "warnings": [], "family_count": len(by_family)}


def prompt_contract_alignment_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    checked = []
    for seed in seeds:
        topic = seed.get("prompt_topic", "generic")
        ontology = PROMPT_CONTRACT_ONTOLOGY.get(topic, PROMPT_CONTRACT_ONTOLOGY["generic"])
        prompt = seed.get("farmer_prompt", "").lower()
        required = ontology["required"]
        forbidden_terms = ontology["forbidden"]
        if required and not any(term in prompt for term in required):
            errors.append(f"prompt lacks required topic cue: {seed['seed_id']} {topic} {prompt}")
        bad_terms = [term for term in forbidden_terms if term in prompt]
        if bad_terms:
            errors.append(f"prompt contains forbidden topic cue: {seed['seed_id']} {topic} {bad_terms}")
        if seed.get("channel") == "image_caption" and re.search(r"\b(can you confirm|confirmed|pakka confirm)\b", prompt):
            errors.append(f"image prompt implies diagnosis: {seed['seed_id']}")
        checked.append({"seed_id": seed.get("seed_id"), "topic": topic})
    return {"errors": errors, "warnings": [], "checked": checked[:20], "checked_count": len(checked)}


def coarse_cluster_leakage_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    exact: dict[str, set[str]] = defaultdict(set)
    coarse: dict[str, set[str]] = defaultdict(set)
    universal: dict[str, bool] = {}
    for seed in seeds:
        exact[seed["behavior_cluster_id"]].add(seed["intended_split"])
        coarse[seed["coarse_behavior_cluster_id"]].add(seed["intended_split"])
        universal[seed["coarse_behavior_cluster_id"]] = seed.get("behavior_cluster_split_policy") == "universal_policy"
    exact_leaks = {cluster: sorted(splits) for cluster, splits in exact.items() if len(splits) > 1}
    coarse_leaks = {cluster: sorted(splits) for cluster, splits in coarse.items() if len(splits) > 1 and not universal.get(cluster)}
    if exact_leaks:
        errors.append(f"exact behavior cluster leakage across splits: {exact_leaks}")
    if coarse_leaks:
        errors.append(f"coarse behavior cluster leakage across splits: {coarse_leaks}")
    return {"errors": errors, "warnings": [], "exact_leaks": exact_leaks, "coarse_leaks": coarse_leaks}


def variant_contract_delta_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        by_family[seed["family_key"]].append(seed)
    for family, rows in by_family.items():
        if len(rows) != 3:
            continue
        axes = rows[0].get("variant_axes", {})
        meaningful = [
            axis for axis in axes
            if axis not in {"species", "language"} and len({row.get("variant_axes", {}).get(axis) for row in rows}) > 1
        ]
        if len(meaningful) < 2:
            errors.append(f"family variants lack two meaningful axes: {family}")
        core_sets = {
            "first_step": {row["safe_supportive_steps"][0]["text"] for row in rows},
            "first_check": {row["must_ask_checks"][0]["text"] for row in rows},
            "red_flag": {row["red_flags"][0]["text"] for row in rows},
            "farmer_pressure": {row.get("farmer_pressure_axis") for row in rows},
        }
        if all(len(values) == 1 for values in core_sets.values()):
            errors.append(f"family variants lack core-contract delta: {family}")
        joined = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        if re.search(r"\b(duration|resource_constraint|herd_context|season_weather|work_context)=", joined):
            errors.append(f"metadata axis leaked into user-facing text: {family}")
    return {"errors": errors, "warnings": [], "family_count": len(by_family)}


def answer_shape_distribution_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    total = len(seeds) or 1
    counts = Counter(seed.get("answer_shape") for seed in seeds)
    if counts and counts.most_common(1)[0][1] / total > 0.30:
        errors.append(f"dominant answer_shape exceeds 30%: {counts.most_common(1)[0]}")
    split_shapes: dict[str, set[str]] = defaultdict(set)
    split_requirements: dict[str, set[str]] = defaultdict(set)
    for seed in seeds:
        split = seed["intended_split"]
        split_shapes[split].add(seed.get("answer_shape"))
        split_requirements[split].update([seed.get("family_bucket"), seed.get("language_style")])
    for split in EXPECTED_SPLIT_COUNTS:
        if len(split_shapes[split]) < 5:
            errors.append(f"{split} has fewer than 5 answer shapes: {sorted(split_shapes[split])}")
        required = {"routine_green", "yellow_triage", "red_emergency", "rural_pressure_myth", "image_caption_uncertainty", "english", "hinglish"}
        missing = required - split_requirements[split]
        if split in {"dev_seed", "final_eval_seed"} and missing:
            errors.append(f"{split} missing required bucket/language coverage: {sorted(missing)}")
    return {"errors": errors, "warnings": [], "answer_shapes": dict(counts), "split_shapes": {split: sorted(values) for split, values in split_shapes.items()}}


def species_context_consistency_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for seed in seeds:
        tags = set(seed.get("tags", []))
        species = seed.get("species")
        flags = seed.get("flags", {})
        if ("calving trouble" in tags or "abortion cluster" in tags) and species not in {"cow", "buffalo"}:
            errors.append(f"calving/abortion seed must be adult female cow/buffalo: {seed['seed_id']}")
        if ("calving trouble" in tags or "abortion cluster" in tags) and seed.get("age_class") != "adult":
            errors.append(f"calving/abortion seed cannot be calf age: {seed['seed_id']}")
        if "milk safety" in tags and (species not in {"cow", "buffalo"} or not flags.get("lactating")):
            errors.append(f"milk safety seed must be lactating cow/buffalo: {seed['seed_id']}")
        if "calf diarrhea/dehydration" in tags and species != "calf":
            errors.append(f"calf diarrhea seed must be calf-only: {seed['seed_id']}")
        if any(tag in tags for tag in ["ox yoke/hoof injury", "ox workload/yoke injury", "working ox"]) and species != "ox":
            if "working ox" in tags or "ox workload/yoke injury" in tags:
                errors.append(f"working ox seed must use ox/bullock context: {seed['seed_id']}")
        if seed.get("channel") == "image_caption" and not flags.get("image_caption"):
            errors.append(f"image channel lacks image flag: {seed['seed_id']}")
    return {"errors": errors, "warnings": [], "checked_count": len(seeds)}


def repetition_caps_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    prompt_frames = Counter(normalize_prompt(seed["farmer_prompt"]) for seed in seeds)
    animal_lang_norm = Counter(normalize_prompt(seed["farmer_prompt"]).replace("{animal}", "{x}") for seed in seeds)
    first_safe = Counter(seed["safe_supportive_steps"][0]["text"] for seed in seeds)
    first_check = Counter(seed["must_ask_checks"][0]["text"] for seed in seeds)
    red_flag_counts = Counter(seed["red_flags"][0]["text"] for seed in seeds)
    must_not_counts = Counter(seed["must_not_say"][0]["text"] for seed in seeds)
    if prompt_frames and prompt_frames.most_common(1)[0][1] > 3:
        errors.append(f"exact normalized prompt frame max 3 exceeded: {prompt_frames.most_common(1)[0]}")
    if animal_lang_norm and animal_lang_norm.most_common(1)[0][1] > 5:
        errors.append(f"animal/language normalized prompt frame max 5 exceeded: {animal_lang_norm.most_common(1)[0]}")
    for label, counts, limit in [
        ("first safe step", first_safe, 3),
        ("first must-ask", first_check, 3),
        ("red flag", red_flag_counts, 3),
        ("must_not_say", must_not_counts, 15),
    ]:
        if counts and counts.most_common(1)[0][1] > limit:
            errors.append(f"{label} repetition cap exceeded: {counts.most_common(1)[0]}")
    return {
        "errors": errors,
        "warnings": [],
        "top_prompts": prompt_frames.most_common(10),
        "top_first_safe_steps": first_safe.most_common(10),
        "top_first_checks": first_check.most_common(10),
        "top_red_flags": red_flag_counts.most_common(10),
        "top_must_not_say": must_not_counts.most_common(10),
    }


def template_similarity_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    total = len(seeds) or 1
    prompt_frames = Counter(normalize_prompt(seed["farmer_prompt"]) for seed in seeds)
    prompt_by_language: dict[str, Counter[str]] = defaultdict(Counter)
    text_fields = {"safe_step": Counter(), "must_ask": Counter(), "red_flag": Counter()}
    answer_shapes = Counter(seed.get("answer_shape") for seed in seeds)
    bucket_frames: dict[str, set[str]] = defaultdict(set)
    for seed in seeds:
        frame = normalize_prompt(seed["farmer_prompt"])
        prompt_by_language[seed["language_style"]][frame] += 1
        bucket_frames[seed["family_bucket"]].add(frame)
        for item in seed["safe_supportive_steps"]:
            if item["text"] not in UNIVERSAL_ALLOWED_TEXT:
                text_fields["safe_step"][item["text"]] += 1
        for item in seed["must_ask_checks"]:
            text_fields["must_ask"][item["text"]] += 1
        for item in seed["red_flags"]:
            text_fields["red_flag"][item["text"]] += 1
    for bucket, frames in bucket_frames.items():
        if len(frames) < 8:
            errors.append(f"bucket has fewer than 8 prompt frames: {bucket} {len(frames)}")
    return {"errors": errors, "warnings": [], "prompt_frame_top": prompt_frames.most_common(10), "answer_shapes": dict(answer_shapes)}


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.lower()
    prompt = re.sub(r"\b(gai|bhains|bail|bachda|cow|buffalo|ox|calf)\b", "{animal}", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


def language_quality_report(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    examples: list[str] = []
    for seed in seeds:
        prompt = seed.get("farmer_prompt", "")
        if MOJIBAKE.search(json.dumps(seed, ensure_ascii=False)):
            errors.append(f"mojibake detected: {seed['seed_id']}")
        for pattern in METADATA_PROMPT_PATTERNS:
            if pattern.search(prompt):
                errors.append(f"metadata-like prompt: {seed['seed_id']} {prompt}")
        if seed["language_style"] == "hinglish" and not re.search(r"\b(kya|kaise|paani|pet|doodh|bail|bhains|gai|bachda|chara|ghav|saans|safe)\b", prompt.lower()):
            errors.append(f"hinglish prompt lacks farmer-natural vocabulary: {seed['seed_id']}")
            examples.append(prompt)
    return {"errors": errors, "warnings": [], "examples": examples[:20]}


def bad_case_gate_report(seeds: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    found = {row.get("case_id") for row in rejected}
    missing = sorted(BAD_CASE_IDS - found)
    if missing:
        errors.append(f"missing bad-case tests: {missing}")
    seed_text = "\n".join(json.dumps(seed, ensure_ascii=False).lower() for seed in seeds)
    required_phrases = ["mustard oil", "local shop", "antibiotic", "pull", "milk", "photo", "carcass", "market"]
    for phrase in required_phrases:
        if phrase not in seed_text and phrase.replace(" ", "_") not in seed_text:
            errors.append(f"bad-case category not exercised in seed contracts: {phrase}")
    return {"errors": errors, "warnings": [], "case_ids": sorted(found)}


def all_expansion_rows(train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [*[( "sft_train", row) for row in train], *[("sft_dev", row) for row in dev], *[("final_eval", row) for row in final_eval]]


def expansion_provenance_report(seeds: list[dict[str, Any]], claims: list[dict[str, Any]], train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    rows = all_expansion_rows(train, dev, final_eval)
    if not rows:
        return {"errors": [], "warnings": [], "row_counts": {"sft_train": 0, "sft_dev": 0, "final_eval": 0}}
    seeds_by_id = {seed["seed_id"]: seed for seed in seeds}
    source_sha = checksum_rows(claims)
    seed_sha = checksum_rows(seeds)
    expected_parent_split = {"sft_train": "train_seed", "sft_dev": "dev_seed", "final_eval": "final_eval_seed"}
    required = {
        "row_id", "parent_seed_id", "parent_seed_family", "parent_seed_split", "parent_behavior_cluster_id",
        "parent_coarse_behavior_cluster_id", "care_claim_ids", "escalation_claim_ids", "policy_claim_ids",
        "parent_seed_cases_sha256", "parent_source_claims_sha256", "generator_version", "generation_config",
        "content_hash", "messages", "answer_sections", "risk_level", "answer_shape", "language_style",
    }
    seen = set()
    counts = Counter()
    for split, row in rows:
        counts[split] += 1
        missing = required - set(row)
        if missing:
            errors.append(f"expanded row missing fields {sorted(missing)}: {row.get('row_id')}")
            continue
        if row["row_id"] in seen:
            errors.append(f"duplicate expanded row id: {row['row_id']}")
        seen.add(row["row_id"])
        if row_content_hash(row) != row.get("content_hash"):
            errors.append(f"content hash mismatch: {row['row_id']}")
        parent = seeds_by_id.get(row["parent_seed_id"])
        if not parent:
            errors.append(f"unknown parent seed: {row['row_id']} {row['parent_seed_id']}")
            continue
        if row.get("parent_seed_split") != expected_parent_split[split] or parent.get("intended_split") != expected_parent_split[split]:
            errors.append(f"cross-split expansion lineage: {row['row_id']} {split} parent={parent.get('intended_split')}")
        if row.get("parent_seed_family") != parent.get("seed_family"):
            errors.append(f"parent seed family mismatch: {row['row_id']}")
        if row.get("parent_behavior_cluster_id") != parent.get("behavior_cluster_id"):
            errors.append(f"parent behavior cluster mismatch: {row['row_id']}")
        if row.get("parent_coarse_behavior_cluster_id") != parent.get("coarse_behavior_cluster_id"):
            errors.append(f"parent coarse behavior cluster mismatch: {row['row_id']}")
        if row.get("parent_seed_cases_sha256") != seed_sha:
            errors.append(f"parent seed checksum mismatch: {row['row_id']}")
        if row.get("parent_source_claims_sha256") != source_sha:
            errors.append(f"parent source checksum mismatch: {row['row_id']}")
    return {"errors": errors, "warnings": [], "row_counts": dict(counts), "pilot_row_counts": PILOT_ROW_SPLIT_COUNTS, "full_row_counts": FULL_ROW_SPLIT_COUNTS}


def row_contract_fidelity_report(seeds: list[dict[str, Any]], claim_by_id: dict[str, dict[str, Any]], train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    seeds_by_id = {seed["seed_id"]: seed for seed in seeds}
    for _, row in all_expansion_rows(train, dev, final_eval):
        parent = seeds_by_id.get(row.get("parent_seed_id"))
        if not parent:
            continue
        for claim_field in ["care_claim_ids", "escalation_claim_ids", "policy_claim_ids"]:
            if set(row.get(claim_field, [])) != set(parent.get(claim_field, [])):
                errors.append(f"row claim inheritance mismatch {claim_field}: {row.get('row_id')}")
        if row.get("answer_shape") != parent.get("answer_shape") or row.get("risk_level") != parent.get("risk_level"):
            errors.append(f"row risk/answer_shape drift: {row.get('row_id')}")
        inherited = set(parent.get("source_claim_ids", []))
        for section in row.get("answer_sections", []):
            section_claims = set(section.get("claim_ids", []))
            if not section_claims or not section_claims <= inherited:
                errors.append(f"answer section has unsupported claim ids: {row.get('row_id')} {section.get('section_type')}")
            if section.get("section_type") in {"safe_step", "must_ask"}:
                if not any(claim_by_id.get(claim_id, {}).get("claim_role") == "care" for claim_id in section_claims):
                    errors.append(f"positive answer section lacks care claim: {row.get('row_id')} {section.get('section_type')}")
            if section.get("section_type") not in {"safe_step", "must_ask", "red_flag", "must_not_say", "boundary", "style"}:
                errors.append(f"unknown answer section type: {row.get('row_id')} {section.get('section_type')}")
    return {"errors": errors, "warnings": [], "checked_rows": len(all_expansion_rows(train, dev, final_eval))}


def answer_safety_drift_report(train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    flagged = []
    for _, row in all_expansion_rows(train, dev, final_eval):
        for item in row.get("answer_sections", []):
            if item.get("section_type") == "red_flag":
                continue
            text = item.get("text", "")
            for pattern in UNSAFE_ADVICE_PATTERNS:
                for match in pattern.finditer(text):
                    window = text[max(0, match.start() - 45): match.end() + 45]
                    if not SAFE_NEGATION.search(window):
                        flagged.append({"row_id": row.get("row_id"), "match": match.group(0)})
                        errors.append(f"unsafe advice drift: {row.get('row_id')} {match.group(0)}")
    return {"errors": errors, "warnings": [], "flagged": flagged[:50], "critical_unsafe_advice_count": len(flagged)}


def escalation_calibration_report(train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    green_over_referrals = 0
    green_total = 0
    for _, row in all_expansion_rows(train, dev, final_eval):
        answer = assistant_text(row).lower()
        risk = row.get("risk_level")
        if risk == "red" and not re.search(r"\b(contact|call|reach|inform)\b.*\b(vet|trained|animal health|pashu sakhi|dairy cooperative|official)\b", answer):
            errors.append(f"red row lacks trained-help escalation: {row.get('row_id')}")
        if risk == "yellow" and not any(phrase in answer for phrase in ["watch", "monitor", "red flag", "if it worsens", "if you see"]):
            errors.append(f"yellow row lacks monitoring/red-flag calibration: {row.get('row_id')}")
        if risk == "green":
            green_total += 1
            if re.search(r"\b(contact|call|go to|visit)\b.*\b(vet|doctor)\b", answer):
                green_over_referrals += 1
    if green_total and green_over_referrals / green_total > 0.10:
        errors.append(f"green over-referral exceeds 10%: {green_over_referrals}/{green_total}")
    return {"errors": errors, "warnings": [], "green_over_referrals": green_over_referrals, "green_total": green_total}


def expansion_split_leakage_report(train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    prompts: dict[str, set[str]] = defaultdict(set)
    answers: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    coarse: dict[str, set[str]] = defaultdict(set)
    for split, row in all_expansion_rows(train, dev, final_eval):
        prompts[normalize_prompt(user_text(row))].add(split)
        answers[normalize_answer(assistant_text(row))].add(split)
        families[row.get("parent_seed_family", "")].add(split)
        coarse[row.get("parent_coarse_behavior_cluster_id", "")].add(split)
    for label, mapping in [("prompt", prompts), ("answer", answers), ("parent family", families), ("coarse cluster", coarse)]:
        leaks = {key: sorted(splits) for key, splits in mapping.items() if key and len(splits) > 1}
        if leaks:
            errors.append(f"cross-split {label} leakage: {dict(list(leaks.items())[:10])}")
    return {"errors": errors, "warnings": [], "checked_rows": len(all_expansion_rows(train, dev, final_eval))}


def expansion_pattern_collapse_report(train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    rows = [row for _, row in all_expansion_rows(train, dev, final_eval)]
    if not rows:
        return {"errors": [], "warnings": [], "row_count": 0}
    openings = Counter(first_sentence(assistant_text(row)) for row in rows)
    closings = Counter(last_sentence(assistant_text(row)) for row in rows)
    structures = Counter(answer_structure(row) for row in rows)
    total = len(rows)
    structure_limit = 0.18 if total > 300 else 0.25
    phrase_limit = 0.12 if total > 300 else 0.20
    if structures and structures.most_common(1)[0][1] / total > structure_limit:
        errors.append(f"dominant answer template exceeds {int(structure_limit * 100)}%: {structures.most_common(1)[0]}")
    if openings and openings.most_common(1)[0][1] / total > phrase_limit:
        errors.append(f"dominant opening phrase exceeds {int(phrase_limit * 100)}%: {openings.most_common(1)[0]}")
    if closings and closings.most_common(1)[0][1] / total > phrase_limit:
        errors.append(f"dominant closing phrase exceeds {int(phrase_limit * 100)}%: {closings.most_common(1)[0]}")
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[row.get("parent_seed_id", "")].append(row)
    for parent_id, parent_rows in by_parent.items():
        if total > 300 and len(parent_rows) != 5:
            errors.append(f"full expansion requires five rows per seed: {parent_id} {len(parent_rows)}")
        uniqueness_fields = {
            "prompt": [normalize_prompt(user_text(row)) for row in parent_rows],
            "template": [row.get("answer_template_id", "") for row in parent_rows],
            "opening": [first_sentence(assistant_text(row)) for row in parent_rows],
            "closing": [last_sentence(assistant_text(row)) for row in parent_rows],
        }
        for label, values in uniqueness_fields.items():
            if len(values) != len(set(values)):
                errors.append(f"same-seed {label} variants are not unique: {parent_id}")
    return {
        "errors": errors,
        "warnings": [],
        "row_count": total,
        "structure_limit": structure_limit,
        "phrase_limit": phrase_limit,
        "top_openings": openings.most_common(10),
        "top_closings": closings.most_common(10),
        "top_structures": structures.most_common(10),
    }


def hinglish_naturalness_report(train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    reviewed = 0
    rejected = 0
    examples = []
    for _, row in all_expansion_rows(train, dev, final_eval):
        if row.get("language_style") != "hinglish":
            continue
        reviewed += 1
        text = f"{user_text(row)}\n{assistant_text(row)}".lower()
        bad = False
        if not re.search(r"\b(kya|kaise|paani|chara|doodh|gai|bhains|bail|bachda|ghav|pet|saans|safe|mat)\b", text):
            bad = True
        if re.search(r"\btherefore|hence|livestock owner|veterinary professional\b", text):
            bad = True
        if bad:
            rejected += 1
            examples.append(row.get("row_id"))
    if reviewed and rejected / reviewed > 0.10:
        errors.append(f"hinglish naturalness rejection rate exceeds 10%: {rejected}/{reviewed}")
    return {"errors": errors, "warnings": [], "reviewed": reviewed, "rejected": rejected, "examples": examples[:20]}


def scale_readiness_report(train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    rows = {"sft_train": len(train), "sft_dev": len(dev), "final_eval": len(final_eval)}
    errors: list[str] = []
    expected = FULL_ROW_SPLIT_COUNTS if manifest.get("approval_state") == "BLOCKED_PENDING_EXPANSION_REVIEW" or sum(rows.values()) > 300 else PILOT_ROW_SPLIT_COUNTS
    if any(rows.values()) and rows != expected:
        errors.append(f"expansion row counts must be {expected}, found {rows}")
    if any(rows.values()) and manifest.get("sft_allowed"):
        errors.append("blocked expansion cannot enable SFT export")
    reason = "1.5K expansion candidate is review-ready only; SFT export requires a new approval bundle." if expected == FULL_ROW_SPLIT_COUNTS else "300-row pilot validates machinery only; 1K/2K requires a new frozen bundle."
    return {"errors": errors, "warnings": [], "row_counts": rows, "expected_row_counts": expected, "scale_ready": False, "reason": reason}


def pilot_approval_audit_report(dataset_dir: Path, claims: list[dict[str, Any]], seeds: list[dict[str, Any]], train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    bundle = artifact_bundle_sha256(claims, seeds, train, dev, final_eval)
    return {"errors": [], "warnings": [], "pilot_bundle_sha256": bundle, "approval_required_for_scaling": bool(train or dev or final_eval), "approval_files_found": len(list((dataset_dir / "pilot_approvals").glob("*_approval.json"))) if (dataset_dir / "pilot_approvals").exists() else 0}


def user_text(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(row.get("user_prompt", ""))


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return str(row.get("assistant_response", ""))


def normalize_answer(answer: str) -> str:
    answer = answer.lower()
    answer = re.sub(r"\b(gai|bhains|bail|bachda|cow|buffalo|ox|calf)\b", "{animal}", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return normalize_answer(parts[0]) if parts and parts[0] else ""


def last_sentence(text: str) -> str:
    parts = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    return normalize_answer(parts[-1]) if parts else ""


def answer_structure(row: dict[str, Any]) -> str:
    return ">".join(section.get("section_type", "") for section in row.get("answer_sections", []))


def approval_audit_report(dataset_dir: Path, claims: list[dict[str, Any]], seeds: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    source_checksum = checksum_rows(claims)
    seed_checksum = checksum_rows(seeds)
    report_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset_dir.glob("*_report.json")
        if path.name not in {"approval_audit_report.json", "phase_gate_report.json"}
    }
    config_sha = validator_config_sha256()
    report_bundle = hashlib.sha256(json.dumps(report_hashes, sort_keys=True).encode("utf-8")).hexdigest()
    bundle = artifact_bundle_sha256(claims, seeds)
    approvals_dir = dataset_dir / "approvals"
    approvals = []
    errors = []
    if approvals_dir.exists():
        for path in approvals_dir.glob("*_approval.json"):
            approvals.append(json.loads(path.read_text(encoding="utf-8")))
    roles = {approval.get("role") for approval in approvals}
    reviewer_ids = [approval.get("reviewer_id") for approval in approvals if approval.get("reviewer_id")]
    if manifest.get("approval_state") != "BLOCKED_PENDING_SEED_APPROVAL":
        if roles != APPROVAL_ROLES:
            errors.append(f"approval files must contain exact roles {APPROVAL_ROLES}, found {roles}")
        if len(reviewer_ids) != len(set(reviewer_ids)):
            errors.append("approval files must use distinct reviewer IDs")
        for approval in approvals:
            if approval.get("approval_bundle_sha256") != bundle:
                errors.append(f"stale approval checksum: {approval.get('role')}")
            if approval.get("decision") != "approved":
                errors.append(f"approval decision must be approved: {approval.get('role')}")
            if not ISO_TIMESTAMP.match(str(approval.get("timestamp", ""))):
                errors.append(f"approval timestamp must be ISO-8601: {approval.get('role')}")
            if approval.get("validator_config_sha256") != config_sha:
                errors.append(f"stale validator config hash: {approval.get('role')}")
            if approval.get("reviewed_report_hashes") != report_hashes:
                errors.append(f"stale or missing reviewed report hashes: {approval.get('role')}")
            unresolved = approval.get("unresolved_risks", [])
            blocking = [risk for risk in unresolved if risk.get("severity") != "non_blocking"]
            if blocking:
                errors.append(f"approval has blocking unresolved risks: {approval.get('role')}")
            for field in ["reviewer_id", "reviewer_name", "timestamp", "decision", "notes", "role_rubric_result", "reviewed_artifact_hashes", "reviewed_report_hashes", "unresolved_risks"]:
                value = approval.get(field)
                if field not in approval or value == "" or value is None:
                    errors.append(f"approval missing {field}: {approval.get('role')}")
    return {
        "errors": errors,
        "warnings": [],
        "approval_bundle_sha256": bundle,
        "validator_config_sha256": config_sha,
        "report_hashes": report_hashes,
        "approval_files_found": len(approvals),
        "roles_found": sorted(role for role in roles if role),
    }


def validate_approval_state(manifest: dict[str, Any], claims: list[dict[str, Any]], seeds: list[dict[str, Any]], reports: dict[str, dict[str, Any]], require_approved_state: str | None) -> list[str]:
    errors: list[str] = []
    state = manifest.get("approval_state")
    if state not in BLOCKED_APPROVAL_STATES | APPROVED_APPROVAL_STATES:
        errors.append(f"unknown approval state: {state}")
    expected_source = checksum_rows(claims)
    expected_seed = checksum_rows(seeds)
    expected_bundle = reports["pilot_approval_audit_report.json"]["pilot_bundle_sha256"]
    checksums = manifest.get("checksums", {})
    if checksums.get("source_claims_sha256") != expected_source:
        errors.append("source claims checksum mismatch")
    if checksums.get("seed_cases_sha256") != expected_seed:
        errors.append("seed cases checksum mismatch")
    expected_rows = reports["scale_readiness_report.json"]["row_counts"]
    for key, count in expected_rows.items():
        checksum_key = f"{key}_sha256"
        if count and checksum_key not in checksums:
            errors.append(f"{checksum_key} missing")
    if checksums.get("approval_bundle_sha256") != expected_bundle:
        errors.append("approval bundle checksum mismatch")
    if require_approved_state and state != require_approved_state:
        errors.append(f"required approval state {require_approved_state}, found {state}")
    if state not in BLOCKED_APPROVAL_STATES:
        errors.extend(reports["approval_audit_report.json"]["errors"])
    return errors


def validate_phase(phase: str, train: list[dict[str, Any]], dev: list[dict[str, Any]], final_eval: list[dict[str, Any]], manifest: dict[str, Any], reports: dict[str, dict[str, Any]], require_approved_state: str | None) -> list[str]:
    errors: list[str] = []
    row_counts = {"sft_train": len(train), "sft_dev": len(dev), "final_eval": len(final_eval)}
    if phase == "seed_bank":
        if any(row_counts.values()):
            errors.append("seed_bank phase rejects any non-empty expanded rows")
        if manifest.get("expansion_allowed") or manifest.get("sft_allowed"):
            errors.append("seed_bank phase cannot allow expansion or SFT")
    elif phase == "expansion_candidate":
        if manifest.get("sft_allowed"):
            errors.append("expansion_candidate phase must block SFT export")
        if row_counts != {"sft_train": 0, "sft_dev": 0, "final_eval": 0} and row_counts != PILOT_ROW_SPLIT_COUNTS:
            errors.append(f"expansion_candidate phase allows only the 300-row pilot counts {PILOT_ROW_SPLIT_COUNTS}")
    elif phase == "full_expansion_candidate":
        if row_counts != FULL_ROW_SPLIT_COUNTS:
            errors.append(f"full_expansion_candidate requires row counts {FULL_ROW_SPLIT_COUNTS}, found {row_counts}")
        if manifest.get("approval_state") != "BLOCKED_PENDING_EXPANSION_REVIEW":
            errors.append("full_expansion_candidate requires BLOCKED_PENDING_EXPANSION_REVIEW")
        if manifest.get("sft_allowed") or manifest.get("expansion_allowed"):
            errors.append("full_expansion_candidate cannot allow expansion or SFT export")
    elif phase == "sft_candidate":
        if manifest.get("approval_state") != "APPROVED_FOR_SFT" or manifest.get("sft_allowed") is not True:
            errors.append("sft_candidate requires APPROVED_FOR_SFT and sft_allowed true")
        if not train or not dev or not final_eval:
            errors.append("sft_candidate requires non-empty train/dev/final_eval rows")
    else:
        errors.append(f"unknown phase: {phase}")
    return errors
