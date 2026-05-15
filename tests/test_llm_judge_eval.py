from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test_runs" / "llm_judge_pytest"
LOCAL_JUDGE_PATH = PROJECT_ROOT / "kaggle_final_eval_local_judge" / "final_eval_local_judge.py"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fresh_case(name: str) -> Path:
    path = TEST_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def load_local_judge_module():
    spec = importlib.util.spec_from_file_location("pashu_final_eval_local_judge", LOCAL_JUDGE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llm_judge_rule_smoke_is_nonreportable_and_scores_rows():
    tmp_path = fresh_case("rule_smoke")
    rubric = PROJECT_ROOT / "kaggle_packages" / "eval_package" / "eval_rubric.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = []
    for row in read_jsonl(rubric)[:3]:
        answer = (
            "Safe step: give clean water, shade, rest, and observe closely. "
            "Do not give injection, medicine, dose, oil drench, cutting, puncturing, or force-feeding on your own."
        )
        if row.get("must_escalate"):
            answer += " Contact a vet or trained animal health worker / Pashu Sakhi / dairy cooperative when reachable."
        rows.append({"eval_id": row["eval_id"], "system": "pashu_saathi_lora", "prediction": answer, "reportable": False})
    predictions.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "judge"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "evals" / "llm_judge_eval.py"),
            "--eval-rubric",
            str(rubric),
            "--predictions-file",
            str(predictions),
            "--out-dir",
            str(out_dir),
            "--systems",
            "pashu_saathi_lora",
            "--max-eval-rows",
            "3",
            "--judge-backend",
            "rule_smoke",
            "--allow-nonreportable",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads((out_dir / "llm_judge_score_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "llm_judge_manifest.json").read_text(encoding="utf-8"))
    judgments = read_jsonl(out_dir / "llm_judge_results.jsonl")
    assert report["reportable"] is False
    assert report["passed_for_model_review"] is False
    assert report["prediction_count"] == 3
    assert report["summary"]["pashu_saathi_lora"]["hard_safety_failure_rate"] == 0
    assert manifest["reportable"] is False
    assert len(judgments) == 3


def test_llm_judge_blocks_unconfigured_reportable_backend():
    tmp_path = fresh_case("unconfigured_backend")
    rubric = PROJECT_ROOT / "kaggle_packages" / "eval_package" / "eval_rubric.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    first = read_jsonl(rubric)[0]
    predictions.write_text(
        json.dumps({"eval_id": first["eval_id"], "system": "pashu_saathi_lora", "prediction": "clean water and rest", "reportable": False}) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "evals" / "llm_judge_eval.py"),
            "--eval-rubric",
            str(rubric),
            "--predictions-file",
            str(predictions),
            "--out-dir",
            str(tmp_path / "judge"),
            "--systems",
            "pashu_saathi_lora",
            "--judge-backend",
            "openai_compatible",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "judge-model" in (result.stderr + result.stdout)


def test_local_final_eval_judge_prompt_contains_expected_contract():
    module = load_local_judge_module()
    rubric = read_jsonl(PROJECT_ROOT / "kaggle_packages" / "eval_package" / "eval_rubric.jsonl")[0]
    final_eval = read_jsonl(PROJECT_ROOT / "data" / "processed" / "full_expansion" / "final_eval.jsonl")[0]
    prompt = module.build_judge_prompt(rubric, final_eval, "Model answer with clean water and shade.")
    assert "EXPECTED_OUTPUT" in prompt
    assert "expected_contract" in prompt
    assert final_eval["assistant_response"] in prompt
    assert rubric["prompt"] in prompt
    assert "Return ONLY valid JSON" in prompt
    assert "medicine, injection, dose" in prompt


def test_local_final_eval_seal_check_detects_train_overlap():
    module = load_local_judge_module()
    final_eval = [{"row_id": "eval_1", "parent_seed_id": "seed_eval"}]
    clean = module.validate_final_eval_is_sealed(final_eval, [{"row_id": "train_1", "parent_seed_id": "seed_train"}], [])
    dirty = module.validate_final_eval_is_sealed(final_eval, [{"row_id": "eval_1", "parent_seed_id": "seed_train"}], [])
    seed_dirty = module.validate_final_eval_is_sealed(final_eval, [{"row_id": "train_1", "parent_seed_id": "seed_eval"}], [])
    assert clean["sealed"] is True
    assert dirty["sealed"] is False
    assert dirty["leaks"]["train_row_overlap"] == ["eval_1"]
    assert seed_dirty["sealed"] is False
    assert seed_dirty["leaks"]["train_seed_overlap"] == ["seed_eval"]


def test_local_final_eval_summary_prefers_conservative_checkpoint_on_tiny_margin():
    module = load_local_judge_module()
    rows = []
    for system in ["base_gemma", "checkpoint_250", "checkpoint_264"]:
        for index in range(10):
            rows.append(
                {
                    "system": system,
                    "overall_pass": system != "base_gemma" or index < 5,
                    "blocking_failure": False,
                    "parse_ok": True,
                    "scores": {
                        "safety": 5,
                        "expected_match": 4 if system == "checkpoint_250" else 4.1,
                        "escalation": 5,
                        "language": 4,
                        "farmer_usefulness": 4,
                    },
                }
            )
    report = module.summarize_scores(rows, parse_failures=0, prediction_count=len(rows), min_valid_json_rate=0.95)
    assert report["valid_json_rate"] == 1.0
    assert report["base_to_checkpoint_delta"]["checkpoint_250"] == 0.5
    assert report["recommended_system"] == "checkpoint_250_conservative_margin"


def test_group_judge_groups_predictions_and_prompt_contains_three_answers():
    module = load_local_judge_module()
    rubric = read_jsonl(PROJECT_ROOT / "kaggle_packages" / "eval_package" / "eval_rubric.jsonl")[0]
    final_eval = read_jsonl(PROJECT_ROOT / "data" / "processed" / "full_expansion" / "final_eval.jsonl")[0]
    predictions = [
        {"eval_id": rubric["eval_id"], "system": "base_gemma", "prediction": "base answer", "prediction_sha256": "b"},
        {"eval_id": rubric["eval_id"], "system": "checkpoint_250", "prediction": "250 answer", "prediction_sha256": "c250"},
        {"eval_id": rubric["eval_id"], "system": "checkpoint_264", "prediction": "264 answer", "prediction_sha256": "c264"},
    ]
    grouped = module.group_predictions_by_eval_id(predictions)
    prompt = module.build_group_judge_prompt(rubric, final_eval, grouped[rubric["eval_id"]])
    assert grouped[rubric["eval_id"]]["base_gemma"]["prediction"] == "base answer"
    assert "base answer" in prompt
    assert "250 answer" in prompt
    assert "264 answer" in prompt
    assert "expected_contract" in prompt
    assert '"winner"' in prompt


def test_group_judge_schema_accepts_minimal_fields():
    module = load_local_judge_module()
    valid = {
        "winner": "checkpoint_250",
        "safety_violation": False,
        "safety_comment": "No safety issue.",
        "reason": "Best matched the expected answer.",
    }
    invalid = {
        "winner": "checkpoint-250",
        "safety_violation": "no",
        "safety_comment": "bad",
        "reason": "bad",
    }
    assert module.validate_group_judge_json(valid) == []
    errors = module.validate_group_judge_json(invalid)
    assert "bad_winner" in errors
    assert "bad_safety_violation" in errors


def test_group_judge_summary_counts_winners_and_safety():
    module = load_local_judge_module()
    rows = [
        {"winner": "base_gemma", "parse_ok": True, "safety_violation": False},
        {"winner": "checkpoint_250", "parse_ok": True, "safety_violation": True, "eval_id": "e2", "safety_comment": "dose mentioned", "reason": "unsafe"},
        {"winner": "checkpoint_250", "parse_ok": True, "safety_violation": False},
        {"winner": "checkpoint_264", "parse_ok": False, "safety_violation": True, "eval_id": "e4", "safety_comment": "parse failed", "reason": "judge_json_parse_failed"},
        {"winner": "tie", "parse_ok": True, "safety_violation": False},
    ]
    report = module.summarize_group_results(rows, parse_failures=1, total=5, min_valid_json_rate=0.9)
    assert report["winner_counts"]["checkpoint_250"] == 2
    assert report["winner_counts"]["base_gemma"] == 1
    assert report["safety_violation_count"] == 2
    assert report["valid_json_rate"] == 0.8
    assert report["final_decision"] == "blocked_by_judge_parse_failures"
    assert report["recommended_system"] == "checkpoint_250"
