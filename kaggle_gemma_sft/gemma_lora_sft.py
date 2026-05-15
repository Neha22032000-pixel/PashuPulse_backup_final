from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


DEFAULT_MODEL_CANDIDATES = (
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
    "/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1",
    "/kaggle/input/gemma-4-e2b-it/transformers/default/1",
    "/kaggle/input/gemma-4-e2b-it",
)
DEFAULT_CHAMPION_CHECKPOINT_CANDIDATES = (
    "/kaggle/input/pashupulse-sft-checkpoint-adapters/checkpoint-250",
    "/kaggle/input/pashu-pulse-lora/checkpoint-250",
    "/kaggle/input/pashu-saathi-gemma-lora-sft/pashu_pulse_lora/checkpoint-250",
    "pashu_saathi/kaggle_outputs/cleaned_full_sft_latest/pashu_pulse_lora/checkpoint-250",
)
PINNED_PACKAGES = {
    "unsloth": "2026.5.2",
    "unsloth_zoo": "2026.5.1",
    "trl": "0.24.0",
    "transformers": "5.5.0",
    "peft": "0.19.1",
    "accelerate": "1.13.0",
    "bitsandbytes": "0.49.2",
    "datasets": "4.3.0",
}
PINNED_REQUIREMENTS = [f"{name}=={version}" for name, version in PINNED_PACKAGES.items()]
# Static gate marker for tests and code review: unsloth==2026.5.2
SYSTEM_PROMPT = (
    "You are PashuPulse, a rural India livestock assistant. Keep answers supportive, safe, grounded, "
    "and practical. Do not diagnose with certainty. Do not give medicines, injections, doses, invasive "
    "procedures, force-feeding, unsafe home remedies, fake contacts, or milk/meat safety guarantees."
)
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"
REQUIRED_GPU_NAME = os.environ.get("PASHU_SAATHI_REQUIRED_GPU_NAME", "t4").strip().lower()
TURN_MARKER_CANDIDATES = (
    ("<|turn>user\n", "<|turn>model\n"),
    ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    ("<start_of_turn>user", "<start_of_turn>model"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_tree(path: Path) -> str:
    if not path.exists():
        return ""
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_clean(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def log_status(out_dir: Path, phase: str, completed: int, total: int, **extra: Any) -> None:
    status = {
        "created_at_utc": utc_now(),
        "phase": phase,
        "completed_steps": completed,
        "total_steps": total,
        "remaining_steps": max(total - completed, 0),
        **extra,
    }
    write_json(out_dir / "run_status.json", status)
    print(
        f"[pashu-sft] phase={phase} step={completed}/{total} steps_left={status['remaining_steps']} "
        f"{json.dumps(json_clean(extra), ensure_ascii=False, sort_keys=True, allow_nan=False)}",
        flush=True,
    )


def install_runtime_dependencies() -> str:
    packages = [*PINNED_REQUIREMENTS, "sentencepiece"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages])
    return "internet:pinned_unsloth_qlora_stack"


def installed_versions() -> dict[str, str]:
    versions = {}
    for name in PINNED_PACKAGES:
        versions[name] = importlib.metadata.version(name)
    return versions


def assert_pinned_versions() -> dict[str, str]:
    versions = installed_versions()
    mismatches = {name: {"expected": expected, "actual": versions.get(name)} for name, expected in PINNED_PACKAGES.items() if versions.get(name) != expected}
    if mismatches:
        raise RuntimeError(f"pinned package version mismatch: {mismatches}")
    return versions


def assert_required_gpu() -> dict[str, Any]:
    import torch

    visible_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not visible_devices:
        raise RuntimeError("No Kaggle GPU is visible; this run requires a T4 GPU.")
    if REQUIRED_GPU_NAME and not all(REQUIRED_GPU_NAME in name.lower() for name in visible_devices):
        raise RuntimeError(
            "Wrong Kaggle GPU family for PashuPulse. "
            f"required_substring={REQUIRED_GPU_NAME!r}; visible_devices={visible_devices}. "
            "Stop this run and rerun until Kaggle assigns a T4."
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"0", ""}:
        raise RuntimeError("PashuPulse SFT expects single-GPU CUDA_VISIBLE_DEVICES=0.")
    return {"required_gpu_name": REQUIRED_GPU_NAME, "visible_devices": visible_devices, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def gpu_memory() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_allocated_gb": 0.0, "gpu_reserved_gb": 0.0}
        return {
            "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        }
    except Exception:
        return {"gpu_allocated_gb": None, "gpu_reserved_gb": None}


def find_model_path() -> str:
    explicit = os.environ.get("PASHU_SAATHI_MODEL_PATH", "").strip()
    if explicit:
        return explicit
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for root in Path("/kaggle/input").rglob("*"):
        if re.search(r"gemma.*(e2b|2b).*it", root.name, re.I) and (root / "config.json").exists():
            return str(root)
    raise RuntimeError("No Gemma model path found. Set PASHU_SAATHI_MODEL_PATH or attach the Gemma 4 E2B Kaggle model.")


def user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(row.get("prompt", ""))


def assistant_text(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return str(row.get("answer", ""))


def validate_training_package(data_dir: Path, mode: str) -> dict[str, Any]:
    data_dir = resolve_training_data_dir(data_dir)
    required = ["sft_train.jsonl", "sft_dev.jsonl", "training_config.json", "sft_package_manifest.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing training package files: {missing}")
    forbidden = {
        "final_eval.jsonl",
        "eval_rubric.jsonl",
        "eval_prompts.jsonl",
        "adversarial_audit_prompts.jsonl",
        "baseline_predictions.jsonl",
        "safety_review_report.json",
        "source_fidelity_review_report.json",
        "language_usefulness_review_report.json",
        "diversity_review_report.json",
    }
    leaked = sorted(path.name for path in data_dir.glob("*") if path.name in forbidden)
    if leaked:
        raise RuntimeError(f"training package contains eval-only leakage: {leaked}")
    train = read_jsonl(data_dir / "sft_train.jsonl")
    dev = read_jsonl(data_dir / "sft_dev.jsonl")
    if not train or not dev:
        raise RuntimeError("training package must contain non-empty train and dev rows")
    for split_name, rows in [("train", train), ("dev", dev)]:
        bad = [
            row.get("row_id")
            for row in rows
            if row.get("EVAL_ONLY_DO_NOT_TRAIN") is True
            or row.get("parent_seed_split") == "final_eval_seed"
            or row.get("split") == "final_eval"
        ]
        if bad:
            raise RuntimeError(f"{split_name} contains eval-only/final lineage rows: {bad[:5]}")
    manifest = read_json(data_dir / "sft_package_manifest.json")
    if manifest.get("package_mode") not in {"smoke", "cleaned_candidate"}:
        raise RuntimeError("invalid package_mode in sft_package_manifest.json; full SFT requires cleaned_candidate and smoke requires smoke")
    if manifest.get("row_counts", {}).get("final_eval", 0) != 0:
        raise RuntimeError("training package manifest includes final_eval rows")
    if mode == "full" and manifest.get("package_mode") != "cleaned_candidate":
        raise RuntimeError("full training requires the cleaned_candidate SFT package; old full/repaired packages are blocked")
    if mode == "full" and manifest.get("status") != "BLOCKED_PENDING_CLEAN_DATA_REVIEW":
        raise RuntimeError("full training requires a checksum-reviewed cleaned candidate manifest")
    if mode == "full" and (manifest.get("sft_allowed") is True or manifest.get("promotion_allowed") is True):
        raise RuntimeError("cleaned candidate package must remain blocked/non-promotable until post-training review")
    if mode == "smoke" and manifest.get("package_mode") != "smoke":
        raise RuntimeError("smoke training requires a smoke SFT package")
    return {"train": train, "dev": dev, "manifest": manifest, "config": read_json(data_dir / "training_config.json"), "data_dir": data_dir}


def resolve_training_data_dir(data_dir: Path) -> Path:
    if (data_dir / "sft_package_manifest.json").exists():
        return data_dir
    preferred = Path(f"/kaggle/input/pashu-saathi-sft-{os.environ.get('PASHU_SAATHI_TRAIN_MODE', 'smoke').strip().lower()}-package")
    if (preferred / "sft_package_manifest.json").exists():
        return preferred
    input_root = Path("/kaggle/input")
    if input_root.exists():
        matches = sorted(input_root.rglob("sft_package_manifest.json"))
        if matches:
            return matches[0].parent
    return data_dir


def assert_full_review(data_dir: Path, package: dict[str, Any], config: dict[str, Any], mode: str) -> dict[str, Any] | None:
    if mode != "full":
        return None
    review_path = data_dir / "sft_param_review_decision.json"
    if not review_path.exists():
        raise RuntimeError("full SFT requires sft_param_review_decision.json")
    review = read_json(review_path)
    if review.get("decision") != "approved_for_full_sft":
        raise RuntimeError(f"full SFT params are not approved: {review.get('decision')}")
    if review.get("training_config_sha256") != sha256_file(data_dir / "training_config.json"):
        raise RuntimeError("full SFT param review has stale training_config_sha256")
    if review.get("sft_package_manifest_sha256") != sha256_file(data_dir / "sft_package_manifest.json"):
        raise RuntimeError("full SFT param review has stale sft_package_manifest_sha256")
    reviewed_package = review.get("package_checksums", {})
    actual_package = package.get("manifest", {}).get("checksums", {})
    for key in ("package_sft_train_sha256", "package_sft_dev_sha256"):
        if reviewed_package.get(key) != actual_package.get(key):
            raise RuntimeError(f"full SFT param review has stale {key}")
    reviewers = review.get("reviewers", [])
    if len(reviewers) != 3 or len({item.get("reviewer_id") for item in reviewers}) != 3:
        raise RuntimeError("full SFT param review requires three distinct reviewer IDs")
    if any(item.get("decision") != "approved" for item in reviewers):
        raise RuntimeError("all full SFT param reviewers must approve")
    return review


def resolve_config(package_config: dict[str, Any], mode: str) -> dict[str, Any]:
    smoke = mode == "smoke"
    defaults = {
        "seed": 76044,
        "max_seq_length": 512 if smoke else 768,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4 if smoke else 8,
        "learning_rate": 2e-4 if smoke else 1.5e-4,
        "num_train_epochs": 1 if smoke else 2,
        "max_steps": int(os.environ.get("PASHU_SAATHI_SMOKE_STEPS", "20")) if smoke else 0,
        "eval_steps": 5 if smoke else 25,
        "save_steps": 10 if smoke else 25,
        "save_total_limit": 2,
        "lora_r": 8 if smoke else 16,
        "lora_alpha": 16 if smoke else 32,
        "lora_dropout": 0.05,
        "gradient_checkpointing": "unsloth",
        "dataset_num_proc": 1,
        "packing": False,
        "fp16": True,
        "bf16": False,
        "load_in_4bit": True,
        "dtype": None,
    }
    defaults.update(package_config)
    env_overrides = {
        "max_seq_length": ("PASHU_SAATHI_MAX_SEQ_LENGTH", int),
        "per_device_train_batch_size": ("PASHU_SAATHI_BATCH_SIZE", int),
        "gradient_accumulation_steps": ("PASHU_SAATHI_GRAD_ACCUM", int),
        "learning_rate": ("PASHU_SAATHI_LEARNING_RATE", float),
        "num_train_epochs": ("PASHU_SAATHI_EPOCHS", float),
        "max_steps": ("PASHU_SAATHI_MAX_STEPS", int),
        "eval_steps": ("PASHU_SAATHI_EVAL_STEPS", int),
        "save_steps": ("PASHU_SAATHI_SAVE_STEPS", int),
        "save_total_limit": ("PASHU_SAATHI_SAVE_TOTAL_LIMIT", int),
        "lora_r": ("PASHU_SAATHI_LORA_R", int),
        "lora_alpha": ("PASHU_SAATHI_LORA_ALPHA", int),
        "lora_dropout": ("PASHU_SAATHI_LORA_DROPOUT", float),
    }
    for key, (env_name, caster) in env_overrides.items():
        if os.environ.get(env_name, "") != "":
            defaults[key] = caster(os.environ[env_name])
    if smoke:
        defaults["max_steps"] = min(int(defaults.get("max_steps") or 20), 30)
    return defaults


def resolve_resume_adapter() -> Path | None:
    explicit = os.environ.get("PASHU_SAATHI_RESUME_ADAPTER", "").strip()
    enabled = os.environ.get("PASHU_SAATHI_CONTINUE_FROM_CHAMPION", "").strip().lower() in {"1", "true", "yes"}
    if not explicit and not enabled:
        return None
    candidates = (explicit, *DEFAULT_CHAMPION_CHECKPOINT_CANDIDATES) if explicit else DEFAULT_CHAMPION_CHECKPOINT_CANDIDATES
    for candidate in candidates:
        path = Path(candidate)
        if (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists():
            return path
    zip_explicit = os.environ.get("PASHU_SAATHI_RESUME_ADAPTER_ZIP", "").strip()
    if zip_explicit:
        extracted = extract_resume_adapter_zip(Path(zip_explicit))
        if extracted:
            return extracted
    if Path("/kaggle/input").exists():
        for adapter_config in sorted(Path("/kaggle/input").rglob("adapter_config.json")):
            path = adapter_config.parent
            if "checkpoint-250" in str(path).lower() and (path / "adapter_model.safetensors").exists():
                return path
        for zip_path in sorted(Path("/kaggle/input").rglob("checkpoint-250.zip")):
            extracted = extract_resume_adapter_zip(zip_path)
            if extracted:
                return extracted
    return None


def extract_resume_adapter_zip(zip_path: Path) -> Path | None:
    import zipfile

    if not zip_path.exists():
        return None
    target = Path("/kaggle/working/pashu_resume_adapters/checkpoint-250")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    if (target / "adapter_config.json").exists():
        return target
    matches = sorted(target.rglob("adapter_config.json"))
    return matches[0].parent if matches else None


def assert_continuation_output_safe(out_dir: Path, resume_adapter: Path | None) -> None:
    if not resume_adapter:
        return
    protected = Path(os.environ.get("PASHU_SAATHI_PROTECTED_CHECKPOINT", "")).resolve() if os.environ.get("PASHU_SAATHI_PROTECTED_CHECKPOINT") else None
    resolved_out = out_dir.resolve()
    if protected and resolved_out == protected:
        raise RuntimeError(f"continuation output would overwrite protected checkpoint: {protected}")
    if "checkpoint-250" in resolved_out.name.lower():
        raise RuntimeError(f"continuation output path must not be named like the protected champion: {resolved_out}")


def render_chat_text(row: dict[str, Any], tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text(row)},
        {"role": "assistant", "content": assistant_text(row)},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return str(text).removeprefix("<bos>")


def detect_turn_markers(rendered_texts: list[str]) -> tuple[str, str]:
    joined = "\n".join(rendered_texts[:5])
    for instruction_part, response_part in TURN_MARKER_CANDIDATES:
        if instruction_part in joined and response_part in joined:
            return instruction_part, response_part
    sample = joined[:1200].replace("\n", "\\n")
    raise RuntimeError(f"Gemma chat user/model markers not found in rendered rows; sample={sample}")


def make_text_dataset(rows: list[dict[str, Any]], tokenizer: Any, markers: tuple[str, str] | None = None) -> tuple[Any, dict[str, Any], tuple[str, str]]:
    from datasets import Dataset

    rendered = []
    for row in rows:
        text = render_chat_text(row, tokenizer)
        rendered.append({"text": text, "row_id": row.get("row_id", "")})
    markers = markers or detect_turn_markers([item["text"] for item in rendered])
    instruction_part, response_part = markers
    marker_failures = [
        item["row_id"]
        for item in rendered
        if instruction_part not in item["text"] or response_part not in item["text"]
    ]
    if marker_failures:
        raise RuntimeError(f"Gemma-4 chat markers missing from rendered rows: {marker_failures[:5]}")
    token_lengths = []
    inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    for item in rendered:
        encoded = inner_tokenizer(item["text"], add_special_tokens=False)
        if encoded is None or "input_ids" not in encoded:
            raise RuntimeError(f"tokenizer did not return input_ids for row {item['row_id']}")
        token_lengths.append(len(encoded["input_ids"]))
    audit = {
        "row_count": len(rendered),
        "row_ids": [item["row_id"] for item in rendered[:20]],
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "avg_tokens": round(sum(token_lengths) / max(len(token_lengths), 1), 2),
        "instruction_part": instruction_part,
        "response_part": response_part,
        "sample_rendered_prefix": rendered[0]["text"][:500] if rendered else "",
    }
    return Dataset.from_list(rendered), audit, markers


def language_attention_lora_targets(model: Any) -> list[str]:
    suffixes = (
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".o_proj",
        ".q_proj.linear",
        ".k_proj.linear",
        ".v_proj.linear",
        ".o_proj.linear",
    )
    candidates = [
        name
        for name, _module in model.named_modules()
        if "language_model" in name
        and "vision_tower" not in name
        and "audio_tower" not in name
        and "tower" not in name
        and ".self_attn." in name
        and name.endswith(suffixes)
    ]
    candidate_set = set(candidates)
    leaf_targets = [
        name for name in candidates if not any(other != name and other.startswith(name + ".") for other in candidate_set)
    ]
    if not leaf_targets:
        raise RuntimeError("No language-model self-attention LoRA targets found.")
    bad = [name for name in leaf_targets if any(token in name for token in ("vision_tower", "audio_tower", "tower", "multi_modal"))]
    if bad:
        raise RuntimeError(f"LoRA target resolver selected non-text tower modules: {bad[:5]}")
    return sorted(leaf_targets)


def trainable_summary(model: Any) -> dict[str, Any]:
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    tower_trainables = [name for name, _count in trainable if any(token in name for token in ("vision_tower", "audio_tower", "tower"))]
    if tower_trainables:
        raise RuntimeError(f"non-text tower parameters are trainable: {tower_trainables[:5]}")
    return {
        "trainable_parameter_count": sum(count for _name, count in trainable),
        "trainable_tensor_count": len(trainable),
        "tower_trainable_count": len(tower_trainables),
        "sample_trainable_names": [name for name, _count in trainable[:20]],
    }


def qlora_summary(model: Any) -> dict[str, Any]:
    quantization_config = getattr(model, "quantization_config", None) or getattr(getattr(model, "base_model", None), "quantization_config", None)
    quant_text = repr(quantization_config)
    four_bit_module_count = sum(1 for _name, module in model.named_modules() if "4bit" in type(module).__name__.lower())
    active = four_bit_module_count > 0 or "load_in_4bit=True" in quant_text or "4bit" in quant_text.lower()
    if not active:
        raise RuntimeError("QLoRA/4-bit quantization is not active.")
    return {"qlora_active": active, "four_bit_module_count": four_bit_module_count, "quantization_config": quant_text}


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model_path = find_model_path()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=int(config["max_seq_length"]),
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if inner_tokenizer.pad_token is None:
        inner_tokenizer.pad_token = inner_tokenizer.eos_token
    inner_tokenizer.padding_side = "right"
    if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    lora_targets = language_attention_lora_targets(model)
    resume_adapter = resolve_resume_adapter()
    continuation_meta: dict[str, Any] = {"continuation_from_adapter": False}
    if resume_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
        if config.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        continuation_meta = {
            "continuation_from_adapter": True,
            "continuation_mode": "checkpoint_250_adapter_weights_fresh_optimizer",
            "protected_parent_adapter_path": str(resume_adapter),
            "protected_parent_adapter_sha256": sha256_tree(resume_adapter),
        }
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(config["lora_r"]),
            target_modules=lora_targets,
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            random_state=int(config["seed"]),
            use_gradient_checkpointing="unsloth" if config.get("gradient_checkpointing") else False,
        )
    if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    runtime = {
        "model_path": model_path,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "lora_target_count": len(lora_targets),
        "lora_targets": lora_targets[:40],
        **continuation_meta,
        **qlora_summary(model),
        **trainable_summary(model),
        **gpu_memory(),
    }
    return model, tokenizer, runtime


class PashuProgressCallback:
    def __init__(self, out_dir: Path, total_steps: int, curve_path: Path, start_time: float, save_steps: int) -> None:
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def on_log(inner_self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
                logs = logs or {}
                step = int(state.global_step or 0)
                elapsed = time.time() - start_time
                eta = elapsed / max(step, 1) * max(total_steps - step, 0)
                row = {
                    "created_at_utc": utc_now(),
                    "phase": "training",
                    "step": step,
                    "total_steps": total_steps,
                    "steps_left": max(total_steps - step, 0),
                    "epoch": float(state.epoch or 0.0),
                    "train_loss": logs.get("loss"),
                    "learning_rate": logs.get("learning_rate"),
                    "elapsed_sec": round(elapsed, 2),
                    "eta_sec": round(eta, 2),
                    **gpu_memory(),
                }
                append_jsonl(curve_path, row)
                checkpoint = str(out_dir / f"checkpoint-{step}") if save_steps and step and step % save_steps == 0 else ""
                status_extra = {key: value for key, value in row.items() if key not in {"created_at_utc", "phase", "step", "total_steps"}}
                log_status(out_dir, "training", step, total_steps, checkpoint_path=checkpoint, **status_extra)

            def on_evaluate(inner_self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:
                metrics = metrics or {}
                step = int(state.global_step or 0)
                row = {
                    "created_at_utc": utc_now(),
                    "phase": "validation",
                    "eval_start_step": step,
                    "eval_batch_done": metrics.get("eval_batches_done"),
                    "eval_batch_total": metrics.get("eval_batches_total"),
                    "eval_batches_left": metrics.get("eval_batches_left"),
                    "eval_loss": metrics.get("eval_loss"),
                    "elapsed_sec": round(time.time() - start_time, 2),
                    "validation_status": "completed",
                    **gpu_memory(),
                }
                append_jsonl(curve_path, row)
                status_extra = {key: value for key, value in row.items() if key not in {"created_at_utc", "phase", "step", "total_steps"}}
                log_status(out_dir, "validation", step, total_steps, **status_extra)

        self.callback = _Callback()


def create_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    config: dict[str, Any],
    out_dir: Path,
    curve_path: Path,
    total_steps: int,
    start_time: float,
    markers: tuple[str, str],
) -> Any:
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    args = SFTConfig(
        output_dir=str(out_dir),
        dataset_text_field="text",
        max_length=int(config["max_seq_length"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        max_steps=int(config["max_steps"]) if int(config.get("max_steps") or 0) > 0 else -1,
        num_train_epochs=float(config["num_train_epochs"]),
        learning_rate=float(config["learning_rate"]),
        warmup_steps=0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=int(config["eval_steps"]),
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config.get("save_total_limit") or 2),
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        max_grad_norm=0.3,
        dataset_num_proc=1,
        packing=False,
        report_to=[],
        seed=int(config["seed"]),
        remove_unused_columns=False,
    )
    if getattr(args, "dataset_num_proc", None) != 1:
        raise RuntimeError("SFTConfig.dataset_num_proc did not stick")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )
    instruction_part, response_part = markers
    masking_kwargs = {"instruction_part": instruction_part, "response_part": response_part}
    if "num_proc" in inspect.signature(train_on_responses_only).parameters:
        masking_kwargs["num_proc"] = 1
    trainer = train_on_responses_only(trainer, **masking_kwargs)
    trainer.add_callback(PashuProgressCallback(out_dir, total_steps, curve_path, start_time, int(config["save_steps"])).callback)
    return trainer


def total_train_steps(train_count: int, config: dict[str, Any]) -> int:
    explicit = int(config.get("max_steps") or 0)
    if explicit > 0:
        return explicit
    effective_batch = int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"])
    return math.ceil(train_count * float(config["num_train_epochs"]) / max(effective_batch, 1))


def adapter_audit(adapter_dir: Path) -> dict[str, Any]:
    files = sorted(path.name for path in adapter_dir.rglob("*") if path.is_file())
    tower_files = [name for name in files if "vision_tower" in name or "audio_tower" in name]
    if tower_files:
        raise RuntimeError(f"adapter contains tower files: {tower_files[:5]}")
    return {"adapter_path": str(adapter_dir), "file_count": len(files), "tower_file_count": len(tower_files), "sample_files": files[:20]}


def main() -> None:
    data_dir = Path(os.environ.get("PASHU_SAATHI_DATA_DIR", "/kaggle/input/pashu-saathi-sft-package"))
    out_dir = Path(os.environ.get("PASHU_SAATHI_OUT_DIR", "/kaggle/working/pashu_pulse_lora"))
    mode = os.environ.get("PASHU_SAATHI_TRAIN_MODE", "full").strip().lower()
    if mode not in {"smoke", "full"}:
        raise SystemExit("PASHU_SAATHI_TRAIN_MODE must be smoke or full")
    resume_adapter = resolve_resume_adapter()
    assert_continuation_output_safe(out_dir, resume_adapter)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "train_eval_curve.jsonl"
    if curve_path.exists():
        curve_path.unlink()
    start = time.time()

    package = validate_training_package(data_dir, mode)
    data_dir = package["data_dir"]
    config = resolve_config(package["config"], mode)
    review = assert_full_review(data_dir, package, config, mode)
    total_steps = total_train_steps(len(package["train"]), config)
    log_status(out_dir, "validated_training_package", 1, total_steps, mode=mode, train_rows=len(package["train"]), dev_rows=len(package["dev"]), config=config)

    gpu_gate = assert_required_gpu()
    log_status(out_dir, "validated_gpu", 2, total_steps, **gpu_gate)
    dependency_note = install_runtime_dependencies()
    versions = assert_pinned_versions()
    log_status(out_dir, "installed_dependencies", 3, total_steps, dependency_note=dependency_note, package_versions=versions)

    from datasets import disable_caching

    disable_caching()
    model, tokenizer, runtime = load_model_and_tokenizer(config)
    log_status(out_dir, "loaded_model", 4, total_steps, **runtime)
    train_dataset, train_audit, markers = make_text_dataset(package["train"], tokenizer)
    dev_dataset, dev_audit, _markers = make_text_dataset(package["dev"], tokenizer, markers)
    write_json(out_dir / "response_mask_audit.json", {"train": train_audit, "dev": dev_audit})
    write_json(out_dir / "lora_target_audit.json", runtime)

    trainer = create_trainer(model, tokenizer, train_dataset, dev_dataset, config, out_dir, curve_path, total_steps, start, markers)
    log_status(out_dir, "trainer_ready", 5, total_steps, train_rows=len(train_dataset), dev_rows=len(dev_dataset), **gpu_memory())
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate()
    if not math.isfinite(float(metrics.get("train_loss", 0.0))):
        raise RuntimeError(f"non-finite train loss: {metrics.get('train_loss')}")
    if "eval_loss" in eval_metrics and not math.isfinite(float(eval_metrics["eval_loss"])):
        raise RuntimeError(f"non-finite eval loss: {eval_metrics.get('eval_loss')}")

    final_adapter = out_dir / "adapter_final"
    trainer.model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    adapter_report = adapter_audit(final_adapter)
    elapsed = round(time.time() - start, 2)
    output_metrics = {
        "created_at_utc": utc_now(),
        "status": "completed",
        "mode": mode,
        "total_steps": total_steps,
        "train_rows": len(package["train"]),
        "dev_rows": len(package["dev"]),
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "elapsed_sec": elapsed,
        **adapter_report,
        **gpu_memory(),
    }
    write_json(out_dir / "metrics.json", output_metrics)
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at_utc": utc_now(),
            "status": "completed",
            "mode": mode,
            "package_manifest": package["manifest"],
            "training_config": config,
            "full_param_review": review,
            "runtime": runtime,
            "protected_champion": {
                "checkpoint_name": "checkpoint-250" if resume_adapter else "",
                "path": str(resume_adapter) if resume_adapter else "",
                "sha256": sha256_tree(resume_adapter) if resume_adapter else "",
                "overwrite_allowed": False if resume_adapter else None,
            },
            "package_versions": versions,
            "checksums": {
                "sft_train_sha256": sha256_file(data_dir / "sft_train.jsonl"),
                "sft_dev_sha256": sha256_file(data_dir / "sft_dev.jsonl"),
                "training_config_sha256": sha256_file(data_dir / "training_config.json"),
                "sft_package_manifest_sha256": sha256_file(data_dir / "sft_package_manifest.json"),
                "curve_sha256": sha256_file(curve_path),
            },
            "prompt_template_sha256": sha256_text(SYSTEM_PROMPT),
            "reportable": True,
            "sft_allowed_at_runtime": mode == "full",
        },
    )
    log_status(out_dir, "completed", total_steps, total_steps, adapter_path=str(final_adapter), elapsed_sec=elapsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure_dir = Path(os.environ.get("PASHU_SAATHI_OUT_DIR", "/kaggle/working/pashu_pulse_lora"))
        write_json(
            failure_dir / "sft_failure.json",
            {
                "created_at_utc": utc_now(),
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "phase": "sft",
                **gpu_memory(),
            },
        )
        raise
