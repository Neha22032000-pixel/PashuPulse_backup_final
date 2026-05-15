from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

MODEL_CANDIDATES = (
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
    "/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1",
    "/kaggle/input/gemma-4-e2b-it/transformers/default/1",
    "/kaggle/input/gemma-4-e2b-it",
)
DATA_CANDIDATES = (
    "/kaggle/input/pashu-saathi-sft-full-package",
    "/kaggle/input/datasets/nehak76044/pashu-saathi-sft-full-package",
)
PINNED_PACKAGES = (
    "unsloth==2026.5.2",
    "unsloth_zoo==2026.5.1",
    "trl==0.24.0",
    "transformers==5.5.0",
    "peft==0.19.1",
    "accelerate==1.13.0",
    "bitsandbytes==0.49.2",
    "datasets==4.3.0",
    "sentencepiece",
)
SYSTEM_PROMPT = (
    "You are PashuPulse, a rural India livestock assistant. Give short, safe, practical livestock help. "
    "Do not diagnose with certainty. Do not suggest medicines, injections, doses, invasive procedures, "
    "force-feeding, unsafe remedies, fake contacts, or milk/meat safety guarantees."
)
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"
OUT_DIR = Path(os.environ.get("PASHU_SAATHI_OUT_DIR", "/kaggle/working/pashu_pulse_throwaway_runtime_probe"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def log_status(phase: str, step: int, total: int, **extra: Any) -> None:
    row = {
        "created_at_utc": utc_now(),
        "phase": phase,
        "completed_steps": step,
        "total_steps": total,
        "remaining_steps": max(total - step, 0),
        **extra,
    }
    write_json(OUT_DIR / "run_status.json", row)
    print(f"[pashu-throwaway-sft] phase={phase} step={step}/{total} left={row['remaining_steps']} {json.dumps(clean(extra), ensure_ascii=False, sort_keys=True, allow_nan=False)}", flush=True)


def install_dependencies() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *PINNED_PACKAGES])


def versions() -> dict[str, str]:
    import importlib.metadata
    import torch

    names = ["unsloth", "unsloth_zoo", "trl", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"]
    out = {name: importlib.metadata.version(name) for name in names}
    out["torch"] = torch.__version__
    return out


def gpu_report() -> dict[str, Any]:
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not names:
        raise RuntimeError("No GPU visible; Kaggle T4 is required.")
    if not all("t4" in name.lower() for name in names):
        raise RuntimeError(f"Wrong GPU. Required T4, visible={names}")
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": names,
        "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
    }


def find_existing(candidates: tuple[str, ...], marker: str) -> Path:
    explicit = os.environ.get(marker, "").strip()
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    if marker == "PASHU_SAATHI_DATA_DIR" and Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").rglob("sft_package_manifest.json"))
        if matches:
            return matches[0].parent
    raise RuntimeError(f"Could not find {marker}; checked={candidates}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_package(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    required = ["sft_train.jsonl", "sft_dev.jsonl", "training_config.json", "sft_package_manifest.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing full package files: {missing}")
    forbidden = ["final_eval.jsonl", "eval_rubric.jsonl", "eval_prompts.jsonl", "adversarial_audit_prompts.jsonl", "baseline_predictions.jsonl"]
    leaked = [name for name in forbidden if (data_dir / name).exists()]
    if leaked:
        raise RuntimeError(f"training package contains eval-only files: {leaked}")
    manifest = read_json(data_dir / "sft_package_manifest.json")
    config = read_json(data_dir / "training_config.json")
    if manifest.get("package_mode") != "full" or config.get("package_mode") != "full":
        raise RuntimeError("throwaway full probe requires package_mode=full")
    train = read_jsonl(data_dir / "sft_train.jsonl")
    dev = read_jsonl(data_dir / "sft_dev.jsonl")
    bad = [row.get("row_id") for row in train + dev if row.get("EVAL_ONLY_DO_NOT_TRAIN") is True or row.get("parent_seed_split") == "final_eval_seed"]
    if bad:
        raise RuntimeError(f"eval-only/final rows leaked into package: {bad[:5]}")
    return train, dev, manifest, config


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


def language_attention_lora_targets(model: Any) -> list[str]:
    suffixes = (
        ".q_proj", ".k_proj", ".v_proj", ".o_proj",
        ".q_proj.linear", ".k_proj.linear", ".v_proj.linear", ".o_proj.linear",
    )
    candidates = [
        name
        for name, _module in model.named_modules()
        if "language_model" in name
        and "vision_tower" not in name
        and "audio_tower" not in name
        and ".self_attn." in name
        and name.endswith(suffixes)
    ]
    candidate_set = set(candidates)
    leaf_targets = [name for name in candidates if not any(other != name and other.startswith(name + ".") for other in candidate_set)]
    if not leaf_targets:
        raise RuntimeError("No language-model self-attention LoRA targets found.")
    return sorted(leaf_targets)


def render_rows(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[Any, dict[str, Any]]:
    from datasets import Dataset

    rendered = []
    token_lengths = []
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text(row)},
            {"role": "assistant", "content": assistant_text(row)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        text = text.removeprefix("<bos>")
        if INSTRUCTION_PART not in text or RESPONSE_PART not in text:
            raise RuntimeError(f"Gemma turn markers missing in row {row.get('row_id')}")
        token_lengths.append(len(inner(text, add_special_tokens=False)["input_ids"]))
        rendered.append({"text": text, "row_id": row.get("row_id", "")})
    return Dataset.from_list(rendered), {
        "row_count": len(rendered),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
        "avg_tokens": round(sum(token_lengths) / len(token_lengths), 2),
    }


def trainable_audit(model: Any) -> dict[str, Any]:
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    tower = [name for name, _count in trainable if "vision_tower" in name or "audio_tower" in name]
    if tower:
        raise RuntimeError(f"vision/audio tower trainables found: {tower[:5]}")
    return {"trainable_parameter_count": sum(count for _name, count in trainable), "trainable_tensor_count": len(trainable), "tower_trainable_count": 0}


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = OUT_DIR / "train_eval_curve.jsonl"
    if curve_path.exists():
        curve_path.unlink()
    data_dir = find_existing(DATA_CANDIDATES, "PASHU_SAATHI_DATA_DIR")
    model_path = find_existing(MODEL_CANDIDATES, "PASHU_SAATHI_MODEL_PATH")
    train_rows, dev_rows, package_manifest, config = validate_package(data_dir)
    max_length = int(config["max_length"])
    effective_batch = int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"])
    total_steps = math.ceil(len(train_rows) * float(config["num_train_epochs"]) / effective_batch)
    log_status("validated_package", 1, total_steps, data_dir=str(data_dir), train_rows=len(train_rows), dev_rows=len(dev_rows), promotion_allowed=False)
    install_dependencies()

    import torch
    from datasets import disable_caching
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    disable_caching()
    torch.cuda.empty_cache()
    log_status("validated_gpu", 2, total_steps, **gpu_report())
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if inner.pad_token is None:
        inner.pad_token = inner.eos_token
    inner.padding_side = "right"
    if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    lora_targets = language_attention_lora_targets(model)
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config["lora_r"]),
        target_modules=lora_targets,
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        random_state=int(config["seed"]),
        use_gradient_checkpointing="unsloth",
    )
    if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    train_dataset, train_audit = render_rows(train_rows, tokenizer)
    dev_dataset, dev_audit = render_rows(dev_rows, tokenizer)
    runtime_audit = {"train": train_audit, "dev": dev_audit, "lora_target_count": len(lora_targets), **trainable_audit(model), **gpu_report()}
    log_status("loaded_and_rendered", 3, total_steps, **runtime_audit)

    class ProgressCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            logs = logs or {}
            if "loss" not in logs:
                return
            step = int(state.global_step or 0)
            elapsed = time.time() - start
            eta = elapsed / max(step, 1) * max(total_steps - step, 0)
            row = {
                "created_at_utc": utc_now(),
                "phase": "training",
                "step": step,
                "total_steps": total_steps,
                "steps_left": max(total_steps - step, 0),
                "epoch": state.epoch,
                "train_loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "grad_norm": logs.get("grad_norm"),
                "elapsed_sec": round(elapsed, 2),
                "eta_sec": round(eta, 2),
                **gpu_report(),
            }
            append_jsonl(curve_path, row)
            status_extra = {key: value for key, value in row.items() if key not in {"created_at_utc", "phase", "step", "total_steps"}}
            log_status("training", step, total_steps, **status_extra)

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:
            metrics = metrics or {}
            step = int(state.global_step or 0)
            row = {"created_at_utc": utc_now(), "phase": "validation", "step": step, "eval_loss": metrics.get("eval_loss"), "elapsed_sec": round(time.time() - start, 2), **gpu_report()}
            append_jsonl(curve_path, row)
            status_extra = {key: value for key, value in row.items() if key not in {"created_at_utc", "phase", "step", "total_steps"}}
            log_status("validation", step, total_steps, **status_extra)

    args = SFTConfig(
        output_dir=str(OUT_DIR),
        dataset_text_field="text",
        max_length=max_length,
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_train_epochs=float(config["num_train_epochs"]),
        learning_rate=float(config["learning_rate"]),
        warmup_steps=0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=int(config["eval_steps"]),
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
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
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=dev_dataset, args=args)
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART, num_proc=1)
    trainer.add_callback(ProgressCallback())
    log_status("trainer_ready", 4, total_steps, **gpu_report())
    result = trainer.train()
    metrics = dict(result.metrics)
    if not math.isfinite(float(metrics.get("train_loss", 0.0))):
        raise RuntimeError(f"non-finite train_loss={metrics.get('train_loss')}")
    adapter_dir = OUT_DIR / "throwaway_runtime_probe_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    write_json(
        OUT_DIR / "metrics.json",
        {"created_at_utc": utc_now(), "status": "completed", "mode": "throwaway_runtime_probe", "promotion_allowed": False, "train_metrics": metrics, "elapsed_sec": round(time.time() - start, 2), "adapter_path": str(adapter_dir), **gpu_report()},
    )
    write_json(
        OUT_DIR / "run_manifest.json",
        {
            "created_at_utc": utc_now(),
            "status": "completed",
            "mode": "THROWAWAY_RUNTIME_ONLY",
            "promotion_allowed": False,
            "final_eval_used": False,
            "model_card_allowed": False,
            "package_manifest": package_manifest,
            "package_versions": versions(),
            "training_config": config,
            "runtime_audit": runtime_audit,
            "checksums": {
                "sft_train_sha256": sha256_file(data_dir / "sft_train.jsonl"),
                "sft_dev_sha256": sha256_file(data_dir / "sft_dev.jsonl"),
                "training_config_sha256": sha256_file(data_dir / "training_config.json"),
                "curve_sha256": sha256_file(curve_path),
            },
        },
    )
    log_status("completed", total_steps, total_steps, adapter_path=str(adapter_dir), promotion_allowed=False, elapsed_sec=round(time.time() - start, 2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(OUT_DIR / "sft_failure.json", {"created_at_utc": utc_now(), "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "promotion_allowed": False})
        raise
