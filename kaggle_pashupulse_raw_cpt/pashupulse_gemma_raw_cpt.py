from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PINNED_PACKAGES = {
    "unsloth": "2026.5.2",
    "unsloth_zoo": "2026.5.1",
    "transformers": "5.5.0",
    "peft": "0.19.1",
    "accelerate": "1.13.0",
    "bitsandbytes": "0.49.2",
    "datasets": "4.3.0",
}

MODEL_CANDIDATES = (
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
    "/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1",
    "/kaggle/input/gemma-4-e2b-it/transformers/default/1",
)

DATASET_ROOT_CANDIDATES = (
    "/kaggle/input/pashupulse-cpt-raw-clean-package",
    "/kaggle/input/datasets/nehak76044/pashupulse-cpt-raw-clean-package",
)

OUTPUT_DIR = Path("/kaggle/working/pashu_pulse_raw_cpt_v1")
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def install_runtime_dependencies() -> dict[str, str]:
    packages = [f"{name}=={version}" for name, version in PINNED_PACKAGES.items()]
    packages.append("sentencepiece")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages])
    return {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}


def assert_t4() -> dict[str, Any]:
    import torch

    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not devices:
        raise RuntimeError("No GPU visible. Run this Kaggle notebook with a T4 GPU.")
    if not all("t4" in device.lower() for device in devices):
        raise RuntimeError(f"This CPT run expects T4. Visible devices: {devices}")
    return {"visible_devices": devices, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def first_existing_dir(candidates: tuple[str, ...], required_file: str | None = None) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        if required_file is None or (path / required_file).exists():
            return path
    raise RuntimeError(f"Could not resolve directory from candidates: {candidates}")


def find_model_path() -> Path:
    return first_existing_dir(MODEL_CANDIDATES, "config.json")


def find_dataset_files() -> dict[str, Path]:
    root = first_existing_dir(DATASET_ROOT_CANDIDATES)
    names = {
        "train": ("train.jsonl", "cpt_train.jsonl", "raw_train.jsonl"),
        "dev": ("dev.jsonl", "validation.jsonl", "cpt_dev.jsonl", "raw_dev.jsonl"),
        "test": ("test.jsonl", "cpt_test.jsonl", "raw_test.jsonl"),
    }
    out: dict[str, Path] = {}
    for split, candidates in names.items():
        for name in candidates:
            path = root / name
            if path.exists():
                out[split] = path
                break
    if "train" not in out:
        jsonl_files = sorted(root.rglob("*.jsonl"))
        if not jsonl_files:
            raise RuntimeError(f"No JSONL files found in CPT dataset root: {root}")
        out["train"] = jsonl_files[0]
    return out


def pick_text(row: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk_text", "raw_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"No text-like field found in row keys: {sorted(row)}")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_source_file"] = str(path)
        row["_line_no"] = line_no
        row["text"] = pick_text(row)
        rows.append(row)
    return rows


def load_datasets() -> tuple[Any, Any | None, dict[str, Any]]:
    from datasets import Dataset

    files = find_dataset_files()
    train_rows = load_jsonl_rows(files["train"])
    dev_rows = load_jsonl_rows(files["dev"]) if "dev" in files else None
    manifest = {
        "files": {split: str(path) for split, path in files.items()},
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows or []),
    }
    return Dataset.from_list(train_rows), Dataset.from_list(dev_rows) if dev_rows else None, manifest


def tokenize_dataset(dataset: Any, tokenizer: Any) -> Any:
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    eos = inner.eos_token or ""

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        texts = [(text if text.endswith(eos) else text + eos) for text in batch["text"]]
        return inner(texts, truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False)

    keep = ["input_ids", "attention_mask"]
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=[col for col in dataset.column_names if col not in keep])
    return tokenized


def load_model_and_tokenizer() -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from unsloth import FastLanguageModel

    model_path = find_model_path()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if inner.pad_token is None:
        inner.pad_token = inner.eos_token
    inner.padding_side = "right"
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer, {"model_path": str(model_path)}


def train() -> None:
    import torch
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    versions = install_runtime_dependencies()
    gpu = assert_t4()
    train_dataset, dev_dataset, dataset_manifest = load_datasets()
    model, tokenizer, model_manifest = load_model_and_tokenizer()
    train_tokens = tokenize_dataset(train_dataset, tokenizer)
    dev_tokens = tokenize_dataset(dev_dataset, tokenizer) if dev_dataset is not None else None
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=inner, mlm=False)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_strategy="steps" if dev_tokens is not None else "no",
        eval_steps=25,
        save_steps=25,
        save_total_limit=4,
        report_to="none",
        seed=3407,
        data_seed=3407,
        max_grad_norm=0.3,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokens,
        eval_dataset=dev_tokens,
        data_collator=collator,
        tokenizer=inner,
    )
    train_result = trainer.train()
    final_dir = OUTPUT_DIR / "adapter_final"
    model.save_pretrained(str(final_dir))
    inner.save_pretrained(str(final_dir))
    if dev_tokens is not None:
        metrics = trainer.evaluate()
    else:
        metrics = {}
    write_json(
        OUTPUT_DIR / "run_manifest.json",
        {
            "created_at_utc": utc_now(),
            "status": "completed",
            "elapsed_sec": round(time.time() - start, 2),
            "package_versions": versions,
            "gpu": gpu,
            "dataset": dataset_manifest,
            "model": model_manifest,
            "training": {
                "method": "raw causal LM CPT with QLoRA",
                "max_seq_length": MAX_SEQ_LENGTH,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": LEARNING_RATE,
                "num_train_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRAD_ACCUM,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
            },
            "train_metrics": train_result.metrics,
            "eval_metrics": metrics,
            "adapter_final": str(final_dir),
        },
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        write_json(
            OUTPUT_DIR / "cpt_failure.json",
            {
                "created_at_utc": utc_now(),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
