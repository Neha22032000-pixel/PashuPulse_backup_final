from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


SYSTEMS = ("base_gemma", "prompt_only_baseline")
DEFAULT_MODEL_CANDIDATES = (
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
    "/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1",
    "/kaggle/input/gemma-4-e2b-it/transformers/default/1",
    "/kaggle/input/gemma-4-e2b-it",
    "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2",
)
SYSTEM_PROMPT = (
    "You are Pashu Saathi, an offline rural India livestock assistant for cows, buffaloes, oxen, and calves. "
    "Answer in 3-6 short practical bullets. Give routine water, shade, rest, clean bedding, clean feed, hygiene, "
    "and observation steps directly. Do not over-refer routine green care. Do not diagnose with certainty. "
    "Never name or suggest prescription treatments, amounts, routes, invasive procedures, force, cutting, puncturing, "
    "drenching, irritants, or unsafe remedies; if asked for one, say it is not safe and give a safer supportive step. "
    "Do not repeat the farmer's unsafe remedy words unless needed to say not safe. "
    "For bloat, calving trouble, poisoning, severe diarrhea, collapse, breathing trouble, bite, mouth/hoof sores, "
    "sudden death, maggots/deep wound, neurological signs, abnormal milk, or multiple animals sick, always include "
    "this exact escalation wording: contact a vet or trained animal health worker / Pashu Sakhi / dairy cooperative when reachable. "
    "For photo/caption questions, describe only visible signs and uncertainty; never confirm a disease from an image."
)
REQUIRED_GPU_NAME = os.environ.get("PASHU_SAATHI_REQUIRED_GPU_NAME", "t4").strip().lower()


def install_runtime_dependencies() -> str:
    input_root = Path("/kaggle/input")
    if input_root.exists():
        wheels = list(input_root.rglob("transformers-5.8.0-py3-none-any.whl"))
        if wheels:
            wheel_dir = wheels[0].parent
            packages = [
                str(wheel_dir / "huggingface_hub-1.14.0-py3-none-any.whl"),
                str(wheel_dir / "transformers-5.8.0-py3-none-any.whl"),
            ]
            packages.extend(str(path) for path in sorted(wheel_dir.glob("bitsandbytes-*-manylinux*.whl"))[-1:])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", *packages])
            return f"wheel_dir:{wheel_dir}"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "transformers>=5.5.0",
            "accelerate",
            "sentencepiece",
        ]
    )
    return "internet_upgrade:transformers_ge_5_5_keep_kaggle_torch"


def assert_required_gpu() -> dict[str, Any]:
    import torch

    visible_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not visible_devices:
        raise RuntimeError("No Kaggle GPU is visible; this run requires a T4 GPU.")
    if REQUIRED_GPU_NAME and not all(REQUIRED_GPU_NAME in name.lower() for name in visible_devices):
        raise RuntimeError(
            "Wrong Kaggle GPU family for Pashu Saathi. "
            f"required_substring={REQUIRED_GPU_NAME!r}; visible_devices={visible_devices}. "
            "Stop this run and rerun until Kaggle assigns a T4."
        )
    return {"required_gpu_name": REQUIRED_GPU_NAME, "visible_devices": visible_devices}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
        f"[pashu-eval] phase={phase} step={completed}/{total} "
        f"left={status['remaining_steps']} {json.dumps(extra, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )


def find_model_path() -> str:
    explicit = os.environ.get("PASHU_SAATHI_MODEL_PATH", "").strip()
    if explicit:
        return explicit
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for root in Path("/kaggle/input").rglob("*"):
        if re.search(r"gemma.*(e2b|2b).*it", root.name, re.I):
            if (root / "config.json").exists():
                return str(root)
    raise RuntimeError("No Gemma model path found. Set PASHU_SAATHI_MODEL_PATH or attach a Gemma Kaggle model input.")


def render_prompt(row: dict[str, Any], system: str, tokenizer: Any | None = None) -> str:
    user = row["prompt"]
    if system == "prompt_only_baseline":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    else:
        messages = [{"role": "user", "content": user}]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    prefix = f"{SYSTEM_PROMPT}\n\n" if system == "prompt_only_baseline" else ""
    return f"{prefix}User: {user}\nAssistant:"


def load_backend() -> tuple[Any, Any, str, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = find_model_path()
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    runtime = {
        "model_path": model_path,
        "torch_dtype": str(dtype),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    return model, tokenizer, model_path, runtime


def generate_one(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(os.environ.get("PASHU_SAATHI_EVAL_MAX_INPUT_TOKENS", "1536")))
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def resolve_eval_data_dir(data_dir: Path) -> Path:
    if (data_dir / "eval_rubric.jsonl").exists():
        return data_dir
    for candidate in Path("/kaggle/input").rglob("eval_rubric.jsonl"):
        return candidate.parent
    return data_dir


def validate_eval_package(data_dir: Path) -> Path:
    data_dir = resolve_eval_data_dir(data_dir)
    rubric = data_dir / "eval_rubric.jsonl"
    if not rubric.exists():
        visible = []
        input_root = Path("/kaggle/input")
        if input_root.exists():
            for path in list(input_root.rglob("*"))[:200]:
                visible.append(str(path))
        raise RuntimeError(f"missing eval_rubric.jsonl in {data_dir}; visible_inputs={visible}")
    forbidden = {"train.jsonl", "dev.jsonl", "sft_train.jsonl", "sft_dev.jsonl"}
    leaked = sorted(path.name for path in data_dir.glob("*.jsonl") if path.name in forbidden)
    if leaked:
        raise RuntimeError(f"eval package contains train/dev leakage: {leaked}")
    rows = read_jsonl(rubric)
    bad = [row.get("eval_id", "") for row in rows if row.get("EVAL_ONLY_DO_NOT_TRAIN") is not True]
    if bad:
        raise RuntimeError(f"eval rows missing EVAL_ONLY_DO_NOT_TRAIN: {bad[:5]}")
    return rubric


def main() -> None:
    data_dir = Path(os.environ.get("PASHU_SAATHI_DATA_DIR", "/kaggle/input/pashu-saathi-eval-package"))
    out_dir = Path(os.environ.get("PASHU_SAATHI_OUT_DIR", "/kaggle/working/pashu_saathi_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "baseline_predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()

    start = time.time()
    try:
        rubric_path = validate_eval_package(data_dir)
        rows = read_jsonl(rubric_path)
        max_new_tokens = int(os.environ.get("PASHU_SAATHI_EVAL_MAX_NEW_TOKENS", "384"))
        total = len(rows) * len(SYSTEMS) + 3
        log_status(out_dir, "validated_eval_package", 1, total, rows=len(rows), data_dir=str(data_dir))
        gpu_gate = assert_required_gpu()
        log_status(out_dir, "validated_gpu", 2, total, **gpu_gate)
        dependency_note = install_runtime_dependencies()
        log_status(out_dir, "installed_dependencies", 3, total, dependency_note=dependency_note)
        model, tokenizer, model_path, runtime = load_backend()
        log_status(out_dir, "loaded_model", 4, total, **runtime)

        step = 4
        for system in SYSTEMS:
            for row in rows:
                step += 1
                prompt = render_prompt(row, system, tokenizer)
                prediction = generate_one(model, tokenizer, prompt, max_new_tokens)
                append_jsonl(
                    predictions_path,
                    {
                        "eval_id": row["eval_id"],
                        "system": system,
                        "prediction": prediction,
                        "backend": "gemma_transformers",
                        "reportable": True,
                        "model_id": model_path,
                        "prompt_template_sha256": sha256_text(SYSTEM_PROMPT if system == "prompt_only_baseline" else "base-user-only"),
                        "generation_params": {"max_new_tokens": max_new_tokens, "do_sample": False},
                    },
                )
                log_status(out_dir, "generating", step, total, system=system, eval_id=row["eval_id"], elapsed_sec=round(time.time() - start, 2))

        manifest = {
            "created_at_utc": utc_now(),
            "status": "completed",
            "reportable": True,
            "systems": list(SYSTEMS),
            "row_count": len(rows),
            "prediction_count": len(rows) * len(SYSTEMS),
            "checksums": {
                "eval_rubric_sha256": sha256_file(rubric_path),
                "predictions_sha256": sha256_file(predictions_path),
            },
            "runtime": runtime,
            "prompt_template_sha256": sha256_text(SYSTEM_PROMPT),
            "elapsed_sec": round(time.time() - start, 2),
            "EVAL_ONLY_DO_NOT_TRAIN": True,
        }
        write_json(out_dir / "eval_run_manifest.json", manifest)
        log_status(out_dir, "completed", total, total, predictions_file=str(predictions_path), elapsed_sec=manifest["elapsed_sec"])
    except Exception as exc:
        failure = {"created_at_utc": utc_now(), "stage": "eval", "status": "failed", "error": str(exc), "reportable": False}
        write_json(out_dir / "eval_failure.json", failure)
        log_status(out_dir, "failed", 0, 1, error=str(exc))
        raise


if __name__ == "__main__":
    main()
