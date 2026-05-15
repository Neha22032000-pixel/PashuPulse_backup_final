# PashuPulse Final-Eval Local Judge

This workflow judges only sealed `final_eval` rows. It does not create training data and every output is marked `EVAL_ONLY_DO_NOT_TRAIN`.

## Compared Systems

- `base_gemma`
- `checkpoint_250`
- `checkpoint_264`

## Judge Order

1. `Qwen2.5-7B-Instruct` in 4-bit
2. `Gemma 4 4B IT` in 4-bit
3. `Gemma 4 E2B IT` as last-resort smoke judge

Attach the judge model as a Kaggle input and set `PASHU_JUDGE_MODEL_PATH` when the default Kaggle path does not match.

## Run Shape

Generate predictions first:

```powershell
python pashu_saathi\kaggle_final_eval_local_judge\final_eval_local_judge.py `
  --stage generate `
  --eval-rubric /kaggle/input/pashu-saathi-eval-package/eval_rubric.jsonl `
  --final-eval /kaggle/input/pashu-saathi-full-expansion/final_eval.jsonl `
  --train-file /kaggle/input/pashupulse-sft-cleaned/sft_train.jsonl `
  --dev-file /kaggle/input/pashupulse-sft-cleaned/sft_dev.jsonl `
  --checkpoint-250 /kaggle/input/pashupulse-sft-output/pashu_pulse_lora/checkpoint-250 `
  --checkpoint-264 /kaggle/input/pashupulse-sft-output/pashu_pulse_lora/checkpoint-264 `
  --max-eval-rows 25
```

Then run the judge in a fresh Kaggle session when possible:

```powershell
python pashu_saathi\kaggle_final_eval_local_judge\final_eval_local_judge.py `
  --stage judge `
  --eval-rubric /kaggle/input/pashu-saathi-eval-package/eval_rubric.jsonl `
  --final-eval /kaggle/input/pashu-saathi-full-expansion/final_eval.jsonl `
  --prediction-dir /kaggle/input/pashupulse-final-eval-predictions `
  --judge-model-path /kaggle/input/qwen2.5-7b-instruct `
  --max-eval-rows 25
```

If memory and JSON parse rate are clean, rerun with `--max-eval-rows 50`.

## Outputs

- `final_eval_predictions_base_gemma.jsonl`
- `final_eval_predictions_checkpoint_250.jsonl`
- `final_eval_predictions_checkpoint_264.jsonl`
- `final_eval_prediction_manifest.json`
- `judge_raw_outputs.jsonl`
- `judge_scores.jsonl`
- `judge_summary.json`
- `model_comparison_report.json`
- `local_llm_judge_manifest.json`

Passing this workflow recommends a checkpoint for post-training review only. It never sets `sft_allowed=true` or promotes a model.
