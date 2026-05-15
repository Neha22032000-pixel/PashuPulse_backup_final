# PashuPulse LLM Judge Eval

This eval is for post-training model review. It does not create training data and every output is marked `EVAL_ONLY_DO_NOT_TRAIN`.

## Inputs

- `eval_rubric.jsonl` from the sealed eval package.
- A predictions JSONL containing `eval_id`, `system`, `prediction`, and `reportable`.
- Optional systems: `base_gemma`, `prompt_only_baseline`, `pashu_saathi_lora`.

## Reportable Judge

Use an OpenAI-compatible LLM endpoint:

```powershell
python pashu_saathi\evals\llm_judge_eval.py `
  --eval-rubric pashu_saathi\kaggle_packages\eval_package\eval_rubric.jsonl `
  --predictions-file path\to\predictions.jsonl `
  --out-dir pashu_saathi\evals\llm_judge_outputs `
  --systems base_gemma,prompt_only_baseline,pashu_saathi_lora `
  --max-eval-rows 50 `
  --judge-backend openai_compatible `
  --judge-model <judge-model>
```

Set `PASHU_LLM_JUDGE_API_KEY` or `OPENAI_API_KEY`. For non-OpenAI providers, set `PASHU_LLM_JUDGE_BASE_URL`.

## Smoke Mode

`rule_smoke` only tests plumbing and is never reportable:

```powershell
python pashu_saathi\evals\llm_judge_eval.py `
  --eval-rubric pashu_saathi\kaggle_packages\eval_package\eval_rubric.jsonl `
  --predictions-file path\to\predictions.jsonl `
  --judge-backend rule_smoke `
  --allow-nonreportable
```

## Outputs

- `llm_judge_results.jsonl`
- `llm_judge_score_report.json`
- `llm_judge_manifest.json`

The LLM judge scores naturalness, usefulness, source fidelity, escalation calibration, and concision. Hard safety gates still block unsafe medicine, injection, dose, procedure, image diagnosis certainty, milk/meat sale guarantees, and red-case under-escalation.
