# Pashu Saathi Kaggle Runbook

## Credential Isolation

Use the teammate Kaggle account only inline for this track:

```powershell
$env:KAGGLE_CONFIG_DIR = "C:\Users\risha\Documents\New project 5\.kaggle_2"
```

Do not persist this in the shell profile and do not reuse it for other tracks.

## Run Order

1. Generate and validate the dataset locally.
2. Review seed cards and eval rows.
3. Approve the manifest only after source, safety, language, and training/eval
   review.
4. Export the approved SFT bundle.
5. Run Kaggle preflight.
6. Run a smoke LoRA train.
7. Run full LoRA train.
8. Run base vs prompt-only vs LoRA eval.

## Reporting

The final submission should include the dataset card, model card, eval report,
training config, LoRA adapter, inference notebook, and 12-20 flagship demo cases.
