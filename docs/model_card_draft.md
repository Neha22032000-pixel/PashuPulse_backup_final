# Model Card Draft: Pashu Saathi Gemma LoRA

## Intended Use

Offline-first rural livestock companion for farmers or field workers who need
plain-language livestock care support when vets or internet are not immediately
available.

## Safety Boundary

The model is not a veterinarian. It should not diagnose, prescribe, dose, inject,
or recommend invasive procedures. It should provide routine care, safe supportive
steps, severity checks, red flags, myth correction, and reachable trained-help
escalation.

## Primary Metrics

- Unsafe advice rate.
- Red-flag escalation recall.
- Unnecessary escalation rate on green cases.
- Myth correction success.
- Farmer-useful answer score.

## Comparison

Report base Gemma, prompt-only baseline, and Pashu Saathi LoRA on the same
locked final eval.
