# PashuPulse Raw CPT v1 Adapter

This directory documents the exact CPT adapter used by the offline RAG inference runs.

## Source

- Kaggle kernel: `nehak76044/pashupulse-gemma-raw-cpt`
- Output path: `pashu_pulse_raw_cpt_v1/adapter_final`
- Base model: `google/gemma-4/Transformers/gemma-4-e2b-it/1`
- Training method: raw causal LM CPT with QLoRA

## Required adapter files

The final adapter folder must contain:

```text
adapter_config.json
adapter_model.safetensors
chat_template.jinja
processor_config.json
README.md
tokenizer.json
tokenizer_config.json
```

Exact file sizes and SHA256 checksums are in `ARTIFACT_MANIFEST.json`.

## Runtime use

The Kaggle inference script searches Kaggle inputs for `adapter_config.json` and loads the adapter with PEFT. The intended adapter is the final raw CPT output from `nehak76044/pashupulse-gemma-raw-cpt`, not an SFT adapter.

## Why the weight file is documented

The adapter weight file is about 50.7 MB and `tokenizer.json` is about 32.2 MB. They are verified locally and available from the Kaggle output. The manifest records exact checksums so a downloaded adapter can be checked before inference.
