# PashuPulse Livestock Source Corpus v1

This is a Beacon-style source-corpus pack for PashuPulse.

Pipeline shape:

```text
trusted livestock sources
-> raw local downloads
-> extracted safe text
-> document cards
-> accepted / quarantined / rejected splits
-> clean livestock corpus rows
-> internal safety rules
-> manifest summary
```

## Important Copyright Boundary

The public repo does not store full raw third-party webpages. Raw pages can be recreated locally with `scripts/download_pashupulse_sft_sources.py` and should remain local unless the source license clearly allows redistribution.

This folder stores source metadata, safe paraphrased extracted text, clean corpus rows, and safety labels.

## Files

- `source_cards.jsonl` - source-level trust and usage policy.
- `document_cards.jsonl` - one document card per source document.
- `accepted_document_cards.jsonl` - clean documents usable for safe corpus/SFT/RAG reference.
- `quarantined_document_cards.jsonl` - useful documents with copyright/safety constraints.
- `rejected_document_cards.jsonl` - unsuitable documents, currently empty.
- `livestock_clean_corpus.jsonl` - safe paraphrased corpus rows.
- `internal_safety_rules.jsonl` - do-not-train-as-answer boundaries.
- `source_claims.jsonl` - claim catalog copied from the source-document bundle.
- `manifest.json` - summary counts and coverage.
- `download_report.md` - human-readable corpus report.
- `raw/` - local-only raw download placeholder.
- `extracted/` - safe extracted/paraphrased text by source.

## Intended Use

- CPT/DAPT candidate rows: only `livestock_clean_corpus.jsonl`, after review.
- SFT data creation: safe rows plus safety rules as constraints.
- RAG cards: use document cards and source claims.
- Audit: use manifest, report, source IDs, and hashes.
