# PashuPulse Livestock Source Corpus v1 Report

Created: 2026-05-18

## Summary

This corpus pack mirrors the Beacon source-corpus pattern for PashuPulse, adapted for livestock safety and public-repo copyright constraints.

```text
trusted livestock sources
-> source/document cards
-> accepted/quarantined/rejected splits
-> safe extracted notes
-> clean corpus rows
-> internal safety rules
-> manifest summary
```

## Counts

| Metric | Count |
|---|---:|
| Sources | 6 |
| Candidate documents | 11 |
| Accepted documents | 9 |
| Quarantined documents | 2 |
| Rejected documents | 0 |
| Clean corpus rows | 13 |
| Internal safety rules | 7 |
| Structured claims | 22 |
| Extracted safe-text files | 11 |
| Raw files committed | 0 |

## Accepted Documents

- `dahd_lhdcp` - official livestock-health escalation and service framing.
- `dahd_nadcp` - official outbreak and national disease-control framing.
- `fao_farmer_first_aid` - first-line support and first-aid scope.
- `nddb_aflatoxicosis` - moldy/spoiled feed safety.
- `nddb_bloat` - bloat danger signs and oil/drench/puncture myth blocking.
- `nddb_calf_nutrition` - calf feeding transition and routine observation.
- `nddb_clean_milk` - clean milk and abnormal milk boundaries.
- `nddb_water` - drinking water and routine observation.
- `tnau_calf_management` - cattle/calf care, housing, feeding, routine checks.

## Quarantined Documents

- `who_rabies` - useful for public-health escalation, but should not become vaccine schedule/dose guidance.
- `msd_mastitis` - useful veterinary concept source, but third-party copyrighted and may contain treatment detail; use only paraphrased concepts and internal safety boundaries.

## Raw Download Policy

Raw HTML/PDF pages are not committed here. Recreate the raw local cache with:

```bash
python scripts/download_pashupulse_sft_sources.py
```

The public repo commits safe extracted notes, document cards, and clean corpus rows.

## Recommended Use

- CPT/DAPT candidate: `livestock_clean_corpus.jsonl`, after manual review.
- SFT generation: use clean rows plus `internal_safety_rules.jsonl` as constraints.
- RAG card creation: use document cards and source claims.
- Audit/storytelling: use `manifest.json` and this report.

## Safety Notes

This corpus is for bounded first-line livestock support. It is not a veterinary treatment manual. It excludes or quarantines medicine doses, injection routes, vaccine schedules, invasive procedures, withdrawal periods, and guaranteed milk/meat safety claims.
