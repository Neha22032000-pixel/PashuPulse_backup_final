# PashuPulse SFT v2 Source Documents

This folder tracks the public source pages used to ground the PashuPulse SFT v2 400-row behavior dataset.

The dataset is not a veterinary textbook copy. These sources were used to define farmer-observable issue families, safety boundaries, escalation patterns, and rural first-line guidance. The SFT rows remain short, non-diagnostic, and non-prescriptive.

Raw HTML downloads are intentionally kept out of git because several sources are third-party copyrighted pages. Use `scripts/download_pashupulse_sft_sources.py` to recreate a local raw cache for audit work.

## Included Source IDs

- `dahd_lhdcp` - DAHD livestock health and disease control grounding
- `dahd_nadcp` - DAHD National Animal Disease Control Programme grounding
- `fao_farmer_first_aid` - FAO farmer first-aid style grounding
- `nddb_aflatoxicosis` - NDDB mouldy feed / aflatoxicosis grounding
- `nddb_bloat` - NDDB bloat issue grounding
- `nddb_calf_nutrition` - NDDB/TNAU calf care grounding
- `nddb_clean_milk` - NDDB clean milk production grounding
- `nddb_water` - NDDB drinking water for dairy animals grounding
- `tnau_calf_management` - TNAU cattle and calf management grounding
- `who_rabies` - WHO rabies / bite exposure grounding
- `msd_mastitis` - Merck/MSD mastitis grounding

## Local Raw Cache

Run:

```bash
python scripts/download_pashupulse_sft_sources.py
```

Default output:

```text
docs/source_documents/pashupulse_sft_v2/raw_downloads/
```

The raw cache is ignored by git. Commit the manifest and downloader, not copied third-party pages.
