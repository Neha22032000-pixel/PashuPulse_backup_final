# PashuPulse SFT v2 Source Documents

This folder tracks the trusted public source pages used to ground PashuPulse livestock behavior data.

The goal is not to copy veterinary textbooks into the repo. The goal is to create an auditable source-document layer that can later support SFT examples, RAG cards, local retrieval, or safety-policy rules.

## What Is In Git

```text
docs/source_documents/pashupulse_sft_v2/
  README.md
  BRIEF_TEMPLATE.md
  source_manifest.json
  source_claims.jsonl
  documents/*.md
  raw_downloads/.gitignore
scripts/download_pashupulse_sft_sources.py
scripts/validate_pashupulse_source_documents.py
```

## What Is Not In Git

Raw downloaded HTML/PDF pages are intentionally excluded, especially for third-party copyrighted sources. Recreate the local raw cache with:

```bash
python scripts/download_pashupulse_sft_sources.py
```

Default output:

```text
docs/source_documents/pashupulse_sft_v2/raw_downloads/
```

## Source Hierarchy

Prefer:

1. Indian government and official livestock institutions: DAHD, NDDB, ICAR, state animal husbandry.
2. International public bodies: FAO, WHO, WOAH.
3. University extension and veterinary education pages.
4. Recognized NGOs or public education manuals when official sources are missing.

Reject or quarantine:

- commercial medicine/product pages,
- miracle-cure blogs,
- forums/social posts,
- SEO farm-health pages with unclear authorship,
- raw drug/dose/procedure instructions,
- anti-vaccine or misinformation content,
- unreadable or garbled OCR.

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

## Source Brief Rule

Every accepted source should have:

- a manifest row in `source_manifest.json`,
- a human-readable Markdown brief in `documents/`,
- structured safe claims in `source_claims.jsonl`,
- license and allowed-use notes,
- safety labels that separate user-facing claims from internal hard bans.

Use `BRIEF_TEMPLATE.md` for new sources.

## Safety Labels

Controlled labels used in `source_claims.jsonl`:

- `safe_general_info`
- `triage_or_referral`
- `requires_veterinarian`
- `diagnosis_claim`
- `drug_dose_or_route`
- `withdrawal_period`
- `procedure_instruction`
- `product_or_commercial_bias`
- `region_specific_regulation`
- `do_not_train_as_answer`

## SFT Use Rule

Use these documents to create farmer-assistant examples that are:

- grounded in source-backed livestock facts,
- concise and rural-helper friendly,
- non-diagnostic unless the source supports the term as a general concept,
- free from medicine dose, injection route, procedure, or guaranteed milk/meat safety claims,
- clear about escalation when red flags appear.

Do not use these briefs to generate raw treatment protocols, dosing instructions, vaccine schedules, withdrawal-period claims, or procedure walkthroughs.

## Validation

Run:

```bash
python scripts/validate_pashupulse_source_documents.py
```

The validator checks required manifest fields, brief file presence, structured claim labels, missing claim coverage, obvious mojibake, and obvious procedural/dose leakage.
