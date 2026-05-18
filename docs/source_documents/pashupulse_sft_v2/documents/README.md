# PashuPulse Source Briefs

This folder contains public-source document briefs for PashuPulse data creation.

These are not raw copied webpages. Each file is a compact source note created from the referenced source: publisher, URL, source tier, PashuPulse use, grounded paraphrased claims, and safety boundaries.

For machine-readable extraction, use:

- `../source_manifest.json`
- `../source_claims.jsonl`
- `../BRIEF_TEMPLATE.md`

Raw downloads are intentionally excluded from git, especially for third-party copyrighted pages. Use `scripts/download_pashupulse_sft_sources.py` to recreate the local raw cache when needed.

## Source Briefs

| File | Source ID | Publisher | Main Use |
| --- | --- | --- | --- |
| `dahd_nadcp.md` | `dahd_nadcp` | DAHD | FMD/brucellosis official programme context and outbreak-safe framing |
| `dahd_lhdcp.md` | `dahd_lhdcp` | DAHD | livestock disease control services, surveillance, escalation framing |
| `fao_farmer_first_aid.md` | `fao_farmer_first_aid` | FAO | first-line support boundaries before trained help |
| `nddb_aflatoxicosis.md` | `nddb_aflatoxicosis` | NDDB | moldy/spoiled feed and aflatoxin risk framing |
| `nddb_bloat.md` | `nddb_bloat` | NDDB | bloat danger signs and oil/drench/puncture myth blocking |
| `nddb_calf_nutrition.md` | `nddb_calf_nutrition` | NDDB | calf feeding transition and routine observation |
| `nddb_clean_milk.md` | `nddb_clean_milk` | NDDB | clean milk production and milking hygiene |
| `nddb_drinking_water.md` | `nddb_water` | NDDB | clean water access and water-related observation guidance |
| `tnau_cattle_care_management.md` | `tnau_calf_management` | TNAU | cattle/calf care, housing, general management notes |
| `who_rabies_recommendations.md` | `who_rabies` | WHO | bite/saliva exposure and rabies escalation boundaries |
| `msd_mastitis_cattle.md` | `msd_mastitis` | MSD/Merck Veterinary Manual | mastitis signs, abnormal milk, udder red-flag framing |

## Use Rule

Use these documents to create farmer-assistant examples that are:

- grounded in source-backed livestock facts,
- concise and rural-helper friendly,
- non-diagnostic unless the source supports the term as a general concept,
- free from medicine dose, injection route, procedure, or guaranteed milk/meat safety claims,
- clear about escalation when red flags appear.

Do not use these briefs to generate raw treatment protocols, dosing instructions, vaccine schedules, withdrawal-period claims, or procedure walkthroughs.
