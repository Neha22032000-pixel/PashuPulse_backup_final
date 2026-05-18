# PashuPulse Source Brief Template

Use this template for every new source brief under `documents/`.

```md
---
source_id: example_source_id
title: Example Source Title
publisher: Example Publisher
publisher_type: government_official | official_dairy_institution | university_extension | intergovernmental | public_health_authority | veterinary_reference | ngo | commercial | other
source_url: https://example.org/source
retrieved_at: YYYY-MM-DD
country_or_region: India | international | other
language: english | hindi | other
species: [cattle, buffalo]
topic: [milk, udder, hygiene]
document_type: guideline | faq | advisory | manual | policy_programme_overview | farmer_education_page | veterinary_reference_page | other
license_status: public_page_reuse_terms_unverified | open_license | government_public_page_reuse_terms_unverified | all_rights_reserved_or_permission_required | unknown
allowed_use: [sft_candidate, rag_reference, internal_safety_rules]
risk_level: low | medium | high
review_status: candidate | downloaded | briefed | qa_passed | quarantined | rejected | approved_for_sft | approved_for_rag
reviewer: unassigned
---

# Short Summary

Two to five sentences. Paraphrase. Do not copy source prose.

# Why This Source Is Trusted

Explain publisher authority and why it fits PashuPulse.

# Safe Source-Derived Claims

| claim_id | claim | species | condition_or_topic | allowed_for_sft | safety_label | evidence_location | notes |
|---|---|---|---|---|---|---|
| example.c01 | Short paraphrased claim. | cattle | milk hygiene | yes | safe_general_info | page section/title if available | Keep non-diagnostic. |

# Safety Boundaries And Exclusions

| boundary_id | source_content_type | safety_label | rule_for_pashupulse | allowed_for_user_answer |
|---|---|---|---|---|
| example.b01 | drug dose table | drug_dose_or_route | Do not train as answer; use only as internal hard ban. | no |

# Regional Or Species Applicability

Note country, climate, production system, and species limits.

# Conflict Or Uncertainty Notes

Record disagreements, outdated content, or assumptions.

# Possible Future Use

- SFT candidate: yes/no and why.
- RAG reference: yes/no and why.
- Internal safety rules: yes/no and why.

# QA Notes

- No long copyrighted excerpts.
- No medicine dose/procedure copied into answer claims.
- Metadata complete.
- Claim IDs present.
```

## Controlled Safety Labels

Use these labels in claim and boundary tables:

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

## Copyright Rule

Prefer paraphrase. Do not recreate source tables, protocols, or long paragraphs unless the license explicitly allows redistribution.