---
source_id: nddb_bloat
title: NDDB Bloat Grounding
publisher: National Dairy Development Board
publisher_type: official_dairy_institution
source_url: https://www.nddb.coop/farmer/animal-health
retrieved_at: 2026-05-17
country_or_region: India
language: english
species: [dairy_cattle, buffalo]
topic: [bloat, feed_risk, urgent_escalation, unsafe_myth_blocking]
document_type: farmer_education_category
license_status: public_page_reuse_terms_unverified
allowed_use: [sft_candidate, rag_reference, internal_safety_rules]
risk_level: high
review_status: briefed
reviewer: unassigned
---

# Short Summary

This brief grounds PashuPulse bloat-style safety behavior from NDDB farmer-facing animal-health context and internal safety review. It is intended for urgent risk recognition and myth blocking, not for treatment instructions.

# Why This Source Is Trusted

NDDB is a major Indian dairy institution with farmer-facing animal-health and dairy guidance. The source category is relevant to Indian cattle and buffalo farming contexts.

# Safe Source-Derived Claims

| claim_id | claim | species | condition_or_topic | allowed_for_sft | safety_label | evidence_location | notes |
|---|---|---|---|---|---|---|
| nddb_bloat.c01 | Left-sided tight belly, restlessness, fast breathing, or severe abdominal distension after feed can be urgent and should trigger trained-help escalation. | cattle, buffalo | bloat danger signs | yes | triage_or_referral | NDDB animal-health grounding category | Use only when explicit bloat anchors appear. |
| nddb_bloat.c02 | Oil, drenching, pipe insertion, puncturing, or cutting for suspected bloat should not be suggested to farmers. | cattle, buffalo | bloat unsafe myths | yes | do_not_train_as_answer | safety boundary derived from bloat risk framing | Render as avoid line for protected myth prompts. |

# Safety Boundaries And Exclusions

| boundary_id | source_content_type | safety_label | rule_for_pashupulse | allowed_for_user_answer |
|---|---|---|---|---|
| nddb_bloat.b01 | emergency treatment/procedure detail | procedure_instruction | Do not provide puncture, tubing, drenching, or force-feeding instructions. | no |
| nddb_bloat.b02 | home remedy myth | do_not_train_as_answer | Directly block oil/drench myths when user asks. | yes, as a negative instruction |

# Regional Or Species Applicability

Applies to Indian dairy cattle and buffalo safety framing. Do not generalize to non-ruminants.

# Conflict Or Uncertainty Notes

This brief is not a complete veterinary bloat protocol. It only supports first-line safety, myth correction, and escalation.

# Possible Future Use

- SFT candidate: yes, especially urgent/myth examples.
- RAG reference: yes, as protected safety card grounding.
- Internal safety rules: yes, for oil/drench/puncture blocking.

# QA Notes

- No drug dose or procedure instructions included.
- No long copied text from source pages.
- Claim IDs mirrored in `source_claims.jsonl`.
