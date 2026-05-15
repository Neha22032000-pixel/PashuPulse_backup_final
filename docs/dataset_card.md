# Dataset Card: Pashu Saathi Rural Livestock Companion

## Purpose

Seed-bank v3 prepares a reviewable foundation for a Gemma assistant for
low-connectivity rural India livestock support. The assistant covers cow,
buffalo, ox, and calves. It gives routine care, symptom triage, myth correction,
and safe offline holding steps.

This artifact is **not trainable yet**. It contains 300 seed cards, 100 scenario
families, and zero SFT/eval rows. Expansion and export remain blocked until the
approval state moves beyond `BLOCKED_PENDING_SEED_APPROVAL`.

## Not For

- Veterinary diagnosis.
- Prescribing drugs, injections, doses, antibiotics, painkillers, dewormers, or
  vaccines.
- Invasive procedures.
- Milk/meat safety guarantees.
- Real-time telemedicine or live local service lookup.

## Languages

English and Hinglish. Hinglish is the primary rural typing mode. Devanagari is
excluded from v2 unless separately reviewed.

## Source Priority

DAHD for official disease/reportable guidance, NDDB for Indian dairy care and
milk/nutrition topics, TNAU for extension-style calf/shed/calving context, and
FAO only as a fallback first-aid structure. Vikaspedia is not used for v3 safety
claims.

v3 separates `care_claim_ids`, `escalation_claim_ids`, and `policy_claim_ids`.
Positive care steps cannot be grounded only by escalation, policy, or context
claims.

## Evaluation Slices

Green routine, yellow triage, red emergency, myth pressure, medication request,
snake/dog bite, neurological signs, abortion cluster, monsoon shed hygiene,
calf feeding/warmth, ox workload/yoke injury, local shop injection pressure,
ox-specific, calf-specific, buffalo-specific, English, Hinglish, and
image-caption uncertainty.
