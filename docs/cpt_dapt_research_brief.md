# PashuPulse CPT/DAPT Research Brief

This is a research-only brief for deciding whether continued pretraining should
be attempted for PashuPulse. It does not approve Kaggle training, SFT, model
promotion, or farmer-facing grounding.

## Research Rationale

CPT/DAPT should answer whether livestock-domain exposure improves the model's
domain familiarity and factual access. It should not be judged as an assistant
behavior layer. Behavior, tone, refusal style, Hinglish mirroring, and product
UX remain later SFT/RAG work.

Held-out livestock loss/perplexity is useful, but it is only Tier 1 evidence.
It shows whether the model predicts livestock text better. It does not prove
that the model can correctly answer source-grounded livestock questions or avoid
unsafe extrapolation.

## Evidence Base

- Gururangan et al., "Don't Stop Pretraining": DAPT/TAPT can improve
  downstream domain/task performance when the pretraining data is relevant.
  <https://arxiv.org/abs/2004.10964>
- AdaptLLM, "Adapting Large Language Models to Domains via Reading
  Comprehension": raw CPT may add domain knowledge while hurting QA prompting;
  source-derived reading-comprehension formats can improve knowledge access.
  <https://arxiv.org/abs/2309.09530>
- BioMedLM: small domain-focused models can be useful with strong domain text
  and careful downstream evaluation. <https://arxiv.org/abs/2403.18421>
- "Investigating Continual Pretraining in Large Language Models": CPT can
  improve domain adaptation, but smaller models are sensitive to learning and
  forgetting. <https://arxiv.org/abs/2402.17400>

## Research Questions

- Does livestock CPT reduce held-out domain loss versus base Gemma?
- Does lower loss correlate with better source-derived factual correctness?
- Does CPT improve terminology familiarity for cow, buffalo, ox/bullock, calf,
  feeding, disease signs, reproduction, milk hygiene, and working-animal care?
- Does CPT help underrepresented slices, or mostly the dominant corpus family?
- Does raw CPT help enough, or is reading-comprehension CPT needed?
- Does CPT introduce unsafe confidence, species confusion, memorization, or
  English/Hinglish retention drift?

## Current Corpus State

The offline library currently contains `60` accepted or accepted-after-stripping
sources, `3,799` clean chunks, and about `2.54M` clean tokens. This is enough to
design a pilot and evaluation, not enough to claim domain mastery.

The library is explicitly not approved as a farmer-facing RAG grounding set. A
later source-card pass must choose a stricter subset.

## Research Decision Frame

Proceed toward CPT only if research shows both:

- Tier 1: better held-out livestock loss.
- Tier 2: better source-grounded correctness without new unsafe confidence.

If retrieval over the offline library matches or beats CPT on factuality, prefer
retrieval-first. If raw CPT improves loss but not knowledge probes, treat it as
domain-language familiarity rather than usable knowledge grounding.
