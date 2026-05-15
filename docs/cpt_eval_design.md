# PashuPulse CPT/DAPT Evaluation Design

This design evaluates knowledge grounding and domain familiarity only. It does
not evaluate final assistant behavior.

## Tier 1: Held-Out Text Loss

Split by `source_id`, not random chunks, to avoid leakage from adjacent text.
Measure loss/perplexity on held-out livestock sources. This shows whether a CPT
candidate has adapted to livestock text distribution.

Tier 1 cannot prove factual answer quality. A lower loss is an early signal, not
a promotion criterion.

## Tier 2: Source-Grounded Knowledge Probes

Use eval-only probes generated from held-out sources:

- Closed-book short factual QA: answer from model weights only.
- Open-book QA: provide a source span in context and test source use.
- Cloze/completion probes: recover source-grounded terminology.
- Multiple-choice probes: distinguish supported livestock facts from distractors.
- Entailment and contradiction probes: supported, contradicted, or not stated.

Every eval item must retain `source_id`, `source_span_id`, species, topic,
language, answer key, and forbidden unsafe extrapolations.

## Safety Sanity Probes

Even in research-only CPT, unsafe veterinary drift is blocking. Probe for:

- medicine, dose, injection, antibiotics, painkillers, dewormers
- bloat oil/drenching
- wound irritants
- hard calf pulling
- unsafe milk/meat sale guarantees
- FMD-like movement or market encouragement
- confident visual or clinical diagnosis

## Retention Probes

Use a small non-livestock English/Hinglish set to detect obvious forgetting.
The goal is not broad benchmark ranking, only catching harmful drift before
training is considered.

## Interpretation

- Loss improves, probes do not: CPT learned style/distribution, not usable
  grounding.
- Probes improve, safety stable: CPT is worth a pilot-training discussion.
- Open-book retrieval beats closed-book CPT: prioritize RAG/source-card work.
- Any unsafe confidence increase: stop and redesign corpus/eval before training.
