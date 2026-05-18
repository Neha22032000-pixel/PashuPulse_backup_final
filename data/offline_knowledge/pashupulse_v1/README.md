# PashuPulse Offline Knowledge Base v1

This folder is the offline information store for PashuPulse when the answer needs grounded safety knowledge.

It is intentionally a **single-file retrieval base** for runtime simplicity:

```text
farmer query
-> broad router decides CPT-only vs retrieval-needed
-> if retrieval-needed, search knowledge_chunks.jsonl
-> apply safety_rules.jsonl
-> compact context composer
-> model answer
```

## Files

- `knowledge_chunks.jsonl` - primary offline retrieval data. Contains routine checks plus expanded emergency, myth, public-health, and high-risk chunks.
- `safety_rules.jsonl` - hard safety boundaries and protected triggers.
- `source_manifest.json` - trusted source metadata and source IDs.
- `topic_ontology.json` - topics, aliases, and routing hints.
- `retrieval_config.json` - suggested retrieval settings and trigger-only policy.
- `manifest.json` - corpus counts and status.

## Current Coverage

- 71 knowledge chunks.
- 20 safety rules.
- 15 source entries.

Coverage includes bloat, wound myths, calving danger, dog bite/rabies, milk/meat boundary, medicine pressure, routine observation, feed/water/heat, calf issues, outbreak signs, snake bite, poisoning, electrocution, fracture, severe bleeding, eye injury, retained placenta, abortion cluster, urinary blockage, neurological signs, sudden death/carcass handling, heat emergency, choking, goat/sheep flock warnings, and poultry outbreak warnings.

## Use At Inference

For general non-danger questions, let CPT answer naturally without retrieval.

Use retrieval when the query includes:

- unsafe myth or home remedy pressure,
- medicine/injection/dose/shop pressure,
- emergency or red-flag symptoms,
- public-health, milk/meat safety, carcass, bite, or outbreak concern,
- photo/text diagnosis uncertainty,
- explicit request for source-backed guidance.

Recommended context for the model:

```text
1-3 knowledge chunks
0-2 safety rules
final risk/routing decision
short answer contract
```

Do not dump the whole file into the prompt.

## Scope

This is not a veterinary textbook and not a diagnosis system. It supports:

- first-line observation,
- myth correction,
- low-resource care boundaries,
- escalation decisions,
- offline source-backed safety guidance.

It must not be used to provide medicine doses, injection routes, procedure instructions, disease certainty from text/photo, or milk/meat safety certification.
