# PashuPulse Offline Knowledge Base v1

This folder is the offline information store for PashuPulse inference.

It is designed to work like a local, trusted web-search/RAG layer:

```text
farmer query
-> local retrieval over knowledge_chunks.jsonl
-> safety rules from safety_rules.jsonl
-> compact context composer
-> model answer
```

## Files

- `knowledge_chunks.jsonl` - primary offline retrieval data. Each row is a small source-grounded livestock guidance chunk.
- `safety_rules.jsonl` - hard safety boundaries and protected triggers.
- `source_manifest.json` - trusted source metadata and source IDs.
- `topic_ontology.json` - topics, aliases, and routing hints.
- `retrieval_config.json` - suggested retrieval settings.
- `manifest.json` - corpus counts and status.

## Chunk Design

Each chunk is intentionally small and retrievable:

- 60-160 words where possible.
- one main topic per chunk.
- species/topic/risk metadata.
- safe for farmer-facing answers.
- no medicine doses, injection routes, surgical steps, or guaranteed milk/meat safety.

## Use At Inference

Retrieve top chunks using keyword/BM25 first, then optional embeddings later.

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
- milk/feed/water/hygiene/calf/bloat/calving/bite/wound guidance.
