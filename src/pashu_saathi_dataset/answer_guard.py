from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pashu_saathi_dataset.router import DEFAULT_KNOWLEDGE_DIR


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    answer: str
    violations: list[str]


def load_guard_policy(knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR) -> dict[str, Any]:
    return json.loads((knowledge_dir / "answer_guard_policy.yaml").read_text(encoding="utf-8"))


def guard_answer(
    answer: str,
    evidence_chunks: list[dict[str, Any]] | None = None,
    knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
) -> GuardResult:
    evidence_chunks = evidence_chunks or []
    policy = load_guard_policy(knowledge_dir)
    supported_text = " ".join(chunk.get("text", "") for chunk in evidence_chunks).lower()
    violations = []
    for group, patterns in policy["patterns"].items():
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                if _is_negated_boundary(answer, match.start(), match.end()):
                    continue
                # The guard still blocks exact unsafe outputs even if a chunk discusses the boundary.
                if group in {"dose_units", "injection_instructions", "procedure_terms", "safety_guarantees", "disease_confirmation"}:
                    violations.append(group)
                    break
                if pattern.lower() not in supported_text:
                    violations.append(group)
                    break
    if violations:
        return GuardResult(False, policy["fallback_template"], sorted(set(violations)))
    return GuardResult(True, answer, [])


def _is_negated_boundary(answer: str, start: int, end: int) -> bool:
    window = answer[max(0, start - 90) : min(len(answer), end + 90)].lower()
    negations = (
        "do not",
        "don't",
        "avoid",
        "never",
        "mat",
        "na ",
        "nahi",
        "nahin",
        "without trained",
        "bina trained",
        "bina vet",
    )
    return any(item in window for item in negations)
