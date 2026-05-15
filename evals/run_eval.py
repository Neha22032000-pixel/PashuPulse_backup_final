from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEMS = ["base_gemma", "prompt_only_baseline", "pashu_saathi_lora"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mock_prediction(row: dict, system: str) -> str:
    if system == "base_gemma":
        if row["risk_level"] == "green":
            return "Call a vet to be safe. Give good feed and water."
        if "myth_pressure" in row["eval_tags"]:
            return "Do not give random medicine. Call a vet."
        return "This may be disease. Call a vet."
    if system == "prompt_only_baseline":
        return row["response_text"].split("Offline note:")[0] + "Escalation: contact trained help if red flags appear."
    return row["response_text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pashu Saathi eval predictions.")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--backend", default="mock", choices=["mock"])
    parser.add_argument("--allow-mock", action="store_true")
    args = parser.parse_args()
    if args.backend == "mock" and not args.allow_mock:
        raise SystemExit("mock backend requires --allow-mock")
    eval_rows = read_jsonl(Path(args.eval_file))
    predictions = []
    for system in SYSTEMS:
        for row in eval_rows:
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "system": system,
                    "prediction": mock_prediction(row, system),
                    "backend": args.backend,
                    "reportable": False,
                }
            )
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
