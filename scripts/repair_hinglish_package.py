from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HINGLISH_TOKENS = {
    "paani", "saaf", "rakho", "rakhein", "dekho", "dekhein", "dhyaan", "agar", "pashu", "gai", "gaay",
    "bhains", "bail", "bachda", "chara", "charah", "sukha", "sookha", "chhaya", "aaram", "alag",
    "mat", "nahi", "jaldi", "turant", "sampark", "bulao", "dikhana", "doctor", "dawa",
    "zabardasti", "khilao", "pilao", "saans", "pet", "doodh", "ghav", "khada", "kamzor", "beemar",
}
BANNED_PHRASES = [
    "as an ai", "in conclusion", "hope this helps", "i understand your concern", "let me explain",
    "mudda ", "first,", "close note", "field note",
]
UNSAFE_ADDED_RE = re.compile(
    r"\b(antibiotic|painkiller|dewormer|meloxicam|calcium|tablet|mg|ml|dose|puncture|cut|kerosene|alcohol|mustard oil|sarson ka tel|haldi laga|mirch|ash|force-feed)\b",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def species(row: dict[str, Any]) -> str:
    tags = set(row.get("tags", []))
    if "buffalo" in tags:
        return "bhains"
    if "ox" in tags:
        return "bail"
    if "calf" in tags:
        return "bachde"
    return "gai"


def topic_key(row: dict[str, Any]) -> str:
    seed = str(row.get("parent_seed_id", row.get("row_id", "")))
    match = re.search(r"ps_v4_\d+_\d+_(.+)$", seed)
    return match.group(1) if match else str(row.get("family_bucket", "care")).replace(" ", "_")


def topic_phrase(key: str) -> str:
    phrases = {
        "shed_floor_dry": "geela bedding hatao, floor sookha rakho aur paani bahar nikalne ka rasta saaf rakho",
        "clean_water_trough": "roz saaf paani do aur paani ka bartan ganda na rehne do",
        "monsoon_ventilation": "shed mein hawa aane do, bheed aur nami kam rakho",
        "mild_appetite_drop": "saaf paani do, kharab chara hatao, aur khana-paani dhyaan se dekho",
        "reduced_milk": "paani, chara, aaram aur udder ki safai dhyaan se check karo",
        "minor_wound_surface": "ghav ke aas-paas ki gandagi saaf paani se dheere hatao aur jagah saaf-sookhi rakho",
        "udder_discomfort": "udder ko saaf rakho, doodh mein clot ya badlav dhyaan se dekho",
        "bloat_after_green_feed": "naya ya zyada hara chara rok do, pashu ko shaant jagah rakho, tel ya zabardasti pilana mat karo",
        "calving_not_progressing": "pashu ko shaant aur saaf jagah rakho, zor se bachda kheenchna mat",
        "calf_diarrhea_dehydration": "bachde ko garam-sookhi jagah rakho, saaf paani paas rakho, kamzori dhyaan se dekho",
        "fmd_like_mouth_hoof": "pashu ko alag rakho, movement aur bazaar le jaana rok do",
        "sudden_death_unknown": "shav ko mat kholo, dusre pashuon ko door rakho, aur official madad bulao",
        "poison_spoiled_feed": "suspect chara turant hatao, saaf paani do, aur baaki pashuon ko bhi dekho",
        "dog_bite_risk": "ghav ko saaf paani se dhona safe hai, pashu ko alag aur shaant rakho",
        "snake_bite_swelling": "pashu ko kam chalne do, shaant rakho, kaatna ya bandhna mat karo",
        "neurological_staggering": "door se dekho, pashu ko chot se bachao, bheed door rakho",
        "heat_collapse": "turant chhaya, hawa aur thandi jagah do; zabardasti paani mat pilao",
        "severe_wound_maggots": "ghav ko dhak kar makkhiyon se bachao, kaatna ya chedna mat karo",
    }
    for needle, phrase in phrases.items():
        if needle in key:
            return phrase
    return "saaf paani, saaf chara, sookhi jagah aur aaram ka dhyaan rakho"


def red_flags(row: dict[str, Any]) -> str:
    key = topic_key(row)
    if "bloat" in key:
        return "pet bahut phool raha ho, saans mein dikkat ho, pashu gir raha ho, ya dard zyada lage"
    if "calving" in key or "abortion" in key:
        return "prasav ruk gaya ho, zyada bleeding ho, pashu bahut kamzor ho, ya bachda atka lage"
    if "diarrhea" in key:
        return "dast bahut zyada ho, aankh dhansi lage, pashu khada na ho, ya khoon dikhe"
    if "fmd" in key:
        return "muh ya khur mein chhale ho, laar zyada ho, ya kai pashu ek saath affected hon"
    if "wound" in key or "bite" in key:
        return "ghav gehra ho, badbu/maggot ho, bleeding ho, ya pashu bahut sust ho"
    if "poison" in key:
        return "collapse, kampan, saans ki dikkat, ya kai pashu ek saath beemar hon"
    return "paani na peena, girna, saans ki dikkat, bahut kamzori, ya kai pashu ek saath beemar lagna"


def escalation(row: dict[str, Any]) -> str:
    risk = row.get("risk_level")
    if risk == "red":
        return "Der mat karo; pashu doctor, trained animal health worker, Pashu Sakhi ya dairy cooperative se turant sampark karo."
    return "Agar yeh red flag dikhe ya halat bigde, to pashu doctor, animal health worker, Pashu Sakhi ya dairy cooperative se sampark karo."


def boundary(row: dict[str, Any]) -> str:
    return "Apni taraf se dawa, injection, dose, cutting, puncture, tel pilana ya zabardasti khilana-pilana mat karo."


def rewrite_hinglish(row: dict[str, Any]) -> str:
    animal = species(row)
    topic = topic_phrase(topic_key(row))
    flags = red_flags(row)
    risk = row.get("risk_level")
    if risk == "green":
        return (
            f"Haan ji, {animal} ke liye yeh routine care hai. {topic.capitalize()}. "
            f"Roz khana-paani aur behavior dhyaan se dekho. Agar {flags}, to isse serious samjho. "
            f"{boundary(row)} {escalation(row)}"
        )
    if risk == "yellow":
        return (
            f"Bhai, {animal} mein yeh halka issue ho sakta hai, par dhyaan se dekhna zaruri hai. {topic.capitalize()}. "
            f"Aaj appetite, paani peena, chalna, saans aur dung/doodh ka badlav note karo. Agar {flags}, to wait mat karo. "
            f"{boundary(row)} {escalation(row)}"
        )
    return (
        f"Yeh {animal} ke liye emergency jaisa case ho sakta hai. {topic.capitalize()}. "
        f"Agar {flags}, to der bilkul mat karo. {boundary(row)} {escalation(row)} "
        f"Jab tak madad aaye, pashu ko shaant, surakshit aur khuli hawa wali jagah par rakho."
    )


def rewrite_english(row: dict[str, Any]) -> str:
    animal = "animal"
    tags = set(row.get("tags", []))
    if "cow" in tags:
        animal = "cow"
    elif "buffalo" in tags:
        animal = "buffalo"
    elif "ox" in tags:
        animal = "ox"
    elif "calf" in tags:
        animal = "calf"
    topic = topic_phrase(topic_key(row)).replace("paani", "water").replace("saaf", "clean").replace("rakho", "keep").replace("mat karo", "do not do")
    risk = row.get("risk_level")
    if risk == "green":
        return (
            f"This looks like routine care for the {animal}. Keep the focus on clean water, clean feed, dry bedding, airflow, and rest. "
            f"Also watch for red flags such as {red_flags(row)}. {boundary_english()} {escalation_english(row)}"
        )
    if risk == "yellow":
        return (
            f"This needs careful watching, but do the safe steps first. Keep clean water nearby, remove suspect feed, give rest, and note appetite, dung, breathing, and movement. "
            f"If you see {red_flags(row)}, do not wait. {boundary_english()} {escalation_english(row)}"
        )
    return (
        f"This may be an emergency for the {animal}. Keep the animal calm, safe, and in fresh air while avoiding force or risky home treatment. "
        f"If you see {red_flags(row)}, act quickly. {boundary_english()} {escalation_english(row)}"
    )


def boundary_english() -> str:
    return "Do not give medicines, injections, doses, cutting, puncturing, oil drenching, or force-feeding on your own."


def escalation_english(row: dict[str, Any]) -> str:
    if row.get("risk_level") == "red":
        return "Contact a vet, trained animal health worker, Pashu Sakhi, or dairy cooperative when reachable."
    return "If the animal worsens or red flags appear, contact a vet, trained animal health worker, Pashu Sakhi, or dairy cooperative when reachable."


def language_mirror_score(text: str) -> float:
    lowered = text.lower()
    words = re.findall(r"[a-zA-Z]+", lowered)
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in HINGLISH_TOKENS)
    english_openers = 1 if lowered.startswith(("first,", "for ", "make the routine", "ask these checks")) else 0
    return max(0.0, min(1.0, (hits / max(len(words), 1) * 5.0) - (0.3 * english_openers)))


def validate_repair(row: dict[str, Any], original: dict[str, Any]) -> tuple[str, list[str], float]:
    answer = row["messages"][1]["content"]
    lowered = answer.lower()
    errors = []
    score = 1.0 if row.get("language_style") == "english" else language_mirror_score(answer)
    if row.get("language_style") == "hinglish" and score < 0.85:
        errors.append("low_language_mirror_score")
    banned = [phrase for phrase in BANNED_PHRASES if phrase in lowered]
    if lowered.startswith("for "):
        banned.append("for_start")
    if banned:
        errors.append("banned_phrase:" + ",".join(banned[:3]))
    for match in UNSAFE_ADDED_RE.finditer(answer):
        window = lowered[max(0, match.start() - 80): match.end() + 80]
        if not re.search(r"\b(mat|nahi|na karo|do not|don't|avoid|without|apni taraf se|on your own)\b", window):
            errors.append("unsafe_treatment_added")
            break
    if row.get("risk_level") == "red" and not re.search(r"\b(turant|jaldi|der mat|emergency|sampark|doctor|pashu sakhi|dairy cooperative)\b", lowered):
        errors.append("red_escalation_missing")
    if "source_claim_ids" in original and row.get("source_claim_ids") != original.get("source_claim_ids"):
        errors.append("source_claim_ids_changed")
    if row.get("row_id") != original.get("row_id") or row.get("parent_seed_id") != original.get("parent_seed_id"):
        errors.append("lineage_changed")
    state = "auto_approved" if not errors else "manual_review_required"
    return state, errors, round(score, 4)


def update_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if out.get("language_style") in {"hinglish", "english"}:
        answer = rewrite_hinglish(out) if out.get("language_style") == "hinglish" else rewrite_english(out)
        out["assistant_response"] = answer
        if out.get("messages") and len(out["messages"]) >= 2:
            out["messages"] = [dict(message) for message in out["messages"]]
            out["messages"][1]["content"] = answer
        if out.get("answer_sections"):
            out["answer_sections"] = [dict(section) for section in out["answer_sections"]]
            for section in out["answer_sections"]:
                if section.get("section_type") == "safe_step":
                    section["text"] = topic_phrase(topic_key(out)).capitalize() + "."
                elif section.get("section_type") == "red_flag":
                    section["text"] = f"Agar {red_flags(out)}, to serious samjho."
                elif section.get("section_type") == "boundary":
                    section["text"] = boundary(out)
                elif section.get("section_type") == "style":
                    section["text"] = escalation(out)
        out["content_hash"] = sha256_text(json.dumps(out.get("messages", []), ensure_ascii=False, sort_keys=True))
    return out


def build_repaired_package(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    report_rows = []
    all_errors = Counter()
    state_counts = Counter()
    lang_scores = []
    for split_name in ["sft_train.jsonl", "sft_dev.jsonl"]:
        repaired = []
        originals = read_jsonl(source_dir / split_name)
        for original in originals:
            row = update_row(original)
            state, errors, score = validate_repair(row, original)
            state_counts[state] += 1
            all_errors.update(errors)
            if row.get("language_style") == "hinglish":
                lang_scores.append(score)
            report_rows.append({"row_id": row.get("row_id"), "split_file": split_name, "language_style": row.get("language_style"), "risk_level": row.get("risk_level"), "repair_state": state, "errors": errors, "language_mirror_score": score})
            if state not in {"auto_approved", "manual_approved"}:
                row["_repair_blocked"] = True
            repaired.append(row)
        write_jsonl(out_dir / split_name, repaired)
    for name in ["training_config.json", "dataset-metadata.json"]:
        shutil.copy2(source_dir / name, out_dir / name)
    config = read_json(out_dir / "training_config.json")
    manifest = {
        "created_at_utc": utc_now(),
        "package_type": "sft_training",
        "package_mode": "full_repaired_candidate",
        "status": "BLOCKED_PENDING_LANGUAGE_REPAIR_REVIEW",
        "source_package_dir": str(source_dir),
        "row_counts": {"sft_train": len(read_jsonl(out_dir / "sft_train.jsonl")), "sft_dev": len(read_jsonl(out_dir / "sft_dev.jsonl")), "final_eval": 0},
        "checksums": {
            "source_sft_train_sha256": sha256_file(source_dir / "sft_train.jsonl"),
            "source_sft_dev_sha256": sha256_file(source_dir / "sft_dev.jsonl"),
            "package_sft_train_sha256": sha256_file(out_dir / "sft_train.jsonl"),
            "package_sft_dev_sha256": sha256_file(out_dir / "sft_dev.jsonl"),
            "training_config_sha256": sha256_file(out_dir / "training_config.json"),
        },
        "sft_allowed": False,
        "promotion_allowed": False,
        "requires_reviews": ["language_quality", "safety_preservation", "data_gate_checksum"],
        "training_config": config,
    }
    write_json(out_dir / "sft_package_manifest.json", manifest)
    repair_report = {
        "created_at_utc": utc_now(),
        "repair_version": "hinglish-repair-v1",
        "state_counts": dict(state_counts),
        "error_counts": dict(all_errors),
        "hinglish_language_mirror": {
            "count": len(lang_scores),
            "min": min(lang_scores) if lang_scores else None,
            "avg": round(sum(lang_scores) / len(lang_scores), 4) if lang_scores else None,
            "max": max(lang_scores) if lang_scores else None,
        },
        "approved_for_real_sft": all_errors == Counter() and state_counts.get("manual_review_required", 0) == 0,
    }
    write_json(out_dir / "language_repair_report.json", repair_report)
    write_jsonl(out_dir / "language_repair_rows.jsonl", report_rows)
    review_request = {
        "created_at_utc": utc_now(),
        "decision": "pending_review",
        "required_decision": "approved_for_repaired_sft",
        "required_reviewer_roles": ["language_quality", "safety_preservation", "data_gate_checksum"],
        "package_checksums": manifest["checksums"],
        "language_repair_report_sha256": sha256_file(out_dir / "language_repair_report.json"),
        "sft_package_manifest_sha256": sha256_file(out_dir / "sft_package_manifest.json"),
    }
    write_json(out_dir / "sft_param_review_request.json", review_request)
    metadata = read_json(out_dir / "dataset-metadata.json")
    metadata["id"] = "nehak76044/pashu-saathi-sft-repaired-package"
    metadata["title"] = "Pashu Saathi SFT Repaired Candidate Package"
    write_json(out_dir / "dataset-metadata.json", metadata)
    return repair_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Hinglish-repaired PashuPulse SFT candidate package.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "kaggle_packages" / "sft_full_package")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "kaggle_packages" / "sft_repaired_candidate")
    args = parser.parse_args()
    report = build_repaired_package(args.source_dir, args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
