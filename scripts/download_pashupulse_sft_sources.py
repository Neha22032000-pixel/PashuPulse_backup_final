from __future__ import annotations

import hashlib
import json
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "source_documents" / "pashupulse_sft_v2" / "raw_downloads"
SOURCES = [
    ("dahd_lhdcp", "https://monitor.dahd.gov.in/livestock-health-disease", "dahd_lhdcp_overview.html"),
    ("dahd_nadcp", "https://www.dahd.gov.in/schemes/programmes/nadcp", "dahd_nadcp.html"),
    ("fao_farmer_first_aid", "https://www.fao.org/4/t1265e/t1285e09.htm", "fao_farmer_first_aid.html"),
    ("nddb_aflatoxicosis", "https://www.nddb.coop/farmer/animal-health/disease/fungal/Aflatoxicosis", "nddb_aflatoxicosis.html"),
    ("nddb_clean_milk", "https://www.nddb.coop/farmer/dairying/cmp", "nddb_clean_milk.html"),
    (
        "nddb_water",
        "https://www.nddb.coop/farmer/animal-nutrition/importance-of-drinking-water-for-dairy-animals",
        "nddb_water.html",
    ),
    (
        "tnau_calf_management",
        "http://agritech.tnau.ac.in/animal_husbandry/animhus_cattle_care%26management.html",
        "tnau_cattle_care_management.html",
    ),
    ("who_rabies", "https://www.who.int/news/item/15-01-2018-who-announces-new-rabies-recommendations", "who_rabies_recommendations.html"),
    (
        "msd_mastitis",
        "https://www.merckvetmanual.com/reproductive-system/mastitis-in-large-animals/mastitis-in-cattle",
        "msd_merck_mastitis_cattle.html",
    ),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, out_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "PashuPulse-source-audit/1.0"})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        out_path.write_bytes(response.read())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for source_id, url, filename in SOURCES:
        out_path = OUT_DIR / filename
        try:
            download(url, out_path)
            results.append(
                {
                    "source_id": source_id,
                    "url": url,
                    "file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
                    "status": "downloaded",
                    "bytes": out_path.stat().st_size,
                    "sha256": sha256_file(out_path),
                }
            )
        except Exception as exc:
            results.append({"source_id": source_id, "url": url, "file": filename, "status": "failed", "error": str(exc)})
    (OUT_DIR / "download_results.json").write_text(
        json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
