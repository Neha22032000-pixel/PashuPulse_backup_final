# Raw Download Cache

Raw source pages are intentionally not committed here.

Run this from the repo root to recreate the local raw cache:

```bash
python scripts/download_pashupulse_sft_sources.py
```

Default raw output path used by the downloader:

```text
docs/source_documents/pashupulse_sft_v2/raw_downloads/
```

For public GitHub, keep raw pages local unless `can_store_raw` is explicitly true for that document.