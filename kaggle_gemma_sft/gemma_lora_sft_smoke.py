from __future__ import annotations

import os

os.environ.setdefault("PASHU_SAATHI_TRAIN_MODE", "smoke")
os.environ.setdefault("PASHU_SAATHI_SMOKE_STEPS", "20")

from gemma_lora_sft import main


if __name__ == "__main__":
    main()
