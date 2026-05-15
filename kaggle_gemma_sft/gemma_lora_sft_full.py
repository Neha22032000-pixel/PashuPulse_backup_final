from __future__ import annotations

import os

os.environ["PASHU_SAATHI_TRAIN_MODE"] = "full"
os.environ.setdefault("PASHU_SAATHI_OUT_DIR", "/kaggle/working/pashu_pulse_cleaned_full_sft")

from gemma_lora_sft import main


if __name__ == "__main__":
    main()
