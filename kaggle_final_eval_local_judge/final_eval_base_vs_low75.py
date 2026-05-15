from __future__ import annotations

import os

os.environ.setdefault("PASHU_LOCAL_JUDGE_STAGE", "all")
os.environ.setdefault("PASHU_JUDGE_MODE", "group_compare")
os.environ.setdefault("PASHU_EVAL_SYSTEMS", "base_gemma,low_lr_checkpoint_75")
os.environ.setdefault("PASHU_LOCAL_JUDGE_MAX_ROWS", "50")
os.environ.setdefault("PASHU_LOCAL_JUDGE_OUT_DIR", "/kaggle/working/pashu_base_vs_low75_judge")
os.environ.setdefault("PASHU_REQUIRED_GPU", "t4")

from final_eval_local_judge import main


if __name__ == "__main__":
    main()
