#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train ACT policy on the gamepad SO-101 dataset.

Usage:
  python examples/gamepad/train_act.py
"""

import subprocess
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = str(Path(__file__).resolve().parent / "records")
REPO_ID = "SonDePoisson/so101_gamepad"
OUTPUT_DIR = "outputs/train/act_gamepad"
POLICY_TYPE = "act"

STEPS = 20000
BATCH_SIZE = 8
SAVE_FREQ = 200
LOG_FREQ = 10
NUM_WORKERS = 2
WANDB = True
RESUME = True


def main():
    if RESUME:
        config_path = str(Path(OUTPUT_DIR) / "checkpoints" / "last" / "pretrained_model" / "train_config.json")
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={config_path}",
            "--resume=true",
            f"--steps={STEPS}",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--policy.type={POLICY_TYPE}",
            f"--dataset.repo_id={REPO_ID}",
            f"--dataset.root={DATASET_ROOT}",
            f"--batch_size={BATCH_SIZE}",
            f"--steps={STEPS}",
            f"--save_freq={SAVE_FREQ}",
            f"--log_freq={LOG_FREQ}",
            f"--num_workers={NUM_WORKERS}",
            f"--output_dir={OUTPUT_DIR}",
            f"--wandb.enable={'true' if WANDB else 'false'}",
            "--policy.push_to_hub=false",
            "--policy.chunk_size=50",
            "--policy.n_action_steps=1",
            "--policy.temporal_ensemble_coeff=0.01",
        ]

    print("Launching ACT training with command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
