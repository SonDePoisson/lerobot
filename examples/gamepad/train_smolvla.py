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
Fine-tune smolVLA on the gamepad SO-101 dataset.

Usage:
  python examples/gamepad/train.py
"""

import subprocess
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = str(Path(__file__).resolve().parent / "records")
REPO_ID = "SonDePoisson/so101_gamepad"
OUTPUT_DIR = "outputs/train/smolvla_gamepad"
POLICY_PATH = "lerobot/smolvla_base"

STEPS = 10000  # Quick test (increase to 10_000 for full training)
BATCH_SIZE = 8  # Small for MPS/CPU, increase on GPU (e.g. 32-64)
SAVE_FREQ = 200  # Save checkpoint every N steps
LOG_FREQ = 10  # Log metrics every N steps
NUM_WORKERS = 2  # Dataloader workers
WANDB = True  # Enable Weights & Biases for training curves
RESUME = False  # Set True to resume from last checkpoint (update STEPS to new total)


def main():
    if RESUME:
        # Resume from last checkpoint — just pass config_path + new steps
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
            f"--policy.path={POLICY_PATH}",
            f"--dataset.repo_id={REPO_ID}",
            f"--dataset.root={DATASET_ROOT}",
            f"--batch_size={BATCH_SIZE}",
            f"--steps={STEPS}",
            f"--save_freq={SAVE_FREQ}",
            f"--log_freq={LOG_FREQ}",
            f"--num_workers={NUM_WORKERS}",
            f"--output_dir={OUTPUT_DIR}",
            "--policy.push_to_hub=false",
            f"--wandb.enable={'true' if WANDB else 'false'}",
            '--rename_map={"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}',
            "--policy.empty_cameras=1",
        ]

    print("Launching training with command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
