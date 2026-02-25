#!/usr/bin/env python

"""
Trim idle frames from the beginning and end of each episode in a LeRobotDataset.

The gamepad recording often has idle frames (no movement) at the start and end
of episodes. This script detects movement and keeps only the active portion.
The original dataset is replaced by the trimmed version.

Usage:
  python examples/gamepad/trim_dataset.py
"""

import shutil

import numpy as np
import pandas as pd
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ID = "SonDePoisson/so101_gamepad"
MOTION_THRESHOLD = 0.001  # 1mm - frames with less movement than this are "idle"
PADDING_FRAMES = 5  # Keep a few frames before first movement and after last


def find_active_range(actions: np.ndarray, threshold: float, padding: int) -> tuple[int, int]:
    """Find the first and last frames with significant movement."""
    deltas = np.diff(actions[:, :3], axis=0)
    moving = np.abs(deltas).max(axis=1) > threshold

    if not moving.any():
        return -1, -1

    first_move = int(np.argmax(moving))
    last_move = int(len(moving) - 1 - np.argmax(moving[::-1]))

    start = max(0, first_move - padding)
    end = min(len(actions), last_move + 1 + padding)

    return start, end


def main():
    dataset_root = Path(__file__).resolve().parent / "records" / REPO_ID
    tmp_root = dataset_root.parent / f"{dataset_root.name}_trimming_tmp"

    # ── Load source dataset ──────────────────────────────────────────────
    print(f"Loading dataset from {dataset_root}...")
    source = LeRobotDataset(repo_id=REPO_ID, root=dataset_root)
    print(f"  Episodes: {source.meta.total_episodes}")
    print(f"  Total frames: {source.meta.total_frames}")

    # ── Pre-analyze with parquet to find trim ranges (fast, no video decoding) ─
    print("\nAnalyzing episodes for idle frames...")
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    trim_ranges = {}
    total_original = 0
    total_kept = 0

    for ep_idx in range(source.meta.total_episodes):
        ep_df = df[df["episode_index"] == ep_idx]
        if len(ep_df) == 0:
            continue

        actions = np.stack(ep_df["action"].values)
        start, end = find_active_range(actions, MOTION_THRESHOLD, PADDING_FRAMES)

        if start == -1:
            print(f"  Episode {ep_idx}: NO movement, skipping entirely")
            continue

        kept = end - start
        trimmed = len(ep_df) - kept
        trim_ranges[ep_idx] = (start, end)
        total_original += len(ep_df)
        total_kept += kept

        print(f"  Episode {ep_idx}: {len(ep_df)} -> {kept} frames (trimmed {trimmed})")

    total_trimmed = total_original - total_kept
    print(f"\nSummary: {total_original} -> {total_kept} frames "
          f"({total_trimmed} trimmed, {100 * total_trimmed / total_original:.0f}% reduction)")

    # ── Create trimmed dataset in temp directory ─────────────────────────
    print(f"\nCreating trimmed dataset...")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    features = {
        k: v for k, v in source.meta.features.items()
        if k not in ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    }

    target = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=source.fps,
        root=tmp_root,
        features=features,
        robot_type=source.meta.robot_type,
        use_videos=True,
        image_writer_threads=4,
    )

    # ── Copy trimmed frames ──────────────────────────────────────────────
    for ep_idx, (start, end) in trim_ranges.items():
        from_idx = source.meta.episodes["dataset_from_index"][ep_idx]
        task = source.meta.episodes["tasks"][ep_idx][0]

        print(f"  Episode {ep_idx} (frames {start}-{end})...", end="", flush=True)

        for rel_frame in range(start, end):
            abs_frame = from_idx + rel_frame
            sample = source[abs_frame]

            frame = {}
            for key in features:
                val = sample[key]
                # Images come as (C, H, W) from dataset, add_frame expects (H, W, C)
                if features[key]["dtype"] in ["image", "video"] and val.ndim == 3:
                    val = val.permute(1, 2, 0)
                frame[key] = val
            frame["task"] = task

            target.add_frame(frame)

        target.save_episode()
        print(f" done ({end - start} frames)")

    target.finalize()

    # ── Replace original with trimmed ────────────────────────────────────
    del source
    print(f"\nReplacing original dataset...")
    shutil.rmtree(dataset_root)
    shutil.move(str(tmp_root), str(dataset_root))

    print(f"Done! Dataset at {dataset_root}")
    print(f"  Episodes: {target.meta.total_episodes}")
    print(f"  Total frames: {target.meta.total_frames}")


if __name__ == "__main__":
    main()
