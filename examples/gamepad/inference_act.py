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
Run a trained ACT policy on the SO-101 robot using EE-space actions.

Async inference: the robot loop runs at 30 FPS while model inference
runs in a background thread. The robot replays the last predicted action
until a new one is available.

Usage:
  SO101_PORT=/dev/tty.usbmodemXXX python examples/gamepad/inference_act.py
"""

import os
import shutil
import threading
import time
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors.feetech import OperatingMode
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_teleop_action_processor,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import get_safe_torch_device
from lerobot.utils.utils import log_say

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "outputs/train/act_gamepad/checkpoints/last/pretrained_model"
DATASET_ROOT = str(Path(__file__).resolve().parent / "records" / "SonDePoisson" / "so101_gamepad")
REPO_ID = "SonDePoisson/so101_gamepad"
FPS = 30
NUM_EPISODES = 5
EPISODE_TIME_SEC = 30
TASK_DESCRIPTION = "Pick up the white rubber and place it in the brown box"

OBS_STR = "observation"
ACTION = "action"

INITIAL_POSITION = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 90.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


def main():
    port = os.environ["SO101_PORT"]
    urdf_path = "./SO101"

    # ── Load trained model ────────────────────────────────────────────────
    print(f"Loading model from {MODEL_PATH}...")
    policy = ACTPolicy.from_pretrained(MODEL_PATH)
    policy.eval()
    device = get_safe_torch_device(policy.config.device)
    print(f"Model loaded (device: {device})")

    # ── Robot config ──────────────────────────────────────────────────────
    camera_config = {
        "top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        "wrist": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
    }
    robot_config = SO101FollowerConfig(
        port=port,
        id="so101_follower",
        cameras=camera_config,
        use_degrees=True,
    )
    robot = SO101Follower(robot_config)

    # ── Kinematics ────────────────────────────────────────────────────────
    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )
    motor_names = list(robot.bus.motors.keys())

    # ── Pipelines ─────────────────────────────────────────────────────────
    robot_ee_to_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=motor_names),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # ── Dataset (needed for stats/features) ───────────────────────────────
    dataset_root = Path(__file__).resolve().parent / "records" / REPO_ID
    dataset = LeRobotDataset(repo_id=REPO_ID, root=dataset_root)

    # ── Pre/post processors ───────────────────────────────────────────────
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=MODEL_PATH,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ── Create eval dataset ───────────────────────────────────────────────
    eval_repo_id = f"{REPO_ID}_eval"
    for eval_path in [
        Path(__file__).resolve().parent / "records" / eval_repo_id,
        HF_LEROBOT_HOME / eval_repo_id,
    ]:
        if eval_path.exists():
            shutil.rmtree(eval_path)
            print(f"Removed previous eval dataset at {eval_path}")

    eval_dataset = LeRobotDataset.create(
        repo_id=eval_repo_id,
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=robot_joints_to_ee,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=make_default_teleop_action_processor(),
                initial_features=create_initial_features(
                    action={
                        f"ee.{k}": PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]
                    }
                ),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # ── Connect robot ─────────────────────────────────────────────────────
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Robot failed to connect!")

    with robot.bus.torque_disabled():
        for motor in robot.bus.motors:
            robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            robot.bus.write("P_Coefficient", motor, 16)

    print("Moving to initial position...")
    robot.send_action(INITIAL_POSITION)
    time.sleep(3.0)

    listener, events = init_keyboard_listener()

    # ── Async inference state ─────────────────────────────────────────────
    lock = threading.Lock()
    latest_action: dict | None = None  # EE action dict from make_robot_action
    inference_obs: dict | None = None  # observation frame to infer on
    inference_ready = threading.Event()
    inference_done = threading.Event()
    stop_inference = threading.Event()

    def inference_worker():
        """Background thread: waits for an observation, runs model, stores action."""
        nonlocal latest_action
        while not stop_inference.is_set():
            # Wait for a new observation to be available
            if not inference_ready.wait(timeout=0.1):
                continue
            inference_ready.clear()

            with lock:
                obs_frame = inference_obs.copy() if inference_obs else None

            if obs_frame is None:
                continue

            # Run model inference (the slow part)
            action_values = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=TASK_DESCRIPTION,
                robot_type=robot.robot_type,
            )
            action_dict = make_robot_action(action_values, dataset.features)

            with lock:
                latest_action = action_dict

            inference_done.set()

    try:
        for episode_idx in range(NUM_EPISODES):
            log_say(f"Running inference episode {episode_idx + 1} of {NUM_EPISODES}")

            # Reset state for this episode
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            latest_action = None
            stop_inference.clear()

            # Start inference thread
            worker = threading.Thread(target=inference_worker, daemon=True)
            worker.start()

            timestamp = 0
            start_episode_t = time.perf_counter()

            while timestamp < EPISODE_TIME_SEC:
                start_loop_t = time.perf_counter()

                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                # 1. Get observation + FK
                obs = robot.get_observation()
                obs_processed = robot_joints_to_ee(obs)

                # 2. Build observation frame for dataset/model
                observation_frame = build_dataset_frame(
                    eval_dataset.features, obs_processed, prefix=OBS_STR
                )

                # 3. Feed observation to inference thread
                with lock:
                    inference_obs = observation_frame
                inference_ready.set()

                # 4. If we have an action, send it to the robot
                with lock:
                    current_action = latest_action

                if current_action is not None:
                    # EE action → joint action via IK
                    robot_action_to_send = robot_ee_to_joints((current_action, obs))
                    robot.send_action(robot_action_to_send)

                    # Write to eval dataset
                    action_frame = build_dataset_frame(
                        eval_dataset.features, current_action, prefix=ACTION
                    )
                    frame = {**observation_frame, **action_frame, "task": TASK_DESCRIPTION}
                    eval_dataset.add_frame(frame)
                else:
                    # First frame: wait for initial inference to complete
                    inference_done.wait(timeout=2.0)
                    inference_done.clear()

                # 5. Rate limiting
                dt_s = time.perf_counter() - start_loop_t
                sleep_time_s = 1 / FPS - dt_s
                if sleep_time_s > 0:
                    time.sleep(sleep_time_s)

                timestamp = time.perf_counter() - start_episode_t

            # Stop inference thread
            stop_inference.set()
            worker.join(timeout=2.0)

            if events.get("rerecord_episode"):
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                eval_dataset.clear_episode_buffer()
                continue

            eval_dataset.save_episode()

            # Move back to initial position between episodes
            print("Resetting to initial position...")
            robot.send_action(INITIAL_POSITION)
            time.sleep(3.0)

    except KeyboardInterrupt:
        print("\nStopping inference.")
    finally:
        stop_inference.set()
        robot.disconnect()
        if listener is not None:
            listener.stop()
        eval_dataset.finalize()
        eval_dataset.push_to_hub()
        print("Done.")


if __name__ == "__main__":
    main()
