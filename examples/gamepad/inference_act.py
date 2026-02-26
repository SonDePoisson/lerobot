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

Based on the official so100_to_so100_EE evaluate.py example from LeRobot.
Uses record_loop with policy inference (same as official evaluation pattern).

Usage:
  SO101_PORT=/dev/tty.usbmodemXXX python examples/gamepad/inference_act.py
"""

import os
import time
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors.feetech import OperatingMode
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
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
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "outputs/train/act_gamepad/checkpoints/last/pretrained_model"
DATASET_ROOT = str(Path(__file__).resolve().parent / "records")
REPO_ID = "SonDePoisson/so101_gamepad"
FPS = 30
NUM_EPISODES = 5
EPISODE_TIME_SEC = 30
TASK_DESCRIPTION = "Pick up the white rubber and place it in the brown box"

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
    print(f"Model loaded (device: {policy.config.device})")

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

    # ── Pipelines (same as evaluate.py) ───────────────────────────────────
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
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # ── Create eval dataset to record inference episodes ──────────────────
    eval_dataset = LeRobotDataset.create(
        repo_id=f"{REPO_ID}_eval",
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

    try:
        for episode_idx in range(NUM_EPISODES):
            log_say(f"Running inference episode {episode_idx + 1} of {NUM_EPISODES}")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=eval_dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=False,
                teleop_action_processor=make_default_teleop_action_processor(),
                robot_action_processor=robot_ee_to_joints,
                robot_observation_processor=robot_joints_to_ee,
            )

            if events["rerecord_episode"]:
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
        robot.disconnect()
        if listener is not None:
            listener.stop()
        eval_dataset.finalize()
        print("Done.")


if __name__ == "__main__":
    main()
