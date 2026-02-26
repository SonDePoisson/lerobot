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

The policy predicts EE-space actions (same format as the recorded dataset),
which are then converted to joint positions via IK before being sent to the robot.

Usage:
  python examples/gamepad/inference_act.py
"""

import os
import time

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors.feetech import OperatingMode
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.robot_utils import precise_sleep

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "outputs/train/act_gamepad/checkpoints/last/pretrained_model"
FPS = 30
MAX_STEPS = 300  # 10 seconds at 30 FPS

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

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ── Load trained model ────────────────────────────────────────────────
    print(f"Loading model from {MODEL_PATH}...")
    model = ACTPolicy.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(device)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_PATH,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ── Robot config (same as record.py) ──────────────────────────────────
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

    # ── FK pipeline: joint observation → EE observation (for model input) ─
    joints_to_ee_observation = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=motor_names),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # ── IK pipeline: EE action → joint positions (for robot) ─────────────
    ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # ── Dataset features (must match what the model was trained on) ───────
    dataset_features = {
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"],
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
    }

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

    # Reset model action queue
    model.reset()

    print(f"Running inference for {MAX_STEPS} steps ({MAX_STEPS / FPS:.1f}s). Press Ctrl+C to stop.")

    try:
        for step in range(MAX_STEPS):
            t0 = time.perf_counter()

            # 1. Get robot observation (joints + cameras)
            obs = robot.get_observation()

            # 2. Convert joints → EE via FK
            obs_ee = joints_to_ee_observation(obs)

            # 3. Build inference frame and run model
            obs_frame = build_inference_frame(
                observation=obs_ee,
                ds_features=dataset_features,
                device=device,
                robot_type="so_follower",
            )
            obs_preprocessed = preprocess(obs_frame)
            action_tensor = model.select_action(obs_preprocessed)
            action_tensor = postprocess(action_tensor)

            # 4. Convert tensor → EE action dict
            ee_action = make_robot_action(action_tensor, dataset_features)

            # 5. Convert EE → joints via IK
            try:
                joint_action = ee_to_joints_processor((ee_action, obs))
            except ValueError as e:
                print(f"Safety skip: {e}")
                continue

            # 6. Send to robot
            robot.send_action(joint_action)

            # 7. Maintain FPS
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

            dt = time.perf_counter() - t0
            if step % 30 == 0:
                print(
                    f"Step {step}/{MAX_STEPS} | dt={dt:.3f}s ({1 / dt:.1f} FPS) | ee=({ee_action.get('ee.x', 0):.3f}, {ee_action.get('ee.y', 0):.3f}, {ee_action.get('ee.z', 0):.3f})"
                )

    except KeyboardInterrupt:
        print("\nStopping inference.")
    finally:
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
