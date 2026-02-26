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
Record episodes on an SO-101 follower arm using a Stadia gamepad.

Based on the official so100_to_so100_EE example from LeRobot.

Controls:
  - Left stick    : move end-effector in X/Y plane
  - Right stick Y : move end-effector in Z axis
  - L2 / R2       : close / open gripper
  - A button      : mark episode SUCCESS (end episode early)
  - Y button      : mark episode FAILURE (end episode early)
  - X button      : re-record current episode

Usage:
    python examples/gamepad/record.py
"""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors.feetech import OperatingMode
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    ProcessorStepRegistry,
    RobotActionProcessorStep,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEE,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.teleoperators.gamepad.teleop_gamepad import GripperAction
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_EPISODES = 45
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 1
TASK_DESCRIPTION = "Pick up the white rubber and place it in the brown box"
HF_REPO_ID = "SonDePoisson/so101_gamepad"
RESUME = True

INITIAL_POSITION = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 90.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


# ── Gamepad wrapper to bridge episode buttons → record_loop events ───────────
class GamepadTeleopWithEvents(GamepadTeleop):
    """Wraps GamepadTeleop so that A/Y/X buttons update the record_loop events dict."""

    def __init__(self, config):
        super().__init__(config)
        self._events = None

    def bind_events(self, events: dict):
        self._events = events

    def get_action(self):
        action = super().get_action()
        if self._events is not None:
            teleop_events = self.get_teleop_events()
            if teleop_events.get(TeleopEvents.SUCCESS):
                self._events["exit_early"] = True
            if teleop_events.get(TeleopEvents.TERMINATE_EPISODE):
                self._events["exit_early"] = True
            if teleop_events.get(TeleopEvents.RERECORD_EPISODE):
                self._events["exit_early"] = True
                self._events["rerecord_episode"] = True
        return action


# ── Gamepad → EE mapping ─────────────────────────────────────────────────────
@ProcessorStepRegistry.register("map_gamepad_action_to_robot_action")
@dataclass
class MapGamepadActionToRobotAction(RobotActionProcessorStep):
    """Accumulates gamepad velocity-like deltas into absolute EE offsets."""

    _acc_x: float = 0.0
    _acc_y: float = 0.0
    _acc_z: float = 0.0

    def action(self, action: RobotAction) -> RobotAction:
        delta_x = float(action.pop("delta_x"))
        delta_y = float(action.pop("delta_y"))
        delta_z = float(action.pop("delta_z"))
        gripper = int(action.pop("gripper"))

        self._acc_x += delta_x
        self._acc_y += delta_y
        self._acc_z += delta_z

        action["enabled"] = True
        action["target_x"] = self._acc_x
        action["target_y"] = self._acc_y
        action["target_z"] = self._acc_z
        action["target_wx"] = 0.0
        action["target_wy"] = 0.0
        action["target_wz"] = 0.0
        action["gripper_vel"] = float(gripper - GripperAction.STAY)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["delta_x", "delta_y", "delta_z", "gripper"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


def main():
    port = os.environ["SO101_PORT"]
    urdf_path = "./SO101"

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

    teleop_config = GamepadTeleopConfig(use_gripper=True)

    robot = SO101Follower(robot_config)
    teleop = GamepadTeleopWithEvents(teleop_config)

    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )
    motor_names = list(robot.bus.motors.keys())

    # ── Pipeline 1: gamepad → EE pose (stored in dataset as action) ──────
    gamepad_to_ee_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapGamepadActionToRobotAction(),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
                motor_names=motor_names,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(speed_factor=20.0),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # ── Pipeline 2: EE pose → joint positions (sent to robot) ────────────
    ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
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

    # ── Pipeline 3: joint observation → EE observation (stored in dataset)
    joints_to_ee_observation = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=motor_names),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # ── Create or resume dataset ─────────────────────────────────────────
    dataset_root = Path(__file__).resolve().parent / "records" / HF_REPO_ID

    if RESUME and dataset_root.exists():
        print(f"Resuming dataset at {dataset_root}")
        dataset = LeRobotDataset(repo_id=HF_REPO_ID, root=dataset_root)
        dataset.start_image_writer(num_threads=4)
        dataset.episode_buffer = dataset.create_episode_buffer()
        print(f"  Existing episodes: {dataset.meta.total_episodes}, frames: {dataset.meta.total_frames}")
    else:
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
            print(f"Removed previous dataset at {dataset_root}")
        dataset = LeRobotDataset.create(
            repo_id=HF_REPO_ID,
            fps=FPS,
            root=dataset_root,
            features=combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=gamepad_to_ee_processor,
                    initial_features=create_initial_features(
                        action={
                            "delta_x": float,
                            "delta_y": float,
                            "delta_z": float,
                            "gripper": float,
                        }
                    ),
                    use_videos=True,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=joints_to_ee_observation,
                    initial_features=create_initial_features(observation=robot.observation_features),
                    use_videos=True,
                ),
            ),
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

    # ── Connect devices ──────────────────────────────────────────────────
    robot.connect()
    teleop.connect()

    if not robot.is_connected or not teleop.is_connected:
        raise RuntimeError("Robot or gamepad failed to connect!")

    with robot.bus.torque_disabled():
        for motor in robot.bus.motors:
            robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            robot.bus.write("P_Coefficient", motor, 32)

    print("Moving to initial position...")
    robot.send_action(INITIAL_POSITION)
    print("Waiting for cameras to initialize...")
    time.sleep(3.0)

    listener, events = init_keyboard_listener()
    teleop.bind_events(events)

    try:
        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.meta.total_episodes + 1}")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=False,
                teleop_action_processor=gamepad_to_ee_processor,
                robot_action_processor=ee_to_joints_processor,
                robot_observation_processor=joints_to_ee_observation,
            )

            # Reset phase between episodes
            if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=False,
                    teleop_action_processor=gamepad_to_ee_processor,
                    robot_action_processor=ee_to_joints_processor,
                    robot_observation_processor=joints_to_ee_observation,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            episode_idx += 1

    finally:
        log_say("Stop recording")
        teleop.disconnect()
        robot.disconnect()
        if listener is not None:
            listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
