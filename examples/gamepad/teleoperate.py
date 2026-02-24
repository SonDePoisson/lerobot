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
Teleoperate an SO-101 follower arm end-effector using a Stadia gamepad.

Controls:
  - Left stick  : move end-effector in X/Y plane
  - Right stick Y: move end-effector in Z axis
  - L2 / R2     : close / open gripper
  - Y button    : mark episode SUCCESS
  - A button    : mark episode FAILURE
  - X button    : rerecord episode

Usage:
  python examples/gamepad/teleoperate.py \
      --port /dev/ttyUSB0 \
      --urdf ./SO101/so101_new_calib.urdf

URDF:
  Download from https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
"""

import os
import time
from dataclasses import dataclass

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    ProcessorStepRegistry,
    RobotActionProcessorStep,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.teleoperators.gamepad.teleop_gamepad import GripperAction
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


@ProcessorStepRegistry.register("map_gamepad_action_to_robot_action")
@dataclass
class MapGamepadActionToRobotAction(RobotActionProcessorStep):
    """
    Maps gamepad delta actions to the format expected by EEReferenceAndDelta.

    The gamepad outputs:
      - delta_x, delta_y, delta_z  (floats in [-1, 1])
      - gripper  (0=close, 1=stay, 2=open)

    EEReferenceAndDelta expects:
      - enabled, target_x, target_y, target_z, target_wx, target_wy, target_wz, gripper_vel
    """

    def action(self, action: RobotAction) -> RobotAction:
        delta_x = float(action.pop("delta_x"))
        delta_y = float(action.pop("delta_y"))
        delta_z = float(action.pop("delta_z"))
        gripper = int(action.pop("gripper"))

        # The gamepad is always "enabled" (no enable/disable toggle like the phone)
        action["enabled"] = True

        # Map gamepad deltas to target deltas
        action["target_x"] = delta_x
        action["target_y"] = delta_y
        action["target_z"] = delta_z

        # No orientation control from gamepad — keep current orientation
        action["target_wx"] = 0.0
        action["target_wy"] = 0.0
        action["target_wz"] = 0.0

        # Map discrete gripper action to velocity:
        #   CLOSE(0) -> -1.0, STAY(1) -> 0.0, OPEN(2) -> +1.0
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
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


def main():
    port = os.environ["SO101_PORT"]
    urdf_path = "./SO101"

    # Robot configuration
    robot_config = SO101FollowerConfig(
        port=port,
        id="so101_follower",
        use_degrees=True,
    )

    # Gamepad configuration
    teleop_config = GamepadTeleopConfig(use_gripper=True)

    # Instantiate robot and gamepad
    robot = SO101Follower(robot_config)
    teleop = GamepadTeleop(teleop_config)

    # Kinematics solver for FK/IK
    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    motor_names = list(robot.bus.motors.keys())

    # Build processing pipeline: gamepad deltas -> EE pose -> joint positions
    gamepad_to_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapGamepadActionToRobotAction(),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
                motor_names=motor_names,
                use_latched_reference=False,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(speed_factor=20.0),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect devices
    robot.connect()
    teleop.connect()

    if not robot.is_connected or not teleop.is_connected:
        raise RuntimeError("Robot or gamepad failed to connect!")

    print("Teleoperation started. Use the Stadia gamepad to control the arm.")
    print("  Left stick  : X / Y movement")
    print("  Right stick Y: Z movement")
    print("  L2 / R2     : close / open gripper")
    print("  Press Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Read current robot state
            robot_obs = robot.get_observation()

            # 2. Read gamepad input
            gamepad_action = teleop.get_action()

            # 3. Convert gamepad deltas -> joint positions via IK pipeline
            try:
                joint_action = gamepad_to_joints_processor((gamepad_action, robot_obs))
            except ValueError as e:
                # EEBoundsAndSafety raises on large jumps — skip this frame
                print(f"Safety skip: {e}")
                continue

            # 4. Send joint positions to robot
            robot.send_action(joint_action)

            # 5. Maintain target FPS
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nStopping teleoperation.")
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
