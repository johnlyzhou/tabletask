"""The DROID robot: a Franka Panda arm with a Robotiq 2F-85 gripper."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from ...language.rotation import unit_quaternions
from ...robots.base import Robot
from .ik import PoseIK

CLOSED_DRIVER_ANGLE = 0.824033  # finger driver joint angle when the gripper is closed, radians
GRIPPER_CTRL_MAX = 255  # finger actuator control for a fully closed gripper
STAND_MIN_HEIGHT = 0.04  # base height above which a pedestal is added under the arm, metres


class Droid(Robot):
    """A Franka Panda arm with a Robotiq 2F-85 gripper, as in the DROID setup.

    Actions are eight values: seven arm values followed by gripper closure in
    [0, 1] (0 open, 1 closed). The arm values depend on ``action_space``:

    - ``joint_position``: absolute targets for the seven joints, in radians.
    - ``joint_delta``: offsets from the current joint positions, in radians.
    - ``cartesian_pose``: a world-frame TCP pose (x, y, z, qw, qx, qy, qz), in
      metres; converted to joint targets by inverse kinematics (requires ``mink``
      and ``daqp``).

    Joint targets are clipped to the joint limits.

    Observations are a dict with:

    - ``joint_position``: (7,) arm joint positions, radians.
    - ``joint_velocity``: (7,) arm joint velocities, radians per second.
    - ``gripper``: closure in [0, 1] (0 open, 1 closed).
    - ``tcp_pose``: (7,) world-frame TCP position and wxyz quaternion.

    Cameras: ``wrist`` and ``external``.

    Args:
        action_space: ``"joint_position"``, ``"joint_delta"`` or ``"cartesian_pose"``.
        name: Scene namespace for the robot.
        position: Arm base position (x, y, z) in the world frame, in metres. A base
            higher than 4 cm stands on a pedestal that reaches the floor.
        quaternion: Arm base orientation as a (w, x, y, z) quaternion.

    Raises:
        ValueError: If ``action_space`` is not one of the supported values.
    """

    model = Path(__file__).parent / "assets/droid/model.xml"
    cameras = ("wrist", "external")
    home = (0.0, -0.7853, 0.0, -2.35619, 0.0, 1.57079, 0.0)  # joints, radians: DROID home, as molmospaces
    action_shape = (8,)

    def __init__(
        self,
        action_space="joint_position",
        *,
        name="droid",
        position=(0.05, 0, 0.75),
        quaternion=(1, 0, 0, 0),
        home_noise=0.0,
    ):
        super().__init__(name, position, quaternion)
        if action_space not in ("joint_position", "joint_delta", "cartesian_pose"):
            raise ValueError(
                "action_space must be 'joint_position', 'joint_delta' or 'cartesian_pose'"
            )
        self.action_space = action_space
        self.home_noise = float(home_noise)  # rad, the most a joint starts away from home
        self.start = None

    def start_pose(self, episode):
        """Where the arm starts in an episode: home, each joint moved by up to `home_noise`.

        Seeded by the episode on its own stream, so an episode always starts the same and the
        scene's own draws are unchanged.
        """
        home = np.asarray(self.home, dtype=float)
        if self.home_noise <= 0:
            return home
        rng = np.random.default_rng([int(episode), 7])
        return home + rng.uniform(-self.home_noise, self.home_noise, len(home))

    def attach(self, scene):
        """Add the arm to an MJCF scene, with a pedestal when its base is raised.

        Args:
            scene: Root ``<mujoco>`` element of the scene, modified in place.
        """
        super().attach(scene)
        height = self.position[2]
        if height > STAND_MIN_HEIGHT:
            stand = ET.SubElement(
                scene.find("worldbody"),
                "body",
                name=f"{self.name}/stand",
                pos=f"{self.position[0]} {self.position[1]} 0",
            )
            ET.SubElement(
                stand,
                "geom",
                type="cylinder",
                size=f".065 {height / 2}",
                pos=f"0 0 {height / 2}",
                rgba=".2 .22 .24 1",
            )
            ET.SubElement(
                stand, "geom", type="box", size=".14 .14 .015", pos="0 0 .015", rgba=".12 .13 .14 1"
            )

    def setup(self):
        """Resolve joint, actuator and site indices in the bound runtime's model."""
        model = self.runtime.model
        joints = [model.joint(self.qualified(f"joint{i}")) for i in range(1, 8)]
        self.qadr = np.array([joint.qposadr[0] for joint in joints])
        self.vadr = np.array([joint.dofadr[0] for joint in joints])
        self.actuators = np.array(
            [model.actuator(self.qualified(f"joint{i}")).id for i in range(1, 8)]
        )
        self.limits = np.array([joint.range for joint in joints])
        self.gripper_actuator = model.actuator(self.qualified("fingers_actuator")).id
        self.gripper_joint = model.joint(self.qualified("right_driver_joint")).qposadr[0]
        self.tcp = model.site(self.qualified("tcp")).id
        if self.action_space == "cartesian_pose":
            self.ik = PoseIK(self.model)
            self.base = model.body(self.qualified("link0")).id

    def reset(self):
        """Move the arm to its start pose, hold it there and open the gripper."""
        data = self.runtime.data
        start = np.asarray(self.home if self.start is None else self.start, dtype=float)
        start = np.clip(start, self.limits[:, 0], self.limits[:, 1])
        data.qpos[self.qadr] = start
        data.ctrl[self.actuators] = start
        data.ctrl[self.gripper_actuator] = 0.0

    def apply(self, action):
        """Set arm and gripper controls from an action.

        Args:
            action: Eight values, seven arm values and gripper closure; see the class
                docstring for the format of each action space.

        Raises:
            ValueError: If the action is malformed, the closure is outside [0, 1], the
                Cartesian orientation is zero, or inverse kinematics fails.
        """
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_shape or not np.isfinite(action).all():
            raise ValueError(
                "Droid action must be eight finite values: "
                "seven joint or pose values and gripper closure"
            )
        if not 0 <= action[7] <= 1:
            raise ValueError("gripper closure must be in [0, 1]: 0 open, 1 closed")
        data = self.runtime.data
        target = action[:7]
        if self.action_space == "joint_delta":
            target = data.qpos[self.qadr] + target
        elif self.action_space == "cartesian_pose":
            pose = target.copy()
            pose[3:7] = unit_quaternions(
                pose[3:7], "Cartesian orientation must be a nonzero wxyz quaternion"
            )
            base_quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(base_quaternion, data.xmat[self.base])
            target = self.ik.solve(
                pose, data.qpos[self.qadr], data.xpos[self.base], base_quaternion
            )
        data.ctrl[self.actuators] = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        data.ctrl[self.gripper_actuator] = GRIPPER_CTRL_MAX * action[7]

    def observe(self):
        """Return joint state, gripper closure and TCP pose; see the class docstring."""
        data = self.runtime.data
        tcp_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(tcp_quaternion, data.site_xmat[self.tcp])
        return {
            "joint_position": data.qpos[self.qadr].copy(),
            "joint_velocity": data.qvel[self.vadr].copy(),
            "gripper": float(np.clip(data.qpos[self.gripper_joint] / CLOSED_DRIVER_ANGLE, 0, 1)),
            "tcp_pose": np.concatenate((data.site_xpos[self.tcp], tcp_quaternion)),
        }
