"""A floating parallel gripper with no arm, one or a pair."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from ...language.rotation import unit_quaternions
from ...robots.base import Robot

RIGHT_MARKER_RGBA = (0.42, 0.035, 0.025, 1)  # red marker colour of the right gripper in a pair


class Gripper(Robot):
    """One or two floating parallel-jaw grippers controlled by world-frame TCP poses.

    Each gripper is a free body welded to a mocap target, so it follows the
    commanded pose without an arm. Its two fingers are position-controlled slides.

    Actions are eight values per gripper: TCP position (x, y, z) in metres, TCP
    orientation (qw, qx, qy, qz), and closure in [0, 1] (0 open, 1 closed). For a
    pair, the action is the flat concatenation of the left then the right gripper,
    shape (16,). The quaternion need not be normalized.

    Observations for one gripper are a dict with:

    - ``joint_position``: empty array; the gripper has no arm joints.
    - ``joint_velocity``: empty array.
    - ``gripper``: closure in [0, 1] (0 open, 1 closed).
    - ``gripper_command``: commanded closure in [0, 1].
    - ``tcp_pose``: (7,) world-frame TCP position and wxyz quaternion.

    For a pair, the observation is ``{"left": ..., "right": ...}`` with one such
    dict per gripper.

    Cameras: ``wrist`` and ``external`` for one gripper; ``left/wrist``,
    ``right/wrist`` and ``external`` for a pair.

    Args:
        action_space: Only ``"cartesian_pose"`` is supported.
        count: 1 for one gripper, 2 for a pair with a blue left and a red right
            marker.
        name: Scene namespace for the robot.
        position: Home TCP position (x, y, z) in metres, or a (2, 3) array for a
            pair. Defaults to (0.4, 0, 1.08) for one gripper, and to y = +0.15 (left)
            and y = -0.15 (right) at the same x and z for a pair.
        quaternion: Home TCP orientation (w, x, y, z), shared by both grippers or
            given as a (2, 4) array, one row per gripper.

    Raises:
        ValueError: If ``count``, ``action_space``, ``position`` or ``quaternion``
            is invalid.
    """

    model = Path(__file__).parent / "assets/gripper/model.xml"
    action_space = "cartesian_pose"

    def __init__(
        self,
        action_space="cartesian_pose",
        *,
        count=1,
        name="gripper",
        position=None,
        quaternion=(0, 0.7071068, -0.7071068, 0),
    ):
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count not in (1, 2)
        ):
            raise ValueError("count must be 1 or 2")
        if action_space != "cartesian_pose":
            raise ValueError("Gripper action_space must be 'cartesian_pose'")
        super().__init__(name)
        self.count = int(count)
        self.action_shape = (8 * self.count,)
        self.cameras = (
            ("wrist", "external") if count == 1 else ("left/wrist", "right/wrist", "external")
        )
        if position is None:
            position = (0.4, 0, 1.08) if count == 1 else ((0.4, 0.15, 1.08), (0.4, -0.15, 1.08))
        self.position = np.asarray(position, dtype=float).copy()
        expected = (3,) if count == 1 else (2, 3)
        if self.position.shape != expected or not np.isfinite(self.position).all():
            raise ValueError(f"position must have shape {expected} and contain finite metres")
        orientations = np.asarray(quaternion, dtype=float)
        if orientations.shape not in ((4,), (self.count, 4)) or not np.isfinite(orientations).all():
            raise ValueError("quaternion must be wxyz, shared or one per gripper")
        orientations = unit_quaternions(
            np.broadcast_to(orientations, (self.count, 4)), "quaternion must be nonzero"
        )
        self.quaternion = orientations[0] if count == 1 else orientations

    def qualified(self, name):
        """The scene name of a robot element, ``<robot name>/<name>``.

        For a pair, ``external`` resolves to the left gripper's copy of the camera,
        since each attached gripper model carries one.
        """
        if self.count == 2 and name == "external":
            name = "left/external"
        return super().qualified(name)

    def attach(self, scene):
        """Add the gripper model to an MJCF scene, once per gripper.

        Args:
            scene: Root ``<mujoco>`` element of the scene, modified in place.
        """
        self.add_model(scene)
        for side in ("",) if self.count == 1 else ("left/", "right/"):
            ET.SubElement(
                scene.find("worldbody"),
                "attach",
                model=f"robot:{self.name}",
                prefix=f"{self.name}/{side}",
            )

    def setup(self):
        """Resolve mocap, joint, actuator and site indices in the bound runtime's model."""
        model = self.runtime.model
        prefixes = (
            [self.name + "/"] if self.count == 1 else [f"{self.name}/left/", f"{self.name}/right/"]
        )
        self.mocaps = np.array([model.body_mocapid[model.body(p + "target").id] for p in prefixes])
        self.free = np.array([model.joint(p + "free").qposadr[0] for p in prefixes])
        self.free_velocity = np.array([model.joint(p + "free").dofadr[0] for p in prefixes])
        self.gripper_actuators = np.array(
            [
                [model.actuator(p + side + "_actuator").id for side in ("left", "right")]
                for p in prefixes
            ]
        )
        joints = [
            [model.joint(p + side + "_slide") for side in ("left", "right")] for p in prefixes
        ]
        self.gripper_joints = np.array([[joint.qposadr[0] for joint in row] for row in joints])
        self.gripper_velocities = np.array([[joint.dofadr[0] for joint in row] for row in joints])
        self.opening = np.array([[joint.range[1] for joint in row] for row in joints])
        self.tcp = np.array([model.site(p + "tcp").id for p in prefixes])
        self.tcp_offset = model.site_pos[self.tcp].copy()
        if self.count == 1:
            self.fingers = tuple(frozenset({model.jnt_bodyid[joint.id]}) for joint in joints[0])
        else:
            model.mat_rgba[model.mat(prefixes[1] + "charcoal").id] = RIGHT_MARKER_RGBA

    def _bases(self, positions, quaternions):
        """Gripper body positions that put each TCP at ``positions``, in metres."""
        rotations = np.zeros((self.count, 9))
        for rotation, quaternion in zip(rotations, quaternions):
            mujoco.mju_quat2Mat(rotation, quaternion)
        return positions - np.einsum("nij,nj->ni", rotations.reshape(-1, 3, 3), self.tcp_offset)

    def reset(self):
        """Place each gripper at its home pose, at rest, with its fingers open."""
        data = self.runtime.data
        quaternions = np.atleast_2d(self.quaternion)
        positions = self._bases(np.atleast_2d(self.position), quaternions)
        for qadr, vadr, position, quaternion in zip(
            self.free, self.free_velocity, positions, quaternions
        ):
            data.qpos[qadr : qadr + 7] = np.r_[position, quaternion]
            data.qvel[vadr : vadr + 6] = 0
        data.mocap_pos[self.mocaps] = positions
        data.mocap_quat[self.mocaps] = quaternions
        data.qpos[self.gripper_joints] = self.opening
        data.qvel[self.gripper_velocities] = 0
        data.ctrl[self.gripper_actuators] = self.opening

    def apply(self, action):
        """Move the mocap targets to the commanded TCP poses and set finger targets.

        Args:
            action: Eight values per gripper, (x, y, z, qw, qx, qy, qz, closure);
                left first for a pair.

        Raises:
            ValueError: If the action has the wrong shape or non-finite values, a
                closure is outside [0, 1], or a quaternion is zero.
        """
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_shape or not np.isfinite(action).all():
            raise ValueError(
                f"Gripper action must have shape {self.action_shape} and finite values"
            )
        actions = action.reshape(self.count, 8)
        if np.any((actions[:, 7] < 0) | (actions[:, 7] > 1)):
            raise ValueError("gripper closure must be in [0, 1]: 0 open, 1 closed")
        quaternions = unit_quaternions(
            actions[:, 3:7], "Cartesian orientation must be a nonzero wxyz quaternion"
        )
        positions = self._bases(actions[:, :3], quaternions)
        data = self.runtime.data
        data.mocap_pos[self.mocaps] = positions
        data.mocap_quat[self.mocaps] = quaternions
        data.ctrl[self.gripper_actuators] = self.opening * (1 - actions[:, 7, None])

    def observe(self):
        """Return gripper closure and command and TCP pose per gripper; see the class docstring."""
        data = self.runtime.data
        states = []
        for i, tcp in enumerate(self.tcp):
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[tcp])
            opening = data.qpos[self.gripper_joints[i]] / self.opening[i]
            command = data.ctrl[self.gripper_actuators[i]] / self.opening[i]
            states.append(
                {
                    "joint_position": np.zeros(0),
                    "joint_velocity": np.zeros(0),
                    "gripper": float(np.clip(1 - np.mean(opening), 0, 1)),
                    "gripper_command": float(1 - np.mean(command)),
                    "tcp_pose": np.r_[data.site_xpos[tcp], quaternion],
                }
            )
        return states[0] if self.count == 1 else dict(zip(("left", "right"), states))
