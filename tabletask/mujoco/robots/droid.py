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
EXTENSION_BEHIND_BASE = 0.25  # how far an extended table top reaches behind the base, metres


class Droid(Robot):
    """A Franka Panda arm with a Robotiq 2F-85 gripper, as in the DROID setup.

    Actions are eight values: seven arm values followed by gripper closure in
    [0, 1] (0 open, 1 closed). The arm values depend on ``action_space``:

    - ``joint_position``: absolute targets for the seven joints, in radians.
    - ``joint_delta``: offsets from the current joint positions, in radians.
    - ``cartesian_pose``: a world-frame TCP pose (x, y, z, qw, qx, qy, qz), in
      metres; converted to joint targets by inverse kinematics (requires ``mink``
      and ``daqp``).

    Joint targets are clipped to the joint limits. Joint speed limits are the FR3's, in
    radians per second.

    Observations are a dict with:

    - ``joint_position``: (7,) arm joint positions, radians.
    - ``joint_velocity``: (7,) arm joint velocities, radians per second.
    - ``gripper``: closure in [0, 1] (0 open, 1 closed).
    - ``gripper_command``: commanded closure in [0, 1].
    - ``tcp_pose``: (7,) world-frame TCP position and wxyz quaternion.

    Cameras: ``wrist`` and ``external``.

    Args:
        action_space: ``"joint_position"``, ``"joint_delta"`` or ``"cartesian_pose"``.
        name: Scene namespace for the robot.
        position: Arm base position (x, y, z) in the world frame, in metres. A base
            higher than 4 cm stands on a pedestal that reaches the floor.
        quaternion: Arm base orientation as a (w, x, y, z) quaternion.
        home_noise: The most each arm joint starts away from home in an episode, radians.
        base_xy_noise: The most the base moves along x and along y in an episode, metres.
        base_yaw_noise: The most the base turns about the vertical in an episode, radians.
        table_extension: The chance an episode extends the table top under the arm, when the
            table is no higher than the base; objects are still placed on the table itself.
        camera_mirror: The chance an episode moves the external camera to the arm's other side.
        camera_azimuth_noise: The most the external camera swings around the point it looks at,
            radians.
        camera_distance_noise: The most the external camera moves towards or away from the point
            it looks at, horizontally, metres.
        camera_height_noise: The most the external camera moves up or down, metres.
        camera_target_noise: The most the point the external camera looks at moves along each
            axis, metres.
        camera_fov_noise: The most the external camera's vertical field of view changes, degrees.

    Raises:
        ValueError: If ``action_space`` is not one of the supported values.
    """

    model = Path(__file__).parent / "assets/droid/model.xml"
    cameras = ("wrist", "external")
    home = (0.0, -0.2 * np.pi, 0.0, -0.8 * np.pi, 0.0, 0.6 * np.pi, 0.0)
    action_shape = (8,)
    speed_limits = np.array((2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26))

    def __init__(
        self,
        action_space="joint_position",
        *,
        name="droid",
        position=(0.05, 0, 0.75),
        quaternion=(1, 0, 0, 0),
        home_noise=0.2,
        base_xy_noise=0.05,
        base_yaw_noise=0.15,
        table_extension=0.5,
        camera_mirror=0.5,
        camera_azimuth_noise=np.pi / 4,
        camera_distance_noise=0.2,
        camera_height_noise=0.2,
        camera_target_noise=0.05,
        camera_fov_noise=4.0,
    ):
        super().__init__(name, position, quaternion)
        if action_space not in ("joint_position", "joint_delta", "cartesian_pose"):
            raise ValueError(
                "action_space must be 'joint_position', 'joint_delta' or 'cartesian_pose'"
            )
        self.action_space = action_space
        self.home_noise = float(home_noise)  # rad, the most a joint starts away from home
        self.base_xy_noise = float(base_xy_noise)
        self.base_yaw_noise = float(base_yaw_noise)
        self.table_extension = float(table_extension)
        self.camera_mirror = float(camera_mirror)
        self.camera_azimuth_noise = float(camera_azimuth_noise)
        self.camera_distance_noise = float(camera_distance_noise)
        self.camera_height_noise = float(camera_height_noise)
        self.camera_target_noise = float(camera_target_noise)
        self.camera_fov_noise = float(camera_fov_noise)
        self.camera = (False, 0.0, 0.0, 0.0, np.zeros(3), 0.0)
        self.nominal = (self.position.copy(), self.quaternion.copy())
        self.extended = False
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

    def place(self, episode):
        """Set the base for an episode: shifted by up to `base_xy_noise` along x and y, turned by up
        to `base_yaw_noise` about the vertical with the external camera it carries, and standing
        over an extended table top with chance `table_extension`. Also draws the external camera's
        side, its swing, distance, height, target and field of view offsets, applied when the
        robot is bound.

        Seeded by the episode on its own stream, like `start_pose`.
        """
        position, quaternion = self.nominal
        rng = np.random.default_rng([int(episode), 8])
        dx, dy = rng.uniform(-self.base_xy_noise, self.base_xy_noise, 2)
        yaw = rng.uniform(-self.base_yaw_noise, self.base_yaw_noise)
        turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.quaternion = np.zeros(4)
        mujoco.mju_mulQuat(self.quaternion, turn, quaternion)
        self.position = position + np.array([dx, dy, 0.0])
        self.extended = bool(rng.random() < self.table_extension)
        self.camera = (
            bool(rng.random() < self.camera_mirror),
            rng.uniform(-self.camera_azimuth_noise, self.camera_azimuth_noise),
            rng.uniform(-self.camera_distance_noise, self.camera_distance_noise),
            rng.uniform(-self.camera_height_noise, self.camera_height_noise),
            rng.uniform(-self.camera_target_noise, self.camera_target_noise, 3),
            rng.uniform(-self.camera_fov_noise, self.camera_fov_noise),
        )

    def attach(self, scene):
        """Add the arm to an MJCF scene, with a pedestal when its base is raised.

        Args:
            scene: Root ``<mujoco>`` element of the scene, modified in place.
        """
        super().attach(scene)
        if self.extended:
            self._extend_table(scene)
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

    def _extend_table(self, scene):
        """Continue the table top and its border under the arm, as visuals only.

        The table keeps its collision block and placement regions, so sampling is unchanged.
        """
        table = scene.find("worldbody/body[@name='play_table']")
        top = None if table is None else table.find("geom[@name='table_top']")
        border = None if table is None else table.find("geom[@name='table_border']")
        if top is None or border is None:
            return
        offset = np.array([float(v) for v in (table.get("pos") or "0 0 0").split()])
        centre = np.array([float(v) for v in top.get("pos").split()])
        size = np.array([float(v) for v in top.get("size").split()])
        if offset[2] + centre[2] + size[2] > self.position[2] + 1e-3:
            return
        front = centre[0] - size[0]
        back = self.position[0] - EXTENSION_BEHIND_BASE
        rim = np.array([float(v) for v in border.get("size").split()]) - size
        for source, grow in ((border, rim), (top, np.zeros(3))):
            far = back - grow[0]
            thickness = float(source.get("size").split()[2])
            extension = ET.SubElement(
                table, "geom", {k: v for k, v in source.attrib.items() if k != "name"}
            )
            extension.set("name", f"{source.get('name')}_extension")
            extension.set(
                "size", f"{(front - far) / 2:.6g} {size[1] + grow[1]:.6g} {thickness:.6g}"
            )
            z = float(source.get("pos").split()[2])
            extension.set("pos", f"{(front + far) / 2:.6g} {centre[1]:.6g} {z:.6g}")

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
        self.fingers = tuple(
            frozenset(
                model.body(self.qualified(f"{side}_{part}")).id for part in ("pad", "follower")
            )
            for side in ("left", "right")
        )
        self._place_camera(model)
        if self.action_space == "cartesian_pose":
            self.ik = PoseIK(self.model)
            self.base = model.body(self.qualified("link0")).id

    def _place_camera(self, model):
        """Move the external camera around the point it looks at, as drawn for this episode.

        The point is where the camera's nominal view axis meets the base's horizontal plane. The
        camera keeps its distance and height from it, mirrored across the base's xz plane on the
        other side, then swings around it, moves in or out and up or down, and looks at it again
        after it moves.
        """
        mirrored, azimuth, distance, height, aim, fov = self.camera
        camera = model.camera(self.qualified("external")).id
        position = model.cam_pos[camera].copy()
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.cam_quat[camera])
        forward = -rotation.reshape(3, 3)[:, 2]
        target = position + forward * (-position[2] / forward[2])
        offset = position - target
        if mirrored:
            offset[1], target[1] = -offset[1], -target[1]
        angle = np.arctan2(offset[1], offset[0]) + azimuth
        reach = max(np.hypot(offset[0], offset[1]) + distance, 0.3)
        swing = np.array([reach * np.cos(angle), reach * np.sin(angle), offset[2] + height])
        position = target + swing
        target = target + aim
        forward = (target - position) / np.linalg.norm(target - position)
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        frame = np.column_stack([right, np.cross(right, forward), -forward])
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, frame.ravel())
        model.cam_pos[camera] = position
        model.cam_quat[camera] = quaternion
        model.cam_fovy[camera] += fov

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
        """Return joint state, gripper closure and command and TCP pose; see the class docstring."""
        data = self.runtime.data
        tcp_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(tcp_quaternion, data.site_xmat[self.tcp])
        return {
            "joint_position": data.qpos[self.qadr].copy(),
            "joint_velocity": data.qvel[self.vadr].copy(),
            "gripper": float(np.clip(data.qpos[self.gripper_joint] / CLOSED_DRIVER_ANGLE, 0, 1)),
            "gripper_command": float(data.ctrl[self.gripper_actuator] / GRIPPER_CTRL_MAX),
            "tcp_pose": np.concatenate((data.site_xpos[self.tcp], tcp_quaternion)),
        }
