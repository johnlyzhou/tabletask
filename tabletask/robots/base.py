"""The base class every robot implements."""

import copy
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Robot(ABC):
    """Base class for a robot embodiment placed in a task scene.

    A subclass sets ``model`` to its MJCF file and ``cameras`` to the camera names
    it defines, and implements ``setup``, ``reset``, ``apply`` and ``observe``. The
    robot's MJCF is attached to the scene under the namespace ``<name>/``, so every
    element of the robot is addressed as ``qualified(element_name)``.

    A robot with an arm also sets ``limits``, each arm joint's (lower, upper) range, and
    ``speed_limits``, each arm joint's top speed; one that senses grasps sets ``fingers``,
    the body ids of each finger in the bound model. ``setup`` sets ``tcp``, the id of the
    end-effector site.

    Args:
        name: Scene namespace for the robot: letters, digits, underscores or hyphens,
            not starting with a digit or hyphen.
        position: Base position (x, y, z) in the world frame, in metres.
        quaternion: Base orientation as a (w, x, y, z) quaternion; normalized on input.

    Raises:
        ValueError: If the name, position or quaternion is invalid.
    """

    model: Path  # MJCF file of the robot
    cameras: tuple[str, ...] = ()  # camera names in the robot's MJCF, unqualified
    fingers: tuple[frozenset[int], ...] = ()
    limits: np.ndarray = np.zeros((0, 2))
    speed_limits: np.ndarray = np.zeros(0)

    def __init__(self, name, position=(0, 0, 0.75), quaternion=(1, 0, 0, 0)):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
            raise ValueError(
                "robot name must be an identifier using letters, digits, underscores or hyphens"
            )
        self.name = name
        self.position = np.asarray(position, dtype=float).copy()
        self.quaternion = np.asarray(quaternion, dtype=float).copy()
        if self.position.shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("position must contain three finite metres")
        if (
            self.quaternion.shape != (4,)
            or not np.isfinite(self.quaternion).all()
            or np.linalg.norm(self.quaternion) < 1e-8
        ):
            raise ValueError("quaternion must be a finite, nonzero wxyz quaternion")
        self.quaternion /= np.linalg.norm(self.quaternion)

    def qualified(self, name):
        """The scene name of a robot element, ``<robot name>/<name>``."""
        return f"{self.name}/{name}"

    def attach(self, scene):
        """Add the robot to an MJCF scene under a world body at its base pose.

        Args:
            scene: Root ``<mujoco>`` element of the scene, modified in place.
        """
        self.add_model(scene)
        body = ET.SubElement(
            scene.find("worldbody"),
            "body",
            name=self.name,
            pos=" ".join(map(str, self.position)),
            quat=" ".join(map(str, self.quaternion)),
        )
        ET.SubElement(body, "attach", model=f"robot:{self.name}", prefix=f"{self.name}/")

    def add_model(self, scene):
        """Declare the robot's MJCF as the scene asset ``robot:<name>``.

        Args:
            scene: Root ``<mujoco>`` element of the scene, modified in place.

        Raises:
            ValueError: If an element of the scene already uses the robot's namespace.
        """
        if any(
            element.get("name", "") == self.name
            or element.get("name", "").startswith(self.name + "/")
            for element in scene.iter()
        ):
            raise ValueError(f"robot namespace {self.name!r} is already used by the scene")
        asset = scene.find("asset")
        if asset is None:
            asset = ET.SubElement(scene, "asset")
        ET.SubElement(
            asset, "model", name=f"robot:{self.name}", file=str(Path(self.model).resolve())
        )

    def bind(self, runtime):
        """Return a copy of this robot bound to a compiled runtime.

        The copy stores ``runtime``, resolves its model indices with ``setup`` and
        checks that each of its cameras exists.

        Args:
            runtime: The compiled MuJoCo runtime that contains this robot.

        Returns:
            A new robot instance owned by that runtime.
        """
        robot = copy.deepcopy(self)
        robot.runtime = runtime
        robot.setup()
        for camera in robot.cameras:
            runtime.model.camera(robot.qualified(camera))
        return robot

    @abstractmethod
    def setup(self):
        """Resolve model indices on the bound robot from ``self.runtime.model``."""

    @abstractmethod
    def reset(self):
        """Set the home joint state and actuator targets before the start is sampled."""

    @abstractmethod
    def apply(self, action):
        """Set actuator controls from an action; the environment steps the physics.

        Args:
            action: Array in the robot's action format.

        Raises:
            ValueError: If the action has the wrong shape or invalid values.
        """

    @abstractmethod
    def observe(self):
        """Return copies of the robot's proprioceptive state.

        Camera images are rendered by the environment, not here.
        """
