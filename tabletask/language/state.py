"""The interface conditions use to read a simulator's state."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class State(Protocol):
    """What conditions read from a simulator. `mujoco.Runtime` implements it.

    Optional, used only by the words that need them: `handled()` for the objects the
    robot holds, and `capabilities` with `observe(capability, *args)` for observations
    declared by vocabulary modules.
    """

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World position (3,) and wxyz quaternion (4,) of a named body or region."""

    def aabb(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World axis-aligned bounds (low, high) of a named body or region."""

    def within(self, name: str, point, slack: float = 0.0) -> bool:
        """Whether a world point lies in a region (or a body's bounds), grown by `slack`."""

    def overlap(self, a: str, b: str) -> float:
        """Signed distance between two regions in metres; negative when they intersect."""

    def contacts(self, a: str, b: str) -> bool:
        """Whether two named bodies or regions are in contact."""

    def touched(self, name: str) -> set[str]:
        """What a task object touches: other task objects, and `table` for anything fixed in the
        scene; the robot is left out."""

    def resting_on(self, name: str) -> list[str]:
        """The objects supporting `name` from below."""

    def surface(self, name: str, at) -> float:
        """The height of a body's surface below a world point; a region's floor height."""

    def joint(self, name: str) -> float:
        """The current value of a named joint."""

    def joint_range(self, name: str) -> tuple[float, float]:
        """The (low, high) limits of a named joint."""

    def start_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Where a named body was at the start of the episode."""

    def rewind(self):
        """A context manager that shows the start state, restoring the current one on exit."""
