"""Robot embodiments: the ``Robot`` base class and the bundled MuJoCo robots."""

from .base import Robot

__all__ = ["Droid", "Gripper", "Robot"]


def __getattr__(name):
    # Import the MuJoCo robots lazily so that importing this package does not load MuJoCo.
    if name in ("Droid", "Gripper"):
        from ..mujoco import robots

        return getattr(robots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
