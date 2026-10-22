"""The privileged state: what the simulator knows about the robot and the task objects, flat.

A learner allowed to see more than a policy does, such as an RL critic, reads it from the
``priv_combined`` observation. Every value is scaled by known limits to lie in about [-1, 1]
while the robot and the objects stay in the workspace; nothing is clipped. In order:

- elapsed fraction of the task's time limit, in [0, 1] (0 when the task has none)
- arm joint positions, scaled to the joint limits
- arm joint velocities, over the robot's speed limits
- gripper closure and gripper command, each in [0, 1]
- end-effector (TCP) position, scaled to the workspace, and orientation, world frame
- for each task object, in ``task.scene.objects`` order:

  - position (body origin), scaled to the workspace, and orientation, world frame
  - offset from the end effector, world frame, over the workspace size
  - orientation in the end-effector frame, R_eefᵀ · R_obj
  - linear and angular velocity, world frame, over ``LINEAR_SPEED`` and ``ANGULAR_SPEED``
  - positions of its hinge and slide joints, if it is articulated, scaled to their limits,
    then zeros up to the most joints any asset it can draw has
  - 1 when both fingers touch it, else 0

  An object with a count is laid out as its copies in order, then zeros for the copies a
  range did not draw, so every episode of a task gives a vector of the same length and each
  copy keeps its place.

The workspace is the table top widened by ``WORKSPACE_MARGIN`` on each side, from
``WORKSPACE_DEPTH`` below its surface to ``WORKSPACE_HEIGHT`` above, in metres.

Every orientation is MuJoCo's rotation matrix for the frame (``xmat`` for a body, ``site_xmat``
for the TCP site): its columns are the frame's own x, y and z axes in world coordinates. It
appears as its first two columns, the x and y axes in world coordinates (six values); the z
axis is left out, since it is the cross product of x and y. Unlike a quaternion, these change
continuously with the rotation.
"""

import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import numpy as np

from . import fetching
from .generation import realize
from .language.schema import Generate

WORKSPACE_MARGIN = 0.15
WORKSPACE_DEPTH = 0.1
WORKSPACE_HEIGHT = 0.7
LINEAR_SPEED = 1.0
ANGULAR_SPEED = 2 * np.pi
OBJECT_SIZE = 25  # values per object without its joints


def privileged_state(runtime, robot, time, template=None):
    """The privileged state of an episode.

    Args:
        runtime: The episode's simulator, with a single-arm robot.
        robot: The robot's observation at this instant.
        time: Seconds elapsed in the episode.
        template: The task the episode was drawn from, whose counts and asset pools set the
            padding; None lays out the episode's objects without padding.

    Returns:
        A float32 vector laid out as the module docstring describes.
    """
    limit = runtime.task.time_limit
    arm, data = runtime.robot, runtime.data
    low, high = workspace(runtime)
    eef_position = data.site_xpos[arm.tcp].reshape(3)
    eef_rotation = data.site_xmat[arm.tcp].reshape(3, 3)
    parts = [
        [time / limit if limit else 0.0],
        _scaled(robot["joint_position"], arm.limits[:, 0], arm.limits[:, 1]),
        robot["joint_velocity"] / arm.speed_limits,
        [robot["gripper"], robot["gripper_command"]],
        _scaled(eef_position, low, high),
        _rot6(eef_rotation),
    ]
    for name, joints in _slots(runtime.task, template):
        if name is None:
            parts.append(np.zeros(OBJECT_SIZE + joints))
        else:
            parts += _object(runtime, name, joints, eef_position, eef_rotation, low, high)
    return np.concatenate(parts).astype(np.float32)


def _slots(task, template):
    """Each object slot in layout order, as (object name, joints to pad to); a copy a count
    range did not draw has no name."""
    if template is None:
        return [(name, 0) for name in task.scene.objects]
    slots = []
    for name, obj in template.scene.objects.items():
        pool = obj.asset if isinstance(obj.asset, list) else [obj.asset]
        joints = max(_joints(_file(a, template.root)) if a else 0 for a in pool)
        if obj.count == 1:
            slots.append((name, joints))
            continue
        most = obj.count[1] if isinstance(obj.count, tuple) else obj.count
        for i in range(1, most + 1):
            copy = f"{name}{i}"
            slots.append((copy if copy in task.scene.objects else None, joints))
    return slots


def _file(asset, root) -> Path:
    """An asset's file; a generator's is its middle draw, as generators keep their joints fixed."""
    if isinstance(asset, Generate):
        return Path(realize(asset, root))
    return fetching.resolve(asset, root)


@cache
def _joints(path) -> int:
    """How many hinge and slide joints an asset file has."""
    world = ET.parse(path).getroot().find("worldbody")
    if world is None:
        return 0
    return sum(j.get("type", "hinge") in ("hinge", "slide") for j in world.iter("joint"))


def _object(runtime, name, joints, eef_position, eef_rotation, low, high):
    """One object's part of the privileged state, its joints zero-padded to `joints`."""
    data = runtime.data
    body = runtime.body(name)
    position = data.xpos[body]
    rotation = data.xmat[body].reshape(3, 3)
    velocity = runtime.velocity(name)
    values, joint_low, joint_high = runtime.articulation(name)
    return [
        _scaled(position, low, high),
        _rot6(rotation),
        (position - eef_position) / (high - low),
        _rot6(eef_rotation.T @ rotation),
        velocity[:3] / LINEAR_SPEED,
        velocity[3:] / ANGULAR_SPEED,
        _scaled(values, joint_low, joint_high),
        np.zeros(max(joints - len(values), 0)),
        [float(runtime.grasped(name))],
    ]


def workspace(runtime):
    """The workspace's lower and upper corners, world frame, in metres."""
    low, high = runtime.aabb("table/top")
    surface = low[2]
    return (
        np.r_[low[:2] - WORKSPACE_MARGIN, surface - WORKSPACE_DEPTH],
        np.r_[high[:2] + WORKSPACE_MARGIN, surface + WORKSPACE_HEIGHT],
    )


def _scaled(values, low, high):
    """Values mapped linearly from [low, high] to [-1, 1]."""
    return 2 * (np.asarray(values) - low) / (high - low) - 1


def _rot6(rotation):
    """A rotation matrix's first two columns, its x and y axes, as six values; z is x cross y."""
    return rotation[:, :2].T.ravel()
