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
  - positions of its hinge and slide joints, if it is articulated, scaled to their limits
  - 1 when both fingers touch it, else 0

The workspace is the table top widened by ``WORKSPACE_MARGIN`` on each side, from
``WORKSPACE_DEPTH`` below its surface to ``WORKSPACE_HEIGHT`` above, in metres.

Every orientation is MuJoCo's rotation matrix for the frame (``xmat`` for a body, ``site_xmat``
for the TCP site): its columns are the frame's own x, y and z axes in world coordinates. It
appears as its first two columns, the x and y axes in world coordinates (six values); the z
axis is left out, since it is the cross product of x and y. Unlike a quaternion, these change
continuously with the rotation.
"""

import numpy as np

WORKSPACE_MARGIN = 0.15
WORKSPACE_DEPTH = 0.1
WORKSPACE_HEIGHT = 0.7
LINEAR_SPEED = 1.0
ANGULAR_SPEED = 2 * np.pi


def privileged_state(runtime, robot, time):
    """The privileged state of an episode.

    Args:
        runtime: The episode's simulator, with a single-arm robot.
        robot: The robot's observation at this instant.
        time: Seconds elapsed in the episode.

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
    for name in runtime.task.scene.objects:
        body = runtime.body(name)
        position = data.xpos[body]
        rotation = data.xmat[body].reshape(3, 3)
        velocity = runtime.velocity(name)
        joints, joint_low, joint_high = runtime.articulation(name)
        parts += [
            _scaled(position, low, high),
            _rot6(rotation),
            (position - eef_position) / (high - low),
            _rot6(eef_rotation.T @ rotation),
            velocity[:3] / LINEAR_SPEED,
            velocity[3:] / ANGULAR_SPEED,
            _scaled(joints, joint_low, joint_high),
            [float(runtime.grasped(name))],
        ]
    return np.concatenate(parts).astype(np.float32)


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
