"""Camera observations and Cartesian motion tools that a language model uses to drive Droid."""

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# How a move is timed; the system prompt below states the same numbers.
LINEAR_SPEED = 0.15  # metres per second
ANGULAR_SPEED = 0.75  # radians per second
GRIPPER_SECONDS = 0.6  # shortest move that changes the gripper
SETTLE_SECONDS = 0.2  # the final pose is held this long
MAX_MOVE_SECONDS = 10.0  # longer moves are rejected

TIMING = (
    f"Translation is interpolated at {LINEAR_SPEED:g} m/s, rotation at {ANGULAR_SPEED:g} rad/s, "
    f"with at least {GRIPPER_SECONDS:g} s for a\n"
    f"changed gripper and {SETTLE_SECONDS:g} s settling. "
    f"A move longer than {MAX_MOVE_SECONDS:g} s is rejected; choose smaller moves."
)

# System prompt sent with every request.
SYSTEM = f"""Control a simulated Droid parallel-jaw robot using the two RGB cameras and robot state.
Use exactly one tool call per turn. move_to takes partial absolute world x,y,z in metres (z up).
move_by takes partial offsets dx,dy,dz (metres) and droll,dpitch,dyaw (radians) from the pose you
are at now, so use it when you can see which way to go but not the world coordinate to go to.
Its gripper field is absolute, not an offset. Omitted offsets stay put.
roll,pitch,yaw are radians: R_target = Rz(yaw) Ry(pitch) Rx(roll) R_initial.
Omitted coordinates retain their measured values; an omitted gripper keeps its last
command. gripper is closure: 0=open, 1=closed, thresholded at 0.5. The gripper command
applies throughout the move, not only at arrival.
{TIMING}
You see fresh images after each chunk. A chunk limit can stop short of the requested target;
check the measured pose before continuing. IK can reject a waypoint, including after partial motion.
Simulation pauses during thinking. You cannot query object coordinates or execute code.
Include a short visual observation/reason in note. done ends the episode; independent task
conditions determine success. Use give_up when unable to proceed.
"""

# Fields of move_to (metres, radians, gripper closure) and move_by (offsets; gripper absolute).
TARGETS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
DELTAS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")

# Tool definitions in a provider-neutral form; examples/api.py converts them per provider.
TOOLS = [
    {
        "name": "move_to",
        "description": "Move toward a partial Cartesian target, then get fresh images.",
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "object",
                    "properties": {k: {"type": "number"} for k in TARGETS},
                    "additionalProperties": False,
                },
                "note": {"type": "string"},
            },
            "required": ["targets", "note"],
            "additionalProperties": False,
        },
    },
    {
        "name": "move_by",
        "description": "Step the gripper by an offset from where it is now, then get fresh images.",
        "parameters": {
            "type": "object",
            "properties": {
                "deltas": {
                    "type": "object",
                    "properties": {k: {"type": "number"} for k in DELTAS},
                    "additionalProperties": False,
                },
                "note": {"type": "string"},
            },
            "required": ["deltas", "note"],
            "additionalProperties": False,
        },
    },
    *[
        {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
                "additionalProperties": False,
            },
        }
        for name, description in [
            ("done", "End the episode when finished."),
            ("give_up", "End the episode when unable to continue."),
        ]
    ],
]


def rotation(pose):
    """Orientation of a (x, y, z, qw, qx, qy, qz) pose as a scipy Rotation."""
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]])


# Motion tool name -> (argument holding the fields, allowed fields).
MOVES = {"move_to": ("targets", TARGETS), "move_by": ("deltas", DELTAS)}


def _validate(name, args):
    """Raise ValueError unless `args` are well-formed arguments for tool `name`."""
    if name not in (*MOVES, "done", "give_up"):
        raise ValueError("Unknown tool")
    key, fields = MOVES.get(name, (None, ()))
    allowed = {"note", key} if key else {"note"}
    if not isinstance(args, dict) or set(args) - allowed or not isinstance(args.get("note"), str):
        raise ValueError("Supply only tool arguments and a string note")
    if key is None:
        return
    asked = args.get(key)
    if not isinstance(asked, dict) or not asked or set(asked) - set(fields):
        raise ValueError(f"Supply at least one valid {key[:-1]} dimension")
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in asked.values()):
        raise ValueError(f"{key.capitalize()} must be finite numbers")
    if not 0 <= asked.get("gripper", 0) <= 1:
        raise ValueError("gripper must be in [0, 1]")


class RobotTools:
    """Observations and tool execution for a Droid built with a Cartesian pose action space.

    Args:
        env: The built environment.
        seconds: Simulated motion budget, in seconds.
        chunk_steps: Control steps after which a move stops early; None runs moves in full.

    Raises:
        ValueError: If the robot is not Cartesian or an argument is out of range.
    """

    dt = 0.05  # control period, seconds

    def __init__(self, env, *, seconds=45.0, chunk_steps=None):
        if (
            env.runtime is None
            or getattr(env.runtime.robot, "action_space", None) != "cartesian_pose"
        ):
            raise ValueError('Build an environment with Droid(action_space="cartesian_pose") first')
        if not math.isfinite(seconds) or seconds < self.dt:
            raise ValueError("seconds must be finite and at least 0.05")
        if chunk_steps is not None and (type(chunk_steps) is not int or chunk_steps < 1):
            raise ValueError("chunk_steps must be a positive integer or None")
        self.env, self.seconds, self.chunk_steps = env, seconds, chunk_steps
        self.started = env.status.time
        robot = env.runtime.robot.observe()
        self.initial = rotation(robot["tcp_pose"])
        # the last closure commanded; a move that omits "gripper" keeps it, since the measured
        # closure of a gripper holding something wide reads as partly open
        self.grip = float(robot["gripper"] >= 0.5)

    @property
    def remaining(self):
        """Simulated seconds left in the motion budget."""
        return max(0.0, self.seconds - (self.env.status.time - self.started))

    @property
    def ended(self):
        """Whether the task failed, succeeded, or the budget has no control step left."""
        return (
            self.env.status.failed
            or self.env.status.succeeded(self.env.task)
            or self.remaining < self.dt - 1e-9
        )

    def observe(self, calls_remaining, image_size):
        """Observation for the model: instruction, robot state, budgets and two RGB images.

        `roll_pitch_yaw` is the grasp orientation relative to its orientation at the start.
        """
        raw = self.env.observe(width=image_size, height=image_size)
        robot = raw["robot"]
        rpy = (rotation(robot["tcp_pose"]) * self.initial.inv()).as_euler("xyz")
        return {
            "instruction": self.env.task.instruction,
            "robot": {
                "joint_position": robot["joint_position"].tolist(),
                "gripper": robot["gripper"],
                "tcp_pose": robot["tcp_pose"].tolist(),
                "roll_pitch_yaw": rpy.tolist(),
            },
            "seconds_remaining": round(self.remaining, 3),
            "calls_remaining": calls_remaining,
            "chunk_steps": self.chunk_steps,
            "control_dt": self.dt,
            "images": {name: raw["images"][name] for name in ("external", "wrist")},
        }

    def execute(self, name, args):
        """Run one tool call and return feedback for the model.

        Invalid arguments and IK failures are reported under "error" rather than raised.
        `done` and `give_up` return {"ended": name} without moving.
        """
        steps, before = 0, self.env.status.time
        result = {}
        try:
            _validate(name, args)
            if name not in MOVES:
                return {"ended": name}
            if self.ended:
                return {"ended": "environment"}
            asked = args["targets"] if name == "move_to" else self._offset(args["deltas"])
            actions = self._motion(asked)
            budget_steps = math.floor((self.remaining + 1e-9) / self.dt)
            limit = min(len(actions), budget_steps, self.chunk_steps or len(actions))
            for action in actions[:limit]:
                self.env.step(action, seconds=self.dt)
                steps += 1
                self.grip = float(action[-1])  # remembered once the robot has taken it
                if self.ended:
                    break
            result["chunk_complete"] = steps == len(actions)
        except ValueError as error:
            result["error"] = str(error)
        return {
            **result,
            "steps": steps,
            "simulated_seconds": self.env.status.time - before,
            "movement_executed": steps > 0,
        }

    def _offset(self, deltas):
        """Convert move_by offsets into move_to targets: the measured pose plus each offset."""
        current = self.env.runtime.robot.observe()
        pose = current["tcp_pose"]
        rpy = (rotation(pose) * self.initial.inv()).as_euler("xyz")
        asked = {}
        for i, key in enumerate(TARGETS[:3]):
            if f"d{key}" in deltas:
                asked[key] = float(pose[i]) + deltas[f"d{key}"]
        for i, key in enumerate(TARGETS[3:6]):
            if f"d{key}" in deltas:
                asked[key] = float(rpy[i]) + deltas[f"d{key}"]
        if "gripper" in deltas:
            asked["gripper"] = deltas["gripper"]
        return asked

    def _motion(self, target):
        """Interpolated pose actions toward `target`, followed by settling steps.

        Raises:
            ValueError: If the move would take longer than `MAX_MOVE_SECONDS`.
        """
        current = self.env.runtime.robot.observe()
        pose = current["tcp_pose"]
        xyz = np.array([target.get(k, pose[i]) for i, k in enumerate(TARGETS[:3])])
        initial = rotation(pose)
        rpy = (initial * self.initial.inv()).as_euler("xyz")
        angles = [target.get(k, rpy[i]) for i, k in enumerate(TARGETS[3:6])]
        final = Rotation.from_euler("xyz", angles) * self.initial
        grip = float(target["gripper"] >= 0.5) if "gripper" in target else self.grip
        gripper_changes = grip != self.grip
        duration = max(
            np.linalg.norm(xyz - pose[:3]) / LINEAR_SPEED,
            (final * initial.inv()).magnitude() / ANGULAR_SPEED,
            GRIPPER_SECONDS if gripper_changes else self.dt,
        )
        if not math.isfinite(duration) or duration > MAX_MOVE_SECONDS:
            raise ValueError(
                f"Motion exceeds {MAX_MOVE_SECONDS:g} seconds; split it into smaller moves"
            )
        count = max(1, math.ceil(duration / self.dt))
        planned = count + math.ceil(SETTLE_SECONDS / self.dt)
        interpolate = Slerp([0, 1], Rotation.from_quat([initial.as_quat(), final.as_quat()]))
        actions = []
        for i in range(1, planned + 1):
            fraction = min(i / count, 1.0)
            quaternion = interpolate(fraction).as_quat()[[3, 0, 1, 2]]
            position = pose[:3] + fraction * (xyz - pose[:3])
            actions.append(np.r_[position, quaternion, grip])
        return actions
