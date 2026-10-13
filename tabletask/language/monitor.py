"""Tracking an episode's progress through the goal stages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .evaluation import evaluate, observe_history, unpack_expression
from .modules import require_capabilities
from .schema import Task
from .transformations import concrete
from .vocabulary import CONNECTIVES

# An object that moved more than this between two checks is still moving.
STILL_METRES = 0.002
STILL_TURN = 0.02  # quaternion distance


@dataclass
class Status:
    """Progress of an episode: elapsed time, completed stages, and any failure."""

    time: float = 0.0
    done: list[str] = field(default_factory=list)
    violated: str = ""
    timed_out: bool = False

    @property
    def failed(self) -> bool:
        """Whether an always condition broke or time ran out."""
        return bool(self.violated) or self.timed_out

    def succeeded(self, task: Task) -> bool:
        """Whether every stage of the task is done and nothing failed."""
        return not self.failed and len(self.done) == len(task.goal.stages)


def _placed_in(tree) -> list[str]:
    """The objects a condition places: the first argument of each relation.

    `inside(mug, bin)` places the mug, not the bin; `aligned([a, b, c])` places all
    three. `touching` places nothing, since a task may want contact made with an
    object still in hand.
    """
    if isinstance(tree, list):  # a list of names, as `aligned` takes, places each of them
        names = []
        for item in tree:
            names += [item] if isinstance(item, str) else _placed_in(item)
        return names
    if not isinstance(tree, dict):
        return []
    name, args = unpack_expression(tree)
    if name == "touching" or not args:
        return []
    if name in CONNECTIVES or isinstance(args[0], (dict, list)):
        return [n for a in args for n in _placed_in(a)]
    return [args[0]] if isinstance(args[0], str) else []


class Monitor:
    """Evaluates a concrete task's stages and always conditions as an episode runs.

    A stage completes once its condition has held for `hold` seconds while none of
    the objects it places is touched by the robot or still moving.
    """

    def __init__(self, task: Task):
        if task.variables or any(obj.count != 1 for obj in task.scene.objects.values()):
            raise ValueError("Monitor needs a concrete task; call instantiate first")
        self.task = concrete(task)
        self._started = False
        self._last_poses: dict[str, tuple] = {}
        self._placed = {
            name: {n.split("/")[0] for n in _placed_in(stage.condition)}
            for name, stage in self.task.goal.stages.items()
        }

    def _moved(self, state) -> set[str]:
        """The placed objects that moved since the last check.

        Displacement, not velocity: contact jitter gives a resting object a small
        nonzero velocity, but it does not move it anywhere.
        """
        moved, seen = set(), {}
        for name in {n for names in self._placed.values() for n in names}:
            try:
                position, turn = state.pose(name)
            except (KeyError, ValueError):
                continue
            seen[name] = (tuple(position), tuple(turn))
            was = self._last_poses.get(name)
            if was is None:
                continue
            drift = sum((a - b) ** 2 for a, b in zip(position, was[0])) ** 0.5
            spin = sum((a - b) ** 2 for a, b in zip(turn, was[1])) ** 0.5
            if drift > STILL_METRES or spin > STILL_TURN:
                moved.add(name)
        self._last_poses = seen
        return moved

    def reset(self, state) -> Status:
        """Start a new episode from `state`, which must satisfy the task's start."""
        require_capabilities(self.task, getattr(state, "capabilities", ()))
        self.status = Status()
        self.history = {}
        self._held = dict.fromkeys(self.task.goal.stages, 0.0)
        self._previous = dict.fromkeys(self.task.goal.stages, False)
        self._started = True
        self._finished_at = None
        self._last_poses = {}
        if self.task.scene.start is not None and not evaluate(self.task.scene.start, state):
            self._started = False
            raise ValueError("the supplied initial state does not satisfy start")
        return self.update(state, 0.0)

    def update(self, state, time: float) -> Status:
        """Evaluate the task at episode time `time` and return the status."""
        if not self._started:
            raise RuntimeError("call reset before updating the monitor")
        if not math.isfinite(time) or time < self.status.time:
            raise ValueError("episode time must be finite and monotonic")
        if self.status.failed:
            return self.status
        dt = time - self.status.time
        self.status.time = time
        trees = [s.condition for s in self.task.goal.stages.values()] + list(
            self.task.goal.always.values()
        )
        observe_history(trees, state, self.history)
        for name, condition in self.task.goal.always.items():
            if not evaluate(condition, state, self.history):
                self.status.violated = name
                return self.status
        handled = set(state.handled()) if hasattr(state, "handled") else set()
        handled |= self._moved(state)
        completed = set(self.status.done)
        for name in self.task.goal.order():
            stage = self.task.goal.stages[name]
            true_now = evaluate(stage.condition, state, self.history)
            if name in completed:
                # a maintained stage breaks when its condition does, held or not
                if stage.maintain and not true_now:
                    self.status.violated = f"maintain {name}"
                    return self.status
                continue
            # an object the robot still touches, or that is still moving, is not yet placed
            holding = true_now and not (handled & self._placed[name])
            eligible = all(p in completed for p in stage.after)
            if eligible and holding:
                self._held[name] = self._held[name] + dt if self._previous[name] else 0.0
                if self._held[name] + 1e-9 >= stage.hold:
                    self.status.done.append(name)
            else:
                self._held[name] = 0.0
            self._previous[name] = eligible and holding
        if self.status.succeeded(self.task) and self._finished_at is None:
            self._finished_at = time
        if self.task.time_limit is not None and time > self.task.time_limit:
            self.status.timed_out = (
                self._finished_at is None or self._finished_at > self.task.time_limit
            )
        return self.status
