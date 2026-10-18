"""The built-in relations (true or false) and terms (numbers) conditions are made of."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np

from .errors import UnsupportedError
from .rotation import compose, quat, spin

TERMS: dict[str, Callable] = {}
RELATIONS: dict[str, Callable] = {}
ARITHMETIC: dict[str, Callable] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}
FLOOR = 0.001  # metres above an object's underside, to test the point it stands on
AXES = {"x": 0, "y": 1, "z": 2}


def term(fn: Callable) -> Callable:
    """Register a function as a term, a measurement conditions can compare."""
    TERMS[fn.__name__] = fn
    return fn


def relation(fn: Callable) -> Callable:
    """Register a function as a relation, a word a condition can use."""
    RELATIONS[fn.__name__] = fn
    return fn


@relation
def at_most(s, quantity, bound):
    """Whether a number is at most a bound."""
    return quantity <= bound


@relation
def at_least(s, quantity, bound):
    """Whether a number is at least a bound."""
    return quantity >= bound


@relation
def between(s, quantity, low, high):
    """Whether a number lies in [low, high]."""
    return low <= quantity <= high


@relation
def touching(s, a, b):
    """Whether two objects or regions are in contact."""
    return s.contacts(a, b)


@relation
def within(s, a, b, slack=0.0):
    """Whether a's origin lies in region b, grown by `slack` metres."""
    return s.within(b, s.pose(a)[0], slack)


@relation
def contained(s, a, b, slack=0.0):
    """Whether a is in region b.

    a counts as in b if its origin, the centre of its bounding box, or the bottom
    centre of its bounding box lies in b, or if it rests on an object that is in b.
    This covers a bottle lying in a bin (origin at one end), a pencil standing in a
    shallow drawer (taller than the drawer), and fruit piled in a basket.
    """
    seen, pending = {b.split("/")[0]}, [a]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if s.within(b, s.pose(name)[0], slack):
            return True
        lo, hi = s.aabb(name)
        middle = (lo + hi) / 2
        if s.within(b, middle, slack) or s.within(b, [middle[0], middle[1], lo[2] + FLOOR], slack):
            return True
        pending.extend(s.resting_on(name))
    return False


@relation
def overlapping(s, a, b, depth=0.0):
    """Whether two regions intersect by at least `depth` metres."""
    return s.overlap(a, b) <= -depth


@term
def offset(s, a, b, axis="z"):
    """Signed distance from b's origin to a's along an axis, in metres."""
    i = AXES[axis]
    return float(s.pose(a)[0][i] - s.pose(b)[0][i])


@term
def gap(s, a, b, plane="xyz"):
    """Distance between two origins in metres, over the axes in `plane`."""
    d = s.pose(a)[0] - s.pose(b)[0]
    return float(np.linalg.norm([d[AXES[c]] for c in plane]))


@term
def clearance(s, a, b):
    """Distance in metres from a's underside to b's surface beneath it."""
    lo, hi = s.aabb(a)
    return float(lo[2] - s.surface(b, (lo[:2] + hi[:2]) / 2))


@term
def depth(s, a, b):
    """How far a's origin is below the bottom of b, in metres."""
    return float(s.aabb(b)[0][2] - s.pose(a)[0][2])


@term
def tilt(s, a):
    """Angle between a's own up axis and the world's, in degrees."""
    w, x, y, z = s.pose(a)[1]
    up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return _angle(up, [0, 0, 1])


@term
def heading(s, a, b):
    """Angle between a's forward axis and the direction to b, in degrees."""
    w, x, y, z = s.pose(a)[1]
    forward = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    toward = s.pose(b)[0] - s.pose(a)[0]
    toward[2] = 0.0
    return 180.0 if float(np.linalg.norm(toward)) < 1e-6 else _angle(forward[:2], toward[:2])


@term
def turn(s, a, rotation, symmetry=0.0):
    """Orientation error in degrees, accounting for symmetry about the object's own z axis."""
    target = quat(rotation)
    steps = max(1, round(360.0 / symmetry)) if symmetry > 0 else 1
    q = s.pose(a)[1]
    return min(
        math.degrees(
            2 * math.acos(min(1.0, abs(float(np.dot(q, compose(target, spin(i * symmetry)))))))
        )
        for i in range(steps)
    )


@term
def twist(s, a, b, symmetry=0.0):
    """How far a is turned from b, in degrees, accounting for rotational symmetry."""
    qa, qb = s.pose(a)[1], s.pose(b)[1]
    steps = max(1, round(360.0 / symmetry)) if symmetry > 0 else 1
    return min(
        math.degrees(
            2 * math.acos(min(1.0, abs(float(np.dot(qa, compose(qb, spin(i * symmetry)))))))
        )
        for i in range(steps)
    )


@term
def value(s, joint):
    """A joint's current value, in its own units."""
    return s.joint(joint)


@term
def fraction(s, joint):
    """Where a joint sits in its travel: 0 at its lower limit, 1 at its upper."""
    lo, hi = s.joint_range(joint)
    if hi <= lo:
        raise ValueError(f"{joint!r} has no range; compare its value instead")
    return (s.joint(joint) - lo) / (hi - lo)


@term
def drift(s, a, b):
    """Distance from a's origin to where b's origin was at the start, in metres."""
    return float(np.linalg.norm(s.pose(a)[0] - s.start_pose(b)[0]))


@term
def spread(s, names, axis="y"):
    """The range of the objects' origins along an axis, in metres."""
    i = AXES[axis]
    values = [s.pose(n)[0][i] for n in names]
    return float(max(values) - min(values))


@term
def step(s, names, axis="y"):
    """The smallest signed gap between consecutive objects' origins along an axis."""
    i = AXES[axis]
    values = [s.pose(n)[0][i] for n in names]
    return float(min(b - a for a, b in itertools.pairwise(values))) if len(values) > 1 else 0.0


@term
def tally(s, facts):
    """How many of a list of conditions are true."""
    return float(sum(1 for f in facts if f))


def _angle(a, b) -> float:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


@term
def reading(s, capability, *args):
    """A finite numeric observation in the units declared by a vocabulary module."""
    if capability not in getattr(s, "capabilities", ()) or not hasattr(s, "observe"):
        raise UnsupportedError("evaluation", [capability])
    answer = s.observe(capability, *args)
    if (
        isinstance(answer, (bool, np.bool_))
        or not isinstance(answer, (int, float, np.number))
        or not np.isfinite(answer)
    ):
        raise ValueError(f"{capability}: expected a finite numeric observation")
    return float(answer)


@relation
def observed(s, capability, *args):
    """A boolean observation declared by a vocabulary module."""
    if capability not in getattr(s, "capabilities", ()) or not hasattr(s, "observe"):
        raise UnsupportedError("evaluation", [capability])
    answer = s.observe(capability, *args)
    if not isinstance(answer, (bool, np.bool_)):
        raise ValueError(f"{capability}: expected a boolean observation")
    return bool(answer)
