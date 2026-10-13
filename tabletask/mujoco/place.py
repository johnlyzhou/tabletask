"""Sampling where each object starts, so the start conditions hold."""

from __future__ import annotations

import mujoco
import numpy as np

from ..language.rotation import quat
from ..language.schema import Object, Task
from .model import STAGING_HEIGHT
from .runtime import Runtime

DROP = 0.005  # height above the support at which free objects are released, metres
GAP = 0.008  # clearance kept between footprints, metres
STEP = 0.005  # height step when searching a mesh region for a level that fits, metres
TRIES = 400  # random positions tried per search
YAW_SAMPLES = 6  # yaw samples per object when its yaw may vary
LEANS = (20.0, 35.0, 50.0, 65.0, 80.0)  # tilts tried for objects inside containers, degrees
LAYOUTS = 12  # full layouts tried before giving up
SIDES = ((90.0, 0.0), (-90.0, 0.0), (0.0, 90.0), (0.0, -90.0))  # (roll, pitch) for "on_side"
STAGING = np.array([0.0, 0.0, STAGING_HEIGHT])  # out-of-the-way spot for measuring objects

# (x, y, half width, half depth, yaw in radians) of an object's yawed bounding rectangle.
Footprint = tuple[float, float, float, float, float]


def place_all(runtime: Runtime, rng: np.random.Generator) -> None:
    """Set initial joint states, then place every object in dependency order.

    Raises:
        RuntimeError: If no layout fits after `LAYOUTS` attempts.
    """
    order = _order(runtime.task, _areas(runtime))
    for _ in range(LAYOUTS):
        try:
            _lay_out(runtime, rng, order)
            runtime.refresh()
            return
        except RuntimeError as error:
            failure = error
    raise RuntimeError(f"{failure} (gave up after {LAYOUTS} layouts)")


def _lay_out(runtime: Runtime, rng: np.random.Generator, order: list[str]) -> None:
    """Set joint states and place each object in `order` once."""
    _joint_states(runtime, rng)
    footprints: dict[str, Footprint] = {}
    for name in order:
        _place(runtime, name, runtime.task.scene.objects[name], rng, footprints)


def _joint_states(runtime: Runtime, rng) -> None:
    """Set each object's joints to their fixed, sampled, "open" or "closed" start values."""
    for name, obj in runtime.task.scene.objects.items():
        for joint, value in obj.joints.items():
            if isinstance(value, tuple):
                value = float(rng.uniform(*value))
            lo, hi = runtime.joint_range(f"{name}/{joint}")
            if isinstance(value, str):
                if value not in ("open", "closed"):
                    raise ValueError(f"unknown joint start {value!r}")
                value = hi if value == "open" else lo
            if not lo <= value <= hi:
                raise ValueError(f"{name}/{joint}: start {value:g} outside range [{lo:g}, {hi:g}]")
            runtime.set_joint(f"{name}/{joint}", value)


def _areas(runtime: Runtime) -> dict[str, float]:
    """Footprint area of each object in its sampled rotation."""
    areas = {}
    for name, obj in runtime.task.scene.objects.items():
        runtime.set_pose(name, STAGING, quat(obj.sample.rotation_degrees), full=False)
        areas[name] = runtime.area(name)
    return areas


def _order(task: Task, areas: dict[str, float]) -> list[str]:
    """Placement order: fixed or pinned objects, then jittered ones, then the rest.

    Within each group larger objects come first, and an object's host always
    precedes it.
    """
    done: list[str] = []

    def priority(name: str) -> tuple:
        obj = task.scene.objects[name]
        area = areas.get(name, 0.0)
        return (not (obj.fixed or obj.sample.pinned), not obj.sample.jittered, -area)

    def visit(name: str) -> None:
        if name in done:
            return
        host = task.scene.objects[name].sample.rel.split("/")[0]
        if host in task.scene.objects and host != name:
            visit(host)
        done.append(name)

    for name in sorted(task.scene.objects, key=priority):
        visit(name)
    return done


def _place(runtime: Runtime, name: str, obj: Object, rng, footprints: dict[str, Footprint]) -> None:
    """Place one object, trying upright, then tilted, then heaped orientations."""
    frame = obj.sample
    turns = YAW_SAMPLES if frame.yaw_jitter else 1
    roll, pitch = _tipped(frame, rng)
    inside = frame.rel.endswith("/inside")
    # (lean in degrees, heap) pairs; only objects inside containers may lean or heap.
    attempts = [(lean, False) for lean in ((0.0, *LEANS) if inside else (0.0,))]
    if inside:
        attempts.append((0.0, True))
    for lean, heap in attempts:
        for _ in range(turns):
            spread = frame.yaw_jitter
            yaw = frame.rotation_degrees[2] + (rng.uniform(-spread, spread) if spread else 0.0)
            try:
                _place_turned(runtime, name, obj, (roll, pitch, yaw), rng, footprints, lean, heap)
                return
            except RuntimeError as error:
                failure = error
    raise failure


def _tipped(frame, rng) -> tuple[float, float]:
    """Roll and pitch for an object: a random side if "on_side", else its rotation."""
    if frame.rotation == "on_side":
        return SIDES[int(rng.integers(len(SIDES)))]
    return frame.rotation_degrees[0], frame.rotation_degrees[1]


def _orient(runtime, name, rotation, lean):
    """Rotate an object and measure it.

    Returns:
        (orientation quaternion, yawed footprint box (half x, half y, yaw), half extents
        of the rotated bounding box, bounding-box centre relative to the body origin).
    """
    roll, pitch, yaw = rotation
    if lean:
        # Lean about the axis that tips the object along its longer side.
        runtime.set_pose(name, STAGING, quat((roll, pitch, 0.0)), full=False)
        lo, hi = runtime.aabb(name)
        if (hi - lo)[0] >= (hi - lo)[1]:
            pitch += lean
        else:
            roll += lean
    orientation = quat((roll, pitch, yaw))

    # Measure the footprint before yaw; the yaw is stored as the box's rotation instead.
    runtime.set_pose(name, STAGING, quat((roll, pitch, 0.0)), full=False)
    lo, hi = runtime.aabb(name)
    box = ((hi - lo)[0] / 2, (hi - lo)[1] / 2, np.radians(yaw))
    runtime.set_pose(name, STAGING, orientation, full=False)
    lo, hi = runtime.aabb(name)
    return orientation, box, (hi - lo) / 2, (lo + hi) / 2 - runtime.pose(name)[0]


def _supports(runtime, rel: str) -> set[str]:
    """Objects this placement rests on, directly or transitively.

    Supports may share a footprint with the object without counting as a collision.
    """
    beneath, host = set(), rel.split("/")[0]
    while host in runtime.task.scene.objects and host not in beneath:
        beneath.add(host)
        host = runtime.task.scene.objects[host].sample.rel.split("/")[0]
    return beneath


def _siblings(runtime, rel: str, footprints) -> list[str]:
    """Objects already placed in the same region, which a heap may land on."""
    return [
        other
        for other in footprints
        if other in runtime.task.scene.objects
        and runtime.task.scene.objects[other].sample.rel == rel
    ]


def _avoid(runtime, obj, footprints, beneath, siblings, half, heap):
    """Footprints this placement must keep clear of."""
    frame = obj.sample
    skip = beneath | set(siblings)
    others = [footprint for other, footprint in footprints.items() if other not in skip]
    if runtime.model.geom_type[runtime.region(frame.rel)] != mujoco.mjtGeom.mjGEOM_MESH and (
        not heap
    ):
        # On a flat region, only objects overlapping this height band can collide.
        floor = runtime.aabb(frame.rel)[0][2] + frame.offset[2]
        ceiling = floor + 2 * half[2] + DROP
        others = [
            footprints[other]
            for other in footprints
            if other not in beneath
            and runtime.aabb(other)[1][2] >= floor - GAP
            and runtime.aabb(other)[0][2] <= ceiling + GAP
        ]
    # Objects in different slots of one fixture may have overlapping bounding boxes, so
    # pinned objects ignore those; two objects in the same slot still exclude each other.
    if not obj.fixed and frame.pinned:
        host = frame.rel.split("/")[0]
        adjacent = {
            footprints[other]
            for other in footprints
            if runtime.task.scene.objects[other].sample.rel != frame.rel
            and runtime.task.scene.objects[other].sample.rel.split("/")[0] == host
        }
        others = [footprint for footprint in others if footprint not in adjacent]
    return others


def _window(name: str, frame, half, region_lo, region_hi):
    """Bounds for the object's centre, and a label describing the placement for errors."""
    shift = np.asarray(frame.offset)
    if not frame.jittered:
        return (
            region_lo + half + shift,
            region_hi - half + shift,
            f"{name!r} in {frame.rel!r}",
        )
    mid = (region_lo + region_hi) / 2 + shift
    reach = np.array([*frame.jitter.span, 0.0])
    lo, hi = mid - reach, mid + reach

    # Clip the jitter window to the region where the object fits; if they do not
    # overlap, use the point of the fitting range nearest the centre.
    room_lo, room_hi = region_lo + half, region_hi - half
    for k in range(2):
        if room_lo[k] <= room_hi[k]:
            lo[k], hi[k] = max(lo[k], room_lo[k]), min(hi[k], room_hi[k])
            if lo[k] > hi[k]:
                lo[k] = hi[k] = min(max(mid[k], room_lo[k]), room_hi[k])
    span = " by ".join(f"{value:g}" for value in frame.jitter.span)
    return lo, hi, f"{name!r} within {span} m of the centre of {frame.rel!r}"


def _ground(runtime, name, obj, half, box, others, siblings, rng, heap):
    """Position on a region where the object lands, and the height of its support."""
    frame = obj.sample
    region_lo, region_hi = runtime.aabb(frame.rel)
    lo, hi, label = _window(name, frame, half, region_lo, region_hi)
    if heap:
        # A heap drops onto the top of the region or of the highest sibling.
        mid = (region_lo + region_hi) / 2 + np.asarray(frame.offset)
        for k in range(2):
            if lo[k] > hi[k]:
                lo[k] = hi[k] = mid[k]
        x, y = _sample(lo, hi, box, others, rng, label)
        return x, y, max([region_hi[2]] + [runtime.aabb(other)[1][2] for other in siblings])
    if runtime.model.geom_type[runtime.region(frame.rel)] == mujoco.mjtGeom.mjGEOM_MESH:
        return _lowest(
            runtime,
            frame.rel,
            lo,
            hi,
            half,
            box,
            others,
            rng,
            label,
            across=frame.jittered or frame.rel.endswith("/top"),
        )
    x, y = _sample(lo, hi, box, others, rng, label)
    return x, y, region_lo[2]


def _place_turned(
    runtime: Runtime,
    name: str,
    obj: Object,
    rotation,
    rng,
    footprints: dict[str, Footprint],
    lean: float = 0.0,
    heap: bool = False,
) -> None:
    """Place an object at one orientation and record its footprint.

    Raises:
        RuntimeError: If there is no free spot for it.
    """
    frame = obj.sample
    orientation, box, half, centre = _orient(runtime, name, rotation, lean)
    if runtime.region(frame.rel) is None:
        # Relative to a frame rather than a region: place at its origin plus offset.
        position = runtime.pose(frame.rel)[0] + np.asarray(frame.offset)
        if frame.jittered:
            reach = np.array(frame.jitter.span)
            position[:2] += rng.uniform(-reach, reach)
        x, y = position[0] + centre[0], position[1] + centre[1]
    else:
        siblings = _siblings(runtime, frame.rel, footprints) if heap else []
        beneath = _supports(runtime, frame.rel)
        others = _avoid(runtime, obj, footprints, beneath, siblings, half, heap)
        x, y, floor = _ground(runtime, name, obj, half, box, others, siblings, rng, heap)
        z = floor + half[2] - centre[2] + (0.0 if obj.fixed else DROP) + frame.offset[2]
        position = np.array([x - centre[0], y - centre[1], z])
    footprints[name] = (float(x), float(y), box[0] + GAP, box[1] + GAP, box[2])
    runtime.set_pose(name, position, orientation, full=False)


def _lowest(
    runtime,
    region: str,
    lo,
    hi,
    half,
    box,
    footprints: list[Footprint],
    rng,
    label: str = "",
    across: bool = False,
) -> tuple[float, float, float]:
    """Lowest free spot in a mesh region, scanning upward in `STEP` increments.

    With `across`, an object that fits nowhere falls back to a clear spot at the
    region's surface instead of raising.

    Returns:
        (x, y, support height) in metres.
    """
    region_lo, region_hi = runtime.aabb(region)
    top = region.endswith("/top")
    surface = region_lo[2] if top else region_hi[2]

    def fits(x, y, z) -> bool:
        if top:
            return runtime.within(region, (x, y, z), -min(half[0], half[1]))
        return runtime.fits(region, (x, y, z), half, GAP)

    lo, hi = _collapse(lo, hi, label, strict=not across)

    # Quick check near the rim: fail early if no position is clear of other objects.
    rim = max(region_lo[2], region_hi[2] - STEP) + 1e-3
    if not across and not any(
        fits(x, y, rim) and _clear((x, y, *box), footprints)
        for x, y in zip(rng.uniform(lo[0], hi[0], TRIES), rng.uniform(lo[1], hi[1], TRIES))
    ):
        raise RuntimeError(f"nowhere to put {label}: no spot clears everything already placed")
    for floor in np.arange(region_lo[2], region_hi[2], STEP):
        for _ in range(TRIES):
            x, y = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
            if not fits(x, y, floor + 1e-3):
                continue
            if _clear((x, y, *box), footprints):
                return float(x), float(y), float(floor)
    if across:
        x, y = _sample(lo, hi, box, footprints, rng, label)
        return x, y, float(surface)
    raise RuntimeError(f"nowhere to put {label}: it does not fit in that region at any height")


def _collapse(lo, hi, label: str, strict: bool = True):
    """Copies of the bounds as float arrays, with inverted x or y bounds set to their midpoint.

    Raises:
        RuntimeError: If `strict` and an axis is inverted by more than `2 * GAP`, meaning the
            object is too wide for the region.
    """
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    if strict:
        for k in range(2):
            if lo[k] > hi[k] + 2 * GAP:
                raise RuntimeError(
                    f"nowhere to put {label}: it is {lo[k] - hi[k]:.2f} m too wide in "
                    f"{'xy'[k]} for that region"
                )
    for k in range(2):
        if lo[k] > hi[k]:
            lo[k] = hi[k] = (lo[k] + hi[k]) / 2
    return lo, hi


def _sample(lo, hi, box, footprints: list[Footprint], rng, label: str = "") -> tuple[float, float]:
    """Random (x, y) within the bounds whose footprint clears all others.

    Bounds inverted by up to `2 * GAP` collapse to their midpoint.
    """
    lo, hi = _collapse(lo, hi, label)
    for _ in range(TRIES):
        x, y = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
        if _clear((x, y, *box), footprints):
            return float(x), float(y)
    raise RuntimeError(f"nowhere to put {label}: no spot clears everything already placed")


def _clear(mine: Footprint, footprints: list[Footprint]) -> bool:
    """Whether a footprint overlaps none of the others (separating-axis test)."""
    if not footprints:
        return True
    x, y, hx, hy, yaw = mine
    others = np.asarray(footprints, dtype=float)
    dx, dy = others[:, 0] - x, others[:, 1] - y
    apart = np.zeros(len(others), dtype=bool)
    their_yaw = others[:, 4]
    # Candidate separating axes: the edge normals of both rectangles.
    for angles in (np.full(len(others), yaw), their_yaw):
        for turn in (0.0, np.pi / 2):
            ax, ay = np.cos(angles + turn), np.sin(angles + turn)

            my_extent = (
                abs(np.cos(yaw) * ax + np.sin(yaw) * ay) * hx
                + abs(-np.sin(yaw) * ax + np.cos(yaw) * ay) * hy
            )
            their_extent = (
                abs(np.cos(their_yaw) * ax + np.sin(their_yaw) * ay) * others[:, 2]
                + abs(-np.sin(their_yaw) * ax + np.cos(their_yaw) * ay) * others[:, 3]
            )
            apart |= abs(dx * ax + dy * ay) >= my_extent + their_extent
    return bool(apart.all())
