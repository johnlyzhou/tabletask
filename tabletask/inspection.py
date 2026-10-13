"""Text reports, pictures and manual object placement for inspecting tasks and assets."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .language.errors import SamplingError
from .language.evaluation import evaluate, evaluate_term, unpack_expression
from .language.kernel import ARITHMETIC, RELATIONS, TERMS
from .language.rotation import quat
from .language.schema import Task
from .language.vocabulary import TIME_WORDS
from .sampling import build

SETTLE = 3.0  # default physics time after moving objects in `try_out` (seconds)
# Side views of a single object: (label, azimuth, elevation), in degrees.
VIEWS = (
    ("front", 0.0, -25.0),
    ("left", -90.0, -25.0),
    ("back", 180.0, -25.0),
    ("right", 90.0, -25.0),
)
COMPARISONS = {"at_most": "<=", "at_least": ">="}  # comparison relations and their symbols


def format_expression(node) -> str:
    """A kernel expression in infix/call notation, e.g. `gap(a, b)` or `(x + y)`."""
    if isinstance(node, list):
        return "[" + ", ".join(format_expression(v) for v in node) + "]"
    if isinstance(node, dict):
        name, args = unpack_expression(node)
        if name in ARITHMETIC:
            return f"({format_expression(args[0])} {name} {format_expression(args[1])})"
        return f"{name}({', '.join(format_expression(a) for a in args)})"
    if isinstance(node, float):
        return f"{node:g}"
    return str(node)


def number(v) -> str:
    """A measured value formatted for a report: 3 decimals for non-integral floats."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return f"{v:.3f}" if isinstance(v, float) and not float(v).is_integer() else f"{v:g}"
    return format_expression(v)


def explain(tree, state, depth: int = 0) -> tuple[bool, list[str]]:
    """Evaluate a compiled condition and describe each measured value.

    Args:
        tree: A compiled condition expression.
        state: The runtime to evaluate against.
        depth: Indentation level of the returned lines.

    Returns:
        Whether the condition holds, and one indented line per sub-condition.
    """
    pad = "  " * depth
    name, args = unpack_expression(tree)
    if name in ("and", "or"):
        results = [explain(t, state, depth + 1) for t in args]
        held = all(r[0] for r in results) if name == "and" else any(r[0] for r in results)
        head = ("all of" if name == "and" else "any of") + f" ({len(args)}):"
        return held, [pad + head] + [line for _, lines in results for line in lines]
    if name == "not":
        held, lines = explain(args[0], state, depth + 1)
        return not held, [pad + "not:", *lines]
    if name in TIME_WORDS:
        held, lines = explain(args[0], state, depth + 1)
        if name == "times":
            note = f"at least {args[1]} times"
        elif name == "once":
            note = "once, ever"
        else:
            note = "at the start"
            held = evaluate(tree, state, {})  # judged at the start, which the state rewinds to
        return held, [pad + f"{note}; right now:", *lines]
    if name in COMPARISONS:
        quantity, bound = evaluate_term(args[0], state), evaluate_term(args[1], state)
        held = RELATIONS[name](state, quantity, bound)
        measured = f"{format_expression(args[0])} = {number(quantity)}"
        return held, [pad + f"{measured}  {COMPARISONS[name]} {number(bound)}  {verdict(held)}"]
    if name == "between":
        quantity, lo, hi = (evaluate_term(a, state) for a in args)
        held = lo <= quantity <= hi
        measured = f"{format_expression(args[0])} = {number(quantity)}"
        return held, [pad + f"{number(lo)} <= {measured} <= {number(hi)}  {verdict(held)}"]
    held = bool(evaluate(tree, state, {}))  # a fresh history, like a first check
    return held, [pad + f"{format_expression(tree)}  {verdict(held)}"]


def verdict(held: bool) -> str:
    """`true`, or `FALSE` in capitals so failures stand out."""
    return "true" if held else "FALSE"


def conditions(runtime) -> list[tuple[str, str, bool, list[str]]]:
    """(section, label, held, detail lines) for every start, goal and always condition.

    Start conditions are labelled as authored when they map one-to-one onto the
    compiled ones, and by their compiled form otherwise; `fixed` entries are skipped.
    """
    task = runtime.task
    result = []
    authored = task.authored or {}
    start = [
        condition
        for condition in authored.get("start", [])
        if not (isinstance(condition, list) and condition and condition[0] == "fixed")
    ]
    compiled_start = task.scene.start
    if not compiled_start:
        parts = []
    elif "and" in compiled_start:
        parts = compiled_start["and"]
    else:
        parts = [compiled_start]
    labels = (
        [format_authored(condition) for condition in start]
        if len(start) == len(parts)
        else [format_expression(part) for part in parts]
    )
    for label, tree in zip(labels, parts):
        held, lines = explain(tree, runtime, 1)
        result.append(("start", label, held, lines))
    for name, stage in task.goal.stages.items():
        held, lines = explain(stage.condition, runtime, 1)
        after = f" (after {', '.join(stage.after)})" if stage.after else ""
        hold = f", hold {stage.hold:g} s" if stage.hold else ""
        result.append(("goal", f"{name}{after}{hold}", held, lines))
    for name, tree in task.goal.always.items():
        held, lines = explain(tree, runtime, 1)
        result.append(("always", name, held, lines))
    return result


def format_authored(condition) -> str:
    """An authored condition as space-separated words."""
    if isinstance(condition, list):
        return " ".join(format_authored(v) if isinstance(v, list) else str(v) for v in condition)
    return str(condition)


def objects(runtime) -> list[str]:
    """One line per object (origin, tilt, size, containing regions) and per object joint."""
    lines = []
    places = runtime.places()
    for name, obj in runtime.task.scene.objects.items():
        origin, _ = runtime.pose(name)
        tilt = TERMS["tilt"](runtime, name)
        span = runtime.extent(name)
        inside = [
            place
            for place in places
            if place.split("/")[0] != name and runtime.within(place, origin, 0.01)
        ]
        asset_name = Path(obj.asset).stem
        lines.append(
            f"{name:<16} {asset_name:<20} at {origin[0]:+.3f} {origin[1]:+.3f} {origin[2]:.3f}"
            f"  tilt {tilt:4.1f} deg  size {span[0]:.3f}x{span[1]:.3f}x{span[2]:.3f}"
            f"  {'in ' + ', '.join(inside) if inside else 'in no region'}"
            + ("  fixed" if obj.fixed else "")
        )
    for name in [joint for joint in runtime.named_joints() if "/" in joint]:
        lo, hi = runtime.joint_range(name)
        lines.append(f"{name:<16} joint at {runtime.joint(name):.3f} of {lo:g}..{hi:g}")
    return lines


def report(runtime) -> list[str]:
    """A text report: object lines, then every condition with its measured values."""
    lines = objects(runtime)
    for section, label, held, detail in conditions(runtime):
        lines.append(f"{section:<7} {label}  {verdict(held)}")
        lines.extend(detail)
    return lines


def put(runtime, name: str, target: str, offset=(0.0, 0.0, 0.0)) -> None:
    """Place an object in a region, keeping its orientation.

    The object's bounding box is centred over the region (plus `offset`, in metres)
    and its bottom is set `DROP` above the region's surface. A bare object name
    such as `bowl` resolves to `bowl/inside`, or else `bowl/top`.

    Raises:
        KeyError: If the region does not exist.
    """
    from .mujoco.place import DROP  # the simulator is imported only when used

    if runtime.region(target) is None and "/" not in target:
        inside = f"{target}/inside"
        target = inside if runtime.region(inside) is not None else f"{target}/top"
    if runtime.region(target) is None:
        raise KeyError(f"no region {target!r}; regions here: {', '.join(runtime.places())}")
    lo, hi = runtime.aabb(target)
    centre = (lo + hi) / 2 + np.asarray(offset, dtype=float)
    floor = runtime.surface(target, centre[:2])
    origin, orientation = runtime.pose(name)
    body_lo, body_hi = runtime.aabb(name)
    shift = np.array(
        [
            centre[0] - (body_lo[0] + body_hi[0]) / 2,
            centre[1] - (body_lo[1] + body_hi[1]) / 2,
            floor + DROP - body_lo[2] + offset[2],
        ]
    )
    runtime.set_pose(name, origin + shift, orientation)


def turn(runtime, name: str, rotation) -> None:
    """Set an object's orientation to roll, pitch, yaw (degrees), keeping its lowest point."""
    origin, _ = runtime.pose(name)
    base = runtime.aabb(name)[0][2]
    runtime.set_pose(name, origin, quat(rotation))
    lift = base - runtime.aabb(name)[0][2]
    runtime.set_pose(name, np.add(origin, [0, 0, lift]), quat(rotation))


def try_out(task: Task, seed: int = 0, puts=(), turns=(), joints=(), settle: float = SETTLE):
    """Sample a start, apply the requested moves, and run physics; returns the runtime.

    Args:
        task: The task to sample.
        seed: Seed of the start.
        puts: (name, region, offset) tuples passed to `put`.
        turns: (name, rotation) tuples passed to `turn`.
        joints: (joint name, position) tuples.
        settle: Maximum physics time after the moves, in seconds.

    Returns:
        The runtime. If sampling fails, the last rejected draw is returned with the
        error message in `runtime.sampling_error`. When moves were made and `settle > 0`,
        `runtime.rested` says whether the scene came to rest.
    """
    try:
        runtime = build(task, seed)
    except SamplingError as error:  # inspect the last rejected draw instead
        runtime = error.runtime
        runtime.sampling_error = str(error)
    for name, target, offset in puts:
        put(runtime, name, target, offset)
    for name, rotation in turns:
        turn(runtime, name, rotation)
    for name, position in joints:
        runtime.set_joint(name, position)
    if settle > 0 and (puts or turns or joints):
        runtime.rested = runtime.run_until_still(settle)
    return runtime


def collision(runtime, out: Path, focus: str | None = None, columns: int = 3):
    """Write a picture of the collision geoms (with convex hulls drawn) to `out`.

    With `focus`, that object is shown from the four `VIEWS`, from above, and as its
    visual; otherwise every object in the task is shown as its visual next to its
    collision geoms.
    """
    if focus:
        views = [
            (label, {"azimuth": azimuth, "elevation": elevation})
            for label, azimuth, elevation in VIEWS
        ]
        views += [("above", {"azimuth": 0.0, "elevation": -89.9}), ("visual", {})]
        frames = [
            runtime.picture(
                width=480,
                height=360,
                camera=None,
                focus=[focus],
                collision=label != "visual",
                **how,
            )
            for label, how in views
        ]
        grid(frames, [label for label, _ in views], columns=columns).save(out)
        return out
    frames, labels = [], []
    for name in runtime.task.scene.objects:
        for label, how in (("visual", {}), ("collision", {"collision": True})):
            frames.append(runtime.picture(width=480, height=360, camera=None, focus=[name], **how))
            labels.append(f"{name} {label}")
    grid(frames, labels, columns=columns).save(out)
    return out


def sheet(runtime, out: Path, focus: str | None = None, regions=None):
    """Write a contact sheet of an object or a task start to `out`.

    With `focus`, the object is shown from the four `VIEWS`, from above, and from a
    corner with its regions painted. Otherwise the task is shown from its camera, then
    closer and from above with the objects' regions painted.

    Args:
        runtime: The runtime to render.
        out: Image path.
        focus: Name of a single object to show.
        regions: Region names to paint, or `"each"` for one panel per region; None
            paints the placement regions.
    """
    if focus:
        views = [
            (label, {"azimuth": azimuth, "elevation": elevation})
            for label, azimuth, elevation in VIEWS
        ]
        views += [("above", {"azimuth": 0.0, "elevation": -89.9})]
        own = _chosen(runtime, regions, lambda p: p.split("/")[0] == focus)
        corner = {"azimuth": 135.0, "elevation": -35.0}
        if regions == "each":
            views += [(p, dict(corner, regions=[p])) for p in own]
        else:
            views += [("regions", dict(corner, regions=own))]
        frames = [
            runtime.picture(width=480, height=360, camera=None, focus=focus, **how)
            for _, how in views
        ]
    else:
        names = list(runtime.task.scene.objects)
        own = _chosen(runtime, regions, lambda p: p.split("/")[0] in runtime.task.scene.objects)
        closer = {"camera": None, "focus": names}
        above = dict(closer, azimuth=0.0, elevation=-89.9)
        if regions == "each":
            views = [("camera", {})] + [(p, dict(closer, regions=[p])) for p in own]
        else:
            views = [
                ("camera", {}),
                ("closer", dict(closer, regions=own)),
                ("above", dict(above, regions=own)),
            ]
        frames = [runtime.picture(width=640, height=360, **how) for _, how in views]
    grid(frames, [label for label, _ in views], columns=3).save(out)
    return out


def _chosen(runtime, regions, belongs):
    """The regions to paint: placement regions, all of them, or those named.

    `places` omits each object's `bounds` region, which is its extent rather than a
    placement region; it is included when regions are named or `"each"` is asked for.
    """
    own = [p for p in runtime.places() if belongs(p)]
    if regions is None:
        return own
    every = own + [
        f"{name}/bounds"
        for name in runtime.task.scene.objects
        if belongs(f"{name}/bounds") and runtime.region(f"{name}/bounds") is not None
    ]
    if regions == "each":
        return every
    # Named regions match on any object, including the table.
    wanted = set(regions)
    return [
        p
        for p in dict.fromkeys(every + runtime.places())
        if p in wanted or p.split("/", 1)[1] in wanted
    ]


def grid(frames: list[np.ndarray], labels: list[str], columns: int | None = None) -> Image.Image:
    """Equal-sized frames tiled into one labelled image; columns default to a square grid."""
    columns = columns or math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / columns)
    h, w = frames[0].shape[:2]
    canvas = Image.new("RGB", (columns * w, rows * h), "black")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(frames, labels)):
        x, y = (i % columns) * w, (i // columns) * h
        canvas.paste(Image.fromarray(image), (x, y))
        draw.rectangle([x + 4, y + 4, x + 12 + draw.textlength(label), y + 20], fill="black")
        draw.text((x + 8, y + 6), label, fill="white")
    return canvas


@contextmanager
def _engine(renderer):
    """Yield one shared path-tracing session for `"rtx"`, or None for rasterising."""
    if renderer is None:
        yield None
        return
    if renderer != "rtx":
        raise ValueError(f"unknown renderer {renderer!r}; use None or 'rtx'")
    from .mujoco.rtx import Session

    with Session() as session:
        yield session


def preview(task: Task, samples=10, out=None, columns=None, renderer=None, **picture_kwargs):
    """A labelled grid of starts sampled with seeds 0 .. samples - 1.

    Args:
        task: The task to sample.
        samples: Number of starts.
        out: Image path to save to, if given.
        columns: Grid columns; defaults to a square grid.
        renderer: None to rasterise, or `"rtx"` to path-trace.
        **picture_kwargs: Passed to `runtime.picture` (e.g. `width`, `height`).

    Returns:
        The grid as a PIL image.

    Raises:
        ValueError: If `samples < 1` or `renderer` is unknown.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    frames = []
    with _engine(renderer) as session:
        for episode in range(samples):
            runtime = build(task, episode)
            try:
                if session is not None:
                    runtime.use_rtx(
                        picture_kwargs.get("width", 960),
                        picture_kwargs.get("height", 540),
                        session=session,
                    )
                frames.append(runtime.picture(**picture_kwargs))
            finally:
                runtime.close_renderer()
    image = grid(frames, [f"episode {episode}" for episode in range(samples)], columns)
    if out:
        image.save(out)
    return image
