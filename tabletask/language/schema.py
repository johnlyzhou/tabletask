"""The compiled form of a task, as the sampler and the monitor read it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import msgspec

Vec3 = tuple[float, float, float]

# Rotations a start can name, as roll, pitch, yaw in degrees.
ROTATIONS = {
    "upright": (0.0, 0.0, 0.0),
    "upside_down": (180.0, 0.0, 0.0),
    "on_side": (90.0, 0.0, 0.0),
}
DEFAULT_YAW_JITTER = 180.0  # degrees either way: any yaw


class Jitter(msgspec.Struct, omit_defaults=True):
    """How far a placement may vary: `xy` in metres, `yaw` in degrees.

    `xy` is the half-width of a square, or `[dx, dy]` for a rectangle.
    """

    xy: float | tuple[float, float] | None = None
    yaw: float | None = None

    @property
    def span(self) -> tuple[float, float]:
        """The half-widths in x and y."""
        return (self.xy, self.xy) if isinstance(self.xy, (int, float)) else tuple(self.xy)


class Frame(msgspec.Struct, omit_defaults=True):
    """Where to place an object: a named frame, an offset in metres, and a rotation."""

    rel: str
    offset: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 | str | None = None
    jitter: Jitter | None = None

    @property
    def rotation_degrees(self) -> Vec3:
        """The rotation in degrees; upright when none is given."""
        if isinstance(self.rotation, str):
            return ROTATIONS[self.rotation]
        return self.rotation if self.rotation is not None else (0.0, 0.0, 0.0)

    @property
    def yaw_jitter(self) -> float:
        """Degrees either way the yaw may vary."""
        if self.jitter is None or self.jitter.yaw is None:
            return DEFAULT_YAW_JITTER
        return self.jitter.yaw

    @property
    def jittered(self) -> bool:
        """Whether the position varies around the centre of `rel` (an `xy` jitter is given)."""
        return self.jitter is not None and self.jitter.xy is not None

    @property
    def pinned(self) -> bool:
        """Whether the position is fixed at the centre of `rel` (an `xy` jitter of zero)."""
        return self.jittered and not any(self.jitter.span)


class Grid(msgspec.Struct, omit_defaults=True):
    """A lattice of named cells inside a region; `pitch` is in metres."""

    columns: str | list[str]
    rows: str | list[str]
    pitch: tuple[float, float]
    plane: str = "xy"

    def cells(self) -> list[tuple[str, Vec3]]:
        """Each cell's name and its offset from the region's centre."""
        columns, rows = list(self.columns), list(self.rows)
        first, second = "xyz".index(self.plane[0]), "xyz".index(self.plane[1])
        cells = []
        for i, column in enumerate(columns):
            for j, row in enumerate(rows):
                offset = [0.0, 0.0, 0.0]
                offset[first] = (i - (len(columns) - 1) / 2) * self.pitch[0]
                offset[second] = (j - (len(rows) - 1) / 2) * self.pitch[1]
                cells.append((f"{column}{row}", tuple(offset)))
        return cells


class Appearance(msgspec.Struct, omit_defaults=True):
    """The table and sky textures, and whether the floor is shown."""

    vary: bool = True
    surface: str | None = None
    sky: str | None = None
    floor: bool = False
    repeat: tuple[float, float] = (2.0, 2.0)
    table_height: float = 0.1  # metres: each episode raises or lowers the table by up to this much


class Region(msgspec.Struct, omit_defaults=True):
    """A non-colliding placement volume; `size` is the full extent in metres."""

    shape: Literal["box", "cylinder", "sphere"]
    size: list[float]
    pos: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    grid: Grid | None = None


class Object(msgspec.Struct, omit_defaults=True):
    """An object in the scene: its asset, where it is placed, and how it is grouped."""

    sample: Frame
    asset: str | None = None
    joints: dict[str, float | tuple[float, float] | str] = {}  # start values by joint name
    regions: dict[str, Region] = {}
    tags: list[str] = []
    properties: dict[str, Any] = {}
    fixed: bool = False
    count: int | tuple[int, int] = 1
    scale: float | list[float | list[float]] = 1.0  # see `mujoco.model.draw_scale`


class Stage(msgspec.Struct, omit_defaults=True):
    """One step of the goal: a condition, the stages before it, and how long it must hold."""

    condition: dict[str, Any]
    after: list[str] = []
    hold: float = 0.5
    maintain: bool = False
    held: bool = False  # counts while the robot still holds the object (a lift, a carry)


class Goal(msgspec.Struct, omit_defaults=True):
    """The stages to complete, and the conditions that must always hold."""

    stages: dict[str, Stage]
    always: dict[str, dict[str, Any]] = {}

    def order(self) -> list[str]:
        """Stage names in an order that respects every `after`."""
        ordered: list[str] = []

        def visit(name: str, trail: tuple = ()) -> None:
            if name in ordered:
                return
            if name in trail:
                raise ValueError(f"stage ordering cycle through {name!r}")
            for prerequisite in self.stages[name].after:
                visit(prerequisite, (*trail, name))
            ordered.append(name)

        for name in self.stages:
            visit(name)
        return ordered


class Scene(msgspec.Struct, omit_defaults=True):
    """The scene file, the objects in it, and the condition every start must satisfy."""

    file: str  # the scene's MJCF file
    library: str | None = None
    objects: dict[str, Object] = {}
    start: dict[str, Any] | None = None
    appearance: Appearance | None = None


class Draw(msgspec.Struct, omit_defaults=True):
    """A variable drawn per episode; variables sharing `apart` take distinct values."""

    one_of: list[str]
    apart: str | None = None


class Task(msgspec.Struct, omit_defaults=True):
    """A compiled task, possibly still holding variables and counts to draw."""

    id: str
    instruction: str
    scene: Scene
    goal: Goal
    variables: dict[str, Draw] = {}
    authored: dict[str, Any] = msgspec.field(default_factory=dict, name="__document__")
    observations: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, Any]] = {}
    time_limit: float | None = None
    source: Any = msgspec.field(default=None, name="__source__")
    milestones: dict[str, Any] = msgspec.field(default_factory=dict, name="__milestones__")

    @property
    def root(self) -> Path:
        """The directory relative paths in this task resolve against."""
        return self.source.parent if self.source else Path.cwd()
