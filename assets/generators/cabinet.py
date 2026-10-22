"""A cabinet with a drawer above a hinged door, built from boxes.

Lengths are metres. The front faces -x, toward the robot; the origin is the middle of
the base. Joints: `drawer` (slides out, 0 closed) and `door` (swings out, 0 closed).
Regions: `drawer` (inside the drawer), `cupboard` (behind the door), `top`,
`drawer_handle`, `door_handle` and `bounds`.
"""

PARAMS = {
    "width": [0.26, 0.5],  # side to side
    "depth": [0.22, 0.38],  # front to back
    "height": [0.22, 0.45],
    "drawer_share": [0.28, 0.45],  # fraction of the inside height the drawer takes
    "handle": ["bar", "knob"],
    "knob": [0.014, 0.035],  # a knob is a cube this wide
    "bar": [0.06, 0.16],  # bar length, at most 70% of the front it sits on
    "hinge": ["left", "right"],
    "color": ["white", "oak", "walnut", "grey", "sage green", "navy blue", "black"],
    "handle_color": ["brass", "steel", "black"],
}

WALL = 0.012  # panel thickness
GAP = 0.003  # clearance around moving parts
STANDOFF = 0.022  # how far a bar handle stands off its front
BAR_RADIUS = 0.006
COLORS = {
    "white": "0.92 0.91 0.88 1",
    "oak": "0.72 0.55 0.36 1",
    "walnut": "0.40 0.27 0.17 1",
    "grey": "0.55 0.56 0.57 1",
    "sage green": "0.55 0.63 0.52 1",
    "navy blue": "0.18 0.24 0.40 1",
    "black": "0.10 0.10 0.11 1",
}
HANDLE_COLORS = {
    "brass": "0.78 0.62 0.30 1",
    "steel": "0.70 0.71 0.72 1",
    "black": "0.08 0.08 0.08 1",
}


def build(p: dict) -> str:
    w, d, h = p["width"] / 2, p["depth"] / 2, p["height"]
    inner = h - 3 * WALL  # bottom, divider and top panels
    split = WALL + inner * (1 - p["drawer_share"]) + WALL  # top of the divider
    drawer_h = h - WALL - split  # the drawer opening's height
    door_h = split - 2 * WALL
    front = -d - WALL / 2  # x of the fronts, just proud of the carcass
    face = front - WALL / 2  # where handles meet the fronts

    carcass = [
        box("bottom", (0, 0, WALL / 2), (d, w, WALL / 2)),
        box("top", (0, 0, h - WALL / 2), (d, w, WALL / 2)),
        box("divider", (0, 0, split - WALL / 2), (d, w - WALL, WALL / 2)),
        box("left", (0, w - WALL / 2, h / 2), (d, WALL / 2, h / 2)),
        box("right", (0, -w + WALL / 2, h / 2), (d, WALL / 2, h / 2)),
        box("back", (d - WALL / 2, 0, h / 2), (WALL / 2, w - WALL, h / 2 - WALL)),
        region("cupboard", (0, 0, WALL + door_h / 2), (d - WALL - GAP, w - WALL - GAP, door_h / 2)),
        region("top", (0, 0, h + 0.01), (d, w, 0.01)),
        region("bounds", (-STANDOFF / 2, 0, h / 2), (d + WALL + STANDOFF / 2, w, h / 2)),
    ]

    # the drawer: a front over the opening and an open tray behind it
    tw, td, th = w - WALL - GAP, d - WALL - GAP, drawer_h - 2 * GAP  # tray half sizes and height
    floor = split + GAP
    middle = split + drawer_h / 2
    drawer = [
        box("drawer_front", (front, 0, middle), (WALL / 2, w - GAP, drawer_h / 2 - GAP)),
        box("drawer_floor", (0, 0, floor + WALL / 4), (td, tw, WALL / 4)),
        box("drawer_left", (0, tw - WALL / 4, floor + th / 2), (td, WALL / 4, th / 2)),
        box("drawer_right", (0, -tw + WALL / 4, floor + th / 2), (td, WALL / 4, th / 2)),
        box("drawer_back", (td - WALL / 4, 0, floor + th / 2), (WALL / 4, tw, th / 2)),
        region("drawer", (0, 0, floor + th / 2 + WALL / 4), (td - WALL / 2, tw - WALL / 2, th / 2)),
        *handle(p, "drawer", (face, 0, middle), "y", w),
    ]

    # the door: hinged on one side edge, its handle near the other
    side = 1 if p["hinge"] == "left" else -1  # left is +y
    hinge = (front - WALL / 2, side * (w - GAP), WALL + door_h / 2)
    grip_y = -side * (w - GAP - max(0.035, p["knob"] / 2 + 0.02))
    door = [
        box("door_panel", (front, 0, WALL + door_h / 2), (WALL / 2, w - GAP, door_h / 2 - GAP)),
        *handle(p, "door", (face, grip_y, WALL + door_h * 0.6), "z", door_h / 2),
    ]

    travel = 2 * td * 0.85
    kind = "bar handles" if p["handle"] == "bar" else "knobs"
    return f"""<mujoco model="cabinet">
  <compiler angle="radian"/>
  <custom>
    <text name="label" data="{p["color"]} cabinet with {kind}"/>
  </custom>
  <default>
    <default class="part">
      <geom type="box" material="body" density="600" condim="4" friction="0.9 0.01 0.002"/>
    </default>
    <default class="handle">
      <geom material="handle" density="2000" condim="4" friction="1.2 0.01 0.002"/>
    </default>
    <default class="place">
      <geom type="box" contype="0" conaffinity="0" rgba="0 0 0 0" group="4" mass="0"/>
    </default>
  </default>
  <asset>
    <material name="body" rgba="{COLORS[p["color"]]}" specular="0.15" shininess="0.25"/>
    <material name="handle" rgba="{HANDLE_COLORS[p["handle_color"]]}" specular="0.6"/>
  </asset>
  <worldbody>
    <body name="cabinet">
      {_lines(carcass, 6)}
      <body name="drawer">
        <joint name="drawer" type="slide" axis="-1 0 0" range="0 {travel:.4f}"
               damping="2" frictionloss="0.2" armature="0.01"/>
        {_lines(drawer, 8)}
      </body>
      <body name="door" pos="{_v(hinge)}">
        <joint name="door" type="hinge" axis="0 0 {-side}" range="0 1.75"
               damping="0.5" frictionloss="0.05" armature="0.01"/>
        {_lines([shift(g, hinge) for g in door], 8)}
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="cabinet" body2="drawer"/>
    <exclude body1="cabinet" body2="door"/>
  </contact>
</mujoco>
"""


def handle(p, prefix, at, along, room):
    """A bar along `along` ("y" or "z") on two posts, or a cube knob on a stem; and its region.

    `at` is where it meets its front, `room` the half length the front leaves for a bar.
    """
    x, y, z = at
    if p["handle"] == "knob":
        k = p["knob"] / 2
        centre = (x - 0.01 - k, y, z)
        return [
            geom(
                f"{prefix}_stem",
                "handle",
                type="cylinder",
                pos=(x - 0.005, y, z),
                size=(k * 0.45, 0.005),
                euler=(0, 1.5708, 0),
            ),
            geom(f"{prefix}_knob", "handle", type="box", pos=centre, size=(k, k, k)),
            region(f"{prefix}_handle", centre, (k + 0.004,) * 3),
        ]
    half = min(p["bar"], 1.4 * room) / 2
    tip = x - STANDOFF
    ends = [(y - half, z), (y + half, z)] if along == "y" else [(y, z - half), (y, z + half)]
    span = (half + 0.006, 0.01) if along == "y" else (0.01, half + 0.006)
    return [
        geom(
            f"{prefix}_bar",
            "handle",
            type="capsule",
            size=(BAR_RADIUS,),
            fromto=(tip, *ends[0], tip, *ends[1]),
        ),
        *[
            geom(
                f"{prefix}_post{i}",
                "handle",
                type="capsule",
                size=(BAR_RADIUS * 0.8,),
                fromto=(x, u, v, tip, u, v),
            )
            for i, (u, v) in enumerate(ends)
        ],
        region(f"{prefix}_handle", (tip, y, z), (BAR_RADIUS + 0.004, *span)),
    ]


def box(name, pos, size):
    return geom(name, "part", pos=pos, size=size)


def region(name, pos, size):
    return geom(f"place:{name}", "place", pos=pos, size=size)


def geom(name, cls, **attrs):
    """A geom element; tuples become space-separated numbers."""
    text = " ".join(f'{k}="{_v(v) if isinstance(v, tuple) else v}"' for k, v in attrs.items())
    return f'<geom name="{name}" class="{cls}" {text}/>'


def shift(element, origin):
    """An element written in the cabinet's frame, moved into a child body's frame at `origin`."""
    import re

    def move(match):
        values = [float(v) for v in match.group(2).split()]
        return f'{match.group(1)}="{_v([v - origin[i % 3] for i, v in enumerate(values)])}"'

    return re.sub(r'(pos|fromto)="([^"]+)"', move, element)


def _v(values):
    return " ".join(f"{v:.4f}" for v in values)


def _lines(elements, spaces):
    return ("\n" + " " * spaces).join(elements)
