"""A drawer in a box frame, built from boxes.

Lengths are metres. The front faces -x, toward the robot; the origin is the middle of
the base. Joint: `drawer` (slides out, 0 closed). Regions: `inside` (inside the drawer),
`top`, `handle` and `bounds`.
"""

PARAMS = {
    "width": [0.2, 0.5],  # side to side
    "depth": [0.2, 0.38],  # front to back
    "height": [0.08, 0.22],
    "handle": ["bar", "knob"],
    "knob": [0.014, 0.035],  # a knob is a cube this wide
    "bar": [0.06, 0.16],  # bar length, at most 70% of the front
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
    opening = h - 2 * WALL  # between the bottom and top panels
    front = -d - WALL / 2  # x of the front, just proud of the frame
    middle = h / 2

    frame = [
        box("bottom", (0, 0, WALL / 2), (d, w, WALL / 2)),
        box("top", (0, 0, h - WALL / 2), (d, w, WALL / 2)),
        box("left", (0, w - WALL / 2, h / 2), (d, WALL / 2, h / 2)),
        box("right", (0, -w + WALL / 2, h / 2), (d, WALL / 2, h / 2)),
        box("back", (d - WALL / 2, 0, h / 2), (WALL / 2, w - WALL, h / 2 - WALL)),
        region("top", (0, 0, h + 0.01), (d, w, 0.01)),
        region("bounds", (-STANDOFF / 2, 0, h / 2), (d + WALL + STANDOFF / 2, w, h / 2)),
    ]

    # the drawer: a front over the opening and an open tray behind it
    tw, td, th = w - WALL - GAP, d - WALL - GAP, opening - 2 * GAP  # tray half sizes and height
    floor = WALL + GAP
    drawer = [
        box("front", (front, 0, middle), (WALL / 2, w - GAP, opening / 2 + WALL / 2 - GAP)),
        box("floor", (0, 0, floor + WALL / 4), (td, tw, WALL / 4)),
        box("tray_left", (0, tw - WALL / 4, floor + th / 2), (td, WALL / 4, th / 2)),
        box("tray_right", (0, -tw + WALL / 4, floor + th / 2), (td, WALL / 4, th / 2)),
        box("tray_back", (td - WALL / 4, 0, floor + th / 2), (WALL / 4, tw, th / 2)),
        region("inside", (0, 0, floor + th / 2 + WALL / 4), (td - WALL / 2, tw - WALL / 2, th / 2)),
        *handle(p, (front - WALL / 2, 0, middle), w),
    ]

    travel = 2 * td * 0.85
    kind = "a bar handle" if p["handle"] == "bar" else "a knob"
    return f"""<mujoco model="drawer">
  <compiler angle="radian"/>
  <custom>
    <text name="label" data="{p["color"]} drawer with {kind}"/>
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
    <body name="drawer_frame">
      {_lines(frame, 6)}
      <body name="tray">
        <joint name="drawer" type="slide" axis="-1 0 0" range="0 {travel:.4f}"
               damping="2" frictionloss="0.2" armature="0.01"/>
        {_lines(drawer, 8)}
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="drawer_frame" body2="tray"/>
  </contact>
</mujoco>
"""


def handle(p, at, room):
    """A horizontal bar on two posts, or a cube knob on a stem; and its region.

    `at` is where it meets the front, `room` the half width the front leaves for a bar.
    """
    x, y, z = at
    if p["handle"] == "knob":
        k = p["knob"] / 2
        centre = (x - 0.01 - k, y, z)
        stem = {"type": "cylinder", "size": (k * 0.45, 0.005), "euler": (0, 1.5708, 0)}
        return [
            geom("stem", "handle", pos=(x - 0.005, y, z), **stem),
            geom("knob", "handle", type="box", pos=centre, size=(k, k, k)),
            region("handle", centre, (k + 0.004,) * 3),
        ]
    half = min(p["bar"], 1.4 * room) / 2
    tip = x - STANDOFF
    bar = {"type": "capsule", "size": (BAR_RADIUS,)}
    post = {"type": "capsule", "size": (BAR_RADIUS * 0.8,)}
    return [
        geom("bar", "handle", fromto=(tip, y - half, z, tip, y + half, z), **bar),
        geom("post0", "handle", fromto=(x, y - half, z, tip, y - half, z), **post),
        geom("post1", "handle", fromto=(x, y + half, z, tip, y + half, z), **post),
        region("handle", (tip, y, z), (BAR_RADIUS + 0.004, half + 0.006, 0.01)),
    ]


def box(name, pos, size):
    return geom(name, "part", pos=pos, size=size)


def region(name, pos, size):
    return geom(f"place:{name}", "place", pos=pos, size=size)


def geom(name, cls, **attrs):
    """A geom element; tuples become space-separated numbers."""
    text = " ".join(f'{k}="{_v(v) if isinstance(v, tuple) else v}"' for k, v in attrs.items())
    return f'<geom name="{name}" class="{cls}" {text}/>'


def _v(values):
    return " ".join(f"{v:.4f}" for v in values)


def _lines(elements, spaces):
    return ("\n" + " " * spaces).join(elements)
