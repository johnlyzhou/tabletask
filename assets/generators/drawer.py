"""A drawer in a box frame, built from boxes.

Lengths are metres. The front faces -x, toward the robot; the origin is the middle of
the base. Joint: `drawer` (slides out, 0 closed). Regions: `inside` (inside the drawer),
`top`, `handle` and `bounds`.
"""

from furniture import (
    DEFAULTS,
    FINISH_PARAMS,
    FRONT_PARAMS,
    HANDLE_FINISH_PARAMS,
    HANDLE_PARAMS,
    STANDOFF,
    WALL,
    box,
    finishes,
    handle,
    lines,
    region,
)

PARAMS = {
    "width": [0.2, 0.5],  # side to side
    "depth": [0.2, 0.38],  # front to back
    "height": [0.08, 0.22],
    **HANDLE_PARAMS,
    "handle_y": [-0.7, 0.7],  # where the handle sits across the front, -1 to 1
    "handle_z": [-0.5, 0.5],  # and up it
    **FINISH_PARAMS,
    **FRONT_PARAMS,
    **HANDLE_FINISH_PARAMS,
}

GAP = 0.003  # clearance around the tray


def build(p: dict) -> str:
    w, d, h = p["width"] / 2, p["depth"] / 2, p["height"]
    opening = h - 2 * WALL  # between the bottom and top panels
    front = -d - WALL / 2  # x of the front, just proud of the frame
    tw, td, th = w - WALL - GAP, d - WALL - GAP, opening - 2 * GAP  # tray half sizes and height
    floor = WALL + GAP

    frame = [
        box("bottom", (0, 0, WALL / 2), (d, w, WALL / 2)),
        box("top", (0, 0, h - WALL / 2), (d, w, WALL / 2)),
        box("left", (0, w - WALL / 2, h / 2), (d, WALL / 2, h / 2)),
        box("right", (0, -w + WALL / 2, h / 2), (d, WALL / 2, h / 2)),
        box("back", (d - WALL / 2, 0, h / 2), (WALL / 2, w - WALL, h / 2 - WALL)),
        region("top", (0, 0, h + 0.01), (d, w, 0.01)),
        region("bounds", (-STANDOFF / 2, 0, h / 2), (d + WALL + STANDOFF / 2, w, h / 2)),
    ]
    half = (WALL / 2, w - GAP, opening / 2 + WALL / 2 - GAP)  # the drawer front
    inside = (td - WALL / 2, tw - WALL / 2, th / 2)
    tray = [
        box("front", (front, 0, h / 2), half, "front"),
        box("floor", (0, 0, floor + WALL / 4), (td, tw, WALL / 4)),
        box("tray_left", (0, tw - WALL / 4, floor + th / 2), (td, WALL / 4, th / 2)),
        box("tray_right", (0, -tw + WALL / 4, floor + th / 2), (td, WALL / 4, th / 2)),
        box("tray_back", (td - WALL / 4, 0, floor + th / 2), (WALL / 4, tw, th / 2)),
        region("inside", (0, 0, floor + th / 2 + WALL / 4), inside),
        *handle(p, front - WALL / 2, 0, h / 2, half[1], half[2]),
    ]
    materials, look, grip = finishes(p)
    return f"""<mujoco model="drawer">
  <compiler angle="radian"/>
  <custom>
    <text name="label" data="{look} drawer with {grip}"/>
  </custom>
  {DEFAULTS}
  <asset>
    {lines(materials, 4)}
  </asset>
  <worldbody>
    <body name="drawer_frame">
      {lines(frame, 6)}
      <body name="tray">
        <joint name="drawer" type="slide" axis="-1 0 0" range="0 {2 * td * 0.85:.4f}"
               damping="2" frictionloss="0.2" armature="0.01"/>
        {lines(tray, 8)}
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="drawer_frame" body2="tray"/>
  </contact>
</mujoco>
"""
