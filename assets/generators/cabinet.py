"""A cabinet with one hinged door, built from boxes.

Lengths are metres. The door faces -x, toward the robot; the origin is the middle of
the base. Joint: `door` (swings out, 0 closed), hinged on the left (+y) or right side.
Regions: `inside` (behind the door, under the shelf if there is one), `shelf`, `top`,
`handle` and `bounds`.
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
    "width": [0.22, 0.45],  # side to side
    "depth": [0.2, 0.36],  # front to back
    "height": [0.22, 0.45],
    "hinge": ["left", "right"],
    "shelf": [None, None, 0.5],  # a shelf at this fraction of the inside height, or none
    **HANDLE_PARAMS,
    "handle_y": [-1.0, 1.0],  # where the handle sits in the strip by the free edge
    "handle_z": [-0.6, 0.6],  # and up the door
    **FINISH_PARAMS,
    **FRONT_PARAMS,
    **HANDLE_FINISH_PARAMS,
}

GAP = 0.003  # clearance around the door


def build(p: dict) -> str:
    w, d, h = p["width"] / 2, p["depth"] / 2, p["height"]
    inner_w, inner_d = w - WALL - GAP, d - WALL - GAP  # inside half width and depth
    frame = [
        box("bottom", (0, 0, WALL / 2), (d, w, WALL / 2)),
        box("top", (0, 0, h - WALL / 2), (d, w, WALL / 2)),
        box("left", (0, w - WALL / 2, h / 2), (d, WALL / 2, h / 2)),
        box("right", (0, -w + WALL / 2, h / 2), (d, WALL / 2, h / 2)),
        box("back", (d - WALL / 2, 0, h / 2), (WALL / 2, w - WALL, h / 2 - WALL)),
        region("top", (0, 0, h + 0.01), (d, w, 0.01)),
        region("bounds", (-STANDOFF / 2, 0, h / 2), (d + WALL + STANDOFF / 2, w, h / 2)),
    ]
    floor, ceiling = WALL, h - WALL
    if p["shelf"] is not None:
        level = floor + (ceiling - floor) * p["shelf"]
        frame.append(box("shelf", (0, 0, level), (d - WALL, w - WALL, WALL / 2)))
        frame.append(
            region(
                "shelf",
                (0, 0, (level + ceiling) / 2),
                (inner_d, inner_w, (ceiling - level - WALL) / 2),
            )
        )
        ceiling = level - WALL / 2
    frame.append(
        region("inside", (0, 0, (floor + ceiling) / 2), (inner_d, inner_w, (ceiling - floor) / 2))
    )

    # the door hangs on its hinge edge; its geoms are written in the hinge's frame
    side = 1 if p["hinge"] == "left" else -1  # left is +y
    span = w - GAP  # half the door's width
    free = -side * 2 * span  # the free edge, across the door from the hinge
    door = [
        box("door", (WALL / 2, -side * span, h / 2), (WALL / 2, span, h / 2 - GAP), "front"),
        *handle(p, 0, free * 0.86, h / 2, span * 0.14, h / 2 - GAP, along="z"),
    ]
    materials, look, grip = finishes(p)
    return f"""<mujoco model="cabinet">
  <compiler angle="radian"/>
  <custom>
    <text name="label" data="{look} cabinet with {grip}"/>
  </custom>
  {DEFAULTS}
  <asset>
    {lines(materials, 4)}
  </asset>
  <worldbody>
    <body name="cabinet">
      {lines(frame, 6)}
      <body name="door" pos="{-d - WALL:.4f} {side * span:.4f} 0">
        <joint name="door" type="hinge" axis="0 0 {-side}" range="0 1.75"
               damping="0.5" frictionloss="0.05" armature="0.01"/>
        {lines(door, 8)}
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="cabinet" body2="door"/>
  </contact>
</mujoco>
"""
