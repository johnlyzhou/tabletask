"""A drawer in a box frame, built from boxes.

Lengths are metres. The front faces -x, toward the robot; the origin is the middle of
the base. Joint: `drawer` (slides out, 0 closed). Regions: `inside` (inside the drawer),
`top`, `handle` and `bounds`.
"""

import colorsys
from pathlib import Path

PARAMS = {
    "width": [0.2, 0.5],  # side to side
    "depth": [0.2, 0.38],  # front to back
    "height": [0.08, 0.22],
    "handle": ["bar", "square bar", "cylinder knob", "cube knob", "round knob"],
    "knob": [0.014, 0.04],  # a knob's width
    "knob_length": [0.5, 1.4],  # a cylinder knob's length, as a fraction of its width
    "bar": [0.06, 0.2],  # bar length, at most 80% of the front
    "bar_thickness": [0.008, 0.016],
    "handle_y": [-0.7, 0.7],  # where the handle sits across the front, -1 to 1
    "handle_z": [-0.5, 0.5],  # and up it
    "finish": ["paint", "paint", "texture"],
    "hue": [0.0, 1.0],
    "saturation": [0.0, 0.8],
    "value": [0.15, 0.95],
    "texture": [
        "AmericanWalnutVeneer",
        "AngliVeneer",
        "AshVeneer",
        "BambooVeneer",
        "BlackOakVeneer",
        "BlackWalnutVeneer01",
        "BlackWalnutVeneer03",
        "BlackPaintedPlanks",
        "BluePaintedPlanks",
        "BlueMetalPlate",
        "MetalPlate",
        "MetalPlate02",
        "PaintedMetalShutter",
        "BrownLeather",
        "FabricLeather01",
        "BambooWall",
        "BeamWall01",
        "GreenMetalRust",
    ],
    "front": ["same", "same", "contrast"],  # the drawer front: the frame's finish, or its own
    "front_hue": [0.0, 1.0],
    "front_value": [0.15, 0.95],
    "handle_finish": ["brass", "steel", "black", "chrome", "paint"],
    "handle_hue": [0.0, 1.0],
}

WALL = 0.012  # panel thickness
GAP = 0.003  # clearance around the tray
STANDOFF = 0.022  # how far a bar stands off the front
METALS = {
    "brass": ("0.78 0.62 0.30 1", "brass"),
    "steel": ("0.62 0.63 0.65 1", "steel"),
    "black": ("0.06 0.06 0.06 1", "black"),
    "chrome": ("0.85 0.86 0.88 1", "chrome"),
}
TEXTURE_WORDS = {
    "AmericanWalnutVeneer": "walnut",
    "AngliVeneer": "light wood",
    "AshVeneer": "ash wood",
    "BambooVeneer": "bamboo",
    "BlackOakVeneer": "dark oak",
    "BlackWalnutVeneer01": "dark walnut",
    "BlackWalnutVeneer03": "dark walnut",
    "BlackPaintedPlanks": "black planked",
    "BluePaintedPlanks": "blue planked",
    "BlueMetalPlate": "blue metal",
    "MetalPlate": "metal",
    "MetalPlate02": "metal",
    "PaintedMetalShutter": "painted metal",
    "BrownLeather": "leather",
    "FabricLeather01": "leather",
    "BambooWall": "bamboo",
    "BeamWall01": "wooden",
    "GreenMetalRust": "rusty green",
}


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
        *handle(p, front - WALL / 2, half[1], half[2], h / 2),
    ]

    body, body_words = finish(p["finish"], p["texture"], p["hue"], p["saturation"], p["value"])
    if p["front"] == "same":
        front_finish, look = body, body_words
    else:
        front_finish, words = finish("paint", None, p["front_hue"], 0.6, p["front_value"])
        look = body_words if words == body_words else f"{body_words} and {words}"
    if p["handle_finish"] == "paint":
        handle_rgba = _rgba(p["handle_hue"], 0.7, 0.6)
        handle_words = _color_word(p["handle_hue"], 0.7, 0.6)
    else:
        handle_rgba, handle_words = METALS[p["handle_finish"]]
    return f"""<mujoco model="drawer">
  <compiler angle="radian"/>
  <custom>
    <text name="label" data="{look} drawer with {_a(handle_words)} {p["handle"]}"/>
  </custom>
  <default>
    <default class="part">
      <geom type="box" material="body" density="600" condim="4" friction="0.9 0.01 0.002"/>
    </default>
    <default class="front">
      <geom type="box" material="front" density="600" condim="4" friction="0.9 0.01 0.002"/>
    </default>
    <default class="handle">
      <geom material="handle" density="2000" condim="4" friction="1.2 0.01 0.002"/>
    </default>
    <default class="place">
      <geom type="box" contype="0" conaffinity="0" rgba="0 0 0 0" group="4" mass="0"/>
    </default>
  </default>
  <asset>
    {_lines(material("body", body) + material("front", front_finish), 4)}
    <material name="handle" rgba="{handle_rgba}" specular="0.6" shininess="0.6"/>
  </asset>
  <worldbody>
    <body name="drawer_frame">
      {_lines(frame, 6)}
      <body name="tray">
        <joint name="drawer" type="slide" axis="-1 0 0" range="0 {2 * td * 0.85:.4f}"
               damping="2" frictionloss="0.2" armature="0.01"/>
        {_lines(tray, 8)}
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="drawer_frame" body2="tray"/>
  </contact>
</mujoco>
"""


def handle(p, x, room_y, room_z, z0):
    """The handle on the front at depth `x`, offset from the front's middle; and its region.

    `room_y` and `room_z` are the front's half width and half height; `z0` its middle.
    """
    kind = p["handle"]
    k = p["knob"] / 2
    t = p["bar_thickness"] / 2
    half = min(p["bar"] / 2, 0.8 * room_y)
    span = (k, k) if kind.endswith("knob") else (half, t)
    y = p["handle_y"] * max(room_y - span[0] - 0.01, 0)
    z = z0 + p["handle_z"] * max(room_z - span[1] - 0.01, 0)
    if not kind.endswith("knob"):
        tip = x - STANDOFF
        if kind == "bar":
            bar = geom(
                "bar",
                "handle",
                type="capsule",
                size=(t,),
                fromto=(tip, y - half, z, tip, y + half, z),
            )
        else:
            bar = geom("bar", "handle", type="box", pos=(tip, y, z), size=(t, half, t))
        posts = [
            geom(
                f"post{i}",
                "handle",
                type="box",
                pos=((x + tip) / 2, y + s * (half - t), z),
                size=((x - tip) / 2, t * 0.8, t * 0.8),
            )
            for i, s in enumerate((-1, 1))
        ]
        return [bar, *posts, region("handle", (tip, y, z), (t + 0.004, half + 0.006, t + 0.004))]
    reach = k * 2 * p["knob_length"] if kind == "cylinder knob" else k * 2
    centre = (x - 0.006 - reach / 2, y, z)
    if kind == "cube knob":
        knob = geom("knob", "handle", type="box", pos=centre, size=(k, k, k))
    elif kind == "round knob":
        knob = geom("knob", "handle", type="sphere", pos=centre, size=(k,))
    else:
        knob = geom(
            "knob", "handle", type="cylinder", pos=centre, size=(k, reach / 2), euler=(0, 1.5708, 0)
        )
    stem = geom(
        "stem",
        "handle",
        type="cylinder",
        pos=(x - 0.003, y, z),
        size=(min(k * 0.5, 0.006), 0.003),
        euler=(0, 1.5708, 0),
    )
    return [stem, knob, region("handle", centre, (reach / 2 + 0.004, k + 0.004, k + 0.004))]


def finish(kind, texture, hue, saturation, value):
    """A finish (a texture name, or an RGBA paint colour) and the words for it."""
    if kind == "texture":
        return {"texture": texture}, TEXTURE_WORDS[texture]
    return {"rgba": _rgba(hue, saturation, value)}, _color_word(hue, saturation, value)


def material(name, settings):
    """A material, and its texture element if it has one."""
    if "texture" not in settings:
        rgba = settings["rgba"]
        return [f'<material name="{name}" rgba="{rgba}" specular="0.15" shininess="0.25"/>']
    file = _texture(settings["texture"])
    return [
        f'<texture name="{name}" type="2d" file="{file}"/>',
        (
            f'<material name="{name}" texture="{name}" texrepeat="2.5 2.5" texuniform="true" '
            'specular="0.1" shininess="0.2"/>'
        ),
    ]


def _texture(name):
    """A library surface texture's file, downloaded if this library lacks it."""
    path = Path(__file__).resolve().parents[1] / "looks" / "surface" / f"{name}.png"
    if not path.exists():
        from tabletask import fetching

        path = fetching.download(fetching.library_path(path)) or path
    return path


def _rgba(hue, saturation, value):
    return " ".join(f"{c:.3f}" for c in colorsys.hsv_to_rgb(hue, saturation, value)) + " 1"


def _color_word(hue, saturation, value):
    if value < 0.22:
        return "black"
    if saturation < 0.15:
        return "white" if value > 0.8 else "grey" if value > 0.4 else "dark grey"
    names = ["red", "orange", "yellow", "green", "green", "teal", "blue", "blue", "purple", "pink"]
    name = names[int(hue * 10) % 10]
    return ("dark " if value < 0.4 else "light " if saturation < 0.4 and value > 0.7 else "") + name


def _a(word):
    return f"{'an' if word[0] in 'aeiou' else 'a'} {word}"


def box(name, pos, size, cls="part"):
    return geom(name, cls, pos=pos, size=size)


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
