"""Pieces the furniture generators share: geoms, regions, finishes and handles.

Lengths are metres; fronts face -x.
"""

import colorsys
from pathlib import Path

WALL = 0.012  # panel thickness
STANDOFF = 0.022  # how far a bar stands off its front

HANDLE_PARAMS = {
    "handle": ["bar", "square bar", "cylinder knob", "cube knob", "round knob"],
    "knob": [0.014, 0.04],  # a knob's width
    "knob_length": [0.5, 1.4],  # a cylinder knob's length, as a fraction of its width
    "bar": [0.06, 0.2],  # bar length, at most 80% of the room it has
    "bar_thickness": [0.008, 0.016],
}
FINISH_PARAMS = {
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
}
FRONT_PARAMS = {
    "front": ["same", "same", "contrast"],  # a front: the body's finish, or its own paint
    "front_hue": [0.0, 1.0],
    "front_value": [0.15, 0.95],
}
HANDLE_FINISH_PARAMS = {
    "handle_finish": ["brass", "steel", "black", "chrome", "paint"],
    "handle_hue": [0.0, 1.0],
}
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
DEFAULTS = """<default>
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
  </default>"""


def handle(p, x, y, z, room_y, room_z, along="y"):
    """The handle on a front at depth `x`, and its region.

    `room_y` and `room_z` are the half width and height of the patch of front around
    (`y`, `z`) it may sit in; `handle_y` and `handle_z` (-1 to 1) place it there. A bar
    runs `along` "y" or "z".
    """
    kind = p["handle"]
    k = p["knob"] / 2
    t = p["bar_thickness"] / 2
    if kind.endswith("knob"):
        span = (k, k)
    else:
        half = min(p["bar"] / 2, 0.8 * (room_y if along == "y" else room_z))
        span = (half, t) if along == "y" else (t, half)
    y += p["handle_y"] * max(room_y - span[0] - 0.01, 0)
    z += p["handle_z"] * max(room_z - span[1] - 0.01, 0)
    if not kind.endswith("knob"):
        tip = x - STANDOFF
        run = (0, half, 0) if along == "y" else (0, 0, half)
        a = (tip, y - run[1], z - run[2])
        b = (tip, y + run[1], z + run[2])
        if kind == "bar":
            bar = geom("bar", "handle", type="capsule", size=(t,), fromto=(*a, *b))
        else:
            size = (t, half, t) if along == "y" else (t, t, half)
            bar = geom("bar", "handle", type="box", pos=(tip, y, z), size=size)
        posts = []
        for i, s in enumerate((-1, 1)):
            at = (y + s * (half - t), z) if along == "y" else (y, z + s * (half - t))
            post = ((x - tip) / 2, t * 0.8, t * 0.8)
            posts.append(
                geom(f"post{i}", "handle", type="box", pos=((x + tip) / 2, *at), size=post)
            )
        reach = (t + 0.004, span[0] + 0.006 if along == "y" else t + 0.004)
        grip = (reach[0], reach[1], t + 0.004 if along == "y" else span[1] + 0.006)
        return [bar, *posts, region("handle", (tip, y, z), grip)]
    reach = k * 2 * p["knob_length"] if kind == "cylinder knob" else k * 2
    centre = (x - 0.006 - reach / 2, y, z)
    sideways = (0, 1.5708, 0)
    if kind == "cube knob":
        knob = geom("knob", "handle", type="box", pos=centre, size=(k, k, k))
    elif kind == "round knob":
        knob = geom("knob", "handle", type="sphere", pos=centre, size=(k,))
    else:
        knob = geom(
            "knob", "handle", type="cylinder", pos=centre, size=(k, reach / 2), euler=sideways
        )
    stem_size = (min(k * 0.5, 0.006), 0.003)
    stem = geom(
        "stem", "handle", type="cylinder", pos=(x - 0.003, y, z), size=stem_size, euler=sideways
    )
    return [stem, knob, region("handle", centre, (reach / 2 + 0.004, k + 0.004, k + 0.004))]


def finishes(p):
    """The body, front and handle materials, and the words for the look and the handle."""
    body, body_words = finish(p["finish"], p["texture"], p["hue"], p["saturation"], p["value"])
    if p["front"] == "same":
        front, look = body, body_words
    else:
        front, words = finish("paint", None, p["front_hue"], 0.6, p["front_value"])
        look = body_words if words == body_words else f"{body_words} and {words}"
    if p["handle_finish"] == "paint":
        handle_rgba = _rgba(p["handle_hue"], 0.7, 0.6)
        handle_words = _color_word(p["handle_hue"], 0.7, 0.6)
    else:
        handle_rgba, handle_words = METALS[p["handle_finish"]]
    materials = [
        *material("body", body),
        *material("front", front),
        f'<material name="handle" rgba="{handle_rgba}" specular="0.6" shininess="0.6"/>',
    ]
    return materials, look, f"{article(handle_words)} {p['handle']}"


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


def article(word):
    return f"{'an' if word[0] in 'aeiou' else 'a'} {word}"


def box(name, pos, size, cls="part"):
    return geom(name, cls, pos=pos, size=size)


def region(name, pos, size):
    return geom(f"place:{name}", "place", pos=pos, size=size)


def geom(name, cls, **attrs):
    """A geom element; tuples become space-separated numbers."""
    text = " ".join(f'{k}="{v4(v) if isinstance(v, tuple) else v}"' for k, v in attrs.items())
    return f'<geom name="{name}" class="{cls}" {text}/>'


def v4(values):
    return " ".join(f"{v:.4f}" for v in values)


def lines(elements, spaces):
    return ("\n" + " " * spaces).join(elements)
