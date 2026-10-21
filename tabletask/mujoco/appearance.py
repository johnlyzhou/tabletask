"""Table and sky textures and lighting drawn per episode."""

from __future__ import annotations

import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .. import fetching
from ..language.schema import Appearance
from .sky import CUBE_LAYOUT

# The textures a scene can draw from, by kind. Listed in the package so a checkout and an
# installed package choose alike; an installed package downloads the one it picks.
INDEX = json.loads((Path(__file__).parents[1] / "scenes/textures.json").read_text())

# Lighting ranges at full variation.
LIGHT_TILT = 1.0  # the most a light's direction turns, radians
LIGHT_BRIGHTNESS = 0.6  # the most a light's brightness changes, as a fraction
LIGHT_TINT = 0.15  # the most each colour channel of a light changes
LIGHT_MIN_DROP = 0.2  # the least downward component of a light's direction
SHADOW_OFF = 0.5  # the chance a light that casts shadows does not


def pool(textures: Path, kind: str) -> tuple[Path, ...]:
    """The texture files of one kind, in the order a seed selects them.

    A library that has the folder is listed directly; without it (an installed package)
    the shipped list stands in, and the chosen file is downloaded.
    """
    folder = textures / kind
    if folder.is_dir():
        return tuple(sorted(folder.glob("*.png")))
    return tuple(folder / name for name in INDEX.get(kind, ()))


def _local(path: Path) -> Path | None:
    """A texture file, downloaded first if it is not here; None if unavailable."""
    if path.exists():
        return path
    name = fetching.library_path(path)
    return fetching.download(name) if name else None


def _select_texture(textures: Path, kind: str, named: str | None, rng) -> Path | None:
    """The texture called `named`, else a random one of that kind (if `rng`), else None."""
    choices = pool(textures, kind)
    if named:
        file = named if Path(named).suffix else f"{named}.png"
        found = _local(textures / kind / file)
        if found is None:
            raise ValueError(f"no {kind} called {named!r} in the library")
        return found
    if rng is None or not choices:
        return None
    chosen = choices[int(rng.integers(len(choices)))]
    found = _local(chosen)
    if found is None:
        warnings.warn(
            f"{kind} texture {chosen.name} is not in {textures / kind} and could not be "
            f"downloaded; the scene keeps its default {kind}",
            stacklevel=2,
        )
    return found


def apply_appearance(mj: ET.Element, appearance, rng, textures: Path) -> None:
    """Apply sampled or named table and sky textures and sampled lighting to the scene XML.

    Args:
        mj: Root `<mujoco>` element of the scene.
        appearance: The scene's `Appearance`, or None for the defaults.
        rng: Random generator used when a texture is not named.
        textures: Folder holding `surface/` and `sky/` texture images.
    """
    appearance = appearance if appearance is not None else Appearance()
    asset, world = mj.find("asset"), mj.find("worldbody")
    if asset is None or world is None:
        return
    if appearance.table_height > 0 and rng is not None:
        _shift_table(world, float(rng.uniform(-appearance.table_height, appearance.table_height)))
    if not appearance.floor:
        for geom in list(world.findall("geom")):
            if geom.get("name") == "floor":
                world.remove(geom)

    pick = rng if appearance.vary else None  # without vary, only named textures apply
    surface = _select_texture(textures, "surface", appearance.surface, pick)
    if surface is not None:
        _set_material_texture(asset, "tabletop", surface, repeat=appearance.repeat)
    sky = _select_texture(textures, "sky", appearance.sky, pick)
    if sky is not None:
        _replace_skybox(asset, {"file": str(sky), "gridsize": "3 4", "gridlayout": CUBE_LAYOUT})
    if appearance.lighting > 0 and rng is not None:
        _vary_lights(mj, appearance.lighting, rng)


def _vary_lights(mj: ET.Element, amount: float, rng) -> None:
    """Turn each light and the headlight, and change their brightness and colour, in place.

    `amount` scales the ranges: each light turns about a random axis by up to `LIGHT_TILT`,
    staying aimed down, its colours are scaled by up to `LIGHT_BRIGHTNESS` and shifted per
    channel by up to `LIGHT_TINT`, and its shadows are turned off with chance `amount *
    SHADOW_OFF`.
    """
    lights = list(mj.iter("light"))
    headlight = mj.find("visual/headlight")
    for light in [*lights, headlight]:
        if light is None:
            continue
        if light is not headlight:
            light.set("dir", _join(_tilt(_values(light, "dir", (0, 0, -1)), amount, rng)))
            if rng.random() < amount * SHADOW_OFF:
                light.set("castshadow", "false")
        brightness = 1 + amount * rng.uniform(-LIGHT_BRIGHTNESS, LIGHT_BRIGHTNESS)
        tint = amount * rng.uniform(-LIGHT_TINT, LIGHT_TINT, 3)
        for key, default in (("ambient", (0, 0, 0)), ("diffuse", (0.7, 0.7, 0.7))):
            if light.get(key) is not None or key == "diffuse":
                colour = np.clip(_values(light, key, default) * brightness + tint, 0, 1)
                light.set(key, _join(colour))


def _tilt(direction: np.ndarray, amount: float, rng) -> np.ndarray:
    """`direction` turned about a random axis by up to `amount * LIGHT_TILT`, kept aimed down."""
    direction = direction / np.linalg.norm(direction)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = amount * rng.uniform(-LIGHT_TILT, LIGHT_TILT)
    turned = (
        direction * np.cos(angle)
        + np.cross(axis, direction) * np.sin(angle)
        + axis * axis.dot(direction) * (1 - np.cos(angle))
    )
    turned[2] = min(turned[2], -LIGHT_MIN_DROP)
    return turned / np.linalg.norm(turned)


def _values(element: ET.Element, key: str, default) -> np.ndarray:
    """A numeric vector attribute, or `default` when it is absent."""
    text = element.get(key)
    return np.array([float(v) for v in text.split()] if text else default, dtype=float)


def _join(values) -> str:
    """Numbers as an MJCF attribute."""
    return " ".join(f"{value:.4g}" for value in values)


def _shift_table(world: ET.Element, dz: float) -> None:
    """Move the table, its placement regions and its collision block up by `dz`; the robot stays."""
    for element in list(world):
        if (element.tag == "body" and element.get("name") == "play_table") or (
            element.tag == "geom" and element.get("name") == "table_plane"
        ):
            x, y, z = (float(v) for v in (element.get("pos") or "0 0 0").split())
            element.set("pos", f"{x} {y} {z + dz}")


def _replace_skybox(asset: ET.Element, sky: dict) -> None:
    """Replace any skybox texture with one built from `sky` attributes."""
    for texture in list(asset.findall("texture")):
        if texture.get("type") == "skybox":
            asset.remove(texture)
    asset.insert(0, ET.Element("texture", {"type": "skybox", **sky}))


def _set_material_texture(asset: ET.Element, material: str, image: Path, repeat) -> None:
    """Texture a material with `image`, creating the material if it does not exist."""
    name = f"appearance:{material}"
    asset.insert(0, ET.Element("texture", {"name": name, "type": "2d", "file": str(image)}))
    element = next(
        (each for each in asset.findall("material") if each.get("name") == material), None
    )
    if element is None:
        element = ET.SubElement(asset, "material", {"name": material})
    element.set("texture", name)
    element.set("rgba", "1 1 1 1")
    element.set("texuniform", "true")
    element.set("texrepeat", " ".join(f"{value:g}" for value in repeat))
