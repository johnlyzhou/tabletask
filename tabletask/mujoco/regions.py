"""Placement-region editing: listing, adding, refitting and removing an asset's regions."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..language.rotation import euler
from .model import PLACE_CLASS, half
from .surface import (
    SurfaceGrid,
    body_vertices,
    convex_mesh,
    find_cavity,
    interior_mesh,
    meshdir,
    support_mesh,
    write_obj,
)

SHAPES = {"box": 3, "cylinder": 2, "sphere": 1}  # number of full extents each region shape takes


def parse_region(text: str) -> dict:
    """Region spec parsed from `name=box:0.1,0.1,0.06@0,0,0.03[/roll,pitch,yaw]`.

    Sizes are full extents and positions are in metres.

    Raises:
        ValueError: If the text is malformed or the extents do not fit the shape.
    """
    name, _, rest = text.partition("=")
    shape_and_size, _, rotation_text = rest.partition("/")
    shape, _, size_text = shape_and_size.partition(":")
    size_text, _, pos_text = size_text.partition("@")
    if not name or shape not in SHAPES:
        raise ValueError(
            "write NAME=SHAPE:extents[@x,y,z][/roll,pitch,yaw]; "
            f"SHAPE is one of {', '.join(SHAPES)}"
        )
    try:
        size = [float(v) for v in size_text.split(",") if v != ""]
        pos = [float(v) for v in pos_text.split(",")] if pos_text else [0.0, 0.0, 0.0]
        rotation = (
            [float(v) for v in rotation_text.split(",")] if rotation_text else [0.0, 0.0, 0.0]
        )
    except ValueError:
        raise ValueError(f"{name}: sizes and positions must be numbers") from None
    if len(size) != SHAPES[shape] or not all(math.isfinite(v) and v > 0 for v in size):
        raise ValueError(
            f"{name}: {shape} takes {SHAPES[shape]} finite positive extent(s), got {size}"
        )
    if len(pos) != 3 or len(rotation) != 3 or not all(map(math.isfinite, [*pos, *rotation])):
        raise ValueError(f"{name}: position and rotation take three finite numbers each")
    if "/" in name:
        raise ValueError(f"{name}: a region name cannot contain '/'")
    return {"name": name, "shape": shape, "size": size, "pos": pos, "rotation": rotation}


def regions(path: Path) -> list[dict]:
    """Every placement region of an asset, with full extents in metres."""
    root = ET.parse(path).getroot()
    meshes = {m.get("name"): m for m in root.iter("mesh")}
    found = []
    for geom in root.iter("geom"):
        name = geom.get("name") or ""
        if not name.startswith("place:"):
            continue
        shape = geom.get("type")
        if shape == "mesh":
            size = None
        else:
            half_size = [float(v) for v in (geom.get("size") or "").split()]
            size = {
                "box": lambda h: [2 * v for v in h],
                "cylinder": lambda h: [2 * h[0], 2 * h[1]],
                "sphere": lambda h: [2 * h[0]],
            }[shape](half_size)
        found.append(
            {
                "name": name[len("place:") :],
                "shape": shape,
                "size": size,
                "pos": [float(v) for v in (geom.get("pos") or "0 0 0").split()],
                "mesh": geom.get("mesh") if shape == "mesh" else None,
                "from_mesh": shape == "mesh" and geom.get("mesh") in meshes,
            }
        )
    return found


def _find_region(root, tag: str):
    """The body holding the region geom named `tag`, and the geom; (None, None) if absent."""
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("name") == tag:
                return body, geom
    return None, None


def _top_level(root, tag: str, path: Path):
    """The asset's top body and its region `tag` (or None), refusing one on a child body.

    Regions are added and fitted in the top body's frame, so one that lives on a child
    body (a lid, a drawer) has to be edited by hand.
    """
    body = root.find("worldbody/body")
    parent, geom = _find_region(root, tag)
    if geom is not None and parent is not body:
        raise ValueError(
            f"{path.stem}: region {tag[len('place:') :]!r} is on part {parent.get('name')!r}; "
            "edit it in the XML"
        )
    return body, geom


def set_region(path: Path, asked: dict) -> str:
    """Add a region from a `parse_region` spec, or replace one of the same name.

    Returns:
        "added" or "replaced".
    """
    root = ET.parse(path).getroot()
    tag = f"place:{asked['name']}"
    body, previous = _top_level(root, tag, path)
    if previous is not None:
        body.remove(previous)
        _drop_unused_mesh(root, previous.get("mesh"))
    geom = ET.SubElement(
        body,
        "geom",
        {
            "name": tag,
            "class": PLACE_CLASS,
            "type": asked["shape"],
            "size": half(asked["shape"], asked["size"]),
            "pos": " ".join(f"{v:g}" for v in asked["pos"]),
        },
    )
    if any(asked["rotation"]):
        geom.set("quat", " ".join(f"{v:.9g}" for v in euler(asked["rotation"])))
    write(path, root)
    return "replaced" if previous is not None else "added"


BUILDERS = ("top", "inside")  # regions that `refit_region` can derive from the mesh


def refit_region(path: Path, name: str) -> str:
    """Rebuild the `top` or `inside` region from the visual mesh, as an import does.

    A hand-written box region is often wider than the face under it, so sampled
    positions can land over a notch or past an edge; the fitted region follows the
    mesh surface instead.

    Returns:
        A description of the region before the refit ("none" if there was none).

    Raises:
        ValueError: If `name` is not fittable or the mesh has no such surface.
    """
    path = Path(path)
    if name not in BUILDERS:
        raise ValueError(f"only {' and '.join(BUILDERS)} can be fitted, not {name!r}")
    points, faces, _ = body_vertices(path)
    surface = SurfaceGrid(points, faces)
    cavity = find_cavity(surface)
    if name == "inside":
        built = interior_mesh(surface, cavity) if cavity is not None else None
    else:
        built = support_mesh(surface, cavity)
    if built is None:
        raise ValueError(f"{path.stem}: no {name} surface found in the visual mesh")
    if name == "top":
        built = _over_solid(surface, built)

    root = ET.parse(path).getroot()
    assets = root.find("asset")
    mesh_folder = path.parent / meshdir(root)
    tag, mesh_name = f"place:{name}", f"{path.stem}_{name}"
    body, previous = _top_level(root, tag, path)

    # Reuse the region's existing file; a new one gets a name unique within a shared meshdir.
    file = None
    # Compare with None: an Element without children is falsy.
    if previous is not None and previous.get("mesh"):
        file = next(
            (
                m.get("file")
                for m in assets.findall("mesh")
                if m.get("name") == previous.get("mesh")
            ),
            None,
        )
    if file is None:
        file = f"regions/{path.stem}_{name}.obj"
    before = "none" if previous is None else _shape_of(root, previous)
    if previous is not None:
        body.remove(previous)
        _drop_unused_mesh(root, previous.get("mesh"))
    (mesh_folder / file).parent.mkdir(parents=True, exist_ok=True)
    write_obj(mesh_folder / file, *built, note=f"{path.stem}: {name} region")
    if not any(m.get("name") == mesh_name for m in assets.findall("mesh")):
        ET.SubElement(assets, "mesh", name=mesh_name, file=file)
    ET.SubElement(
        body, "geom", {"name": tag, "class": PLACE_CLASS, "type": "mesh", "mesh": mesh_name}
    )
    write(path, root)
    return before


def _over_solid(surface: SurfaceGrid, built, min_scale: float = 0.4):
    """Shrink a region about its centre until no part of it lies over a hole in the mesh.

    Regions are convex (`within` tests against the convex hull), so a notch or gap
    cannot be cut out; the region is scaled down instead, by at most `min_scale`.
    """
    points, _ = built
    empty = ~np.isfinite(surface.height)
    xs, ys = np.meshgrid(surface.xs, surface.ys, indexing="ij")
    holes = np.column_stack([xs[empty], ys[empty]])
    if not holes.size:
        return built
    middle = points[:, :2].mean(0)

    def clear_of_holes(scale: float) -> bool:
        scaled = (points[:, :2] - middle) * scale + middle
        try:  # The same half-space test as `within`, in two dimensions.
            sides = ConvexHull(scaled).equations
        except QhullError:
            return False
        return not np.all(holes @ sides[:, :2].T + sides[:, 2] <= 0, axis=1).any()

    if clear_of_holes(1.0):
        return built
    low, high = min_scale, 1.0
    for _ in range(12):  # Bisection on the scale factor.
        mid = (low + high) / 2
        if clear_of_holes(mid):
            low = mid
        else:
            high = mid
    scaled = (points[:, :2] - middle) * low + middle
    return convex_mesh(np.column_stack([scaled, points[:, 2]]))


def _shape_of(root, geom) -> str:
    """Short description of a region geom's shape, e.g. `box 0.1 0.1 0.02`."""
    geom_type = geom.get("type")
    if geom_type != "mesh":
        return f"{geom_type} {geom.get('size', '')}".strip()
    mesh = next((m for m in root.iter("mesh") if m.get("name") == geom.get("mesh")), None)
    return f"mesh {mesh.get('file')}" if mesh is not None else "mesh"


def drop_region(path: Path, name: str) -> bool:
    """Remove a region and any mesh only it used; False if there was no such region."""
    root = ET.parse(path).getroot()
    body, geom = _find_region(root, f"place:{name}")
    if geom is None:
        return False
    body.remove(geom)
    _drop_unused_mesh(root, geom.get("mesh"))
    write(path, root)
    return True


def _drop_unused_mesh(root, mesh: str | None) -> None:
    """Remove a mesh asset if no geom references it."""
    if mesh and not any(g.get("mesh") == mesh for g in root.iter("geom")):
        asset = root.find("asset")
        for element in asset.findall("mesh"):
            if element.get("name") == mesh:
                asset.remove(element)


def write(path: Path, root) -> None:
    """Write an indented XML tree to `path`."""
    ET.indent(root, space="  ")
    Path(path).write_text(ET.tostring(root, encoding="unicode") + "\n")
