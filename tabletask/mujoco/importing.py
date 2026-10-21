"""Mesh import: turning meshes into assets, and fitting their collision geometry."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, QhullError

from ..language.rotation import mat
from .checks import study
from .model import PLACE_CLASS
from .regions import write
from .surface import (
    SurfaceGrid,
    body_vertices,
    find_cavity,
    interior_mesh,
    meshdir,
    support_mesh,
    write_obj,
)

# Mesh import and collision decomposition.
MAX_REORIENTATIONS = 3  # times an import re-orients a mesh that falls over
IMPORT_MAX_TILT = 15.0  # degrees of tilt after settling that make an import re-orient
MAX_HULL_VERTICES = 128  # vertex limit per collision hull (MuJoCo maxhullvert)
MAX_HULLS = 32  # collision hulls per asset; CoACD merges the closest pieces beyond this
DECOMPOSITION_THRESHOLDS = (0.10, 0.07, 0.05, 0.035, 0.02)  # CoACD concavity, coarse to fine
MIN_VOLUME_IMPROVEMENT = 0.001  # volume saved per extra hull, as a fraction of the convex hull
MIN_HULL_VOLUME = 1e-9  # cubic metres; smaller hulls are dropped
BASE_DEPTH = 0.008  # metres above the lowest point that count as the base
BASE_FLATTEN_DEPTH = 0.003  # metres above the lowest point flattened into the support hull
DEFAULT_MASS = 0.25  # kilograms
CONTACT_ATTRIBUTES = ("condim", "friction", "solref", "solimp", "priority")  # kept on rebuild
DEFAULT_RGBA = "0.75 0.72 0.68 1"  # material colour of untextured assets


def load_mesh(
    path: Path, scale: float = 1.0, up: str = "z"
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Image.Image | None]:
    """Vertices, triangles, UVs and texture of a mesh file, in metres with z up."""
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    points = np.asarray(mesh.vertices, dtype=float) * scale
    if up == "y":
        points = points[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    elif up != "z":
        raise ValueError(f"up must be z or y, not {up!r}")
    triangles = np.asarray(mesh.faces, dtype=np.int32)
    uvs, image = None, None
    visual = getattr(mesh, "visual", None)
    if getattr(visual, "uv", None) is not None and len(visual.uv) == len(points):
        uvs = np.asarray(visual.uv, dtype=float)
        material = getattr(visual, "material", None)
        image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
        if image is not None and min(image.size) < 4:
            image = None  # Ignore placeholder texture swatches.
    return points, triangles, uvs, image


def center_mesh(points: np.ndarray) -> np.ndarray:
    """Points shifted so the horizontal bounds are centred and the lowest vertex is at z=0."""
    lo, hi = points.min(0), points.max(0)
    return points - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])


def settled_rotation(rotation: np.ndarray) -> np.ndarray:
    """Rotation that aligns the mesh with its settled up axis, without adding yaw."""
    up = rotation[2]  # World up in object coordinates.
    axis = np.cross(up, [0.0, 0.0, 1.0])
    sine, cosine = np.linalg.norm(axis), float(np.clip(up[2], -1.0, 1.0))
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sine
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sine * skew + (1 - cosine) * skew @ skew


def decompose_collision(points: np.ndarray, triangles: np.ndarray, folder: Path) -> list[Path]:
    """Write convex collision hulls, refining until the volume saved per hull is too small.

    Returns:
        Paths of the written hull `.obj` files, including a flattened base hull if any.
    """
    import coacd
    import trimesh

    coacd.set_log_level("error")
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    collision_mesh = coacd.Mesh(points, triangles)
    convex_volume = trimesh.Trimesh(points, triangles, process=False).convex_hull.volume
    best, best_volume, best_count = [], None, 0
    for threshold in DECOMPOSITION_THRESHOLDS:
        pieces = [
            trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False).convex_hull
            for v, f in coacd.run_coacd(
                collision_mesh,
                threshold=threshold,
                preprocess_mode="auto",
                merge=True,
                max_convex_hull=MAX_HULLS,
            )
            if len(v) >= 4
        ]
        pieces = [p for p in pieces if p.volume >= MIN_HULL_VOLUME]
        if not pieces:
            continue
        volume, count = sum(p.volume for p in pieces), len(pieces)
        if best_volume is None:
            best, best_volume, best_count = pieces, volume, count
            continue
        if count <= best_count:
            continue
        if (best_volume - volume) / (count - best_count) < MIN_VOLUME_IMPROVEMENT * convex_volume:
            break
        best, best_volume, best_count = pieces, volume, count
    if not best:
        raise ValueError("collision decomposition produced no usable hulls")
    written = []
    for hull in [*best, base_hull(points, best)]:
        if hull is not None:
            written.append(folder / f"h{len(written):02d}.obj")
            hull.export(written[-1])
    return written


def base_hull(points: np.ndarray, pieces: list):
    """Flattened support hull under the mesh, or None when the base is too narrow."""
    import trimesh

    low = points[:, 2].min()
    base = points[points[:, 2] < low + BASE_DEPTH].copy()
    seat = base[:, 2] < low + BASE_FLATTEN_DEPTH
    if seat.sum() < 3 or len(base) < 4:
        return None
    span_seat, span_base = np.ptp(base[seat, :2], axis=0), np.ptp(base[:, :2], axis=0)
    if np.any(span_seat < 0.01) or np.any(span_seat < 0.7 * span_base):
        return None  # The contact patch is too narrow to flatten.
    floor = min(p.vertices[:, 2].min() for p in pieces) - 0.0005 if pieces else low
    base[seat, 2] = floor
    if np.linalg.matrix_rank(base - base[0]) < 3:
        return None
    try:
        hull = trimesh.convex.convex_hull(base)
    except (QhullError, ValueError):
        return None
    return hull if hull.volume >= MIN_HULL_VOLUME else None


def _colliders(path: Path, root) -> list[ET.Element]:
    """The geoms of an asset that physics collides with, in any group.

    Raises:
        ValueError: If there are none, or if some belong to a moving part (a body with a
            joint): their replacement would all go on the asset's top body.
    """
    solid = [
        g
        for g in root.find("worldbody").iter("geom")
        if g.get("contype") != "0" and g.get("class") != PLACE_CLASS
    ]
    if not solid:
        raise ValueError(f"{path.stem}: no collision geometry to replace")
    moving = [
        body.get("name") or "?"
        for body in root.iter("body")
        if body.find("joint") is not None and any(g in solid for g in body.iter("geom"))
    ]
    if moving:
        raise ValueError(
            f"{path.stem}: moving parts {moving} have colliders of their own; this asset is "
            "articulated, so fit its collision geometry by hand"
        )
    return solid


def _collider_mass(path: Path, solid: list[ET.Element]) -> float:
    """Total mass of the collision geoms: their explicit masses, else what MuJoCo computes.

    Without explicit masses, the asset is compiled with every other geom weightless, so
    a visual geom's own mass is not counted twice.
    """
    explicit = sum(float(g.get("mass", 0) or 0) for g in solid)
    if explicit > 0:
        return explicit
    root = ET.parse(path).getroot()
    colliders = {
        i
        for i, g in enumerate(root.find("worldbody").iter("geom"))
        if g.get("contype") != "0" and g.get("class") != PLACE_CLASS
    }
    for i, geom in enumerate(root.find("worldbody").iter("geom")):
        if i not in colliders:
            geom.set("mass", "0")
            geom.attrib.pop("density", None)
    probe = path.with_name(f".{path.stem}.mass.xml")  # beside the asset, so its paths resolve
    try:
        ET.ElementTree(root).write(probe)
        return float(mujoco.MjModel.from_xml_path(str(probe)).body_subtreemass[1])
    finally:
        probe.unlink(missing_ok=True)


def distribute_mass(pieces: list[Path], mass: float) -> list[float]:
    """Per-hull masses proportional to hull volume, with a minimum share for small hulls."""
    volumes = []
    for piece in pieces:
        try:
            volumes.append(ConvexHull(_read_obj_vertices(piece)).volume)
        except QhullError:
            volumes.append(0.0)
    volumes = np.maximum(np.asarray(volumes), 0.002 * max(sum(volumes), 1e-12))
    return [float(mass * v / volumes.sum()) for v in volumes]


def _read_obj_vertices(obj: Path) -> np.ndarray:
    """Vertex positions of a Wavefront `.obj` file."""
    return np.array(
        [
            [float(x) for x in line.split()[1:4]]
            for line in obj.read_text().splitlines()
            if line.startswith("v ")
        ]
    )


def _write_meshes(name: str, library: Path, points, triangles, uvs, image, mass: float):
    """Write the visual mesh, its texture, the collision hulls and any region meshes."""
    meshes = library / "mesh" / name
    meshes.mkdir(parents=True, exist_ok=True)
    write_obj(meshes / "0.obj", points, triangles, uvs, name)
    if image is not None and uvs is not None:
        image.convert("RGB").save(meshes / "0.png")  # MuJoCo only reads PNG textures.
    pieces = decompose_collision(points, triangles, meshes / "hulls")

    surface = SurfaceGrid(points, triangles)
    cavity = find_cavity(surface)
    places = []
    for stale in ("inside.obj", "top.obj"):
        (meshes / stale).unlink(missing_ok=True)
    if cavity is not None:
        write_obj(
            meshes / "inside.obj", *interior_mesh(surface, cavity), note=f"{name}: interior region"
        )
        places.append("inside")
    if (top := support_mesh(surface, cavity)) is not None:
        write_obj(meshes / "top.obj", *top, note=f"{name}: support region")
        places.append("top")
    return pieces, distribute_mass(pieces, mass), places, surface


def _add_hulls(assets, body, prefix: str, folder: str, pieces, shares, contact: dict) -> None:
    """Add a mesh asset and an invisible group-3 collision geom for each hull.

    Args:
        assets: The `<asset>` element the hull meshes go in.
        body: The body the collision geoms go in.
        prefix: Start of each hull mesh name, `<prefix>_hull_<index>`.
        folder: Folder of the hull files, relative to the mesh directory.
        pieces: Hull `.obj` paths.
        shares: Mass of each hull, in kilograms.
        contact: Contact attributes set on every geom, in order; empty values are skipped.
    """
    for index, (piece, share) in enumerate(zip(pieces, shares)):
        mesh_name = f"{prefix}_hull_{index:02d}"
        ET.SubElement(
            assets,
            "mesh",
            name=mesh_name,
            file=f"{folder}/{piece.name}",
            maxhullvert=str(MAX_HULL_VERTICES),
        )
        attributes = {
            "type": "mesh",
            "mesh": mesh_name,
            "rgba": "0 0 0 0",
            "group": "3",
            "mass": f"{share:.9g}",
            **{key: value for key, value in contact.items() if value},
        }
        ET.SubElement(body, "geom", attributes)


def _document(name: str, textured: bool, pieces, shares, places, surface) -> ET.Element:
    """The asset's MJCF: one visual geom, the collision hulls, and a geom per region."""
    root = ET.Element("mujoco", model=name)
    ET.SubElement(
        root, "compiler", angle="radian", meshdir=f"../mesh/{name}", texturedir=f"../mesh/{name}"
    )
    defaults = ET.SubElement(ET.SubElement(root, "default"), "default", {"class": PLACE_CLASS})
    ET.SubElement(
        defaults, "geom", contype="0", conaffinity="0", rgba="0 0 0 0", group="4", mass="0"
    )
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "mesh", name=f"{name}_0_mesh", file="0.obj")
    material = ET.SubElement(
        assets, "material", name=f"{name}_0_skin", specular="0.2", shininess="0.3"
    )
    if textured:
        ET.SubElement(assets, "texture", name=f"{name}_0_tex", type="2d", file="0.png")
        material.set("roughness", "0.5")
        material.set("metallic", "0")
        ET.SubElement(material, "layer", texture=f"{name}_0_tex", role="rgb")
    else:
        material.set("rgba", DEFAULT_RGBA)

    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name=name)
    ET.SubElement(
        body,
        "geom",
        type="mesh",
        mesh=f"{name}_0_mesh",
        material=f"{name}_0_skin",
        contype="0",
        conaffinity="0",
        mass="0",
        condim="4",
    )
    _add_hulls(
        assets, body, name, "hulls", pieces, shares, {"friction": "0.9 0.01 0.002", "condim": "4"}
    )
    middle = (surface.bounds_min + surface.bounds_max) / 2
    half_extents = (surface.bounds_max - surface.bounds_min) / 2
    ET.SubElement(
        body,
        "geom",
        {
            "name": "place:bounds",
            "class": PLACE_CLASS,
            "type": "box",
            "size": " ".join(f"{v:.4f}" for v in half_extents),
            "pos": " ".join(f"{v:.4f}" for v in middle),
        },
    )
    for region in places:
        ET.SubElement(assets, "mesh", name=f"{name}_{region}", file=f"{region}.obj")
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"place:{region}",
                "class": PLACE_CLASS,
                "type": "mesh",
                "mesh": f"{name}_{region}",
            },
        )
    return root


def write_asset(name: str, library: Path, points, triangles, uvs, image, mass: float) -> Path:
    """Write visual geometry, collision hulls, regions and the MuJoCo asset XML."""
    pieces, shares, places, surface = _write_meshes(
        name, library, points, triangles, uvs, image, mass
    )
    textured = image is not None and uvs is not None
    root = _document(name, textured, pieces, shares, places, surface)
    path = library / "objects" / f"{name}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, root)
    return path


def import_mesh(
    name: str,
    mesh: Path,
    library: Path,
    texture: Path | None = None,
    mass: float = DEFAULT_MASS,
    scale: float = 1.0,
    up: str = "z",
) -> tuple[Path, list[str]]:
    """Import a mesh as an asset, re-orienting it if it falls over when set down.

    Args:
        name: Asset name; used for the XML file, the body and the mesh folder.
        mesh: Mesh file to import (any format trimesh reads).
        library: Library root; writes `objects/<name>.xml` and `mesh/<name>/`.
        texture: Optional image that replaces the mesh's own texture.
        mass: Total mass in kilograms.
        scale: Factor that converts mesh units to metres.
        up: Up axis of the mesh file, "z" or "y".

    Returns:
        The asset XML path and the tilt measured after each import attempt.

    Raises:
        ValueError: On a non-positive mass or scale, or a texture without UVs.
    """
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be finite and positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    points, triangles, uvs, image = load_mesh(mesh, scale, up)
    if texture is not None:
        with Image.open(texture) as source:
            image = source.convert("RGB")
    if image is not None and uvs is None:
        raise ValueError(
            f"{mesh.name} has no texture coordinates, so a texture cannot be put on it"
        )
    points = center_mesh(points)
    checks = []
    for attempt in range(MAX_REORIENTATIONS + 1):
        xml = write_asset(name, library, points, triangles, uvs, image, mass)
        facts, problems, runtime = study(xml)
        checks.append(f"tilt {facts['tilt']:.1f} deg")
        # a fresh import should stand as authored, so a large tilt alone means re-orient
        fell = facts["tilt"] > IMPORT_MAX_TILT or any(p.startswith("falls over") for p in problems)
        if not fell or attempt == MAX_REORIENTATIONS:
            break
        _, quat = runtime.pose(name)
        points = center_mesh(points @ settled_rotation(mat(quat)).T)
    return xml, checks


SOLIDS = ("box", "cylinder", "sphere", "ellipsoid")  # primitives `collide_as` can fit


def _pick_solid(half_extents: np.ndarray, fill: float) -> str:
    """Primitive guessed from the fraction of its bounding box the mesh fills."""
    round_xy = abs(half_extents[0] - half_extents[1]) / max(half_extents[0], half_extents[1]) < 0.12
    if fill > 0.85:
        return "box"
    if fill > 0.68 and round_xy:
        return "cylinder"  # A cylinder fills pi/4 = 0.785 of its box.
    if fill > 0.44:  # A sphere fills pi/6 = 0.524 of its box.
        # A sphere takes its radius from the height, so it must be nearly equal in all
        # axes; otherwise an ellipsoid fits better.
        round_enough = (
            round_xy
            and abs(half_extents[2] - half_extents[0]) / max(half_extents[2], half_extents[0])
            < 0.04
        )
        return "sphere" if round_enough else "ellipsoid"
    return "box"


def collide_as(path: Path, shape: str | None = None) -> tuple[int, str, np.ndarray]:
    """Replace an asset's collision hulls with one primitive fitted to its visual mesh.

    Hulls of round objects catch on their facets, and hulls of tapered objects can be
    too wide; a primitive avoids both. The primitive is sized so its bottom matches
    the mesh's bottom, and the asset keeps its total mass.

    Args:
        path: Asset XML file, rewritten in place.
        shape: One of `SOLIDS`; guessed from the mesh when None.

    Returns:
        The number of collision geoms replaced, the shape used, and its MuJoCo size.

    Raises:
        ValueError: On an unknown shape, or when the asset has no collision geoms or
            has collision geoms outside group 3 (hand-built colliders).
    """
    path = Path(path)
    points, _, _ = body_vertices(path)
    lo, hi = points.min(0), points.max(0)
    half_extents, middle = (hi - lo) / 2, (lo + hi) / 2 + 0.0
    try:
        fill = float(ConvexHull(points).volume / np.prod(hi - lo))
    except QhullError:
        fill = 1.0
    shape = shape or _pick_solid(half_extents, fill)
    if shape not in SOLIDS:
        raise ValueError(f"shape must be one of {', '.join(SOLIDS)}, not {shape!r}")

    root = ET.parse(path).getroot()
    body = root.find("worldbody/body")
    # All geoms physics collides with, in any group. Hand-built colliders (e.g. maze
    # walls in group 0) must not be merged into one primitive, so they raise below.
    solid = _colliders(path, root)
    hand_built = [g for g in solid if g.get("group") != "3"]
    if hand_built:
        kinds = sorted({g.get("type") or "mesh" for g in hand_built})
        raise ValueError(
            f"{path.stem}: {len(hand_built)} of {len(solid)} collision geoms are outside group 3 "
            f"({', '.join(kinds)}) -- this asset was built by hand, so fit it by hand"
        )
    mass = _collider_mass(path, solid)
    contact_attrs = {k: solid[0].get(k) for k in CONTACT_ATTRIBUTES}
    before = len(solid)

    radius = float((half_extents[0] + half_extents[1]) / 2)
    size = {
        "box": half_extents,
        "cylinder": np.array([radius, half_extents[2]]),
        "sphere": np.array([half_extents[2]]),  # Radius from the height, so it rests at z=0.
        "ellipsoid": half_extents,
    }[shape]
    for geom in solid:
        next(b for b in root.iter() if geom in list(b)).remove(geom)
    for mesh in [
        m for m in root.find("asset").findall("mesh") if "_hull_" in (m.get("name") or "")
    ]:
        root.find("asset").remove(mesh)
    body.append(
        ET.Element(
            "geom",
            {
                "type": shape,
                "size": " ".join(f"{v:.5f}" for v in np.atleast_1d(size)),
                "pos": " ".join(f"{v:.4f}" for v in middle),
                "rgba": "0 0 0 0",
                "group": "3",
                "mass": f"{mass:.9g}",
                **{k: v for k, v in contact_attrs.items() if v},
            },
        )
    )
    write(path, root)
    _drop_loose_hulls(path, root)
    return before, shape, np.atleast_1d(size)


def _drop_loose_hulls(path: Path, root) -> None:
    """Delete hull files that no asset sharing this mesh folder references.

    Several assets can share one mesh folder, so files referenced by any of them
    are kept.
    """
    mesh_folder = (path.parent / meshdir(root)).resolve()
    used = set()
    for other in sorted(path.parent.glob("*.xml")):
        try:
            other_root = ET.parse(other).getroot()
        except ET.ParseError:
            continue
        other_meshdir = other.parent / meshdir(other_root)
        if other_meshdir.resolve() != mesh_folder:
            continue
        used |= {m.get("file") for m in other_root.iter("mesh") if m.get("file")}
    for folder in (mesh_folder / "hulls", mesh_folder / f"{path.stem}_hulls"):
        if not folder.is_dir():
            continue
        for obj in sorted(folder.iterdir()):
            if f"{folder.name}/{obj.name}" not in used:
                obj.unlink()
        if not any(folder.iterdir()):
            folder.rmdir()


def rehull(path: Path) -> tuple[int, int]:
    """Replace an asset's collision geometry with convex hulls of its visual mesh.

    The asset keeps its total mass and the contact attributes of its first collision
    geom. Hand-built collision geoms (outside group 3) stay as visual-only geoms.

    Returns:
        The number of collision geoms before and the number of hulls after.

    Raises:
        ValueError: If the asset has no collision geometry or no visual mesh.
    """
    path = Path(path)
    points, faces, _ = body_vertices(path)
    root = ET.parse(path).getroot()
    body = root.find("worldbody/body")
    mesh_folder = path.parent / meshdir(root)

    solid = _colliders(path, root)
    mass = _collider_mass(path, solid)
    contact_attrs = {k: solid[0].get(k) for k in CONTACT_ATTRIBUTES}
    before = len(solid)

    pieces = decompose_collision(points, faces, mesh_folder / f"{path.stem}_hulls")
    for geom in solid:  # Old hulls are removed; hand-built colliders become visual only.
        parent = next(b for b in root.iter() if geom in list(b))
        if geom.get("group") == "3":
            parent.remove(geom)
        else:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("mass", "0")
            for attr in (*CONTACT_ATTRIBUTES, "density"):
                geom.attrib.pop(attr, None)
    asset = root.find("asset")
    for old in [m for m in asset.findall("mesh") if "_hull_" in (m.get("name") or "")]:
        asset.remove(old)
    shares = distribute_mass(pieces, mass)
    _add_hulls(asset, body, path.stem, f"{path.stem}_hulls", pieces, shares, contact_attrs)
    write(path, root)
    return before, len(pieces)
