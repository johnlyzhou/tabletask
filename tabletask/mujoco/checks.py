"""Asset checks: placing an asset on the table and reporting its facts and faults."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..language.errors import SamplingError
from ..language.format import load
from ..language.kernel import TERMS
from ..language.rotation import mat
from ..sampling import build
from .geometry import collides, geoms_under
from .runtime import SETTLE_SECONDS
from .surface import body_vertices

# Asset checks.
MAX_SETTLED_TILT = 3.0  # degrees of tilt after settling before an asset counts as fallen
MAX_ORIGIN_OFFSET = 0.003  # metres the origin may sit above the asset's lowest point
HULL_TOLERANCE = 0.0005  # metres a surface point may lie outside a hull and still count as inside
MAX_UNCOVERED_DEPTH = 0.003  # metres outside the hulls beyond which surface counts as missing
# Names of the region shapes reported by `look`; other geom types print their numeric code.
REGION_SHAPES = {
    int(geom_type): geom_type.name.removeprefix("mjGEOM_").lower()
    for geom_type in (
        mujoco.mjtGeom.mjGEOM_MESH,
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_SPHERE,
    )
}


def asset_task(path: Path) -> dict:
    """A one-object task document that places an asset on the table, unjittered."""
    path = Path(path).resolve()
    library = path.parent.parent if path.parent.name == "objects" else path.parent
    name = path.stem
    return {
        "id": name,
        "instruction": f"Look at the {name}.",
        "library": str(library),
        "appearance": {"vary": False},
        "objects": {
            name: {
                "asset": str(path),
                "sample": {"rel": "table/top", "jitter": {"xy": 0.0, "yaw": 0.0}},
            }
        },
        "start": [["on", name, "table"]],
        "goal": [["not", ["on", name, "table"]]],
    }


def study(path: Path) -> tuple[dict, list[str], object]:
    """Facts about an asset, the problems found, and the runtime it was placed in.

    The asset is placed on the table in a one-object task and left to settle.

    Returns:
        A dict of measured facts (size, mass, tilt, regions, joints, ...), a list of
        problem descriptions for the author to fix, and the runtime.
    """
    path = Path(path).resolve()
    name = path.stem
    facts = {"name": name, "file": str(path)}
    problems = _naming_problems(path, name)
    try:
        runtime = build(load(asset_task(path)), 0)
    except SamplingError as error:  # measure the last attempt and report why it failed
        runtime = error.runtime
        problems.append(f"no settled placement: {error}")
    facts.update(_measure(runtime, name))
    share, deepest, hulls = uncovered(path)
    facts["uncovered"] = [round(share, 4), round(deepest, 4)]
    problems += _problems(facts, share, deepest, hulls)
    return facts, problems, runtime


def _naming_problems(path: Path, name: str) -> list[str]:
    """A problem if the top-level bodies are not one body named after the file."""
    root = ET.parse(path).getroot()
    bodies = [b.get("name") for b in root.find("worldbody").findall("body")]
    if bodies == [name]:
        return []
    problem = (
        f"the top-level body is named {bodies!r}; "
        f"tasks expect one body named after the file, {name!r}"
    )
    return [problem]


def _measure(runtime, name: str) -> dict:
    """Size, mass, settled pose, collision hulls, regions and joints of a settled asset."""
    facts = {}
    origin, quat = runtime.pose(name)
    points = runtime.shape(name)
    local = (points - origin) @ mat(quat)
    lo, hi = local.min(0), local.max(0)
    facts["size"] = [round(float(v), 4) for v in hi - lo]
    facts["origin_above_base"] = round(float(-lo[2]), 4)
    facts["mass"] = round(float(runtime.model.body_subtreemass[runtime.body(name)]), 4)
    facts["tilt"] = round(float(TERMS["tilt"](runtime, name)), 2)
    # Standing height after settling on the table.
    facts["height"] = round(float(runtime.extent(name)[2]), 4)
    facts["rested"] = bool(runtime.rested)
    facts["hulls"] = sum(
        1 for g in geoms_under(runtime.model, runtime.body(name)) if collides(runtime.model, g)
    )
    base = origin[2] + lo[2]  # World height of the lowest point.
    facts["regions"] = _regions(runtime, name, base)
    facts["joints"] = {
        j.split("/", 1)[1]: [round(v, 4) for v in runtime.joint_range(j)]
        for j in runtime.named_joints()
        if j.startswith(f"{name}/")
    }
    return facts


def _regions(runtime, name: str, base: float) -> dict:
    """Shape, size and floor height above `base` of each of the asset's regions."""
    regions = {}
    for place in runtime.places() + (
        [f"{name}/bounds"] if runtime.region(f"{name}/bounds") is not None else []
    ):
        if place.split("/")[0] != name:
            continue
        region_lo, region_hi = runtime.aabb(place)
        geom_type = int(runtime.model.geom_type[runtime.region(place)])
        regions[place.split("/", 1)[1]] = {
            "shape": REGION_SHAPES.get(geom_type, str(geom_type)),
            "size": [round(float(v), 4) for v in region_hi - region_lo],
            "floor": round(float(region_lo[2] - base), 4),
        }
    return regions


def _problems(facts: dict, share: float, deepest: float, hulls) -> list[str]:
    """Problems found in the measured facts and in the collision coverage."""
    problems = []
    # Mesh decimation leaves shallow gaps over much of the surface; a missing part leaves
    # a deep one.
    if hulls and deepest > MAX_UNCOVERED_DEPTH and share > 0.005:
        problems.append(
            f"{share * 100:.0f}% of the surface lies outside the collision geometry, "
            f"by up to {deepest * 1000:.0f} mm; something is not covered"
        )
    if abs(facts["origin_above_base"]) > MAX_ORIGIN_OFFSET:
        problems.append(
            f"the origin sits {facts['origin_above_base'] * 1000:.0f} mm above the lowest point; "
            "an asset rests on the table at z=0"
        )
    if not facts["rested"]:
        problems.append(
            f"did not come to rest within {SETTLE_SECONDS:g} s of being set on the table"
        )
    if facts["tilt"] > MAX_SETTLED_TILT and abs(facts["height"] - facts["size"][2]) > max(
        0.005, 0.1 * facts["size"][2]
    ):
        problems.append(
            f"falls over when set down (tilts {facts['tilt']:.0f} deg, "
            f"{facts['size'][2] * 100:.1f} cm tall "
            f"as authored, {facts['height'] * 100:.1f} cm standing): check which way is up, "
            "and that it rests on its base"
        )
    if facts["mass"] <= 0:
        problems.append("has no mass")
    if facts["hulls"] == 0:
        problems.append("has no collision geometry")
    return problems


def describe(facts: dict, problems: list[str]) -> list[str]:
    """Human-readable lines for the output of `study`."""
    sx, sy, sz = facts["size"]
    lines = [
        f"{facts['name']}  {facts['file']}",
        f"  size      {sx:.3f} x {sy:.3f} x {sz:.3f} m (x, y, z)",
        f"  mass      {facts['mass']:.3f} kg, {facts['hulls']} collision hulls",
        f"  origin    {facts['origin_above_base'] * 1000:.1f} mm above the lowest point",
        (
            f"  set down  tilt {facts['tilt']:.1f} deg, "
            f"stands {facts['height'] * 100:.1f} cm tall, "
            f"{'at rest' if facts['rested'] else 'still moving'}"
        ),
    ]
    for region, info in facts["regions"].items():
        size = info["size"]
        lines.append(
            f"  region    {region:<10} {info['shape']:<9} "
            f"{size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m,"
            f" floor {info['floor'] * 1000:+.0f} mm above the base"
        )
    joints = facts["joints"].items()
    lines += [f"  joint     {joint:<10} range {lo:g}..{hi:g}" for joint, (lo, hi) in joints]
    lines += [f"  PROBLEM   {problem}" for problem in problems]
    return lines


def uncovered(path: Path, sample: int = 5000) -> tuple[float, float, int]:
    """How much of an asset's visual surface lies outside its collision hulls.

    Returns:
        The fraction of sampled visual-mesh vertices outside every hull, the largest
        distance outside in metres, and the number of hulls checked. Assets with a
        primitive collider return zeros.
    """
    path = Path(path)
    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hull_planes = []
    for gid in range(model.ngeom):
        if not collides(model, gid):
            continue
        if model.geom_dataid[gid] < 0:
            return 0.0, 0.0, 0  # Primitive colliders are chosen by hand, not fitted.
        mesh_id = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start : start + count].reshape(-1, 3)
        vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        try:
            hull_planes.append(ConvexHull(vertices).equations)
        except QhullError:
            continue
    if not hull_planes:
        return 0.0, 0.0, 0
    points, _, _ = body_vertices(path)
    if len(points) > sample:
        points = points[np.random.default_rng(0).choice(len(points), sample, replace=False)]
    outside = np.full(len(points), np.inf)
    for planes in hull_planes:
        outside = np.minimum(outside, (points @ planes[:, :3].T + planes[:, 3]).max(1))
    missed = outside > HULL_TOLERANCE
    return float(missed.mean()), float(outside.max()), len(hull_planes)
