"""Geometry queries on a compiled MuJoCo model: hulls, footprints, heights and static contacts."""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

TOUCH = 0.0001  # distance at or below which two geoms count as touching, in metres


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Unit vectors spread evenly over the sphere (Fibonacci lattice)."""
    i = np.arange(count) + 0.5
    phi, z = np.pi * (1 + 5**0.5) * i, 1 - 2 * i / count
    r = np.sqrt(1 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


DIRECTIONS = _fibonacci_sphere(1024)  # unit directions used to sample a hull's extreme points
_BOX_CORNERS = np.array(
    [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float
)  # corners of the unit cube [-1, 1]^3
_HULL_CACHE: dict[tuple, tuple] = {}  # (vertex count, vertex hash) -> (half-spaces, corners)


def hull(model: mujoco.MjModel, mesh: int) -> tuple[np.ndarray | None, np.ndarray]:
    """Convex hull of a mesh in its own frame, as (half-spaces, corners); cached per mesh.

    A mesh too flat to hull has no half-spaces, and every vertex is a corner.
    """
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start : start + count]
    key = (int(count), hashlib.blake2b(vertices.tobytes(), digest_size=16).digest())
    if key not in _HULL_CACHE:
        try:
            found = ConvexHull(vertices)
            _HULL_CACHE[key] = (found.equations, vertices[found.vertices])
        except Exception:
            _HULL_CACHE[key] = (None, vertices)
    return _HULL_CACHE[key]


def under(points: np.ndarray, at, pad: float) -> bool:
    """Whether an (x, y) point lies within the horizontal footprint of `points`, grown by `pad`."""
    at = np.asarray(at, dtype=float)
    lo, hi = points.min(0), points.max(0)
    if not (np.all(at >= lo[:2] - pad) and np.all(at <= hi[:2] + pad)):
        return False
    flat = np.unique(np.round(points[:, :2], 6), axis=0)
    if len(flat) < 3:
        return True
    try:
        footprint = ConvexHull(flat)
    except Exception:
        return True

    # Convex-hull normals point outward.
    return bool(np.all(footprint.equations[:, :2] @ at + footprint.equations[:, 2] <= pad))


def points(model, data, geom: int) -> np.ndarray:
    """World-frame mesh vertices of a geom, or the corners of its bounding box."""
    rotation, origin = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return model.mesh_vert[start : start + count] @ rotation.T + origin
    params = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        half_extent = np.array([params[0], params[0], params[1]])
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        half_extent = np.array([params[0]] * 3)
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half_extent = np.array([params[0], params[0], params[1] + params[0]])
    else:
        half_extent = params[:3]
    return (_BOX_CORNERS * half_extent) @ rotation.T + origin


def height(model, data, geom: int, at, top: float) -> tuple[float, bool]:
    """Height of a geom's surface under an (x, y) point, and whether a downward ray hit it.

    The ray starts 1 mm above `top`; on a miss, `top` is returned.
    """
    origin = np.array([at[0], at[1], top + 0.001])
    down = np.array([0.0, 0.0, -1.0])
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        dist = mujoco.mj_rayMesh(model, data, geom, origin, down)
    else:
        dist = mujoco.mju_rayGeom(
            data.geom_xpos[geom],
            data.geom_xmat[geom],
            model.geom_size[geom],
            origin,
            down,
            int(model.geom_type[geom]),
        )
    return (float(origin[2] - dist), True) if dist >= 0 else (top, False)


def collides(model, geom: int) -> bool:
    """Whether a geom takes part in collisions."""
    return bool(model.geom_contype[geom] or model.geom_conaffinity[geom])


def geoms_under(model, root: int) -> list[int]:
    """Every geom of a body and its descendants, excluding region geoms."""
    bodies, frontier = {root}, [root]
    while frontier:
        parent = frontier.pop()
        for child in range(parent + 1, model.nbody):
            if model.body_parentid[child] == parent and child not in bodies:
                bodies.add(child)
                frontier.append(child)
    geoms = []
    for body in sorted(bodies):
        start = model.body_geomadr[body]
        geoms += [
            g
            for g in range(start, start + model.body_geomnum[body])
            if "place:" not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        ]
    return geoms


def _flush_contact(model, data, support: int, solid: int):
    """Contact point of a geom lying flush on top of an axis-aligned box, or None.

    `mj_geomDistance` gives no well-defined closest point for flush faces, so this case
    is handled separately.
    """
    if model.geom_type[support] != mujoco.mjtGeom.mjGEOM_BOX:
        return None
    if not np.allclose(data.geom_xmat[support].reshape(3, 3), np.eye(3), atol=1e-8):
        return None
    corners = points(model, data, solid)
    centre, half = data.geom_xpos[support], model.geom_size[support]
    top = centre[2] + half[2]
    if abs(float(corners[:, 2].min()) - top) > TOUCH:
        return None
    foot = corners[corners[:, 2] <= top + TOUCH]
    lo = np.maximum(foot[:, :2].min(0), centre[:2] - half[:2])
    hi = np.minimum(foot[:, :2].max(0), centre[:2] + half[:2])
    if np.any(lo > hi):
        return None
    where = (lo + hi) / 2
    return where if under(foot, where, TOUCH) else None


def static_contacts(model, data, geoms_a, geoms_b):
    """Yield contact points between static geoms of two sets.

    MuJoCo does not report contacts between static bodies, so these are computed directly.
    """

    def is_static(geom):
        return model.body_weldid[model.geom_bodyid[geom]] == 0 and collides(model, geom)

    segment = np.zeros(6)
    for g1 in filter(is_static, geoms_a):
        for g2 in filter(is_static, geoms_b):
            if g1 == g2:
                continue
            for support, solid in ((g1, g2), (g2, g1)):
                where = _flush_contact(model, data, support, solid)
                if where is not None:
                    yield where
            if mujoco.mj_geomDistance(model, data, g1, g2, 0.001, segment) <= TOUCH:
                yield (segment[:3] + segment[3:]) / 2
