"""Height-map analysis of mesh surfaces, and the mesh reading and writing it shares."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from ..fetching import asset_dirs

# Placement region detection.
GRID_RESOLUTION = 64  # height-map cells along each horizontal axis
MIN_CAVITY_DEPTH = 0.012  # metres below the rim
MIN_WALL_COVERAGE = 0.9  # fraction of horizontal rays from a cavity that must hit a wall
MIN_CAVITY_VOLUME_RATIO = 0.15  # cavity volume as a fraction of the volume under the rim
SUPPORT_REGION_HEIGHT = 0.10  # metres a `top` region extends above the asset


def meshdir(root) -> str:
    """An asset's mesh directory relative to its file (`meshdir`, else `assetdir`, else ".")."""
    return asset_dirs(root.find("compiler"))["mesh"] or "."


class SurfaceGrid:
    """Height map of a mesh, sampled with downward rays on a regular XY grid."""

    def __init__(self, points: np.ndarray, triangles: np.ndarray, across: int = GRID_RESOLUTION):
        spec = mujoco.MjSpec()
        mesh = spec.add_mesh(name="shape")
        mesh.uservert = points.ravel().astype(np.float32).tolist()
        mesh.userface = triangles.ravel().astype(np.int32).tolist()
        spec.worldbody.add_body(name="object").add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname="shape"
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.bounds_min, self.bounds_max = points.min(0), points.max(0)
        self.xs = self.bounds_min[0] + (np.arange(across) + 0.5) / across * (
            self.bounds_max[0] - self.bounds_min[0]
        )
        self.ys = self.bounds_min[1] + (np.arange(across) + 0.5) / across * (
            self.bounds_max[1] - self.bounds_min[1]
        )
        top = float(self.bounds_max[2]) + 0.05
        self.height = np.full((across, across), np.nan)  # NaN where the ray misses
        hit, down = np.zeros(1, dtype=np.int32), np.array([0.0, 0.0, -1.0])
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                dist = mujoco.mj_ray(
                    self.model, self.data, np.array([x, y, top]), down, None, 1, -1, hit
                )
                if hit[0] >= 0:
                    self.height[i, j] = top - dist
        self.height_span = float(self.bounds_max[2] - self.bounds_min[2])
        self.max_height = float(self.bounds_max[2])
        self.cell_size = np.array([self.xs[1] - self.xs[0], self.ys[1] - self.ys[0]])

    def ray_intersects(self, start, heading) -> bool:
        """Whether a ray from `start` along `heading` hits the mesh."""
        hit = np.zeros(1, dtype=np.int32)
        mujoco.mj_ray(
            self.model,
            self.data,
            np.asarray(start, dtype=float),
            np.asarray(heading, dtype=float),
            None,
            1,
            -1,
            hit,
        )
        return bool(hit[0] >= 0)


def find_cavity(surface: SurfaceGrid):
    """An enclosed depression as `(cell mask, floor, rim)`, or None if there is none."""
    if surface.height_span < 0.02:
        return None
    heights = surface.height[np.isfinite(surface.height)]
    if not heights.size:
        return None
    tried: list[float] = []
    for rim in [surface.max_height] + [
        float(np.percentile(heights, p)) for p in (95, 90, 85, 80, 70, 60)
    ]:
        if rim - surface.bounds_min[2] < 0.02 or any(abs(rim - t) < 0.005 for t in tried):
            continue
        tried.append(rim)
        found = _cavity_below(surface, rim)
        if found is not None:
            return found
    return None


def _cavity_below(surface: SurfaceGrid, rim: float):
    """The largest enclosed depression below a rim height, or None."""
    depth = rim - float(surface.bounds_min[2])
    surface_mask = np.isfinite(surface.height) & (surface.height <= rim + 1e-6)
    low = surface_mask & (surface.height < rim - max(MIN_CAVITY_DEPTH, 0.2 * depth))
    labels, count = ndimage.label(low)
    best, best_volume = None, 0.0
    for k in range(1, count + 1):
        region_mask = labels == k
        volume = float((rim - surface.height[region_mask]).sum())
        if volume <= best_volume or region_mask.sum() < 6:
            continue
        floor = float(np.percentile(surface.height[region_mask], 2))
        i, j = np.unravel_index(
            np.argmax(ndimage.distance_transform_cdt(region_mask)), region_mask.shape
        )
        ray_origin = np.array([surface.xs[i], surface.ys[j], floor + 0.3 * (rim - floor)])
        if not surface.height[i, j] <= ray_origin[2]:
            continue  # Ray origin lies below the surface.
        angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
        if sum(
            surface.ray_intersects(ray_origin, [np.cos(t), np.sin(t), 0.0]) for t in angles
        ) < MIN_WALL_COVERAGE * len(angles):
            continue  # The region is not enclosed.
        best, best_volume = region_mask, volume
    if best is None or best_volume / (surface_mask.sum() * depth) < MIN_CAVITY_VOLUME_RATIO:
        return None
    floor = float(np.percentile(surface.height[best], 2))
    if rim - floor < 0.015:
        return None
    deep = best & (surface.height < floor + 0.5 * (rim - floor))
    columns, rows = np.nonzero(deep)
    span = (
        np.array([surface.xs[columns.max()], surface.ys[rows.max()]])
        - np.array([surface.xs[columns.min()], surface.ys[rows.min()]])
        + surface.cell_size
    )
    if np.any(span < 0.02):
        return None
    return best, floor, rim


def interior_mesh(surface: SurfaceGrid, cavity) -> tuple[np.ndarray, np.ndarray]:
    """Convex `inside` region between the cavity surface and its rim."""
    cells, floor, rim = cavity
    i, j = np.nonzero(cells)
    x, y = surface.xs[i], surface.ys[j]
    heights = np.maximum(surface.height[i, j], floor)
    return convex_mesh(
        np.vstack([np.column_stack([x, y, heights]), np.column_stack([x, y, np.full_like(x, rim)])])
    )


def find_support_height(surface: SurfaceGrid) -> float | None:
    """Height of the dominant horizontal support surface near the top, or None."""
    if surface.height_span < 0.005:
        return surface.max_height  # A flat object is all top surface.
    heights = np.sort(surface.height[np.isfinite(surface.height)])
    if heights.size < 8:
        return None
    # Only the upper half counts, so a lower face does not win over the top.
    upper_heights = heights[heights > surface.max_height - 0.5 * surface.height_span]
    window = max(0.003, 0.05 * surface.height_span)
    best_count, best_height = 0, None
    for k, height in enumerate(upper_heights):
        count = np.searchsorted(upper_heights, height + window, side="right") - k
        if count > best_count:
            best_count, best_height = count, height
    if best_count / heights.size <= 0.45:
        return None
    return float(best_height + window / 2)


def support_mesh(surface: SurfaceGrid, cavity) -> tuple[np.ndarray, np.ndarray] | None:
    """Convex `top` region above the support surface, or None if there is none."""
    high = surface.height > surface.max_height - max(0.005, 0.1 * surface.height_span)
    floor = surface.max_height
    level = find_support_height(surface) if cavity is None else None
    if level is None and cavity is None:
        return None  # No flat surface to place objects on.
    if level is not None:
        high |= np.abs(surface.height - level) <= max(0.003, 0.05 * surface.height_span)
        floor = min(level, surface.max_height)
    i, j = np.nonzero(np.isfinite(surface.height) & high)
    if i.size < 3:
        return None
    x, y = surface.xs[i], surface.ys[j]
    try:
        return convex_mesh(
            np.vstack(
                [
                    np.column_stack([x, y, np.full_like(x, floor)]),
                    np.column_stack(
                        [x, y, np.full_like(x, surface.max_height + SUPPORT_REGION_HEIGHT)]
                    ),
                ]
            )
        )
    except QhullError:
        return None


def convex_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull vertices of `points` and outward-facing triangles."""
    hull = ConvexHull(points)
    new_index = {old: new for new, old in enumerate(hull.vertices)}
    faces = []
    for corners, plane in zip(hull.simplices, hull.equations):
        a, b, c = points[corners]
        if np.dot(np.cross(b - a, c - a), plane[:3]) < 0:
            corners = corners[[0, 2, 1]]
        faces.append([new_index[k] for k in corners])
    return points[hull.vertices], np.asarray(faces)


def write_obj(
    path: Path, points: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None = None, note: str = ""
):
    """Write a Wavefront `.obj` file, with UVs if given and `note` as a header comment."""
    with path.open("w") as f:
        if note:
            f.write(f"# {note}\n")
        for x, y, z in points:
            f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
        if uvs is not None:
            for u, v in uvs:
                f.write(f"vt {u:.5f} {v:.5f}\n")
        for face in faces:
            f.write(
                "f "
                + " ".join(f"{k + 1}/{k + 1}" if uvs is not None else str(k + 1) for k in face)
                + "\n"
            )


def body_vertices(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Vertices, triangles and geom names of an asset's visual meshes, in the body frame.

    MuJoCo compiles the asset, so mesh scale, rotation and nested bodies are already
    applied.
    """
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = 1  # The asset's single top-level body.
    origin, rotation = data.xpos[body], data.xmat[body].reshape(3, 3)
    points, faces, names = [], [], []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or "place:" in name:
            continue
        if model.geom_group[gid] == 3:  # A collision hull, not visual geometry.
            continue
        mesh_id = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[start : start + count]
        world = local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        face_start, face_count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        faces.append(
            model.mesh_face[face_start : face_start + face_count] + sum(len(p) for p in points)
        )
        points.append((world - origin) @ rotation)
        names.append(name)
    if not points:
        raise ValueError(f"{path.name}: no visual mesh to decompose")
    return np.vstack(points), np.vstack(faces).astype(np.int32), names
