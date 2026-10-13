"""Rasterised rendering: camera placement, scene options and single frames."""

from __future__ import annotations

import colorsys
from contextlib import contextmanager

import mujoco
import numpy as np

from .geometry import collides, geoms_under, points

# Default orbit camera: azimuth and elevation in degrees, and distance as a multiple of
# the framed extent.
AZIMUTH, ELEVATION, MARGIN = 150.0, -30.0, 1.6
DEFAULT_CAMERA = "top"  # scene camera used when none is given

# RGBA colours for collision views: the focused objects (or everything, without a focus)
# and the remaining scenery.
SUBJECT, SCENERY = (0.25, 0.70, 1.00, 0.60), (0.45, 0.45, 0.48, 0.30)
REGION_RGBA = (0.2, 0.7, 1.0, 0.25)  # translucent blue for highlighted regions
REGION_GROUP = 4  # geom group that holds region geoms


def bounds(runtime) -> tuple[np.ndarray, np.ndarray]:
    """World bounds of the task's objects and the regions they start in."""
    boxes = [runtime.aabb(name) for name in runtime.task.scene.objects]
    boxes += [
        runtime.aabb(obj.sample.rel)
        for obj in runtime.task.scene.objects.values()
        if runtime.region(obj.sample.rel) is not None
    ]
    if not boxes:
        finite = [
            g
            for g in range(runtime.model.ngeom)
            if runtime.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
        ]
        corners = [points(runtime.model, runtime.data, g) for g in finite]
        boxes = [(c.min(0), c.max(0)) for c in corners]
    return (np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0))


def orbit(
    runtime, azimuth: float = AZIMUTH, elevation: float = ELEVATION, focus=None
) -> mujoco.MjvCamera:
    """Free camera framing the whole task, or the object(s) named in `focus`."""
    if focus is None:
        lo, hi = bounds(runtime)
    else:
        boxes = [runtime.aabb(name) for name in ([focus] if isinstance(focus, str) else focus)]
        lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
    view = mujoco.MjvCamera()
    view.lookat[:] = [
        (lo[0] + hi[0]) / 2,
        (lo[1] + hi[1]) / 2,
        hi[2] if focus is None else (lo[2] + hi[2]) / 2,
    ]
    view.distance = MARGIN * float(np.linalg.norm((hi - lo)[:2] if focus is None else hi - lo))
    view.azimuth, view.elevation = azimuth, elevation
    return view


MAX_SHADOW_PER_PIXEL = 4  # shadow map texels per image pixel, beyond which detail is wasted
MIN_SHADOW_SIZE = 2048


def make_renderer(model, width: int, height: int) -> mujoco.Renderer:
    """A rasteriser for images of this size.

    The offscreen buffer and shadow map are sized for the image rather than the scene's
    defaults, which are set for large pictures: at 320x240 those would take gigabytes
    of GPU memory per renderer.
    """
    visual, quality = model.vis.global_, model.vis.quality
    visual.offwidth, visual.offheight = width, height
    quality.shadowsize = min(
        quality.shadowsize, max(MIN_SHADOW_SIZE, MAX_SHADOW_PER_PIXEL * max(width, height))
    )
    return mujoco.Renderer(model, height=height, width=width)


def camera_view(runtime, camera, azimuth=AZIMUTH, elevation=ELEVATION, focus=None):
    """The named model camera, or an `orbit` view when it is None or not in the model."""
    camera_id = (
        -1
        if camera is None
        else mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    )
    if camera_id < 0:
        return orbit(runtime, azimuth, elevation, focus)
    view = mujoco.MjvCamera()
    view.type = mujoco.mjtCamera.mjCAMERA_FIXED
    view.fixedcamid = camera_id
    return view


def hull_colours(count: int, alpha: float = 0.92) -> list[tuple]:
    """RGBA colours for `count` convex pieces, with consecutive hues far apart."""
    step = 0.618034  # golden-ratio conjugate
    return [
        (*colorsys.hsv_to_rgb((i * step) % 1.0, 0.62, 0.95 if i % 2 else 0.78), alpha)
        for i in range(count)
    ]


@contextmanager
def scene_options(runtime, regions=False, collision=False, focus=None):
    """Yield an `MjvOption` for the requested view; geom colours are restored on exit.

    Args:
        runtime: The `Runtime` to render.
        regions: True to highlight every region, or an iterable of region names.
        collision: Show collision geoms (meshes as convex hulls) instead of visual geoms.
        focus: Object name or names to colour per convex piece in the collision view.
    """
    options = mujoco.MjvOption()
    saved_rgba = runtime.model.geom_rgba.copy() if (regions or collision) else None
    try:
        if collision:
            options.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
            options.geomgroup[:] = 1  # collision hulls are in group 3, hidden by default
            options.geomgroup[REGION_GROUP] = 0  # regions do not collide
            # Focused objects get one colour per convex piece; other collision geoms are grey.
            focus_geoms = []
            for name in [focus] if isinstance(focus, str) else (focus or []):
                try:
                    focus_geoms += geoms_under(runtime.model, runtime.body(name))
                except KeyError:
                    continue
            colliding = [gid for gid in focus_geoms if collides(runtime.model, gid)]
            palette = dict(zip(colliding, hull_colours(len(colliding))))
            for gid in range(runtime.model.ngeom):
                if not collides(runtime.model, gid):
                    runtime.model.geom_rgba[gid] = (0, 0, 0, 0)
                else:
                    runtime.model.geom_rgba[gid] = palette.get(
                        gid, SUBJECT if not focus_geoms else SCENERY
                    )
        if regions:
            options.geomgroup[REGION_GROUP] = 1
            chosen = None if regions is True else {runtime.region(name) for name in regions}
            for gid in range(runtime.model.ngeom):
                if runtime.model.geom_group[gid] == REGION_GROUP and (
                    chosen is None or gid in chosen
                ):
                    runtime.model.geom_rgba[gid] = REGION_RGBA
        yield options
    finally:
        if saved_rgba is not None:
            runtime.model.geom_rgba[:] = saved_rgba


def frame(
    runtime,
    width: int = 960,
    height: int = 540,
    camera: str | None = DEFAULT_CAMERA,
    azimuth: float = AZIMUTH,
    elevation: float = ELEVATION,
    regions=False,
    collision=False,
    focus=None,
) -> np.ndarray:
    """Render the current state as an RGB array.

    Args:
        runtime: The `Runtime` to render.
        width: Image width in pixels.
        height: Image height in pixels.
        camera: Model camera name; None (or an unknown name) uses an orbit view.
        azimuth: Orbit azimuth in degrees.
        elevation: Orbit elevation in degrees.
        regions: True to paint every region translucent blue, or an iterable of region names.
        collision: Show the collision geoms, meshes as the convex hulls physics uses.
        focus: Object name or names to frame instead of the whole task.

    Returns:
        A (height, width, 3) uint8 array.
    """
    renderer = make_renderer(runtime.model, width, height)
    try:
        with scene_options(runtime, regions, collision, focus) as options:
            renderer.update_scene(
                runtime.data, camera_view(runtime, camera, azimuth, elevation, focus), options
            )
            return renderer.render()
    finally:
        renderer.close()
