"""Path-traced rendering with NVIDIA RTX (ovrtx) from a USD export of the scene."""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from pathlib import Path

import mujoco
import numpy as np

from .render import AZIMUTH, DEFAULT_CAMERA, ELEVATION, scene_options
from .usd import CAMERA, PRODUCT, Scene

SAMPLES = 4  # path-tracing steps accumulated per frame


def _staged(version: str) -> bool:
    """Whether this ovrtx version renders through an OVStage (0.4 and later).

    Only the leading digits of each version part are read, so `0.4rc1` parses as 0.4.
    """
    parts = [re.match(r"\d+", part) for part in version.split(".")[:2]]
    return tuple(int(part.group()) if part else 0 for part in parts) >= (0, 4)


def _engine():
    """A new RTX engine on the GPUs listed in `TABLETASK_RTX_GPU`, or on all GPUs."""
    import ovrtx

    return ovrtx.Renderer(
        config=ovrtx.RendererConfig(active_cuda_gpus=os.environ.get("TABLETASK_RTX_GPU"))
    )


def _unavailable_reason() -> str | None:
    """Why RTX rendering is unavailable, or None if its dependencies load."""
    try:
        import ovrtx
        from mujoco.usd.exporter import USDExporter  # noqa: F401
        from pxr import UsdGeom  # noqa: F401

        if _staged(ovrtx.__version__):
            import ovstage  # noqa: F401
    except ImportError as error:
        return f"{error}; install ovrtx ovstage usd-core"
    except OSError as error:
        return f"the RTX libraries could not load ({error})"
    return None


class Session:
    """Context manager that shares one RTX engine across renderers used one after another.

    The engine is created on first use and destroyed on exit.
    """

    def __init__(self):
        self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.engine is not None:
            self.engine.destroy()
            self.engine = None

    def get(self):
        """The shared RTX engine, created on the first call."""
        if self.engine is None:
            self.engine = _engine()
        return self.engine


class Renderer:
    """Path-traced renderer for one runtime.

    Args:
        runtime: The MuJoCo runtime to render.
        width: Default image width, in pixels.
        height: Default image height, in pixels.
        session: Optional `Session` whose engine is shared; without one the renderer
            creates and owns its own engine.
    """

    def __init__(self, runtime, width: int = 960, height: int = 540, session=None):
        import ovrtx

        self.runtime, self.width, self.height = runtime, width, height
        self._temporary = tempfile.TemporaryDirectory(prefix="tabletask-rtx-")
        self._rtx = self._live = None
        self._owns_engine = session is None
        self._ordinal = 0
        self._paths = None
        self._queries = {}
        self._values = {}
        self._scene_key = None
        self._xforms = []
        try:
            self._scene = Scene(runtime, Path(self._temporary.name))
            self._rtx = session.get() if session is not None else _engine()
            if _staged(ovrtx.__version__):
                import ovstage

                self._live = ovstage.Stage("tabletask")
                self._rtx.attach_ovstage(self._live)
        except Exception:
            self.close()
            raise

    def _query(self, paths):
        """A cached OVStage query for a list of prim paths."""
        key = tuple(paths)
        if key not in self._queries:
            path_list = self._paths.create_path_list_from_strings(paths)
            self._queries[key] = (self._live.query_from_path_list(path_list), path_list)
        return self._queries[key][0]

    def _sync(self, stage, regions, skybox):
        """Update the live OVStage from the USD stage.

        The whole stage is reloaded when anything other than transforms and camera
        parameters changed; otherwise only those attributes are written.
        """
        import ovstage
        from pxr import UsdGeom

        model = self.runtime.model
        key = (
            str(regions),
            str(skybox),
            *(
                getattr(model, name).tobytes()
                for name in (
                    "geom_rgba",
                    "geom_size",
                    "mat_rgba",
                    "mat_metallic",
                    "mat_roughness",
                    "mat_texid",
                    "mat_texrepeat",
                    "mat_shininess",
                )
            ),
            tuple(
                (str(ref.usd_xform.GetPath()), ref.usd_xform.GetVisibilityAttr().Get())
                for ref in self._scene.export.geom_refs.values()
            ),
            UsdGeom.Camera(stage.GetPrimAtPath(CAMERA)).GetProjectionAttr().Get(),
        )
        self._ordinal += 1
        if key != self._scene_key:
            self._release_queries()
            ovstage.population.open_usd_from_string(
                self._live,
                stage.GetRootLayer().ExportToString(),
                ordinal=self._ordinal,
                time_code=0.0,
            )
            self._paths = ovstage.PathDictionary(self._live)
            self._xforms = [
                UsdGeom.Xformable(prim)
                for prim in stage.Traverse()
                if prim.HasAttribute("xformOpOrder")
            ]
            self._scene_key = key
        else:
            # OVRTX consumes local transforms as omni:xform, not USD xformOp attributes.
            transforms = np.array(
                [xf.GetLocalTransformation() for xf in self._xforms], dtype=np.float64
            )
            self._write([str(xf.GetPath()) for xf in self._xforms], "omni:xform", transforms)
            lens = stage.GetPrimAtPath(CAMERA)
            for name in (
                "focalLength",
                "verticalAperture",
                "horizontalAperture",
                "horizontalApertureOffset",
                "verticalApertureOffset",
                "clippingRange",
            ):
                value = np.asarray([lens.GetAttribute(name).Get()], dtype=np.float32)
                self._write([CAMERA], name, value)
            self._write(
                [PRODUCT], "resolution", np.array([[self.width, self.height]], dtype=np.int32)
            )
        self._live.advance_write_floor(self._ordinal, ovstage.Scope.ALL).wait()

    def _write(self, paths, name, value):
        """Write an attribute for all paths if any value changed since the last write.

        All rows are sent together because a partial write leaves moved prims at their
        previous transforms.
        """
        key = (tuple(paths), name)
        previous = self._values.get(key)
        if previous is not None and np.array_equal(previous, value):
            return
        self._live.write_attribute(
            self._query(paths), name, self._ordinal, value, is_array=False
        ).wait()
        self._values[key] = value

    def _release_queries(self):
        """Release the cached queries, written values and path dictionary."""
        for query, paths in self._queries.values():
            query.release().wait()
            paths.release()
        self._queries.clear()
        self._values.clear()
        if self._paths is not None:
            self._paths.destroy()
            self._paths = None

    def frame(
        self,
        camera=DEFAULT_CAMERA,
        width=None,
        height=None,
        samples=SAMPLES,
        *,
        azimuth=AZIMUTH,
        elevation=ELEVATION,
        regions=False,
        focus=None,
        skybox=None,
    ):
        """Render one path-traced RGB frame.

        Args:
            camera: Model camera name; None or an unknown name uses an orbit view.
            width: Image width in pixels; defaults to the last width used.
            height: Image height in pixels; defaults to the last height used.
            samples: Number of path-tracing steps to accumulate.
            azimuth: Orbit-view azimuth, in degrees.
            elevation: Orbit-view elevation, in degrees.
            regions: True to highlight every region, or an iterable of region names.
            focus: Object name or names the orbit view frames.
            skybox: Sky image path; None uses the scene's own sky and False uses none.
                An explicit sky hides the floor geom.

        Returns:
            A (height, width, 3) RGB image.

        Raises:
            ValueError: If the image size or sample count is not a positive integer.
            RuntimeError: If the renderer returns no colour frame.
        """
        from ovrtx import Device

        # An explicit sky replaces the floor, so the floor is hidden while staging.
        custom_sky = skybox is not None
        if skybox is None:
            skybox = self._scene.sky

        width = self.width if width is None else width
        height = self.height if height is None else height
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (width, height, samples)
        ):
            raise ValueError("image dimensions and samples must be positive integers")
        self.width, self.height = width, height

        floor = mujoco.mj_name2id(self.runtime.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        rgba = self.runtime.model.geom_rgba[floor].copy() if floor >= 0 else None
        try:
            if skybox and custom_sky and floor >= 0:
                self.runtime.model.geom_rgba[floor, 3] = 0
            with scene_options(self.runtime, regions) as options:
                scene = self._scene.stage(
                    camera, options, azimuth, elevation, focus, skybox, width, height
                )
        finally:
            if rgba is not None:
                self.runtime.model.geom_rgba[floor] = rgba
        if self._live is not None:
            self._sync(scene, regions, skybox)
        else:
            self._rtx.open_usd_from_string(scene.GetRootLayer().ExportToString())
        for _ in range(samples):
            kwargs = {"ordinal": self._ordinal} if self._live is not None else {}
            products = self._rtx.step(render_products={PRODUCT}, delta_time=1 / 60, **kwargs)
        # The returned mapping has no .get().
        product = next((value for name, value in products.items() if name == PRODUCT), None)
        if product:
            for rendered in reversed(product.frames):
                key = next((var for var in rendered.render_vars if var.endswith("LdrColor")), None)
                if key is not None:
                    with rendered.render_vars[key].map(device=Device.CPU) as mapped:
                        pixels = np.from_dlpack(mapped).copy()
                    return pixels.reshape(self.height, self.width, 4)[:, :, :3].copy()
        raise RuntimeError("the RTX renderer returned no colour frame")

    def images(self, cameras, width, height, regions=False):
        """Render one frame per camera name, as a dict from name to image."""
        return {
            name: self.frame(self.runtime.camera(name), width, height, regions=regions)
            for name in cameras
        }

    def close(self):
        """Release the live stage, the engine if owned, and the temporary directory."""
        live, renderer = self._live, self._rtx
        self._live = self._rtx = None
        try:
            try:
                if live is not None:
                    try:
                        self._release_queries()
                        if renderer is not None:
                            renderer.detach_ovstage()
                    finally:
                        live.destroy()
            finally:
                if renderer is not None and self._owns_engine:
                    renderer.destroy()
        finally:
            self._temporary.cleanup()


def attach(runtime, width=960, height=540, session=None):
    """Create an RTX renderer for a runtime, or warn and return None if RTX is unavailable.

    Args:
        runtime: The MuJoCo runtime to render.
        width: Default image width, in pixels.
        height: Default image height, in pixels.
        session: Optional `Session` that shares its engine.

    Returns:
        A `Renderer`, or None when the caller should fall back to the rasteriser.
    """
    reason = _unavailable_reason()
    if reason is None:
        try:
            return Renderer(runtime, width, height, session)
        except Exception as error:
            message = str(error).strip()
            reason = message.splitlines()[0] if message else type(error).__name__
    warnings.warn(
        f"rendering with the rasteriser: RTX is unavailable because {reason}",
        RuntimeWarning,
        stacklevel=2,
    )
    return None
