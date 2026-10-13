"""USD export of a MuJoCo runtime for the RTX renderer: geometry, materials, lights, camera."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .render import camera_view
from .sky import latlong

# Prim paths of the render product that ovrtx reads and of the camera it renders from.
PRODUCT = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"
CAMERA = "/World/RtxCamera"


class Scene:
    """USD stage of a runtime: exported once, then updated, lit and aimed for each frame.

    Args:
        runtime: The MuJoCo runtime to export.
        directory: Directory for the exported stage, its assets and converted skies.
    """

    def __init__(self, runtime, directory: Path):
        from mujoco.usd.exporter import USDExporter
        from pxr import Tf

        # MuJoCo geom names may contain characters that are invalid in USD prim names.
        class Exporter(USDExporter):
            def _get_geom_name(self, geom):
                return Tf.MakeValidIdentifier(super()._get_geom_name(geom))

        self.runtime = runtime
        self._directory = directory
        texture = ET.fromstring(runtime.xml).find("asset/texture[@type='skybox']")
        self.sky = texture.get("file") if texture is not None else None
        self._skies = {}
        self._baked_refs = None
        self._animated = []
        self.export = Exporter(
            model=runtime.model,
            width=min(480, runtime.model.vis.global_.offwidth),
            height=min(480, runtime.model.vis.global_.offheight),
            camera_names=[],
            light_intensity=0,
            output_directory="scene",
            output_directory_root=str(directory),
            verbose=False,
        )

    def _pose(self, options):
        """Update the stage to the current pose, as static values rather than time samples.

        Prims are post-processed once per set of exported geoms and lights: asset
        paths are made absolute and mesh normals are set from the model.
        """
        from pxr import Sdf, UsdGeom

        self.export.update_scene(self.runtime.data, scene_option=options)
        stage = self.export.stage
        time = self.export.updates - 1
        refs = (len(self.export.geom_refs), len(self.export.usd_lights))
        if refs != self._baked_refs:
            animated = {attr.GetPath(): attr for attr in self._animated}
            for prim in stage.Traverse():
                for attr in prim.GetAttributes():
                    if attr.GetNumTimeSamples() or attr.GetName() == "visibility":
                        animated[attr.GetPath()] = attr
                    if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                        value = attr.Get()
                        if value and value.path.startswith("../assets/"):
                            attr.Set(
                                Sdf.AssetPath(
                                    str(self._directory / "scene/assets" / Path(value.path).name)
                                )
                            )
            for ref in self.export.geom_refs.values():
                if not hasattr(ref, "dataid"):
                    continue
                mesh = ref.usd_mesh
                mesh.GetSubdivisionSchemeAttr().Set("none")
                model = self.runtime.model
                mesh_id = ref.dataid
                face_start = model.mesh_faceadr[mesh_id]
                face_end = face_start + model.mesh_facenum[mesh_id]
                normal_ids = model.mesh_facenormal[face_start:face_end].ravel()
                if len(normal_ids) and np.all(normal_ids >= 0):
                    normals = model.mesh_normal[model.mesh_normaladr[mesh_id] + normal_ids]
                    mesh.GetNormalsAttr().Set(normals)
                    mesh.SetNormalsInterpolation("faceVarying")
            self._animated = list(animated.values())
            self._baked_refs = refs
        # Replace the exporter's latest time sample with a static value.
        for attr in self._animated:
            if attr.GetNumTimeSamples():
                value = attr.Get(time)
                attr.Clear()
                attr.Set(value)
        for ref in self.export.geom_refs.values():
            ref.usd_xform.GetVisibilityAttr().Set(
                "inherited" if ref.last_visible_frame == time else "invisible"
            )
        # Copy the current geom colours and materials into the shaders.
        for geom in self.export.scene.geoms[: self.export.scene.ngeom]:
            name = self.export._get_geom_name(geom)
            shader = stage.GetPrimAtPath(f"/World/_materials/Material_{name}/Principled_BSDF")
            if shader:
                color = shader.GetAttribute("inputs:diffuseColor")
                if not color.HasAuthoredConnections():
                    color.Set(tuple(float(channel) for channel in geom.rgba[:3]))
                else:
                    texture = stage.GetPrimAtPath(
                        f"/World/_materials/Material_{name}/Image_Texture"
                    )
                    texture.CreateAttribute("inputs:scale", Sdf.ValueTypeNames.Float4).Set(
                        (*map(float, geom.rgba[:3]), 1.0)
                    )
                shader.GetAttribute("inputs:opacity").Set(float(geom.rgba[3]))
                self._material(geom, shader)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        return stage

    def _material(self, geom, prim):
        """Set a geom shader's metallic and roughness, and bind its PBR texture maps."""
        from pxr import Sdf, UsdGeom, UsdShade, Vt

        model = self.runtime.model
        shader = UsdShade.Shader(prim)
        material_id = geom.matid
        has_material = material_id >= 0
        for name, value in (
            ("metallic", float(model.mat_metallic[material_id]) if has_material else 0.0),
            (
                "roughness",
                float(model.mat_roughness[material_id]) if has_material else 1.0 - geom.shininess,
            ),
        ):
            scalar = shader.CreateInput(name, Sdf.ValueTypeNames.Float)
            scalar.DisconnectSource()
            scalar.Set(float(value))
        if not has_material:
            return
        stage, path = prim.GetStage(), prim.GetParent().GetPath()
        for name in ("normal", "roughness", "metallic", "occlusion"):
            role = getattr(mujoco.mjtTextureRole, f"mjTEXROLE_{name.upper()}")
            texture_id = int(model.mat_texid[material_id, role])
            if texture_id < 0:
                # Without a texture, normal and occlusion inputs fall back to neutral values.
                existing = shader.GetInput(name)
                if existing and name in ("normal", "occlusion"):
                    existing.DisconnectSource()
                    existing.Set((0, 0, 1) if name == "normal" else 1.0)
                continue
            uv = UsdShade.Shader.Define(stage, path.AppendChild("uvmap"))
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("UVMap")
            uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            ref = self.export.geom_refs[self.export._get_geom_name(geom)]
            if not UsdGeom.PrimvarsAPI(ref.usd_mesh).HasPrimvar("UVMap"):
                coords, indices = ref._get_uv_geometry()
                texcoords = UsdGeom.PrimvarsAPI(ref.usd_mesh).CreatePrimvar(
                    "UVMap", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
                texcoords.Set(coords)
                texcoords.SetIndices(Vt.IntArray(np.asarray(indices).ravel().tolist()))
            texture = UsdShade.Shader.Define(stage, path.AppendChild(f"{name}_Texture"))
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(
                    str(
                        self._directory
                        / "scene/assets"
                        / Path(self.export.texture_files[texture_id]).name
                    )
                )
            )
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv.ConnectableAPI(), "result"
            )
            if name == "normal":
                # Map normal-map texels from [0, 1] to [-1, 1].
                texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2, 2, 2, 1))
                texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1, -1, -1, 0))
                channel, output_type, input_type = (
                    "rgb",
                    Sdf.ValueTypeNames.Float3,
                    Sdf.ValueTypeNames.Normal3f,
                )
            else:
                channel, output_type, input_type = (
                    "r",
                    Sdf.ValueTypeNames.Float,
                    Sdf.ValueTypeNames.Float,
                )
            texture.CreateOutput(channel, output_type)
            shader.CreateInput(name, input_type).ConnectToSource(texture.ConnectableAPI(), channel)

    def _aim(self, stage, camera, options, azimuth, elevation, focus, width, height):
        """Define the USD camera to match MuJoCo's view from `camera`."""
        from pxr import Gf, UsdGeom

        runtime = self.runtime
        view = camera_view(runtime, camera, azimuth, elevation, focus)
        mujoco.mjv_updateScene(
            runtime.model,
            runtime.data,
            options,
            None,
            view,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.export.scene,
        )
        cam = mujoco.mjv_averageCamera(*self.export.scene.camera)
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((np.cross(cam.forward, cam.up), cam.up, -cam.forward))
        transform[:3, 3] = cam.pos
        lens = UsdGeom.Camera.Define(stage, CAMERA)
        xform = UsdGeom.Xformable(lens)
        xform.ClearXformOpOrder()
        op = xform.GetPrim().GetAttribute("xformOp:transform")
        if not op:
            op = xform.AddTransformOp().GetAttr()
        else:
            xform.SetXformOpOrder([UsdGeom.XformOp(op)])
        op.Set(Gf.Matrix4d(transform.T.tolist()))
        # USD apertures are in tenths of a scene unit; MuJoCo gives the frustum at the
        # near plane, so a perspective frustum is scaled out to the focal length.
        near = float(cam.frustum_near)
        focal = 20.0
        scale = 10.0 if cam.orthographic else focal / near
        tall = float(cam.frustum_top - cam.frustum_bottom) * scale
        wide = tall * width / height
        lens.CreateProjectionAttr("orthographic" if cam.orthographic else "perspective")
        lens.CreateFocalLengthAttr(focal)
        lens.CreateVerticalApertureAttr(tall)
        lens.CreateHorizontalApertureAttr(wide)
        lens.CreateHorizontalApertureOffsetAttr(float(cam.frustum_center) * scale)
        lens.CreateVerticalApertureOffsetAttr(
            float(cam.frustum_top + cam.frustum_bottom) * scale / 2
        )
        lens.CreateClippingRangeAttr(Gf.Vec2f(near, float(cam.frustum_far)))

    def stage(self, camera, options, azimuth, elevation, focus, skybox, width, height):
        """Return the USD stage posed, lit and aimed for one frame.

        Args:
            camera: Model camera name; None or an unknown name uses an orbit view.
            options: `MjvOption` for the MuJoCo scene update.
            azimuth: Orbit-view azimuth, in degrees.
            elevation: Orbit-view elevation, in degrees.
            focus: Object name or names the orbit view frames.
            skybox: Sky image path for the dome light, or a false value for none.
            width: Render product width, in pixels.
            height: Render product height, in pixels.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        stage = self._pose(options)
        dome = UsdLux.DomeLight.Define(stage, "/World/RtxDome")
        dome.CreateIntensityAttr(300 if skybox else 180)
        dome.CreateEnableColorTemperatureAttr(not bool(skybox))
        dome.CreateColorTemperatureAttr(7500)
        texture = ""
        if skybox:
            source = Path(skybox).resolve()
            if source not in self._skies:
                destination = self._directory / f"sky_{len(self._skies)}.png"
                latlong(source, destination)
                self._skies[source] = destination
            texture = str(self._skies[source])
        dome.CreateTextureFileAttr(Sdf.AssetPath(texture))
        dome.CreateTextureFormatAttr("latlong")
        # A rectangular area light above the table, aimed at it like a window.
        window = UsdLux.RectLight.Define(stage, "/World/DaylightWindow")
        window.CreateWidthAttr(1.6)
        window.CreateHeightAttr(1.2)
        window.CreateIntensityAttr(4500)
        window.CreateEnableColorTemperatureAttr(True)
        window.CreateColorTemperatureAttr(5600)
        if not window.GetPrim().GetAttribute("xformOp:transform"):
            eye = np.array([-0.05, -1.25, 2.1])
            forward = np.array([0.6, 0, 0.8]) - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 0, 1])
            right /= np.linalg.norm(right)
            transform = np.eye(4)
            transform[:3, :3] = np.column_stack((right, np.cross(right, forward), -forward))
            transform[:3, 3] = eye
            UsdGeom.Xformable(window).AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
        self._aim(stage, camera, options, azimuth, elevation, focus, width, height)
        for path in ("/Render", "/Render/OmniverseKit", "/Render/OmniverseKit/HydraTextures"):
            stage.DefinePrim(path)
        product = stage.DefinePrim(PRODUCT, "RenderProduct")
        product.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(
                prependedItems=[
                    "OmniRtxSettingsCommonAdvancedAPI_1",
                    "OmniRtxSettingsPtAdvancedAPI_1",
                ]
            ),
        )
        product.CreateRelationship("camera").SetTargets([Sdf.Path(CAMERA)])
        product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2, custom=False).Set(
            Gf.Vec2i(width, height)
        )
        color = stage.DefinePrim("/Render/Vars/LdrColor", "RenderVar")
        color.CreateAttribute("sourceName", Sdf.ValueTypeNames.String, custom=False).Set("LdrColor")
        product.CreateRelationship("orderedVars").SetTargets([color.GetPath()])
        product.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set(
            "RealTimePathTracing"
        )
        return stage
