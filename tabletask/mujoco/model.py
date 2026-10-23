"""Assembling a task's MuJoCo scene: the scene file, objects and robot."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .. import fetching
from ..language.rotation import euler
from ..language.schema import Object, Region, Task
from .appearance import apply_appearance
from .render import bounds

PLACE_CLASS = "place"  # MJCF default class of region geoms
QUARTER_TURN = 1e-6  # how far a rotation matrix entry may be from 0 or 1 for a quarter turn
STAGING_HEIGHT = 5.0  # height at which object bodies are created before placement, metres


def mjcf(task: Task, rng=None, robot=None) -> str:
    """The full scene for a task as MJCF text.

    Args:
        task: The task whose scene and objects to assemble.
        rng: Random generator for appearance and object scales; None leaves
            the scene's textures and colours as is and gives each scale range its middle.
        robot: Optional robot to attach to the scene.

    Returns:
        The scene as an MJCF string with absolute file paths.
    """
    scene_path = fetching.resolve(task.scene.file, task.root)
    mj = ET.parse(scene_path).getroot()
    absolutize(mj, scene_path.parent)
    asset, world = mj.find("asset"), mj.find("worldbody")
    textures = None
    if rng is not None:
        # Textures get their own stream, seeded by one draw that is always made, so the
        # placement draws after it are the same whether or not textures are available.
        textures = np.random.default_rng(int(rng.integers(2**63)))
        apply_appearance(
            mj,
            task.scene.appearance,
            textures,
            fetching.resolve(task.scene.library or ".", task.root) / "looks",
        )
    for name, obj in task.scene.objects.items():
        if obj.asset:
            path = fetching.resolve(obj.asset, task.root)
            asset.append(ET.Element("model", {"name": f"asset:{name}", "file": str(path)}))
            if obj.fixed:
                _exclude_joint_parent_contacts(mj, name, path)
        world.append(_body(name, obj))
    if robot is not None:
        robot.attach(mj)
    text = ET.tostring(mj, encoding="unicode")
    scaled = {name: obj.scale for name, obj in task.scene.objects.items() if obj.scale != 1}
    if not scaled:
        return text
    spec = mujoco.MjSpec.from_string(text)
    # Scales get their own stream, drawn from only by tasks that scale an object.
    sizes = np.random.default_rng(int(rng.integers(2**63))) if rng is not None else None
    for name, scale in scaled.items():
        scale_object(spec, name, draw_scale(scale, sizes))
    spec.compile()
    _absolute_files(spec, mj)
    return spec.to_xml()


def _absolute_files(spec: mujoco.MjSpec, mj: ET.Element) -> None:
    """Point attached models' files at absolute paths, which the spec's flat XML needs."""
    models = {m.get("name"): Path(m.get("file")) for m in mj.find("asset").findall("model")}
    folders = {}
    for attach in mj.iter("attach"):
        path = models[attach.get("model")]
        compiler = ET.parse(path).getroot().find("compiler")
        folders[attach.get("prefix")] = (path.parent, fetching.asset_dirs(compiler))
    for kind, elements in (("mesh", spec.meshes), ("texture", spec.textures)):
        for element in elements:
            prefix = element.name.split("/")[0] + "/"
            if element.file and prefix in folders and not Path(element.file).is_absolute():
                base, dirs = folders[prefix]
                element.file = str((base / dirs[kind] / element.file).resolve())


def draw_scale(scale, rng=None) -> np.ndarray:
    """An object's scale along x, y and z, drawn from its ranges (their middle without `rng`).

    Args:
        scale: A factor, a `[low, high]` range for one factor on every axis, or three
            entries, one per axis, each a factor or a `[low, high]` range.
        rng: Random generator, or None.
    """

    def pick(value) -> float:
        if not isinstance(value, list):
            return float(value)
        low, high = value
        return float(rng.uniform(low, high)) if rng is not None else (low + high) / 2

    if not isinstance(scale, list):
        return np.full(3, float(scale))
    if len(scale) == 2:
        return np.full(3, pick(scale))
    return np.array([pick(value) for value in scale])


def scale_object(spec: mujoco.MjSpec, name: str, scale: np.ndarray) -> None:
    """Scale a task object along its own x, y and z axes: meshes, shapes, frames and masses.

    Raises:
        ValueError: If an uneven scale meets a part turned by other than quarter turns, or a
            round shape it would make oval.
    """
    even = np.ptp(scale) < 1e-12
    degrees = bool(spec.compiler.degree)
    sequence = "".join(spec.compiler.eulerseq) or "xyz"
    volume = float(np.prod(scale))

    def along(rotation: np.ndarray, what: str) -> np.ndarray:
        turns = np.abs(rotation)
        if not even and not np.allclose(turns, np.round(turns), atol=QUARTER_TURN):
            raise ValueError(f"{name}: {what} is turned by other than quarter turns")
        return turns.T @ scale

    def walk(body, frame: np.ndarray) -> None:
        local = along(frame, f"body {body.name!r}")
        for geom in body.geoms:
            geom.pos = geom.pos * local
            own = along(frame @ _rotation(geom, degrees, sequence), f"geom {geom.name!r}")
            _scale_shape(spec, geom, own, even, name)
            if geom.mass > 0:
                geom.mass *= volume
        for site in body.sites:
            site.pos = site.pos * local
        for joint in body.joints:
            joint.pos = joint.pos * local
            if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                joint.range = joint.range * float(
                    np.abs(joint.axis) @ local / np.linalg.norm(joint.axis)
                )
        if body.explicitinertial:
            body.ipos = body.ipos * local
            widths = along(frame @ _matrix(body.iquat), f"body {body.name!r} inertia")
            squares = np.maximum(6 / body.mass * (body.inertia.sum() - 2 * body.inertia), 0)
            body.mass *= volume
            squares *= widths**2
            body.inertia = body.mass / 12 * (squares.sum() - squares)
        for child in body.bodies:
            child.pos = child.pos * local
            walk(child, frame @ _rotation(child, degrees, sequence))

    walk(spec.body(name), np.eye(3))


def _scale_shape(spec, geom, own: np.ndarray, even: bool, name: str) -> None:
    """Scale a geom's mesh or primitive size along its own axes."""
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = spec.mesh(geom.meshname)
        turns = np.abs(_matrix(mesh.refquat))
        if not even and not np.allclose(turns, np.round(turns), atol=QUARTER_TURN):
            raise ValueError(f"{name}: mesh {mesh.name!r} is turned by other than quarter turns")
        mesh.scale = mesh.scale * (turns.T @ own)
        return
    if not np.isnan(geom.fromto[0]):
        raise ValueError(f"{name}: geom {geom.name!r} is placed by fromto")
    size = np.array(geom.size)
    if geom.type in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        size *= own
    elif geom.type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        if not even and abs(own[0] - own[1]) > 1e-12:
            raise ValueError(f"{name}: geom {geom.name!r} would turn oval")
        size[:2] *= own[[0, 2]]
    elif geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
        if not even:
            raise ValueError(f"{name}: geom {geom.name!r} is a sphere")
        size[0] *= own[0]
    geom.size = size


def _rotation(element, degrees: bool, sequence: str) -> np.ndarray:
    """An element's rotation matrix relative to its parent, from its quat or euler angles."""
    alt = element.alt
    if alt.type == mujoco.mjtOrientation.mjORIENTATION_EULER:
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.radians(alt.euler) if degrees else alt.euler, sequence)
        return _matrix(quat)
    if alt.type == mujoco.mjtOrientation.mjORIENTATION_AXISANGLE:
        quat = np.zeros(4)
        angle = np.radians(alt.axisangle[3]) if degrees else alt.axisangle[3]
        mujoco.mju_axisAngle2Quat(
            quat, alt.axisangle[:3] / np.linalg.norm(alt.axisangle[:3]), angle
        )
        return _matrix(quat)
    return _matrix(element.quat)


def _matrix(quat) -> np.ndarray:
    """The rotation matrix of a (w, x, y, z) quaternion, which need not be normalized."""
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float) / np.linalg.norm(quat))
    return matrix.reshape(3, 3)


def _exclude_joint_parent_contacts(mj: ET.Element, name: str, path: Path) -> None:
    """Exclude contacts between each jointed body of an asset and its parent body."""
    contact = mj.find("contact")
    if contact is None:
        contact = ET.SubElement(mj, "contact")
    root = ET.parse(path).getroot()
    for parent in root.iter("body"):
        for child in parent.findall("body"):
            if child.find("joint") is not None and parent.get("name") and child.get("name"):
                ET.SubElement(
                    contact,
                    "exclude",
                    {
                        "body1": f"{name}/{parent.get('name')}",
                        "body2": f"{name}/{child.get('name')}",
                    },
                )


def absolutize(mj: ET.Element, base: Path) -> None:
    """Make every file path in the MJCF absolute and drop the compiler's asset dirs."""
    compiler = mj.find("compiler")
    folder = fetching.asset_dirs(compiler)
    for element in mj.iter():
        for key, value in list(element.attrib.items()):
            if key.startswith("file") and value and not Path(value).is_absolute():
                element.set(key, str((base / folder.get(element.tag, "") / value).resolve()))
    if compiler is not None:
        for key in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib.pop(key, None)


def _body(name: str, obj: Object) -> ET.Element:
    """The world body for a task object: its joint, attached asset and region geoms."""
    body = ET.Element("body", {"name": name, "pos": f"0 0 {STAGING_HEIGHT:g}"})
    if not obj.fixed:
        ET.SubElement(body, "freejoint")
    if obj.asset:
        # Near-zero inertial: the wrapper body adds no mass to the attached asset.
        ET.SubElement(
            body, "inertial", {"pos": "0 0 0", "mass": "1e-6", "diaginertia": "1e-9 1e-9 1e-9"}
        )
        ET.SubElement(body, "attach", {"model": f"asset:{name}", "prefix": f"{name}/"})
    for region_name, region in obj.regions.items():
        if region.grid is None:
            ET.SubElement(body, "geom", _region(f"{name}/place:{region_name}", region))
            continue
        for cell, offset in region.grid.cells():
            geom = ET.SubElement(body, "geom", _region(f"{name}/place:{cell}", region))
            geom.set("pos", " ".join(f"{a + b:g}" for a, b in zip(region.pos, offset)))
    return body


def half(shape: str, s: list[float]) -> str:
    """MuJoCo's size attribute for a shape given its full extents in metres."""
    if shape == "box":
        return f"{s[0] / 2:g} {s[1] / 2:g} {s[2] / 2:g}"
    if shape == "cylinder":
        return f"{s[0] / 2:g} {s[1] / 2:g}"
    return f"{s[0] / 2:g}"


def _region(name: str, region: Region) -> dict[str, str]:
    """Geom attributes for a region."""
    return {
        "name": name,
        "class": PLACE_CLASS,
        "type": region.shape,
        "size": half(region.shape, region.size),
        "pos": " ".join(f"{value:g}" for value in region.pos),
        "quat": " ".join(f"{value:.9g}" for value in euler(region.rotation)),
    }


def save(runtime, path: Path, portable: bool = False, view: str | None = None) -> None:
    """Write the compiled scene with the current state as its keyframe.

    Args:
        runtime: The runtime whose scene and state to write.
        path: Output MJCF file.
        portable: Write file references relative to `path` and set each task body's
            pose to its current one, so the file loads from any location (as long as
            the referenced assets stay put) and viewers that ignore keyframes still
            show this state.
        view: Name of a camera whose viewpoint the free camera opens on, for a
            portable file. None frames the table.
    """
    qpos = " ".join(f"{value:.6g}" for value in runtime.data.qpos)
    ctrl = (
        ' ctrl="' + " ".join(f"{value:.6g}" for value in runtime.data.ctrl) + '"'
        if runtime.model.nu
        else ""
    )
    key = f'\n  <keyframe>\n    <key name="start" qpos="{qpos}"{ctrl}/>\n  </keyframe>\n'
    text = re.sub(r"<keyframe>.*?</keyframe>", "", runtime.xml, flags=re.S)
    if portable:
        text = _portable(runtime, text, Path(path).resolve().parent, view)
    text = text.replace("</mujoco>", key + "</mujoco>")
    Path(path).write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def _portable(runtime, text: str, folder: Path, view: str | None = None) -> str:
    """Rewrite scene MJCF to load from `folder` and open on the current state."""
    root = ET.fromstring(text)
    for element in root.iter():
        for name, value in element.attrib.items():
            if name.startswith("file") and Path(value).is_absolute():
                element.set(name, os.path.relpath(value, folder))
    for body in root.find("worldbody").findall("body"):
        if body.get("name") in runtime.task.scene.objects:
            position, quaternion = runtime.pose(body.get("name"))
            body.set("pos", " ".join(f"{value:.6g}" for value in position))
            body.set("quat", " ".join(f"{value:.6g}" for value in quaternion))
    # Invisible, non-colliding geoms (e.g. the rig's camera housing) still cast shadows
    # in viewers with full lighting, so the export drops them.
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if (
                geom.get("rgba", "").split()[-1:] == ["0"]
                and geom.get("contype") == "0"
                and geom.get("conaffinity") == "0"
                and geom.get("class") != PLACE_CLASS
            ):
                body.remove(geom)
    # Set the free camera to frame the table, or to stand where the named camera stands.
    # `cameraid` is removed because it locks the view to a fixed camera that cannot orbit.
    lo, hi = bounds(runtime)
    table = runtime.aabb("table/top")
    lo, hi = np.minimum(lo, table[0]), np.maximum(hi, table[1])
    lookat, distance = (lo + hi) / 2, float(np.max(hi - lo)) * 1.5
    visual_global = root.find("visual/global")
    if visual_global is None:
        visual_global = ET.SubElement(root.find("visual"), "global")
    visual_global.attrib.pop("cameraid", None)
    if view is not None:
        camera_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_CAMERA, view)
        if camera_id < 0:
            raise KeyError(f"no camera {view!r} to open the scene from")
        eye = runtime.data.cam_xpos[camera_id]
        forward = -runtime.data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        # Look at where the camera axis crosses table height, so orbiting pivots on the table.
        distance = (
            (eye[2] - lookat[2]) / -forward[2]
            if forward[2] < -1e-6
            else float(np.linalg.norm(lookat - eye))
        )
        lookat = eye + forward * distance
        visual_global.set("azimuth", f"{np.degrees(np.arctan2(forward[1], forward[0])):.4g}")
        visual_global.set("elevation", f"{np.degrees(np.arcsin(np.clip(forward[2], -1, 1))):.4g}")
        visual_global.set("fovy", f"{float(runtime.model.cam_fovy[camera_id]):.4g}")
    statistic = root.find("statistic")
    if statistic is None:
        statistic = ET.SubElement(root, "statistic")
    statistic.set("center", " ".join(f"{value:.4g}" for value in lookat))
    statistic.set("extent", f"{distance / 1.5:.4g}")  # the free camera starts 1.5 extents out
    return ET.tostring(root, encoding="unicode")
