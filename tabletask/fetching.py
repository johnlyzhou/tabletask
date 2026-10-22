"""Download asset files from the published library into a local cache on first use."""

from __future__ import annotations

import ast
import importlib.util
import os
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

# Base URL of the asset library; override with TABLETASK_LIBRARY_URL to serve assets elsewhere.
LIBRARY_URL = os.environ.get(
    "TABLETASK_LIBRARY_URL", "https://raw.githubusercontent.com/omarrayyann/tabletask/main"
).rstrip("/")
TIMEOUT = 60  # seconds per download


def cache() -> Path:
    """The directory where fetched assets are kept between runs."""
    given = os.environ.get("TABLETASK_ASSETS")
    if given:
        return Path(given).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "tabletask"


def library_path(path: Path) -> str | None:
    """The library-relative name of a path (from its last `assets/` on), or None.

    An installed package has no repository around it, so a file is identified by
    its position under `assets/` rather than by its location relative to a task.
    """
    parts = Path(path).parts
    if "assets" not in parts:
        return None
    return "/".join(parts[len(parts) - 1 - parts[::-1].index("assets") :])


def download(name: str) -> Path | None:
    """The cached copy of one library file, downloading it if needed; None if unavailable."""
    if not Path(name).suffix:
        return None  # directories are not fetched, only the files under them
    out = cache() / name
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(f"{LIBRARY_URL}/{name}", timeout=TIMEOUT) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    # write under a name no other process uses, then rename it into place: atomic, so
    # neither an interrupted download nor two processes fetching one file leave a broken copy
    partial = out.with_name(f"{out.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    partial.write_bytes(body)
    os.replace(partial, out)
    return out


def asset_dirs(compiler: ET.Element | None) -> dict[str, str]:
    """The folder each kind of MJCF asset file resolves against, keyed by element tag.

    Follows MuJoCo's `<compiler>` rules: `meshdir` applies to mesh, hfield and skin
    files, `texturedir` to textures, and `assetdir` stands in for either when it is not
    given. Folders are relative to the MJCF file; "" means the file's own folder.
    """
    if compiler is None:
        compiler = ET.Element("compiler")
    assetdir = compiler.get("assetdir") or ""
    meshdir = compiler.get("meshdir") or assetdir
    texturedir = compiler.get("texturedir") or assetdir
    return {"mesh": meshdir, "texture": texturedir, "hfield": meshdir, "skin": meshdir}


def siblings(script: Path, name: str) -> None:
    """Fetch the modules beside a generator script that it imports and Python cannot find."""
    try:
        tree = ast.parse(script.read_text())
    except SyntaxError:
        return
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module.split(".")[0])
    for module in sorted(modules):
        if importlib.util.find_spec(module) is None:
            download(f"{Path(name).parent.as_posix()}/{module}.py")


def asset(name: str) -> Path | None:
    """The cached copy of an asset xml, after fetching every mesh and texture it references.

    A generator script also brings the modules beside it that it imports.

    Returns None if the file itself is unavailable.

    Raises:
        FileNotFoundError: If the file downloads but a mesh or texture it references
            does not.
    """
    xml = download(name)
    if xml is not None and xml.suffix == ".py":
        siblings(xml, name)
    if xml is None or xml.suffix != ".xml":
        return xml
    try:
        root = ET.parse(xml).getroot()
    except ET.ParseError:
        return xml
    parent = Path(name).parent
    missing = []
    for tag, folder in asset_dirs(root.find("compiler")).items():
        for node in root.iter(tag):
            file = node.get("file")
            if not file:
                continue
            needed = os.path.normpath(parent / folder / file).replace(os.sep, "/")
            if download(needed) is None:
                missing.append(f"{LIBRARY_URL}/{needed}")
    if missing:
        raise FileNotFoundError(f"{name} needs files that could not be downloaded: {missing}")
    return xml


def resolve(path: str, root: Path) -> Path:
    """Resolve a task-relative path, searching the task's folder and then its ancestors.

    Tasks are grouped in subfolders by tier, so shared assets are found by
    walking up rather than by a fixed relative path. If nothing is found locally,
    the asset is fetched (an installed package ships without assets).
    """
    relative = Path(path)
    if relative.is_absolute():
        return relative
    here = (root / relative).resolve()
    if here.exists():
        return here
    from .language.format import TASKS  # a task given as a dict has the working directory as root

    bases = [*root.resolve().parents]
    if TASKS is not None:
        bases += [TASKS / "base", *(TASKS / "base").resolve().parents]
    for base in bases:
        candidate = (base / relative).resolve()
        if candidate.exists():
            return candidate
    name = library_path(here)
    if name and Path(name).suffix:  # a file; folders are not downloaded
        fetched = asset(name)
        if fetched is None:
            raise FileNotFoundError(
                f"{path} was not found locally and could not be downloaded from "
                f"{LIBRARY_URL}/{name}"
            )
        return fetched
    return here
