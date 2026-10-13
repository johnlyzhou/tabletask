"""Command-line tools: `python -m tabletask <command>`."""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import tabletask
from tabletask import inspection, validation
from tabletask.language.errors import SamplingError, UnsupportedError
from tabletask.language.format import task_file

SOURCE_SUFFIXES = (".json", ".xml")  # task documents and asset files


def _paths(given, suffixes=SOURCE_SUFFIXES):
    """Expand files, directories and shipped task ids into a list of file paths."""
    found = []
    for item in given:
        path = Path(item)
        if path.is_dir():
            # Tasks are grouped in subfolders by tier, so search recursively.
            found.extend(p for p in sorted(path.rglob("*")) if p.suffix in suffixes)
        elif path.is_file():
            found.append(path)
        elif not path.suffix and task_file(item) is not None:
            found.append(task_file(item))  # a shipped task id
        else:
            raise SystemExit(f"no file or task id {item}")
    return found


def _picture_path(given, path, several=False):
    """The `--out` path if given, else `<stem>.png` in the working directory.

    With several inputs, `--out` gets each input's name appended, so none is overwritten.
    """
    if not given:
        return Path(f"{path.stem}.png")
    given = Path(given)
    return given.with_name(f"{given.stem}-{path.stem}{given.suffix}") if several else given


def _positive(text):
    """An integer of at least 1, for counts such as --samples."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _parse_put(text):
    """Parse NAME=REGION or NAME=REGION+dx,dy,dz into (name, region, offset)."""
    name, _, region = text.partition("=")
    region, _, offset = region.partition("+")
    if not name or not region:
        raise argparse.ArgumentTypeError(f"--put wants NAME=REGION[+dx,dy,dz], got {text!r}")
    try:
        shift = tuple(float(v) for v in offset.split(",")) if offset else (0.0, 0.0, 0.0)
        if len(shift) != 3 or not all(math.isfinite(v) for v in shift):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--put offset wants three finite numbers, got {offset!r}"
        ) from None
    return name, region, shift


def _parse_turn(text):
    """Parse NAME=roll,pitch,yaw (degrees) into (name, rotation)."""
    name, _, angles = text.partition("=")
    try:
        rotation = tuple(float(v) for v in angles.split(","))
        if not name or len(rotation) != 3 or not all(math.isfinite(v) for v in rotation):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--turn wants NAME=roll,pitch,yaw in finite degrees, got {text!r}"
        ) from None
    return name, rotation


def _parse_joint(text):
    """Parse NAME=VALUE into (joint name, position)."""
    name, _, raw_position = text.partition("=")
    try:
        position = float(raw_position)
        if not name or not math.isfinite(position):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--joint wants NAME=VALUE with a finite value, got {text!r}"
        ) from None
    return name, position


def _parse_regions(text: str):
    """Parse `each`, or comma-separated region names (`inside`, `bowl/inside`)."""
    return (
        "each"
        if text == "each"
        else tuple(part.strip() for part in text.split(",") if part.strip())
    )


def look(args):
    """Print a report and write a picture for an asset, or for one start of a task."""
    failures = 0
    paths = _paths([args.path])
    for path in paths:
        out = _picture_path(args.out, path, several=len(paths) > 1)
        if path.suffix == ".xml":
            failures += _inspect_asset(path, out, args.collision, args.regions)
        else:
            runtime = inspection.try_out(
                tabletask.load(path), args.seed, args.put, args.turn, args.joint, args.settle
            )
            try:
                if runtime.sampling_error:
                    print(
                        f"no valid start for seed {args.seed}: {runtime.sampling_error}\n"
                        "the last draw, rejected:"
                    )
                    failures += 1
                if args.put or args.turn or args.joint:
                    motion = "at rest" if runtime.rested else "still moving"
                    print(f"after moving objects and {args.settle:g} s of physics: {motion}")
                print("\n".join(inspection.report(runtime)))
                if args.collision:
                    inspection.collision(runtime, out)
                else:
                    inspection.sheet(runtime, out, regions=args.regions)
            finally:
                runtime.close_renderer()
        print(f"picture   {out}")
    return int(failures > 0)


def _inspect_asset(path, out, collision=False, regions=None):
    """Write an asset's picture, print its facts, and return whether it has problems."""
    from tabletask.mujoco.checks import describe, study

    facts, problems, runtime = study(path)
    try:
        if collision:
            inspection.collision(runtime, out, focus=path.stem)
        else:
            inspection.sheet(runtime, out, focus=path.stem, regions=regions)
        print("\n".join(describe(facts, problems)))
        return bool(problems)
    finally:
        runtime.close_renderer()


def asset(args):
    """Import a mesh as an asset, or rebuild existing assets' collision geometry."""
    from tabletask.mujoco.importing import collide_as, import_mesh, rehull

    if args.collide:
        for path in _paths(args.collide, (".xml",)):
            before, shape, size = collide_as(path, args.shape)
            extents = " x ".join(f"{v * 1000:.1f}" for v in size)
            print(f"{path.stem}: {before} collision geom(s) -> {shape} {extents} mm", flush=True)
        return 0
    if args.rehull:
        paths = _paths(args.rehull, (".xml",))
        if len(paths) == 1 or args.jobs == 1:
            for path in paths:
                before, after = rehull(path)
                counts = f"{before} collision geom(s) -> {after} hulls"
                print(f"{path.stem}: {counts}", flush=True)
            return 0
        # Each asset is decomposed in its own process; they write to different files.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for path, (before, after) in zip(paths, pool.map(rehull, paths)):
                print(f"{path.stem}: {before} collision geom(s) -> {after} hulls", flush=True)
        return 0
    library = Path(args.library).resolve()
    xml, checks = import_mesh(
        args.name,
        Path(args.mesh),
        library,
        texture=Path(args.texture) if args.texture else None,
        mass=args.mass,
        scale=args.scale,
        up=args.up,
    )
    print(f"created {xml} ({' -> '.join(checks)})")
    out = _picture_path(args.out, xml)
    problems = _inspect_asset(xml, out)
    print(f"picture   {out}")
    return int(problems)


def region(args):
    """Remove, add, refit, list and picture an asset's placement regions, in that order."""
    from tabletask.mujoco.regions import (
        drop_region,
        parse_region,
        refit_region,
        regions,
        set_region,
    )

    path = Path(args.asset)
    for text in args.remove:
        print(f"removed   {text}" if drop_region(path, text) else f"no region {text!r} to remove")
    for text in args.add:
        spec = parse_region(text)
        print(f"{set_region(path, spec):9s} {spec['name']}")
    for name in args.fit:
        print(f"fitted    {name} (was {refit_region(path, name)})")
    found = regions(path)
    if not found:
        print(f"{path.stem}: no regions")
    for entry in found:
        size = entry["size"]
        extent = "mesh" if size is None else " x ".join(f"{v:.3f}" for v in size)
        position = " ".join(f"{v:+.3f}" for v in entry["pos"])
        print(f"  {entry['name']:<12} {entry['shape']:<9} {extent:<21} at {position}")
    if args.out or args.add or args.remove or args.fit:
        return _inspect_asset(path, _picture_path(args.out, path))
    return 0


def check(args):
    """Validate tasks and assets, printing a summary per file or JSON for all."""
    results = []
    for path in _paths(args.paths):
        result = validation.check(
            path,
            build=args.build,
            samples=args.samples,
            seed=args.seed,
            lint_goal=args.lint,
        )
        results.append(result)
        if not args.json:
            print(validation.summary(result), flush=True)
    if args.json:
        print(json.dumps(results, indent=2))
    return int(any(result["problems"] for result in results))


def preview(args):
    """Write a grid of sampled starts for each task."""
    paths = _paths(args.tasks, (".json",))
    for path in paths:
        out = _picture_path(args.out, path, several=len(paths) > 1)
        tabletask.preview(tabletask.load(path), args.samples, out)
        print(f"{path.stem}: {out}")
    return 0


def _look_command(commands):
    look_parser = commands.add_parser(
        "look", help="report and picture an asset xml, or one start of a task json"
    )
    look_parser.add_argument("path", help="task json, asset xml, directory, or shipped task id")
    look_parser.add_argument(
        "--seed", type=int, default=0, help="seed of the start to sample (tasks only)"
    )
    look_parser.add_argument("--out", help="picture path (default: <name>.png here)")
    look_parser.add_argument(
        "--collision", action="store_true", help="show collision geoms instead of visuals"
    )
    look_parser.add_argument(
        "--regions",
        type=_parse_regions,
        help="paint only these regions (comma separated), or 'each' for one view per region",
    )
    look_parser.set_defaults(run=look, put=(), turn=(), joint=(), settle=0.0)


def _try_command(commands):
    try_parser = commands.add_parser(
        "try", help="move objects in a task start, run physics, and report every condition"
    )
    try_parser.add_argument("path", help="task json or shipped task id")
    try_parser.add_argument("--seed", type=int, default=0, help="seed of the start to sample")
    try_parser.add_argument(
        "--put",
        type=_parse_put,
        action="append",
        default=[],
        metavar="NAME=REGION[+dx,dy,dz]",
        help="place an object in a region, e.g. banana=bowl/inside or cup=table/left+0.05,0,0",
    )
    try_parser.add_argument(
        "--turn",
        type=_parse_turn,
        action="append",
        default=[],
        metavar="NAME=roll,pitch,yaw",
        help="rotate an object, in degrees, e.g. bottle=90,0,0",
    )
    try_parser.add_argument(
        "--joint",
        type=_parse_joint,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set a joint position, e.g. laptop/lid=1.2",
    )
    try_parser.add_argument(
        "--settle",
        type=float,
        default=inspection.SETTLE,
        help=f"seconds of physics to run after moving objects (default {inspection.SETTLE:g})",
    )
    try_parser.add_argument("--out", help="picture path (default: <name>.png here)")
    try_parser.set_defaults(run=look, collision=False, regions=None)


def _asset_command(commands):
    asset_parser = commands.add_parser(
        "asset", help="import a mesh with collision geometry, mass and placement regions"
    )
    asset_parser.add_argument(
        "name",
        nargs="?",
        help="asset name; writes <library>/objects/<name>.xml and <library>/mesh/<name>/",
    )
    asset_parser.add_argument(
        "--rehull",
        nargs="+",
        metavar="ASSET",
        help="instead, rebuild these assets' collision geoms as convex hulls of the visual mesh",
    )
    asset_parser.add_argument(
        "--jobs", type=int, default=8, help="parallel processes for --rehull (default 8)"
    )
    asset_parser.add_argument(
        "--collide",
        nargs="+",
        metavar="ASSET",
        help="instead, replace these assets' collision geoms with one primitive fitted to the mesh",
    )
    asset_parser.add_argument(
        "--shape",
        choices=("box", "cylinder", "sphere", "ellipsoid"),
        help="primitive for --collide (default: inferred from the mesh)",
    )
    asset_parser.add_argument("--mesh", help="mesh file (obj, glb, stl, ...)")
    asset_parser.add_argument("--texture", help="texture image (default: the mesh's material)")
    asset_parser.add_argument("--mass", type=float, default=0.25, help="mass in kg (default 0.25)")
    asset_parser.add_argument(
        "--scale", type=float, default=1.0, help="factor converting mesh units to metres"
    )
    asset_parser.add_argument(
        "--up", choices=("z", "y"), default="z", help="up axis of the mesh file (default z)"
    )
    asset_parser.add_argument(
        "--library", default="assets", help="asset library directory (default assets)"
    )
    asset_parser.add_argument("--out", help="picture path (default: <name>.png here)")
    asset_parser.set_defaults(run=asset)


def _region_command(commands):
    region_parser = commands.add_parser(
        "region", help="add, remove, refit and list an asset's placement regions"
    )
    region_parser.add_argument("asset", help="asset xml to edit")
    region_parser.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="NAME=SHAPE:EXTENTS[@X,Y,Z][/R,P,Y]",
        help="add or replace a region, e.g. top=box:0.1,0.1,0.06@0,0,0.03 (metres, degrees)",
    )
    region_parser.add_argument(
        "--remove", action="append", default=[], metavar="NAME", help="remove a region by name"
    )
    region_parser.add_argument(
        "--fit",
        action="append",
        default=[],
        choices=("top", "inside"),
        help="rebuild a region from the asset's mesh surface",
    )
    region_parser.add_argument("--out", help="picture path, with regions painted")
    region_parser.set_defaults(run=region)


def _check_command(commands):
    check_parser = commands.add_parser(
        "check", help="validate task documents and asset files; exit 1 on any problem"
    )
    check_parser.add_argument(
        "paths", nargs="+", help="task json, asset xml, directories, or shipped task ids"
    )
    check_parser.add_argument(
        "--build", action="store_true", help="also sample starts in the simulator"
    )
    check_parser.add_argument(
        "--samples",
        type=_positive,
        default=6,
        help="starts to sample per task with --build (default 6)",
    )
    check_parser.add_argument("--seed", type=int, default=0, help="first seed to sample")
    check_parser.add_argument(
        "--lint",
        action="store_true",
        help="also flag goals that accept more than the instruction asks",
    )
    check_parser.add_argument("--json", action="store_true", help="print results as JSON")
    check_parser.set_defaults(run=check)


def _preview_command(commands):
    preview_parser = commands.add_parser("preview", help="render a grid of sampled starts")
    preview_parser.add_argument("tasks", nargs="+", help="task json, directories, or task ids")
    preview_parser.add_argument(
        "--samples", type=_positive, default=6, help="starts to sample per task (default 6)"
    )
    preview_parser.add_argument("--out", help="picture path (default: <name>.png here)")
    preview_parser.set_defaults(run=preview)


def main(argv=None):
    """Parse the command line, run the chosen command and return its exit code."""
    parser = argparse.ArgumentParser(
        prog="tabletask", description="Inspect, sample and validate tabletop tasks and assets."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for add in (
        _look_command,
        _try_command,
        _asset_command,
        _region_command,
        _check_command,
        _preview_command,
    ):
        add(commands)
    commands.add_parser("words", help="print the vocabulary").set_defaults(
        run=lambda args: print(tabletask.words()) or 0
    )
    commands.add_parser("schema", help="print the authoring JSON Schema").set_defaults(
        run=lambda args: print(json.dumps(tabletask.json_schema(), indent=2)) or 0
    )
    args = parser.parse_args(argv)
    if (
        args.command == "asset"
        and not args.rehull
        and not args.collide
        and not (args.name and args.mesh)
    ):
        parser.error(
            "give a name and --mesh to import, --rehull to redo hulls, or --collide for a primitive"
        )
    try:
        return args.run(args)
    except (ValueError, KeyError, OSError, ET.ParseError, SamplingError, UnsupportedError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        raise SystemExit(f"error: {message}") from None


if __name__ == "__main__":
    raise SystemExit(main())
