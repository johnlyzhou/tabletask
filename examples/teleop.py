"""Teleoperate a task with an iOS device running Teledex, starting a new episode after each ends.

python -m examples.teleop robolab-banana_in_bowl --episode 6
"""

import argparse
import time

import tabletask
from tabletask.inspection import conditions
from tabletask.language.grammar import names_in
from tabletask.robots import Droid, Gripper

from .phone import Phone

RATE = 0.05  # simulated seconds per streamed pose; Droid's Cartesian control period
HIDDEN = 5  # geom group the renderer does not draw
KEEP = ("link7", "base")  # robot bodies whose subtrees stay visible when the arm is hidden
FONT_SIZE = 22  # instruction caption, pixels


def _hide_arm(runtime):
    """Hide the arm's geoms except the gripper; geom groups do not affect contacts."""
    model, robot = runtime.model, runtime.robot
    names = {model.body(body).name for body in range(model.nbody)}
    kept = {model.body(robot.qualified(name)).id for name in KEEP if robot.qualified(name) in names}
    for geom in range(model.ngeom):
        body = model.geom_bodyid[geom]
        if not model.body(body).name.startswith(robot.name + "/"):
            continue
        while body and body not in kept:  # walk up to a kept body or the world (0)
            body = model.body_parentid[body]
        if not body:
            model.geom_group[geom] = HIDDEN


def _goal_regions(env):
    """Regions named by the goal, excluding fixture tops, which would cover the others."""
    named = set()
    for stage in env.task.goal.stages.values():
        named |= set(names_in(stage.condition))
    for tree in env.task.goal.always.values():
        named |= set(names_in(tree))
    return sorted(
        name for name in named if name != "table/top" and env.runtime.region(name) is not None
    )


class Checks:
    """Print goal and `always` checks to the terminal whenever one changes."""

    def __init__(self):
        self.held = {}  # check key -> last printed verdict

    def show(self, env):
        """Print every check, at the start of an episode."""
        self.held = {}
        for key, (held, line) in self._read(env).items():
            self.held[key] = held
            print(f"  {line}")

    def update(self, env):
        """Print the checks whose verdict changed since the last call."""
        for key, (held, line) in self._read(env).items():
            if self.held.get(key) != held:
                self.held[key] = held
                print(f"{env.status.time:7.2f}s  {line}")

    @staticmethod
    def _read(env):
        """Current checks as {key: (held, line)}, including each condition's detail lines."""
        checks = {}
        for section, label, held, detail in conditions(env.runtime):
            if section == "start":
                continue
            checks[(section, label)] = held, f"{section} {label}: {'MET' if held else 'not met'}"
            for line in detail:
                text = line.strip()
                if text.endswith((" true", " FALSE")):
                    # Key on the text before the measured value, which changes every step.
                    key = text.split(" = ")[0] if " = " in text else text.rsplit(" ", 1)[0]
                    checks[(section, label, key)] = text.endswith(" true"), f"    {text}"
        return checks


class Window:
    """A tkinter window showing one camera, redrawn on every step."""

    def __init__(self, camera="dashboard", width=1280, height=720, regions=False):
        import tkinter

        self.camera, self.size, self.regions = camera, (width, height), regions
        self.root = tkinter.Tk()
        self.root.title("teleop")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.label = tkinter.Label(self.root)
        self.label.pack()
        self.open = True
        self.model = None
        self.photo = None
        self.painted = False
        self.banner = None
        self.root.lift()  # tkinter opens behind the terminal otherwise
        self.root.attributes("-topmost", True)
        self.root.after(500, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def _caption(self, image, text):
        """Paste the instruction band across the top of `image`."""
        if self.banner is None or self.banner[0] != text:
            self.banner = (text, self._band(text, image.width))
        band = self.banner[1]
        image.paste(band, (0, 0), band)
        return image

    def _band(self, text, width):
        """Instruction text wrapped on a translucent dark band, cached by `_caption`."""
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new("RGBA", (width, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default(size=FONT_SIZE)
        margin, lines, line = 12, [], ""
        for word in text.split():
            wider = f"{line} {word}".strip()
            if line and draw.textlength(wider, font=font) > image.width - 2 * margin:
                lines.append(line)
                line = word
            else:
                line = wider
        lines.append(line)
        wrapped = "\n".join(lines)
        box = draw.textbbox((margin, margin), wrapped, font=font)
        draw.rectangle((0, 0, image.width, box[3] + margin), fill=(0, 0, 0, 150))
        draw.text((margin, margin), wrapped, font=font, fill=(255, 255, 255))
        return image.crop((0, 0, width, box[3] + margin))

    def show(self, env, title):
        """Render the camera and redraw the window; does nothing once it is closed."""
        from PIL import Image, ImageTk

        if not self.open:
            return
        if env.runtime.model is not self.model:
            self.model = env.runtime.model
            _hide_arm(env.runtime)
            self.painted = _goal_regions(env) if self.regions else False
        rgb = env.runtime.images([self.camera], *self.size, regions=self.painted)[self.camera]
        picture = self._caption(Image.fromarray(rgb), env.task.instruction)
        # Pasting into one PhotoImage is about four times faster than creating a new one.
        if self.photo is None:
            self.photo = ImageTk.PhotoImage(picture)
            self.label.configure(image=self.photo)
        else:
            self.photo.paste(picture)
        self.root.title(title)
        self.root.update()

    def close(self):
        """Destroy the window."""
        if self.open:
            self.open = False
            self.root.destroy()


class _Session:
    """Episode bookkeeping and per-tick control for one teleoperation session."""

    def __init__(self, env, phone, checks):
        self.env, self.phone, self.checks = env, phone, checks
        self.solved = self.played = 0
        self.clutched = self.restarting = False

    def anchor(self):
        """Anchor the phone pose to the current gripper pose."""
        self.phone.anchor_to(self.env.runtime.robot.observe()["tcp_pose"])

    def announce(self):
        """Print the episode number and, in debug mode, every goal check."""
        print(f"episode {self.env.episode}")
        if self.checks:
            self.checks.show(self.env)

    def begin(self, episode):
        """Reset to `episode` and anchor the phone to the new start."""
        self.env.reset(episode)
        self.anchor()
        self.announce()

    def tick(self):
        """Run one control period: restart, clutch, or step toward the phone pose."""
        phone, env = self.phone, self.env
        if phone.restarting:
            if not self.restarting:
                self.restarting = True
                phone.buzz()
                print(f"episode {env.episode}: dropped")
                self.begin(env.episode + 1)
            time.sleep(RATE)
            return
        self.restarting = False
        if phone.clutching:
            if not self.clutched:
                phone.buzz()
            self.clutched = True
            self.anchor()
            time.sleep(RATE)
            return
        self.clutched = False
        try:
            _, _, terminated, truncated, info = env.step(phone.target(), RATE)
        except ValueError:  # skip an unreachable pose rather than end the session
            return
        if self.checks:
            self.checks.update(env)
        if terminated or truncated:
            self._end_episode(info)

    def _end_episode(self, info):
        """Count and print the episode outcome, then start the next episode."""
        self.played += 1
        self.solved += bool(info["success"])
        outcome = "solved" if info["success"] else f"failed: {info['violated'] or 'out of time'}"
        print(f"episode {self.env.episode}: {outcome}")
        self.phone.buzz()
        self.begin(self.env.episode + 1)


def teleoperate(
    env,
    *,
    scale=2.0,
    port=8888,
    view=True,
    camera="dashboard",
    width=1280,
    debug=False,
    phone=None,
):
    """Stream phone poses into an environment until the window closes or Ctrl-C.

    Starts the next episode whenever one ends. Button bindings are listed in examples/README.md.

    Args:
        env: An environment built with `Gripper()` or `Droid(action_space="cartesian_pose")`.
        scale: Metres of gripper motion per metre of phone motion.
        port: Port the phone connects to.
        view: Show a camera window.
        camera: Camera the window shows.
        width: Window width in pixels; the height is 9/16 of it.
        debug: Print goal checks as they change and draw the goal regions.
        phone: Input device to use instead of a `Phone`, with the same methods.

    Returns:
        (solved, played): episodes solved and episodes ended; dropped episodes are not counted.

    Raises:
        ValueError: If the robot does not use a Cartesian pose action space.
    """
    if env.runtime is None or getattr(env.runtime.robot, "action_space", None) != "cartesian_pose":
        raise ValueError(
            'Build an environment with Gripper() or Droid(action_space="cartesian_pose") first'
        )
    phone = phone or Phone(scale=scale, port=port)
    window = Window(camera, width, round(width * 9 / 16), regions=debug) if view else None
    session = _Session(env, phone, Checks() if debug else None)
    cameras, env.cameras = env.cameras, False
    interval, env.monitor_interval = env.monitor_interval, RATE
    try:
        phone.start()
        session.anchor()
        print("connected; move the phone to move the gripper, toggle for the fingers")
        session.announce()
        while window is None or window.open:
            if window is not None:
                window.show(env, f"{env.task.id} · episode {env.episode}")
            session.tick()
    except KeyboardInterrupt:
        pass
    finally:
        env.cameras, env.monitor_interval = cameras, interval
        phone.stop()
        if window is not None:
            window.close()
    print(f"solved {session.solved} of {session.played}")
    return session.solved, session.played


def main(argv=None):
    """Parse the command line, build the environment and teleoperate it."""
    parser = argparse.ArgumentParser(
        prog="python -m examples.teleop",
        description="Drive a task with an iOS device running Teledex.",
    )
    parser.add_argument("task", help="task id or task file")
    parser.add_argument(
        "--robot",
        choices=("gripper", "droid"),
        default="gripper",
        help="the floating Gripper (no arm, every pose reachable) or the full DROID arm",
    )
    parser.add_argument("--episode", type=int, default=0, help="first episode to play")
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="metres of gripper motion per metre of phone motion",
    )
    parser.add_argument("--port", type=int, default=8888, help="port the phone connects to")
    parser.add_argument("--camera", default="dashboard", help="camera the window shows")
    parser.add_argument("--width", type=int, default=1280, help="window width in pixels")
    parser.add_argument("--no-view", action="store_true", help="skip the camera window")
    parser.add_argument(
        "--debug", action="store_true", help="print each goal check as it turns true or false"
    )
    args = parser.parse_args(argv)

    robot = Gripper() if args.robot == "gripper" else Droid(action_space="cartesian_pose")
    env = tabletask.make(args.task).add(robot)
    env.reset(args.episode)
    try:
        teleoperate(
            env,
            scale=args.scale,
            port=args.port,
            view=not args.no_view,
            camera=args.camera,
            width=args.width,
            debug=args.debug,
        )
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
