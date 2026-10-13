"""Inverse kinematics for Cartesian control of an arm, with mink."""

import importlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..model import absolutize

ITERATIONS = 100  # maximum differential IK steps per solve
STEP = 0.02  # integration time step of each IK step, seconds
POSITION_TOLERANCE = 1e-4  # TCP position error at convergence, metres
ORIENTATION_TOLERANCE = 1e-3  # TCP orientation error at convergence, radians


class PoseIK:
    """Inverse kinematics from a world-frame TCP pose to the seven arm joint positions.

    Solves on a reduced copy of the arm model with ``mink`` and the ``daqp`` QP
    solver, under joint position and velocity limits.

    Args:
        path: Path to the arm's MJCF file.

    Raises:
        ImportError: If ``mink`` or ``daqp`` is not installed.
    """

    def __init__(self, path):
        try:
            import mink

            importlib.import_module("daqp")
        except ImportError as error:
            raise ImportError(
                'Cartesian actions require mink and daqp: pip install "tabletask[control]"'
            ) from error
        self.mink = mink
        root = ET.parse(path).getroot()
        absolutize(root, path.parent)
        # TCP kinematics only depend on the seven arm joints, not the finger linkage.
        arm = {f"joint{i}" for i in range(1, 8)}
        for body in root.find("worldbody").iter("body"):
            for joint in list(body.findall("joint")):
                if joint.get("name") not in arm:
                    body.remove(joint)
        for tag in ("actuator", "equality", "tendon", "keyframe"):
            for element in list(root.findall(tag)):
                root.remove(element)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.configuration = mink.Configuration(model)
        self.base = model.body("link0").id
        self.qadr = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.frame = mink.FrameTask(
            "tcp", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1e-3
        )
        # A small posture cost resolves redundancy toward the current joint positions.
        self.posture = mink.PostureTask(model, cost=1e-4)
        self.limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, {f"joint{i}": 2.0 for i in range(1, 8)}),  # rad/s
        ]
        # Upper bound on the TCP distance from the base: the sum of the link offsets.
        site = model.site("tcp")
        self.reach = np.linalg.norm(site.pos)
        body = site.bodyid[0]
        while body != self.base:
            self.reach += np.linalg.norm(model.body_pos[body])
            body = model.body_parentid[body]

    def solve(self, pose, joints, base_position, base_quaternion):
        """Joint positions that place the TCP at a world-frame pose.

        Args:
            pose: Target TCP pose (x, y, z, qw, qx, qy, qz), with a unit quaternion.
            joints: Current seven arm joint positions, used as the initial guess and
                as the posture target.
            base_position: World position of the arm base, in metres.
            base_quaternion: World orientation of the arm base, wxyz.

        Returns:
            The seven arm joint positions, in radians.

        Raises:
            ValueError: If the target is out of reach, no joint-limited solution
                exists, or the solver does not converge.
        """
        config = self.configuration
        displacement = pose[:3] - base_position
        if np.linalg.norm(displacement) > self.reach:
            raise ValueError("Cartesian target is outside the arm reach")
        config.model.body_pos[self.base] = base_position
        config.model.body_quat[self.base] = base_quaternion
        q = config.q.copy()
        q[self.qadr] = joints
        config.update(q)
        self.posture.set_target_from_configuration(config)
        self.frame.set_target(
            self.mink.SE3.from_rotation_and_translation(self.mink.SO3(pose[3:7]), pose[:3])
        )

        def converged() -> bool:
            """Whether the TCP is within tolerance of the target pose."""
            error = self.frame.compute_error(config)
            return (
                np.linalg.norm(error[:3]) < POSITION_TOLERANCE
                and np.linalg.norm(error[3:]) < ORIENTATION_TOLERANCE
            )

        for _ in range(ITERATIONS):
            if converged():
                return config.q[self.qadr].copy()
            try:
                velocity = self.mink.solve_ik(
                    config,
                    [self.frame, self.posture],
                    STEP,
                    "daqp",
                    damping=1e-5,
                    limits=self.limits,
                    safety_break=True,
                )
            except (self.mink.NoSolutionFound, self.mink.NotWithinConfigurationLimits) as error:
                raise ValueError("Cartesian IK could not find a joint-limited solution") from error
            if not np.isfinite(velocity).all():
                raise ValueError("Cartesian IK produced a non-finite solution")
            config.integrate_inplace(velocity, STEP)
        if converged():
            return config.q[self.qadr].copy()
        raise ValueError(f"Cartesian IK did not converge within {ITERATIONS} iterations")
