"""Quaternion helpers; quaternions are wxyz and angles degrees."""

import math

import numpy as np


def quat(rotation_deg) -> np.ndarray:
    """wxyz quaternion from roll/pitch/yaw in degrees."""
    r, p, y = (math.radians(v) / 2 for v in rotation_deg)
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def compose(a, b) -> np.ndarray:
    """The wxyz quaternion for turning by `a`, then by `b` about the turned axes."""
    w1, v1, w2, v2 = a[0], np.asarray(a[1:]), b[0], np.asarray(b[1:])
    return np.concatenate(([w1 * w2 - v1 @ v2], w1 * v2 + w2 * v1 + np.cross(v1, v2)))


def spin(degrees: float) -> np.ndarray:
    """The wxyz quaternion for a turn about the vertical."""
    half = math.radians(degrees) / 2
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def euler(rotation_deg) -> np.ndarray:
    """wxyz quaternion from a turn about x, then y, then z -- MuJoCo's `euler` sequence.

    This is not `quat`: that one turns about z last, as task rotations are written.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for i, degrees in enumerate(rotation_deg):
        half = math.radians(degrees) / 2
        turn = np.zeros(4)
        turn[0], turn[i + 1] = math.cos(half), math.sin(half)
        q = compose(q, turn)
    return q


def mat(q) -> np.ndarray:
    """The rotation a wxyz quaternion stands for, as a 3x3."""
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ]
    )


def unit_quaternions(quaternions, error: str) -> np.ndarray:
    """Normalise wxyz quaternions along the last axis.

    Dividing by the largest component first keeps tiny quaternions stable.

    Raises:
        ValueError: With message `error`, if any quaternion is zero.
    """
    quaternions = np.array(quaternions, dtype=float)
    scales = np.max(np.abs(quaternions), axis=-1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError(error)
    quaternions /= scales
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return quaternions
