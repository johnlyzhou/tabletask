"""Converting cube-cross skyboxes to equirectangular dome-light textures."""

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

CUBE_LAYOUT = ".U..LFRB.D.."  # MuJoCo gridlayout of a 4x3 cube-cross skybox


def latlong(source, target):
    """Convert a 4x3 cube-cross skybox into a 2:1 equirectangular dome-light texture.

    Args:
        source: Path of the cube-cross image, laid out as `CUBE_LAYOUT`.
        target: Path to write the 2048x1024 equirectangular image to.

    Raises:
        ValueError: If the source image is not four faces wide by three tall.
    """
    with Image.open(source) as source_image:
        image = np.asarray(source_image.convert("RGB"))
    if image.shape[1] * 3 != image.shape[0] * 4:
        raise ValueError(f"Expected a four-by-three skybox cross: {source}")
    edge = image.shape[1] // 4
    width, height = 2048, 1024
    # Unit view direction for every output pixel (y up, z forward).
    lon = ((np.arange(width) + 0.5) / width * 2 - 1) * np.pi
    lat = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    x = np.cos(lat)[:, None] * np.sin(lon)[None, :]
    y = np.broadcast_to(np.sin(lat)[:, None], x.shape)
    z = np.cos(lat)[:, None] * np.cos(lon)[None, :]
    # The dominant axis of each direction picks the cube face it hits.
    axis = np.argmax(np.abs(np.stack([x, y, z])), axis=0)
    result = np.empty((height, width, 3), dtype=np.uint8)
    # (face, dominant axis, hemisphere mask, u, v, projection denominator)
    faces = [
        ("F", 2, z > 0, x, -y, z),
        ("B", 2, z < 0, -x, -y, -z),
        ("R", 0, x > 0, -z, -y, x),
        ("L", 0, x < 0, z, -y, -x),
        ("U", 1, y > 0, x, z, y),
        ("D", 1, y < 0, x, -z, -y),
    ]
    layout = CUBE_LAYOUT
    for face, component, side, u, v, denominator in faces:
        mask = (axis == component) & side
        column, row = layout.index(face) % 4, layout.index(face) // 4
        tile = image[row * edge : (row + 1) * edge, column * edge : (column + 1) * edge]
        px = (u[mask] / denominator[mask] + 1) * 0.5 * edge - 0.5
        py = (v[mask] / denominator[mask] + 1) * 0.5 * edge - 0.5
        for channel in range(3):
            result[:, :, channel][mask] = map_coordinates(
                tile[:, :, channel], [py, px], order=1, mode="nearest"
            )
    Image.fromarray(result).save(target)
