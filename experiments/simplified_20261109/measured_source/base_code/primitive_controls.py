"""Explicit non-physical crop control; never calls an intensity-based detector."""
import numpy as np

from saqre_phys import extract_primitives


def grid_boxes(width, height, rows, columns):
    if width < columns or height < rows:
        raise ValueError("Image is too small for the requested uniform grid")
    return [(c * width // columns, r * height // rows,
             (c + 1) * width // columns, (r + 1) * height // rows)
            for r in range(rows) for c in range(columns)]


def extract_for_mode(image, max_regions=6, max_sc=8, mode="physics"):
    if mode == "physics":
        return extract_primitives(image, max_regions=max_regions, max_sc=max_sc)
    if mode != "uniform_grid":
        raise ValueError(f"Unknown primitive mode: {mode}")
    if (max_regions, max_sc) != (6, 8):
        raise ValueError("The predefined grid control uses six coarse and eight fine tiles")
    image = image.convert("RGB")
    width, height = image.size

    def unused_stats():
        return {"mu": 0.0, "cv": 0.0, "t": 0.0, "typefeat": np.zeros(4, dtype=np.float32)}

    return {
        "scene": (image, unused_stats()),
        "regions": [(image.crop(box), unused_stats()) for box in grid_boxes(width, height, 2, 3)],
        "scatterers": [(image.crop(box), unused_stats()) for box in grid_boxes(width, height, 2, 4)],
        "L": 0.0, "c_star": 0.0, "nu": 0.0, "cfar_frac": 0.0,
    }
