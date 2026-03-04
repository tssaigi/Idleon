from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from .models import SheetImportOptions, TemplateImage


def load_templates_from_path(path: str | Path) -> list[TemplateImage]:
    path = Path(path)
    templates: list[TemplateImage] = []
    for frame_index, rgba in iter_image_frames(path):
        suffix = f"_f{frame_index:03d}" if frame_index is not None else ""
        templates.append(
            TemplateImage(
                name=f"{path.stem}{suffix}",
                source=str(path),
                image_rgba=rgba,
            )
        )
    return templates


def slice_sprite_sheet(path: str | Path, options: SheetImportOptions) -> list[TemplateImage]:
    path = Path(path)
    templates: list[TemplateImage] = []
    for frame_index, rgba in iter_image_frames(path):
        sheet_name = path.stem if frame_index is None else f"{path.stem}_f{frame_index:03d}"
        templates.extend(slice_sprite_sheet_array(rgba, sheet_name, str(path), options))
    return templates


def slice_sprite_sheet_array(
    rgba: np.ndarray,
    sheet_name: str,
    source: str,
    options: SheetImportOptions,
) -> list[TemplateImage]:
    height, width = rgba.shape[:2]
    usable_width = width - (options.margin * 2) - (options.spacing * (options.cols - 1))
    usable_height = height - (options.margin * 2) - (options.spacing * (options.rows - 1))
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("Sheet dimensions are smaller than the requested margins/spacings.")

    cell_width = usable_width // options.cols
    cell_height = usable_height // options.rows
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("Rows/cols create empty sprite cells.")

    templates: list[TemplateImage] = []
    index = 0
    for row in range(options.rows):
        for col in range(options.cols):
            left = options.margin + col * (cell_width + options.spacing)
            top = options.margin + row * (cell_height + options.spacing)
            right = left + cell_width
            bottom = top + cell_height
            cell = rgba[top:bottom, left:right].copy()

            if options.trim_empty:
                cell = trim_sprite(cell, options.background_tolerance)

            if cell.size == 0 or cell.shape[0] < 3 or cell.shape[1] < 3:
                continue

            if is_effectively_empty(cell):
                continue

            templates.append(
                TemplateImage(
                    name=f"{sheet_name}_{index:03d}",
                    source=source,
                    image_rgba=cell,
                )
            )
            index += 1

    return templates


def trim_sprite(rgba: np.ndarray, tolerance: int = 12) -> np.ndarray:
    alpha = rgba[:, :, 3]
    if np.any(alpha > 0) and np.any(alpha < 255):
        ys, xs = np.where(alpha > 0)
    elif np.any(alpha > 0):
        bg = rgba[0, 0, :3].astype(np.int16)
        diff = np.max(np.abs(rgba[:, :, :3].astype(np.int16) - bg), axis=2)
        ys, xs = np.where(diff > tolerance)
    else:
        ys, xs = np.where(alpha > 0)

    if len(xs) == 0 or len(ys) == 0:
        return rgba[0:0, 0:0]

    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    return rgba[top:bottom, left:right].copy()


def is_effectively_empty(rgba: np.ndarray) -> bool:
    alpha = rgba[:, :, 3]
    if np.any(alpha > 8):
        return False
    rgb = rgba[:, :, :3]
    return bool(np.std(rgb) < 1.0)


def iter_image_frames(path: str | Path) -> list[tuple[int | None, np.ndarray]]:
    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1) or 1)
        if frame_count <= 1:
            return [(None, np.array(image.convert("RGBA")))]

        frames: list[tuple[int | None, np.ndarray]] = []
        for index, frame in enumerate(ImageSequence.Iterator(image)):
            frames.append((index, np.array(frame.copy().convert("RGBA"))))
        return frames
