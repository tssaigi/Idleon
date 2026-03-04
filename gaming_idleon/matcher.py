from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import MatchResult, RunSettings, TemplateImage


@dataclass(slots=True)
class TemplateVariant:
    template_name: str
    template_source: str
    priority: int
    kind: str
    scale: float
    rotation: float
    edge_map: np.ndarray
    mask: np.ndarray | None
    width: int
    height: int


def prepare_templates(
    templates: list[TemplateImage],
    settings: RunSettings,
) -> list[TemplateVariant]:
    variants: list[TemplateVariant] = []
    scales = build_scales(settings)
    for template in templates:
        base_edge_map, base_mask = build_template_edge_map(template.image_rgba, settings)
        for rotation in build_rotations(settings):
            rotated_edge_map = rotate_image(base_edge_map, rotation, cv2.INTER_LINEAR, 0)
            rotated_mask = None
            if base_mask is not None:
                rotated_mask = rotate_image(base_mask, rotation, cv2.INTER_NEAREST, 0)
                if np.count_nonzero(rotated_mask) == 0:
                    rotated_mask = None

            for scale in scales:
                scaled_edge_map = resize_image(rotated_edge_map, scale, cv2.INTER_LINEAR)
                if scaled_edge_map.shape[0] < 3 or scaled_edge_map.shape[1] < 3:
                    continue
                scaled_mask = None
                if rotated_mask is not None:
                    scaled_mask = resize_image(rotated_mask, scale, cv2.INTER_NEAREST)
                    if np.count_nonzero(scaled_mask) == 0:
                        scaled_mask = None
                variants.append(
                    TemplateVariant(
                        template_name=template.name,
                        template_source=template.source,
                        priority=template.priority,
                        kind=template.kind,
                        scale=scale,
                        rotation=rotation,
                        edge_map=scaled_edge_map,
                        mask=scaled_mask,
                        width=int(scaled_edge_map.shape[1]),
                        height=int(scaled_edge_map.shape[0]),
                    )
                )
    return variants


def find_matches(
    screen_rgb: np.ndarray,
    variants: list[TemplateVariant],
    settings: RunSettings,
) -> list[MatchResult]:
    screen_edges = build_screen_edge_map(screen_rgb, settings)
    accepted: list[MatchResult] = []
    counts_by_template: dict[str, int] = {}

    for variant in variants:
        if variant.width > screen_edges.shape[1] or variant.height > screen_edges.shape[0]:
            continue

        method = cv2.TM_CCORR_NORMED
        if variant.mask is not None:
            result = cv2.matchTemplate(screen_edges, variant.edge_map, method, mask=variant.mask)
        else:
            result = cv2.matchTemplate(screen_edges, variant.edge_map, method)

        ys, xs = extract_peak_locations(
            result,
            threshold=settings.threshold,
            min_distance=max(
                int(max(variant.width, variant.height) * max(settings.dedupe_distance_ratio, 0.35)),
                8,
            ),
        )
        if len(xs) == 0:
            continue

        candidates = sorted(
            (
                (
                    float(result[y, x]),
                    int(x),
                    int(y),
                )
                for y, x in zip(ys, xs)
            ),
            reverse=True,
        )

        for score, x, y in candidates:
            template_count = counts_by_template.get(variant.template_name, 0)
            if settings.max_matches_per_template > 0 and template_count >= settings.max_matches_per_template:
                break

            center_x = x + (variant.width // 2)
            center_y = y + (variant.height // 2)

            if is_duplicate_match(
                accepted,
                variant.template_name,
                center_x,
                center_y,
                variant.width,
                variant.height,
                settings.dedupe_distance_ratio,
            ):
                continue

            accepted.append(
                MatchResult(
                    template_name=variant.template_name,
                    template_source=variant.template_source,
                    score=score,
                    priority=variant.priority,
                    center_x=center_x,
                    center_y=center_y,
                    width=variant.width,
                    height=variant.height,
                    scale=variant.scale,
                )
            )
            counts_by_template[variant.template_name] = template_count + 1

    accepted.sort(key=lambda item: (item.priority, item.score), reverse=True)
    return accepted


def extract_peak_locations(
    result: np.ndarray,
    threshold: float,
    min_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    if result.size == 0:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    threshold_mask = result >= threshold
    if not np.any(threshold_mask):
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    kernel_size = max(int(min_distance), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(result, kernel)
    peak_mask = threshold_mask & (result >= (dilated - 1e-6))
    ys, xs = np.where(peak_mask)
    return (ys, xs)


def build_scales(settings: RunSettings) -> list[float]:
    min_scale = min(settings.min_scale, settings.max_scale)
    max_scale = max(settings.min_scale, settings.max_scale)
    if abs(min_scale - max_scale) < 0.001:
        return [round(min_scale, 3)]

    scales: list[float] = []
    current = min_scale
    while current <= max_scale + 1e-9:
        scales.append(round(current, 3))
        current += max(settings.scale_step, 0.01)
    return scales


def build_rotations(settings: RunSettings) -> list[float]:
    max_degrees = max(float(settings.rotation_max_degrees), 0.0)
    step = max(float(settings.rotation_step), 1.0)
    if max_degrees <= 0.01:
        return [0.0]

    rotations: list[float] = [0.0]
    current = step
    while current <= max_degrees + 1e-9:
        rotations.append(round(current, 3))
        rotations.append(round(-current, 3))
        current += step
    return rotations


def resize_image(image: np.ndarray, scale: float, interpolation: int) -> np.ndarray:
    if abs(scale - 1.0) < 0.001:
        return image
    return cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=interpolation)


def rotate_image(image: np.ndarray, angle: float, interpolation: int, fill_value: int) -> np.ndarray:
    if abs(angle) < 0.001:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    bound_width = int((height * sin) + (width * cos))
    bound_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (bound_width / 2.0) - center[0]
    matrix[1, 2] += (bound_height / 2.0) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (bound_width, bound_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value,
    )


def is_duplicate_match(
    matches: list[MatchResult],
    template_name: str,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    ratio: float,
) -> bool:
    min_distance = max(int(max(width, height) * ratio), 8)
    min_distance_sq = min_distance * min_distance
    for match in matches:
        if match.template_name != template_name:
            continue
        dx = match.center_x - center_x
        dy = match.center_y - center_y
        if (dx * dx) + (dy * dy) <= min_distance_sq:
            return True
    return False


def build_template_edge_map(
    rgba: np.ndarray,
    settings: RunSettings,
) -> tuple[np.ndarray, np.ndarray | None]:
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    alpha = rgba[:, :, 3]
    mask = (alpha > 18).astype(np.uint8) * 255 if np.any(alpha < 250) else None

    if mask is not None:
        gray = gray.copy()
        gray[mask == 0] = 0

    edges = build_edges(gray, settings)
    if mask is not None:
        silhouette = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edges = np.maximum(edges, silhouette)
        edges[mask == 0] = 0

    return (edges, mask)


def build_screen_edge_map(screen_rgb: np.ndarray, settings: RunSettings) -> np.ndarray:
    gray = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2GRAY)
    return build_edges(gray, settings)


def build_edges(gray: np.ndarray, settings: RunSettings) -> np.ndarray:
    blur_size = max(int(settings.edge_blur_size), 1)
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges = cv2.Canny(blurred, 40, 120)
    iterations = max(int(settings.edge_dilate_iterations), 0)
    if iterations:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=iterations)
    return cv2.GaussianBlur(edges, (3, 3), 0)
