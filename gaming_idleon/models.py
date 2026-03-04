from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MIN_FERTILIZER_SETTLE_SECONDS = 0.25
FERTILIZER_ROW_NAMES = ("Value", "Speed", "Capacity")


@dataclass(slots=True)
class TemplateImage:
    name: str
    source: str
    image_rgba: np.ndarray
    priority: int = 0
    kind: str = "plant"

    @property
    def width(self) -> int:
        return int(self.image_rgba.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_rgba.shape[0])


@dataclass(slots=True)
class SheetImportOptions:
    rows: int
    cols: int
    margin: int = 0
    spacing: int = 0
    trim_empty: bool = True
    background_tolerance: int = 12


@dataclass(slots=True)
class RunSettings:
    threshold: float = 0.56
    min_scale: float = 1.0
    max_scale: float = 1.0
    scale_step: float = 0.05
    max_matches_per_template: int = 0
    max_clicks_per_cycle: int = 0
    scan_interval: float = 0.40
    cooldown_seconds: float = 1.2
    start_delay: float = 3.0
    move_duration_min: float = 0.10
    move_duration_max: float = 0.22
    mouse_speed: float = 1.0
    jitter_pixels: int = 4
    dedupe_distance_ratio: float = 0.50
    edge_blur_size: int = 3
    edge_dilate_iterations: int = 1
    fertilizer_check_interval: float = 8.0
    fertilizer_toggle_delay: float = 0.65
    upgrade_delay: float = 0.85
    bits_retry_interval: float = 0.25
    affordable_margin: float = 1.2
    sprinkler_interval: float = 0.0
    sprinkler_growth_delay: float = 0.30
    rotation_max_degrees: float = 18.0
    rotation_step: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "scale_step": self.scale_step,
            "max_matches_per_template": self.max_matches_per_template,
            "max_clicks_per_cycle": self.max_clicks_per_cycle,
            "scan_interval": self.scan_interval,
            "cooldown_seconds": self.cooldown_seconds,
            "start_delay": self.start_delay,
            "move_duration_min": self.move_duration_min,
            "move_duration_max": self.move_duration_max,
            "mouse_speed": self.mouse_speed,
            "jitter_pixels": self.jitter_pixels,
            "dedupe_distance_ratio": self.dedupe_distance_ratio,
            "edge_blur_size": self.edge_blur_size,
            "edge_dilate_iterations": self.edge_dilate_iterations,
            "fertilizer_check_interval": self.fertilizer_check_interval,
            "fertilizer_toggle_delay": self.fertilizer_toggle_delay,
            "upgrade_delay": self.upgrade_delay,
            "bits_retry_interval": self.bits_retry_interval,
            "affordable_margin": self.affordable_margin,
            "sprinkler_interval": self.sprinkler_interval,
            "sprinkler_growth_delay": self.sprinkler_growth_delay,
            "rotation_max_degrees": self.rotation_max_degrees,
            "rotation_step": self.rotation_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunSettings":
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass(slots=True)
class MatchResult:
    template_name: str
    template_source: str
    score: float
    priority: int
    center_x: int
    center_y: int
    width: int
    height: int
    scale: float


@dataclass(slots=True)
class PreviewTarget:
    template_name: str
    score: float
    priority: int
    image_x: int
    image_y: int
    box_left: int
    box_top: int
    box_width: int
    box_height: int


@dataclass(slots=True)
class CyclePlanPreview:
    planned_targets: list[PreviewTarget]
    raw_match_count: int = 0
    filtered_match_count: int = 0


@dataclass(slots=True)
class PixelRect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + (self.width // 2), self.top + (self.height // 2))


@dataclass(slots=True)
class NormalizedRect:
    left: float
    top: float
    width: float
    height: float

    def clamp(self) -> "NormalizedRect":
        left = min(max(self.left, 0.0), 1.0)
        top = min(max(self.top, 0.0), 1.0)
        width = min(max(self.width, 0.0), 1.0 - left)
        height = min(max(self.height, 0.0), 1.0 - top)
        return NormalizedRect(left=left, top=top, width=width, height=height)

    def to_pixel_rect(self, image_width: int, image_height: int) -> PixelRect:
        left = int(round(self.left * image_width))
        top = int(round(self.top * image_height))
        right = int(round((self.left + self.width) * image_width))
        bottom = int(round((self.top + self.height) * image_height))
        return PixelRect(
            left=max(left, 0),
            top=max(top, 0),
            width=max(right - left, 1),
            height=max(bottom - top, 1),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NormalizedRect | None":
        if not data:
            return None
        return cls(
            left=float(data["left"]),
            top=float(data["top"]),
            width=float(data["width"]),
            height=float(data["height"]),
        )


@dataclass(slots=True)
class WindowInfo:
    window_id: int
    owner_name: str
    title: str
    bounds: PixelRect
    thumbnail_size: tuple[int, int]
    owner_pid: int | None = None

    @property
    def label(self) -> str:
        title = self.title.strip() or "Untitled"
        return f"{self.owner_name} - {title}"


@dataclass(slots=True)
class WindowCapture:
    window_id: int
    image_rgb: np.ndarray
    bounds: PixelRect
    scale_x: float
    scale_y: float

    @property
    def image_width(self) -> int:
        return int(self.image_rgb.shape[1])

    @property
    def image_height(self) -> int:
        return int(self.image_rgb.shape[0])

    def rect_from_normalized(self, rect: NormalizedRect) -> PixelRect:
        return rect.to_pixel_rect(self.image_width, self.image_height)

    def crop(self, rect: NormalizedRect) -> np.ndarray:
        pixel_rect = self.rect_from_normalized(rect)
        return self.image_rgb[pixel_rect.top:pixel_rect.bottom, pixel_rect.left:pixel_rect.right].copy()

    def image_to_screen(self, x: int, y: int) -> tuple[int, int]:
        screen_x = int(round(self.bounds.left + (x / max(self.scale_x, 1e-6))))
        screen_y = int(round(self.bounds.top + (y / max(self.scale_y, 1e-6))))
        return (screen_x, screen_y)


@dataclass(slots=True)
class DisplayAmount:
    raw_text: str
    mantissa: float
    suffix: str
    tier_name: str
    tier_index: int
    rank: int

    def is_at_least(self, other: "DisplayAmount", margin: float = 1.0) -> bool:
        if self.rank != other.rank:
            return self.rank > other.rank
        return self.mantissa >= (other.mantissa * margin)

    def short_label(self) -> str:
        return f"{self.mantissa:g}{self.suffix} [{self.tier_name}]"


@dataclass(slots=True)
class FertilizerOffer:
    row_index: int
    cost: DisplayAmount
    cost_rect: NormalizedRect
    button_rect: NormalizedRect

    @property
    def row_name(self) -> str:
        return fertilizer_row_name(self.row_index)


def fertilizer_row_name(row_index: int) -> str:
    if 0 <= row_index < len(FERTILIZER_ROW_NAMES):
        return FERTILIZER_ROW_NAMES[row_index]
    return f"Row {row_index + 1}"


@dataclass(slots=True)
class GameCalibration:
    field_rect: NormalizedRect | None = None
    bits_rect: NormalizedRect | None = None
    fertilizer_button_rect: NormalizedRect | None = None
    fertilizer_cost_1_rect: NormalizedRect | None = None
    fertilizer_purchase_1_rect: NormalizedRect | None = None
    fertilizer_purchase_1_ref: str | None = None
    fertilizer_cost_2_rect: NormalizedRect | None = None
    fertilizer_purchase_2_rect: NormalizedRect | None = None
    fertilizer_purchase_2_ref: str | None = None
    fertilizer_cost_3_rect: NormalizedRect | None = None
    fertilizer_purchase_3_rect: NormalizedRect | None = None
    fertilizer_purchase_3_ref: str | None = None
    sprinkler_rect: NormalizedRect | None = None
    imports_rect: NormalizedRect | None = None

    def fertilizer_offer_slots(self) -> list[tuple[NormalizedRect | None, NormalizedRect | None]]:
        return [
            (self.fertilizer_cost_1_rect, self.fertilizer_purchase_1_rect),
            (self.fertilizer_cost_2_rect, self.fertilizer_purchase_2_rect),
            (self.fertilizer_cost_3_rect, self.fertilizer_purchase_3_rect),
        ]

    def fertilizer_ready(self) -> bool:
        return all(
            cost_rect is not None and button_rect is not None
            for cost_rect, button_rect in self.fertilizer_offer_slots()
        )

    def fertilizer_purchase_refs(self) -> list[str | None]:
        return [
            self.fertilizer_purchase_1_ref,
            self.fertilizer_purchase_2_ref,
            self.fertilizer_purchase_3_ref,
        ]

    def ready_for_run(self) -> bool:
        return all(
            rect is not None
            for rect in [
                self.field_rect,
                self.bits_rect,
                self.fertilizer_button_rect,
            ]
        ) and self.fertilizer_ready()

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_rect": self.field_rect.to_dict() if self.field_rect else None,
            "bits_rect": self.bits_rect.to_dict() if self.bits_rect else None,
            "fertilizer_button_rect": (
                self.fertilizer_button_rect.to_dict() if self.fertilizer_button_rect else None
            ),
            "fertilizer_cost_1_rect": (
                self.fertilizer_cost_1_rect.to_dict() if self.fertilizer_cost_1_rect else None
            ),
            "fertilizer_purchase_1_rect": (
                self.fertilizer_purchase_1_rect.to_dict() if self.fertilizer_purchase_1_rect else None
            ),
            "fertilizer_cost_2_rect": (
                self.fertilizer_cost_2_rect.to_dict() if self.fertilizer_cost_2_rect else None
            ),
            "fertilizer_purchase_2_rect": (
                self.fertilizer_purchase_2_rect.to_dict() if self.fertilizer_purchase_2_rect else None
            ),
            "fertilizer_purchase_1_ref": self.fertilizer_purchase_1_ref,
            "fertilizer_purchase_2_ref": self.fertilizer_purchase_2_ref,
            "fertilizer_cost_3_rect": (
                self.fertilizer_cost_3_rect.to_dict() if self.fertilizer_cost_3_rect else None
            ),
            "fertilizer_purchase_3_rect": (
                self.fertilizer_purchase_3_rect.to_dict() if self.fertilizer_purchase_3_rect else None
            ),
            "fertilizer_purchase_3_ref": self.fertilizer_purchase_3_ref,
            "sprinkler_rect": self.sprinkler_rect.to_dict() if self.sprinkler_rect else None,
            "imports_rect": self.imports_rect.to_dict() if self.imports_rect else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GameCalibration":
        if not data:
            return cls()

        cost_1_rect = NormalizedRect.from_dict(data.get("fertilizer_cost_1_rect"))
        purchase_1_rect = NormalizedRect.from_dict(data.get("fertilizer_purchase_1_rect"))
        cost_2_rect = NormalizedRect.from_dict(data.get("fertilizer_cost_2_rect"))
        purchase_2_rect = NormalizedRect.from_dict(data.get("fertilizer_purchase_2_rect"))
        cost_3_rect = NormalizedRect.from_dict(data.get("fertilizer_cost_3_rect"))
        purchase_3_rect = NormalizedRect.from_dict(data.get("fertilizer_purchase_3_rect"))

        if any(
            rect is None
            for rect in [
                cost_1_rect,
                purchase_1_rect,
                cost_2_rect,
                purchase_2_rect,
                cost_3_rect,
                purchase_3_rect,
            ]
        ):
            legacy_panel_rect = NormalizedRect.from_dict(data.get("fertilizer_panel_rect"))
            if legacy_panel_rect is not None:
                legacy_slots = split_legacy_fertilizer_panel(legacy_panel_rect)
                cost_1_rect = cost_1_rect or legacy_slots[0][0]
                purchase_1_rect = purchase_1_rect or legacy_slots[0][1]
                cost_2_rect = cost_2_rect or legacy_slots[1][0]
                purchase_2_rect = purchase_2_rect or legacy_slots[1][1]
                cost_3_rect = cost_3_rect or legacy_slots[2][0]
                purchase_3_rect = purchase_3_rect or legacy_slots[2][1]

        return cls(
            field_rect=NormalizedRect.from_dict(data.get("field_rect")),
            bits_rect=NormalizedRect.from_dict(data.get("bits_rect")),
            fertilizer_button_rect=NormalizedRect.from_dict(data.get("fertilizer_button_rect")),
            fertilizer_cost_1_rect=cost_1_rect,
            fertilizer_purchase_1_rect=purchase_1_rect,
            fertilizer_purchase_1_ref=str(data.get("fertilizer_purchase_1_ref") or "") or None,
            fertilizer_cost_2_rect=cost_2_rect,
            fertilizer_purchase_2_rect=purchase_2_rect,
            fertilizer_purchase_2_ref=str(data.get("fertilizer_purchase_2_ref") or "") or None,
            fertilizer_cost_3_rect=cost_3_rect,
            fertilizer_purchase_3_rect=purchase_3_rect,
            fertilizer_purchase_3_ref=str(data.get("fertilizer_purchase_3_ref") or "") or None,
            sprinkler_rect=NormalizedRect.from_dict(data.get("sprinkler_rect")),
            imports_rect=NormalizedRect.from_dict(data.get("imports_rect")),
        )


def split_legacy_fertilizer_panel(
    panel_rect: NormalizedRect,
) -> list[tuple[NormalizedRect, NormalizedRect]]:
    slots: list[tuple[NormalizedRect, NormalizedRect]] = []
    row_height = panel_rect.height / 3.0
    for row_index in range(3):
        row_top = panel_rect.top + (row_index * row_height)
        slots.append(
            (
                NormalizedRect(
                    left=panel_rect.left,
                    top=row_top,
                    width=panel_rect.width * 0.68,
                    height=row_height,
                ).clamp(),
                NormalizedRect(
                    left=panel_rect.left + (panel_rect.width * 0.72),
                    top=row_top,
                    width=panel_rect.width * 0.28,
                    height=row_height,
                ).clamp(),
            )
        )
    return slots
