from __future__ import annotations

import base64
import io
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

try:
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
except Exception:  # pragma: no cover - optional OCR backend
    EngineType = None
    LangRec = None
    ModelType = None
    OCRVersion = None
    RapidOCR = None

try:
    import Foundation
    import Vision
    import objc
except Exception:  # pragma: no cover - optional macOS OCR backend
    Foundation = None
    Vision = None
    objc = None

from .models import DisplayAmount, FertilizerOffer, GameCalibration, NormalizedRect, WindowCapture


SUFFIX_ORDER = ["", "K", "M", "B", "T", "Q", "QQ"]
TIER_NAMES = ["white", "green", "red", "purple", "blue"]
COLOR_ANCHORS = {
    "white": np.array([245, 245, 245], dtype=np.float32),
    "green": np.array([125, 235, 120], dtype=np.float32),
    "red": np.array([250, 110, 110], dtype=np.float32),
    "purple": np.array([205, 130, 255], dtype=np.float32),
    "blue": np.array([120, 185, 255], dtype=np.float32),
}
AMOUNT_RE = re.compile(r"([0-9OILSD]+(?:[.][0-9OILSD]+)?)\s*(QQ|K|M|B|T|Q)?")
OCR_CONFIGS = [
    (
        "--psm 7 --oem 3 "
        "-c tessedit_char_whitelist=0123456789.,KMBTQOILSDqobtm "
        "-c load_system_dawg=0 -c load_freq_dawg=0 -c classify_bln_numeric_mode=1"
    ),
    (
        "--psm 8 --oem 3 "
        "-c tessedit_char_whitelist=0123456789.,KMBTQOILSDqobtm "
        "-c load_system_dawg=0 -c load_freq_dawg=0 -c classify_bln_numeric_mode=1"
    ),
    (
        "--psm 13 --oem 3 "
        "-c tessedit_char_whitelist=0123456789.,KMBTQOILSDqobtm "
        "-c load_system_dawg=0 -c load_freq_dawg=0 -c classify_bln_numeric_mode=1"
    ),
]
FAST_OCR_CONFIGS = OCR_CONFIGS[:1]
FERTILIZER_ICON_FRACTIONS = (0.18, 0.24, 0.30)
VISION_LANGUAGES = ["en-US"]
VISION_CUSTOM_WORDS = ["K", "M", "B", "T", "Q", "QQ"]
RAPIDOCR_MIN_SCORE = 0.45
_RAPIDOCR_ENGINE = None
_RAPIDOCR_ENGINE_INITIALIZED = False


def read_bits(crop_rgb: np.ndarray) -> DisplayAmount | None:
    return read_amount_with_icon(crop_rgb, icon_fraction=0.24)


def read_fertilizer_offers(
    capture: WindowCapture,
    calibration: GameCalibration,
) -> list[FertilizerOffer]:
    offers: list[FertilizerOffer] = []
    for row_index, (cost_rect, button_rect) in enumerate(calibration.fertilizer_offer_slots()):
        if cost_rect is None or button_rect is None:
            continue
        amount = read_fertilizer_cost_amount(capture, cost_rect, button_rect)
        if amount is None or amount.mantissa <= 0:
            continue
        offers.append(
            FertilizerOffer(
                row_index=row_index,
                cost=amount,
                cost_rect=cost_rect,
                button_rect=button_rect,
            )
        )
    return offers


def is_fertilizer_view(
    capture: WindowCapture,
    calibration: GameCalibration,
    minimum_rows: int = 2,
) -> bool:
    calibrated_rows = sum(
        1
        for cost_rect, button_rect in calibration.fertilizer_offer_slots()
        if cost_rect is not None and button_rect is not None
    )
    if calibrated_rows == 0:
        return False
    required_rows = min(max(minimum_rows, 1), calibrated_rows)
    offers = read_fertilizer_offers(capture, calibration)
    button_refs = [ref for ref in calibration.fertilizer_purchase_refs() if ref]
    button_matches = count_matching_fertilizer_buttons(capture, calibration)
    if len(button_refs) >= 2:
        if button_matches >= 2:
            return True
        if button_matches >= 1 and len(offers) >= 1 and required_rows <= 1:
            return True
        return False
    if len(button_refs) == 1:
        return button_matches >= 1 and (len(offers) >= 1 or required_rows <= 1)
    return len(offers) >= required_rows


def read_amount(
    crop_rgb: np.ndarray,
    tier_override: str | None = None,
    *,
    fast: bool = False,
) -> DisplayAmount | None:
    tier = tier_override or detect_color_tier(crop_rgb)
    rapid_candidates: list[DisplayAmount] = []
    vision_candidates: list[DisplayAmount] = []
    parsed_candidates: list[DisplayAmount] = []
    configs = FAST_OCR_CONFIGS if fast else OCR_CONFIGS
    timeout = 0.35 if fast else None
    for candidate in build_rapidocr_variants(crop_rgb, fast=fast):
        for text in rapidocr_image_to_strings(candidate):
            parsed = parse_display_amount(text, tier)
            if parsed is not None:
                rapid_candidates.append(parsed)
    best_rapid = choose_best_amount(rapid_candidates)
    if best_rapid is not None and not is_suspicious_amount(best_rapid):
        if candidate_consensus_count(rapid_candidates, best_rapid) >= 2:
            return best_rapid

    for candidate in build_vision_variants(crop_rgb, fast=fast):
        image = Image.fromarray(candidate)
        for text in vision_image_to_strings(image, fast=fast):
            parsed = parse_display_amount(text, tier)
            if parsed is not None:
                vision_candidates.append(parsed)
    best_vision = choose_best_amount(vision_candidates)
    if best_vision is not None and not is_suspicious_amount(best_vision):
        if candidate_consensus_count(vision_candidates, best_vision) >= 2:
            return best_vision

    for candidate in build_ocr_variants(crop_rgb, fast=fast):
        image = Image.fromarray(candidate)
        for config in configs:
            text = image_to_string_safe(image, config=config, timeout=timeout)
            parsed = parse_display_amount(text, tier)
            if parsed is not None:
                parsed_candidates.append(parsed)
    return choose_best_amount(rapid_candidates + vision_candidates + parsed_candidates)


def read_amount_with_icon(
    crop_rgb: np.ndarray,
    icon_fraction: float = 0.24,
) -> DisplayAmount | None:
    amount_crop, icon_crop = split_amount_icon_crop(crop_rgb, icon_fraction)
    amount_crop = tighten_amount_crop(amount_crop)
    tier = detect_bits_icon_tier(icon_crop)
    candidates: list[DisplayAmount] = []
    direct_amount = read_amount(amount_crop, tier_override=tier)
    if direct_amount is not None:
        candidates.append(direct_amount)
    segmented_candidates: list[DisplayAmount] = []
    for segmented_crop in extract_right_aligned_amount_crops(amount_crop):
        segmented_amount = read_amount(segmented_crop, tier_override=tier, fast=True)
        if segmented_amount is not None:
            segmented_candidates.append(segmented_amount)
    suffixed_segmented = [amount for amount in segmented_candidates if amount.suffix]
    if suffixed_segmented:
        return choose_best_amount(suffixed_segmented)
    candidates.extend(segmented_candidates)
    return choose_best_amount(candidates)


def read_fertilizer_cost_amount(
    capture: WindowCapture,
    cost_rect: NormalizedRect,
    button_rect: NormalizedRect,
) -> DisplayAmount | None:
    source_crops = build_fertilizer_cost_crops(capture, cost_rect, button_rect)
    if not source_crops:
        return None
    source_results: list[tuple[int, DisplayAmount]] = []
    for source_index, source_crop in enumerate(source_crops):
        source_candidates: list[DisplayAmount] = []
        source_segmented_candidates: list[DisplayAmount] = []
        source_tier = detect_fertilizer_cost_tier(
            capture,
            cost_rect,
            button_rect,
            source_crop[:, -max(source_crop.shape[1] // 5, 1):, :],
        )
        for icon_fraction in FERTILIZER_ICON_FRACTIONS:
            amount_crop, icon_crop = split_amount_icon_crop(source_crop, icon_fraction=icon_fraction)
            tier = detect_fertilizer_cost_tier(capture, cost_rect, button_rect, icon_crop)
            amount = read_amount(amount_crop, tier_override=tier, fast=True)
            if amount is not None:
                source_candidates.append(amount)
        fallback_tier = source_tier
        fallback_width = max(int(round(source_crop.shape[1] * 0.82)), 1)
        fallback_crop = source_crop[:, :fallback_width, :].copy()
        fallback_amount = read_amount(fallback_crop, tier_override=fallback_tier, fast=True)
        if fallback_amount is not None:
            source_candidates.append(fallback_amount)
        for segmented_crop in extract_fertilizer_amount_crops(source_crop):
            segmented_amount = read_amount(segmented_crop, tier_override=source_tier, fast=True)
            if segmented_amount is not None:
                source_segmented_candidates.append(segmented_amount)

        best_source_amount = choose_preferred_amount(source_candidates, source_segmented_candidates)
        if best_source_amount is not None:
            source_results.append((source_index, best_source_amount))

    if not source_results:
        fallback_crop = max(source_crops, key=lambda crop: crop.shape[1] * crop.shape[0])
        fallback_tier = detect_fertilizer_cost_tier(
            capture,
            cost_rect,
            button_rect,
            fallback_crop[:, -max(fallback_crop.shape[1] // 5, 1):, :],
        )
        slow_amount = read_amount(fallback_crop, tier_override=fallback_tier, fast=False)
        if slow_amount is not None:
            source_results.append((len(source_crops), slow_amount))

    for preferred_sources in ((0,), (0, 1), tuple(range(len(source_crops) + 1))):
        direct_suffixed = [
            amount
            for source_index, amount in source_results
            if source_index in preferred_sources and amount.suffix
        ]
        if direct_suffixed:
            return choose_best_amount(direct_suffixed)

    for preferred_sources in ((0,), (0, 1), tuple(range(len(source_crops) + 1))):
        direct_candidates = [
            amount
            for source_index, amount in source_results
            if source_index in preferred_sources
        ]
        if direct_candidates:
            return choose_best_amount(direct_candidates)
    return None


def choose_preferred_amount(
    candidates: list[DisplayAmount],
    segmented_candidates: list[DisplayAmount],
) -> DisplayAmount | None:
    direct_suffixed = [amount for amount in candidates if amount.suffix]
    if direct_suffixed:
        return choose_best_amount(direct_suffixed)
    direct_any = choose_best_amount(candidates)
    if direct_any is not None:
        return direct_any
    segmented_suffixed = [amount for amount in segmented_candidates if amount.suffix]
    if segmented_suffixed:
        return choose_best_amount(segmented_suffixed)
    return choose_best_amount(segmented_candidates)


def build_fertilizer_cost_crops(
    capture: WindowCapture,
    cost_rect: NormalizedRect,
    button_rect: NormalizedRect,
) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    seen_shapes: set[tuple[int, int, int]] = set()
    for crop in (
        capture.crop(cost_rect),
        expand_fertilizer_cost_crop(capture, cost_rect, button_rect),
        build_fertilizer_row_band_crop(capture, cost_rect, button_rect),
    ):
        if crop.size == 0:
            continue
        shape = tuple(crop.shape)
        if shape in seen_shapes:
            continue
        crops.append(crop)
        seen_shapes.add(shape)
    return crops


def expand_fertilizer_cost_crop(
    capture: WindowCapture,
    cost_rect: NormalizedRect,
    button_rect: NormalizedRect,
) -> np.ndarray:
    cost_px = capture.rect_from_normalized(cost_rect)
    button_px = capture.rect_from_normalized(button_rect)
    pad_x = max(int(round(cost_px.width * 0.12)), 4)
    pad_y = max(int(round(cost_px.height * 0.18)), 3)
    left = max(cost_px.left - pad_x, 0)
    top = max(cost_px.top - pad_y, 0)
    right_limit = capture.image_rgb.shape[1]
    if button_px.left > cost_px.right:
        right_limit = min(button_px.left, right_limit)
    right = min(cost_px.right + pad_x, right_limit)
    bottom = min(cost_px.bottom + pad_y, capture.image_rgb.shape[0])
    if right <= left or bottom <= top:
        return capture.crop(cost_rect)
    return capture.image_rgb[top:bottom, left:right].copy()


def build_fertilizer_row_band_crop(
    capture: WindowCapture,
    cost_rect: NormalizedRect,
    button_rect: NormalizedRect,
) -> np.ndarray:
    cost_px = capture.rect_from_normalized(cost_rect)
    button_px = capture.rect_from_normalized(button_rect)
    if cost_px.width <= 0 or cost_px.height <= 0:
        return capture.crop(cost_rect)

    pad_x = max(int(round(cost_px.width * 0.22)), 6)
    pad_y = max(int(round(cost_px.height * 0.28)), 4)
    left = max(cost_px.left - pad_x, 0)
    top = max(min(cost_px.top, button_px.top) - pad_y, 0)
    bottom = min(max(cost_px.bottom, button_px.bottom) + pad_y, capture.image_rgb.shape[0])

    right_limit = capture.image_rgb.shape[1]
    if button_px.left > cost_px.left:
        right_limit = min(button_px.left, right_limit)
    right = min(max(cost_px.right + pad_x, left + 1), right_limit)
    if right <= left or bottom <= top:
        return capture.crop(cost_rect)
    return capture.image_rgb[top:bottom, left:right].copy()


def extract_fertilizer_amount_crops(source_crop: np.ndarray) -> list[np.ndarray]:
    if source_crop.size == 0:
        return []

    text_end = detect_probable_fertilizer_icon_left(source_crop)
    text_crop = source_crop[:, :text_end, :].copy() if text_end > 0 else source_crop
    if text_crop.size == 0:
        return []
    return extract_right_aligned_amount_crops(text_crop)


def extract_right_aligned_amount_crops(text_crop: np.ndarray) -> list[np.ndarray]:
    if text_crop.size == 0:
        return []

    gray = cv2.cvtColor(text_crop, cv2.COLOR_RGB2GRAY)
    mask = build_foreground_mask(gray)
    bright_mask = (gray > 130).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, bright_mask)
    segments = find_foreground_column_segments(mask)
    if not segments:
        tightened = tighten_amount_crop(text_crop)
        return [tightened] if tightened.size != 0 else []

    crops: list[np.ndarray] = []
    seen_shapes: set[tuple[int, int, int]] = set()
    candidate_segments = [segments[-1]]
    if len(segments) >= 2:
        previous_left, previous_right = segments[-2]
        current_left, current_right = segments[-1]
        if current_left - previous_right <= max(int(round(text_crop.shape[1] * 0.04)), 6):
            candidate_segments.append((previous_left, current_right))

    for left, right in candidate_segments:
        segment_mask = mask[:, left:right]
        bbox = foreground_bbox(segment_mask)
        if bbox is None:
            continue
        seg_left, seg_top, seg_right, seg_bottom = bbox
        crop = text_crop[seg_top:seg_bottom, left + seg_left:left + seg_right].copy()
        tightened = tighten_amount_crop(crop)
        if tightened.size == 0:
            continue
        shape = tuple(tightened.shape)
        if shape in seen_shapes:
            continue
        crops.append(tightened)
        seen_shapes.add(shape)
    return crops


def detect_probable_fertilizer_icon_left(source_crop: np.ndarray) -> int:
    hsv = cv2.cvtColor(source_crop, cv2.COLOR_RGB2HSV)
    icon_mask = (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 110)
    column_energy = np.count_nonzero(icon_mask, axis=0)
    if column_energy.size == 0:
        return source_crop.shape[1]
    threshold = max(int(round(source_crop.shape[0] * 0.06)), 2)
    active_columns = np.flatnonzero(column_energy >= threshold)
    if active_columns.size == 0:
        return source_crop.shape[1]
    icon_left = int(active_columns[0])
    if icon_left < int(round(source_crop.shape[1] * 0.45)):
        return source_crop.shape[1]
    return max(icon_left - 2, 1)


def find_foreground_column_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    column_energy = np.count_nonzero(mask > 0, axis=0)
    threshold = max(int(round(mask.shape[0] * 0.05)), 1)
    active = column_energy >= threshold
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            if index - start >= 4:
                segments.append((start, index))
            start = None
    if start is not None and len(active) - start >= 4:
        segments.append((start, len(active)))
    return segments


def detect_fertilizer_cost_tier(
    capture: WindowCapture,
    cost_rect: NormalizedRect,
    button_rect: NormalizedRect,
    icon_crop: np.ndarray,
) -> str:
    tier = detect_bits_icon_tier(icon_crop)
    if tier != "white":
        return tier

    cost_px = capture.rect_from_normalized(cost_rect)
    button_px = capture.rect_from_normalized(button_rect)
    if button_px.left > cost_px.right:
        gap_right = min(
            button_px.left,
            cost_px.right + max(int(round(cost_px.width * 0.35)), 8),
        )
        gap_crop = capture.image_rgb[cost_px.top:cost_px.bottom, cost_px.right:gap_right].copy()
        gap_tier = detect_bits_icon_tier(gap_crop)
        if gap_tier != "white":
            return gap_tier

    right_strip_width = max(int(round(cost_px.width * 0.28)), 6)
    right_strip = capture.image_rgb[
        cost_px.top:cost_px.bottom,
        max(cost_px.right - right_strip_width, cost_px.left):cost_px.right,
    ].copy()
    strip_tier = detect_bits_icon_tier(right_strip)
    if strip_tier != "white":
        return strip_tier
    return tier


def build_fertilizer_button_reference(crop_rgb: np.ndarray) -> str | None:
    if crop_rgb.size == 0:
        return None
    prepared = prepare_fertilizer_button_reference(crop_rgb)
    success, encoded = cv2.imencode(".png", prepared)
    if not success:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def count_matching_fertilizer_buttons(
    capture: WindowCapture,
    calibration: GameCalibration,
    threshold: float = 0.72,
) -> int:
    matches = 0
    for (_, button_rect), button_ref in zip(
        calibration.fertilizer_offer_slots(),
        calibration.fertilizer_purchase_refs(),
    ):
        if button_rect is None or not button_ref:
            continue
        current_crop = capture.crop(button_rect)
        similarity = fertilizer_button_similarity(current_crop, button_ref)
        if similarity >= threshold:
            matches += 1
    return matches


def fertilizer_button_similarity(crop_rgb: np.ndarray, button_ref: str) -> float:
    if crop_rgb.size == 0 or not button_ref:
        return 0.0
    reference = decode_reference_image(button_ref)
    if reference is None:
        return 0.0
    current = prepare_fertilizer_button_reference(crop_rgb)
    result = cv2.matchTemplate(current, reference, cv2.TM_CCOEFF_NORMED)
    return float(result[0, 0])


def decode_reference_image(button_ref: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(button_ref.encode("ascii"))
    except Exception:
        return None
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        return None
    return decoded


def prepare_fertilizer_button_reference(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (96, 40), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def split_amount_icon_crop(
    crop_rgb: np.ndarray,
    icon_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_rgb.size == 0:
        return (crop_rgb, crop_rgb)
    width = crop_rgb.shape[1]
    icon_fraction = min(max(icon_fraction, 0.12), 0.45)
    split_x = min(max(int(round(width * (1.0 - icon_fraction))), 1), max(width - 1, 1))
    amount_crop = crop_rgb[:, :split_x, :].copy()
    icon_crop = crop_rgb[:, split_x:, :].copy()
    return (amount_crop, icon_crop)


def detect_bits_icon_tier(icon_crop_rgb: np.ndarray) -> str:
    if icon_crop_rgb.size == 0:
        return "white"

    hsv = cv2.cvtColor(icon_crop_rgb, cv2.COLOR_RGB2HSV)
    bright_mask = hsv[:, :, 2] > 120
    color_mask = hsv[:, :, 1] > 35
    candidate_mask = bright_mask & (color_mask | (hsv[:, :, 2] > 210))
    pixels = icon_crop_rgb[candidate_mask]
    if pixels.size == 0:
        return "white"
    return detect_color_tier(pixels.reshape(-1, 1, 3))


def parse_display_amount(text: str, tier_name: str) -> DisplayAmount | None:
    cleaned = sanitize_ocr_text(text)
    match = search_amount_match(cleaned)
    if match is None:
        return None

    raw_number = normalize_numeric_token(match.group(1))
    suffix = (match.group(2) or "").upper()
    suffix = suffix.replace("0", "Q").replace("O", "Q")
    raw_number, suffix = recover_suffix_from_suffixless_ocr(raw_number, suffix, tier_name)
    raw_number, suffix = recover_suffix_from_overflowing_mantissa(raw_number, suffix)
    if suffix == "QQQ":
        suffix = "QQ"
    if suffix not in SUFFIX_ORDER:
        return None

    try:
        mantissa = float(raw_number)
    except ValueError:
        return None

    tier_name = tier_name if tier_name in TIER_NAMES else "white"
    tier_index = TIER_NAMES.index(tier_name)
    rank = tier_index * len(SUFFIX_ORDER) + SUFFIX_ORDER.index(suffix)
    return DisplayAmount(
        raw_text=cleaned,
        mantissa=mantissa,
        suffix=suffix,
        tier_name=tier_name,
        tier_index=tier_index,
        rank=rank,
    )


def sanitize_ocr_text(text: str) -> str:
    cleaned = text.upper().strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("OQ", "QQ")
    cleaned = cleaned.replace("0Q", "QQ")
    cleaned = cleaned.replace("I", "1")
    cleaned = cleaned.replace("L", "1")
    cleaned = cleaned.replace("S", "5")
    cleaned = cleaned.replace("OB", "0B")
    return cleaned


def search_amount_match(cleaned: str) -> re.Match[str] | None:
    return AMOUNT_RE.search(cleaned)


def normalize_numeric_token(raw_number: str) -> str:
    translation = str.maketrans(
        {
            "O": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "S": "5",
        }
    )
    return raw_number.translate(translation)


def recover_suffix_from_suffixless_ocr(
    raw_number: str,
    suffix: str,
    tier_name: str,
) -> tuple[str, str]:
    if suffix:
        return (raw_number, suffix)
    if tier_name not in {"red", "purple", "blue"}:
        return (raw_number, suffix)
    if "." in raw_number or len(raw_number) < 4:
        return (raw_number, suffix)
    if raw_number.endswith("80") and len(raw_number) >= 5:
        return (raw_number[:-2], "B")
    if raw_number.endswith("8"):
        return (raw_number[:-1], "B")
    return (raw_number, suffix)


def recover_suffix_from_overflowing_mantissa(
    raw_number: str,
    suffix: str,
) -> tuple[str, str]:
    if not suffix or "." in raw_number or len(raw_number) <= 3:
        return (raw_number, suffix)

    candidate = raw_number
    if candidate.endswith("80") and len(candidate) >= 5:
        candidate = candidate[:-2]
    elif candidate.endswith("8"):
        candidate = candidate[:-1]
    elif len(candidate) >= 4:
        candidate = candidate[:-1]

    if not candidate:
        return (raw_number, suffix)
    try:
        mantissa = float(candidate)
    except ValueError:
        return (raw_number, suffix)
    if mantissa >= 1000:
        return (raw_number, suffix)
    return (candidate, suffix)


def choose_best_amount(candidates: list[DisplayAmount]) -> DisplayAmount | None:
    if not candidates:
        return None
    counts: dict[tuple[int, int], int] = {}
    for candidate in candidates:
        key = amount_vote_key(candidate)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(
        candidates,
        key=lambda item: (
            amount_plausibility_score(item),
            amount_text_confidence(item),
            counts[amount_vote_key(item)],
            amount_suffix_confidence(item),
            int(item.mantissa > 0),
            mantissa_digit_count(item),
            -item.rank,
            item.mantissa,
        ),
        reverse=True,
    )
    return ranked[0]


def amount_vote_key(amount: DisplayAmount) -> tuple[int, int]:
    return (amount.rank, int(round(amount.mantissa * 1000)))


def amount_plausibility_score(amount: DisplayAmount) -> int:
    return 0 if is_suspicious_amount(amount) else 1


def amount_suffix_confidence(amount: DisplayAmount) -> int:
    if amount.suffix:
        return 2
    if amount.mantissa < 1000:
        return 1
    return 0


def amount_text_confidence(amount: DisplayAmount) -> int:
    canonical = canonical_amount_text(amount)
    raw = sanitize_ocr_text(amount.raw_text)
    if raw == canonical:
        return 2
    if canonical in raw or raw in canonical:
        return 1
    return 0


def canonical_amount_text(amount: DisplayAmount) -> str:
    return f"{amount.mantissa:g}{amount.suffix}"


def is_suspicious_amount(amount: DisplayAmount) -> bool:
    return (
        amount.tier_name != "white"
        and amount.mantissa >= 1000
        and (not amount.suffix or amount.suffix in SUFFIX_ORDER[1:])
    )


def mantissa_digit_count(amount: DisplayAmount) -> int:
    if abs(amount.mantissa - round(amount.mantissa)) < 1e-6:
        return len(str(int(round(amount.mantissa))))
    return len("".join(char for char in f"{amount.mantissa:g}" if char.isdigit()))


def detect_color_tier(crop_rgb: np.ndarray) -> str:
    if crop_rgb.size == 0:
        return "white"

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    value_mask = hsv[:, :, 2] > 140
    sat_mask = hsv[:, :, 1] > 30
    text_mask = value_mask & (sat_mask | (hsv[:, :, 2] > 210))
    pixels = crop_rgb[text_mask]
    if pixels.size == 0:
        flat = crop_rgb.reshape(-1, 3)
        brightness = np.mean(flat, axis=1)
        indices = np.argsort(brightness)[-max(len(flat) // 12, 12) :]
        pixels = flat[indices]
    mean_color = pixels.astype(np.float32).mean(axis=0)

    best_tier = "white"
    best_distance = float("inf")
    for tier_name, anchor in COLOR_ANCHORS.items():
        distance = float(np.linalg.norm(mean_color - anchor))
        if distance < best_distance:
            best_distance = distance
            best_tier = tier_name
    return best_tier


def build_ocr_variants(crop_rgb: np.ndarray, fast: bool = False) -> list[np.ndarray]:
    if crop_rgb.size == 0:
        return []

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = tighten_grayscale_crop(gray)
    scale = 3.0 if fast else 4.0
    gray = cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    if fast:
        return [add_ocr_border(otsu), add_ocr_border(gray)]
    inverted = cv2.bitwise_not(otsu)
    return [add_ocr_border(gray), add_ocr_border(adaptive), add_ocr_border(otsu), add_ocr_border(inverted)]


def build_rapidocr_variants(crop_rgb: np.ndarray, fast: bool = False) -> list[np.ndarray]:
    if crop_rgb.size == 0:
        return []

    rgb = tighten_amount_crop(crop_rgb)
    if rgb.size == 0:
        return []

    scale = 3.5 if fast else 5.0
    rgb = cv2.resize(rgb, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.45, cv2.GaussianBlur(gray, (0, 0), 2.0), -0.45, 0)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return [
        add_ocr_border(rgb),
        add_ocr_border(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)),
        add_ocr_border(cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)),
    ]


def build_vision_variants(crop_rgb: np.ndarray, fast: bool = False) -> list[np.ndarray]:
    if crop_rgb.size == 0:
        return []

    rgb = tighten_amount_crop(crop_rgb)
    if rgb.size == 0:
        return []

    scale = 3.5 if fast else 5.0
    rgb = cv2.resize(rgb, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.45, cv2.GaussianBlur(gray, (0, 0), 2.0), -0.45, 0)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    return [
        add_ocr_border(rgb),
        add_ocr_border(gray),
        add_ocr_border(sharpened),
        add_ocr_border(otsu),
        add_ocr_border(adaptive),
    ]


def image_to_string_safe(image: Image.Image, config: str, timeout: float | None) -> str:
    try:
        if timeout is None:
            return pytesseract.image_to_string(image, config=config)
        return pytesseract.image_to_string(image, config=config, timeout=timeout)
    except TypeError:
        return pytesseract.image_to_string(image, config=config)
    except RuntimeError:
        return ""


def rapidocr_image_to_strings(image: np.ndarray) -> list[str]:
    engine = get_rapidocr_engine()
    if engine is None:
        return []
    try:
        output = engine(image, use_det=False, use_cls=False, use_rec=True)
    except Exception:
        return []

    texts = getattr(output, "txts", ()) or ()
    scores = getattr(output, "scores", ()) or ()
    if not scores:
        scores = (1.0,) * len(texts)

    candidates: list[str] = []
    for text, score in zip(texts, scores):
        if score is None or float(score) >= RAPIDOCR_MIN_SCORE:
            normalized = str(text).strip()
            if normalized:
                candidates.append(normalized)
    return dedupe_text_candidates(candidates)


def get_rapidocr_engine():
    global _RAPIDOCR_ENGINE, _RAPIDOCR_ENGINE_INITIALIZED
    if _RAPIDOCR_ENGINE_INITIALIZED:
        return _RAPIDOCR_ENGINE

    _RAPIDOCR_ENGINE_INITIALIZED = True
    if RapidOCR is None or EngineType is None or LangRec is None or ModelType is None or OCRVersion is None:
        return None

    try:
        _RAPIDOCR_ENGINE = RapidOCR(
            params={
                "Global.log_level": "error",
                "Global.text_score": 0.0,
                "Global.min_height": 8,
                "Global.width_height_ratio": 20,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.EN,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
        )
    except Exception:
        _RAPIDOCR_ENGINE = None
    return _RAPIDOCR_ENGINE


def vision_image_to_strings(image: Image.Image, fast: bool) -> list[str]:
    if Vision is None or Foundation is None or objc is None:
        return []

    candidate_strings: list[str] = []
    with objc.autorelease_pool():
        png_data = pil_image_to_nsdata(image)
        if png_data is None:
            return []

        def completion_handler(request, error) -> None:
            if error is not None:
                return
            for observation in request.results() or []:
                top_candidates = observation.topCandidates_(3 if fast else 5)
                for candidate in top_candidates:
                    text = str(candidate.string()).strip()
                    if text:
                        candidate_strings.append(text)

        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(completion_handler)
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        request.setRecognitionLanguages_(VISION_LANGUAGES)
        if hasattr(request, "setCustomWords_"):
            request.setCustomWords_(VISION_CUSTOM_WORDS)
        if hasattr(request, "setMinimumTextHeight_"):
            request.setMinimumTextHeight_(0.05)
        for revision_name in (
            "VNRecognizeTextRequestRevision4",
            "VNRecognizeTextRequestRevision3",
            "VNRecognizeTextRequestRevision2",
        ):
            revision = getattr(Vision, revision_name, None)
            if revision is not None and hasattr(request, "setRevision_"):
                request.setRevision_(revision)
                break

        request_handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(png_data, None)
        try:
            ok, error = request_handler.performRequests_error_([request], None)
        except Exception:
            return []
        if not ok or error is not None:
            return []

    return dedupe_text_candidates(candidate_strings)


def pil_image_to_nsdata(image: Image.Image):
    normalized = image
    if normalized.mode not in {"L", "RGB"}:
        normalized = normalized.convert("RGB")
    buffer = io.BytesIO()
    normalized.save(buffer, format="PNG")
    raw = buffer.getvalue()
    return Foundation.NSData.dataWithBytes_length_(raw, len(raw))


def dedupe_text_candidates(candidates: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def candidate_consensus_count(candidates: list[DisplayAmount], best: DisplayAmount | None) -> int:
    if best is None:
        return 0
    key = amount_vote_key(best)
    return sum(1 for candidate in candidates if amount_vote_key(candidate) == key)


def add_ocr_border(image: np.ndarray) -> np.ndarray:
    return cv2.copyMakeBorder(
        image,
        top=8,
        bottom=8,
        left=8,
        right=8,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )


def tighten_amount_crop(crop_rgb: np.ndarray) -> np.ndarray:
    if crop_rgb.size == 0:
        return crop_rgb
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    mask = build_foreground_mask(gray)
    bbox = foreground_bbox(mask)
    if bbox is None:
        return crop_rgb
    left, top, right, bottom = bbox
    return crop_rgb[top:bottom, left:right].copy()


def tighten_grayscale_crop(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray
    mask = build_foreground_mask(gray)
    bbox = foreground_bbox(mask)
    if bbox is None:
        return gray
    left, top, right, bottom = bbox
    return gray[top:bottom, left:right].copy()


def build_foreground_mask(gray: np.ndarray) -> np.ndarray:
    border_pixels = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )
    background_level = int(np.median(border_pixels))
    diff_mask = cv2.absdiff(gray, np.full_like(gray, background_level))
    _, mask = cv2.threshold(diff_mask, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    pad = 2
    left = max(x - pad, 0)
    top = max(y - pad, 0)
    right = min(x + width + pad, mask.shape[1])
    bottom = min(y + height + pad, mask.shape[0])
    if right - left < 4 or bottom - top < 4:
        return None
    return (left, top, right, bottom)
