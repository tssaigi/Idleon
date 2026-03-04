from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from gaming_idleon.app import (
    TIER_BASE_PRIORITIES,
    UNRANKED_TIER,
    apply_tier_priorities,
    build_detection_settings_help_text,
    build_tier_assignments,
    priority_to_tier,
    restore_templates_from_manifest,
    serialize_templates,
)
from gaming_idleon.clicker import ClickWorker, aggregate_offer_samples, choose_stable_display_amount, filter_clickable_matches
from gaming_idleon.economy import (
    build_fertilizer_button_reference,
    choose_best_amount,
    dedupe_text_candidates,
    extract_fertilizer_amount_crops,
    extract_right_aligned_amount_crops,
    is_fertilizer_view,
    parse_display_amount,
    read_bits,
    read_amount,
    read_fertilizer_cost_amount,
    read_fertilizer_offers,
)
from gaming_idleon.matcher import extract_peak_locations, find_matches, prepare_templates, rotate_image
from gaming_idleon.models import (
    DisplayAmount,
    MIN_FERTILIZER_SETTLE_SECONDS,
    FertilizerOffer,
    GameCalibration,
    MatchResult,
    NormalizedRect,
    PixelRect,
    RunSettings,
    SheetImportOptions,
    TemplateImage,
    WindowCapture,
    fertilizer_row_name,
)
from gaming_idleon.sprites import load_templates_from_path, slice_sprite_sheet, slice_sprite_sheet_array


class SpriteSheetTests(unittest.TestCase):
    def test_slice_sprite_sheet_array_skips_empty_cells(self) -> None:
        rgba = np.zeros((40, 40, 4), dtype=np.uint8)
        rgba[2:18, 2:18] = [255, 0, 0, 255]
        rgba[22:38, 22:38] = [0, 255, 0, 255]

        templates = slice_sprite_sheet_array(
            rgba=rgba,
            sheet_name="sheet",
            source="memory",
            options=SheetImportOptions(rows=2, cols=2, trim_empty=True),
        )

        self.assertEqual(len(templates), 2)
        self.assertEqual(templates[0].width, 16)
        self.assertEqual(templates[0].height, 16)


class MatcherTests(unittest.TestCase):
    def test_find_matches_locates_template(self) -> None:
        screen = np.full((120, 200, 3), 255, dtype=np.uint8)
        screen[50:70, 80:100] = [220, 220, 220]
        screen[56:64, 88:92] = [20, 20, 20]
        screen[58:62, 84:96] = [20, 20, 20]

        template_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        template_rgba[:, :] = [220, 220, 220, 255]
        template_rgba[6:14, 8:12] = [20, 20, 20, 255]
        template_rgba[8:12, 4:16] = [20, 20, 20, 255]
        template = TemplateImage(name="square", source="memory", image_rgba=template_rgba)

        settings = RunSettings(threshold=0.60)
        variants = prepare_templates([template], settings)
        matches = find_matches(screen, variants, settings)

        self.assertTrue(matches)
        self.assertEqual(matches[0].center_x, 90)
        self.assertEqual(matches[0].center_y, 60)

    def test_find_matches_prefers_higher_priority(self) -> None:
        screen = np.full((120, 220, 3), 255, dtype=np.uint8)

        high_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        high_rgba[:, :] = [210, 210, 210, 255]
        high_rgba[3:17, 8:12] = [10, 10, 10, 255]
        high_rgba[8:12, 3:17] = [10, 10, 10, 255]
        screen[30:50, 30:50] = high_rgba[:, :, :3]

        low_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        low_rgba[:, :] = [210, 210, 210, 255]
        low_rgba[4:16, 4:16] = [10, 10, 10, 255]
        low_rgba[8:12, 8:12] = [210, 210, 210, 255]
        screen[30:50, 130:150] = low_rgba[:, :, :3]

        templates = [
            TemplateImage(name="low", source="memory", image_rgba=low_rgba, priority=1),
            TemplateImage(name="high", source="memory", image_rgba=high_rgba, priority=9),
        ]
        settings = RunSettings(threshold=0.55)
        variants = prepare_templates(templates, settings)
        matches = find_matches(screen, variants, settings)

        self.assertGreaterEqual(len(matches), 2)
        self.assertEqual(matches[0].template_name, "high")
        self.assertGreaterEqual(matches[0].priority, matches[1].priority)

    def test_find_matches_does_not_cap_template_hits_when_limit_is_zero(self) -> None:
        screen = np.full((120, 220, 3), 255, dtype=np.uint8)

        template_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        template_rgba[:, :] = [210, 210, 210, 255]
        template_rgba[3:17, 8:12] = [10, 10, 10, 255]
        template_rgba[8:12, 3:17] = [10, 10, 10, 255]
        for left in (10, 80, 150):
            screen[30:50, left:left + 20] = template_rgba[:, :, :3]

        template = TemplateImage(name="many", source="memory", image_rgba=template_rgba, priority=1)
        settings = RunSettings(threshold=0.55, max_matches_per_template=0)
        variants = prepare_templates([template], settings)
        matches = find_matches(screen, variants, settings)

        self.assertGreaterEqual(len(matches), 3)

    def test_find_matches_handles_rotation(self) -> None:
        template_rgba = np.zeros((22, 22, 4), dtype=np.uint8)
        template_rgba[:, :] = [220, 220, 220, 255]
        template_rgba[4:18, 10:12] = [20, 20, 20, 255]
        template_rgba[10:12, 4:18] = [20, 20, 20, 255]
        rotated = rotate_image(template_rgba, 12.0, 1, 0)

        screen = np.full((160, 220, 3), 255, dtype=np.uint8)
        h, w = rotated.shape[:2]
        screen[50:50 + h, 100:100 + w] = rotated[:, :, :3]

        template = TemplateImage(name="rotated", source="memory", image_rgba=template_rgba, priority=5)
        settings = RunSettings(threshold=0.45)
        variants = prepare_templates([template], settings)
        matches = find_matches(screen, variants, settings)

        self.assertTrue(matches)
        self.assertEqual(matches[0].template_name, "rotated")

    def test_extract_peak_locations_keeps_local_maxima_only(self) -> None:
        result = np.zeros((12, 12), dtype=np.float32)
        result[2, 2] = 0.91
        result[2, 3] = 0.84
        result[8, 8] = 0.89
        result[8, 9] = 0.83

        ys, xs = extract_peak_locations(result, threshold=0.8, min_distance=3)

        points = sorted(zip(xs.tolist(), ys.tolist()))
        self.assertEqual(points, [(2, 2), (8, 8)])


class EconomyTests(unittest.TestCase):
    def test_parse_display_amount_keeps_color_tier(self) -> None:
        amount = parse_display_amount("12.5QQ", "red")

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.suffix, "QQ")
        self.assertEqual(amount.tier_name, "red")
        self.assertTrue(amount.rank > 0)

    def test_parse_display_amount_normalizes_ocr_zeroes(self) -> None:
        amount = parse_display_amount("9OOB", "white")

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 900.0)
        self.assertEqual(amount.suffix, "B")

    def test_parse_display_amount_recovers_suffix_when_b_looks_like_8(self) -> None:
        amount = parse_display_amount("9008", "purple")

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 900.0)
        self.assertEqual(amount.suffix, "B")

    def test_parse_display_amount_recovers_suffix_when_b_looks_like_80(self) -> None:
        amount = parse_display_amount("90080", "purple")

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 900.0)
        self.assertEqual(amount.suffix, "B")

    def test_parse_display_amount_recovers_overflowing_suffix_mantissa(self) -> None:
        amount = parse_display_amount("2708B", "purple")

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 270.0)
        self.assertEqual(amount.suffix, "B")

    def test_parse_display_amount_recovers_overflowing_suffix_mantissa_without_trailing_eight(self) -> None:
        amount = parse_display_amount("4133B", "purple")

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 413.0)
        self.assertEqual(amount.suffix, "B")

    def test_parse_display_amount_keeps_valid_decimal_suffix_amount(self) -> None:
        amount = parse_display_amount("9.99T", "purple")

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 9.99)
        self.assertEqual(amount.suffix, "T")

    def test_choose_best_amount_prefers_fuller_900b_over_10t(self) -> None:
        low = parse_display_amount("900B", "purple")
        high = parse_display_amount("10T", "purple")
        assert low is not None
        assert high is not None

        best = choose_best_amount([high, low])

        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.mantissa, 900.0)
        self.assertEqual(best.suffix, "B")

    def test_choose_best_amount_prefers_suffixed_value_over_large_suffixless_parse(self) -> None:
        suspicious = parse_display_amount("90080", "purple")
        valid = parse_display_amount("900B", "purple")
        assert suspicious is not None
        assert valid is not None

        best = choose_best_amount([suspicious, suspicious, valid])

        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.mantissa, 900.0)
        self.assertEqual(best.suffix, "B")

    def test_choose_best_amount_prefers_clean_text_over_repaired_noisy_text(self) -> None:
        noisy = parse_display_amount("4133B", "purple")
        clean = parse_display_amount("133B", "purple")
        assert noisy is not None
        assert clean is not None

        best = choose_best_amount([noisy, noisy, clean])

        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.mantissa, 133.0)
        self.assertEqual(best.suffix, "B")

    def test_read_bits_uses_icon_color_not_text_color(self) -> None:
        crop = np.zeros((24, 120, 3), dtype=np.uint8)
        crop[:, :90] = [245, 245, 245]
        crop[4:20, 95:115] = [250, 110, 110]

        with patch("gaming_idleon.economy.pytesseract.image_to_string", return_value="291B"):
            amount = read_bits(crop)

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.tier_name, "red")
        self.assertEqual(amount.suffix, "B")

    def test_read_amount_can_use_vision_candidates(self) -> None:
        crop = np.full((24, 120, 3), 255, dtype=np.uint8)

        with patch("gaming_idleon.economy.vision_image_to_strings", return_value=["278B"]):
            with patch("gaming_idleon.economy.pytesseract.image_to_string", return_value=""):
                amount = read_amount(crop, tier_override="purple", fast=True)

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 278.0)
        self.assertEqual(amount.suffix, "B")

    def test_read_amount_prefers_rapidocr_consensus_over_single_vision_misread(self) -> None:
        crop = np.full((24, 120, 3), 255, dtype=np.uint8)

        with patch(
            "gaming_idleon.economy.build_rapidocr_variants",
            return_value=[crop, crop],
        ):
            with patch("gaming_idleon.economy.rapidocr_image_to_strings", return_value=["340B"]):
                with patch("gaming_idleon.economy.vision_image_to_strings", return_value=["5T"]):
                    with patch("gaming_idleon.economy.pytesseract.image_to_string", return_value=""):
                        amount = read_amount(crop, tier_override="purple", fast=True)

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 340.0)
        self.assertEqual(amount.suffix, "B")

    def test_dedupe_text_candidates_keeps_first_unique_text(self) -> None:
        deduped = dedupe_text_candidates(["278B", "278B", " 278B ", "540B"])

        self.assertEqual(deduped, ["278B", "540B"])

    def test_read_bits_prefers_suffixed_segmented_candidate_over_full_strip_noise(self) -> None:
        crop = np.zeros((24, 120, 3), dtype=np.uint8)
        crop[:, :90] = [245, 245, 245]
        crop[4:20, 95:115] = [205, 130, 255]
        noisy = parse_display_amount("433B", "purple")
        clean = parse_display_amount("133B", "purple")
        assert noisy is not None
        assert clean is not None

        with patch("gaming_idleon.economy.extract_right_aligned_amount_crops", return_value=[np.zeros((8, 20, 3), dtype=np.uint8)]):
            with patch("gaming_idleon.economy.read_amount", side_effect=[noisy, clean]):
                amount = read_bits(crop)

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 133.0)
        self.assertEqual(amount.suffix, "B")

    def test_extract_right_aligned_amount_crops_prefers_amount_near_icon(self) -> None:
        crop = np.zeros((32, 120, 3), dtype=np.uint8)
        crop[:, :] = [35, 35, 35]
        crop[8:24, 8:26] = [220, 220, 220]
        crop[8:24, 56:86] = [245, 245, 245]

        extracted = extract_right_aligned_amount_crops(crop)

        self.assertTrue(extracted)
        rightmost = extracted[0]
        self.assertLessEqual(rightmost.shape[1], 34)
        self.assertGreaterEqual(rightmost.shape[1], 20)
        self.assertGreater(np.mean(rightmost), 150)

    def test_read_fertilizer_offers_uses_explicit_regions(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((100, 100, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 100, 100),
            scale_x=1.0,
            scale_y=1.0,
        )
        calibration = GameCalibration(
            fertilizer_cost_1_rect=NormalizedRect(0.10, 0.10, 0.10, 0.10),
            fertilizer_purchase_1_rect=NormalizedRect(0.25, 0.10, 0.10, 0.10),
            fertilizer_cost_2_rect=NormalizedRect(0.10, 0.30, 0.10, 0.10),
            fertilizer_purchase_2_rect=NormalizedRect(0.25, 0.30, 0.10, 0.10),
            fertilizer_cost_3_rect=NormalizedRect(0.10, 0.50, 0.10, 0.10),
            fertilizer_purchase_3_rect=NormalizedRect(0.25, 0.50, 0.10, 0.10),
        )
        amount_a = parse_display_amount("7.7Q", "white")
        amount_b = parse_display_amount("2B", "red")
        assert amount_a is not None
        assert amount_b is not None

        with patch("gaming_idleon.economy.read_fertilizer_cost_amount", side_effect=[amount_a, None, amount_b]):
            offers = read_fertilizer_offers(capture, calibration)

        self.assertEqual([offer.row_index for offer in offers], [0, 2])
        self.assertEqual(offers[0].button_rect, calibration.fertilizer_purchase_1_rect)
        self.assertEqual(offers[1].button_rect, calibration.fertilizer_purchase_3_rect)

    def test_fertilizer_cost_uses_icon_color_not_text_color(self) -> None:
        image = np.zeros((60, 120, 3), dtype=np.uint8)
        image[10:30, 10:50] = [245, 245, 245]
        image[10:30, 50:62] = [250, 110, 110]
        capture = WindowCapture(
            window_id=1,
            image_rgb=image,
            bounds=PixelRect(0, 0, 120, 60),
            scale_x=1.0,
            scale_y=1.0,
        )
        calibration = GameCalibration(
            fertilizer_cost_1_rect=NormalizedRect(10 / 120, 10 / 60, 40 / 120, 20 / 60),
            fertilizer_purchase_1_rect=NormalizedRect(70 / 120, 10 / 60, 20 / 120, 20 / 60),
        )

        with patch("gaming_idleon.economy.pytesseract.image_to_string", return_value="7.7Q"):
            offers = read_fertilizer_offers(capture, calibration)

        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].cost.tier_name, "red")

    def test_read_fertilizer_cost_amount_uses_best_split_candidate(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((80, 180, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 180, 80),
            scale_x=1.0,
            scale_y=1.0,
        )
        cost_rect = NormalizedRect(0.10, 0.10, 0.30, 0.20)
        button_rect = NormalizedRect(0.50, 0.10, 0.20, 0.20)
        high = parse_display_amount("10T", "purple")
        low = parse_display_amount("900B", "purple")
        assert high is not None
        assert low is not None

        with patch(
            "gaming_idleon.economy.detect_fertilizer_cost_tier",
            return_value="purple",
        ):
            with patch(
                "gaming_idleon.economy.read_amount",
                side_effect=[high] + ([low] * 16),
            ):
                amount = read_fertilizer_cost_amount(capture, cost_rect, button_rect)

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 900.0)
        self.assertEqual(amount.suffix, "B")

    def test_read_fertilizer_cost_amount_prefers_non_segmented_suffixed_value_over_truncated_segment(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((80, 180, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 180, 80),
            scale_x=1.0,
            scale_y=1.0,
        )
        cost_rect = NormalizedRect(0.10, 0.10, 0.30, 0.20)
        button_rect = NormalizedRect(0.50, 0.10, 0.20, 0.20)
        base = parse_display_amount("2708B", "purple")
        segmented = parse_display_amount("70", "purple")
        assert base is not None
        assert segmented is not None

        with patch("gaming_idleon.economy.build_fertilizer_cost_crops", return_value=[np.zeros((12, 40, 3), dtype=np.uint8)]):
            with patch("gaming_idleon.economy.detect_fertilizer_cost_tier", return_value="purple"):
                with patch("gaming_idleon.economy.extract_fertilizer_amount_crops", return_value=[np.zeros((8, 16, 3), dtype=np.uint8)]):
                    with patch("gaming_idleon.economy.read_amount", side_effect=[base, None, None, None, segmented]):
                        amount = read_fertilizer_cost_amount(capture, cost_rect, button_rect)

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 270.0)
        self.assertEqual(amount.suffix, "B")

    def test_read_fertilizer_cost_amount_prefers_raw_or_expanded_crop_over_row_band_outlier(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((80, 180, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 180, 80),
            scale_x=1.0,
            scale_y=1.0,
        )
        cost_rect = NormalizedRect(0.10, 0.10, 0.30, 0.20)
        button_rect = NormalizedRect(0.50, 0.10, 0.20, 0.20)
        valid = parse_display_amount("900B", "purple")
        outlier = parse_display_amount("5T", "purple")
        assert valid is not None
        assert outlier is not None

        with patch(
            "gaming_idleon.economy.build_fertilizer_cost_crops",
            return_value=[
                np.zeros((12, 40, 3), dtype=np.uint8),
                np.zeros((14, 48, 3), dtype=np.uint8),
                np.zeros((18, 64, 3), dtype=np.uint8),
            ],
        ):
            with patch("gaming_idleon.economy.detect_fertilizer_cost_tier", return_value="purple"):
                with patch("gaming_idleon.economy.extract_fertilizer_amount_crops", return_value=[]):
                    with patch(
                        "gaming_idleon.economy.read_amount",
                        side_effect=[
                            valid, None, None, None,
                            valid, None, None, None,
                            outlier, outlier, outlier, outlier,
                        ],
                    ):
                        amount = read_fertilizer_cost_amount(capture, cost_rect, button_rect)

        self.assertIsNotNone(amount)
        assert amount is not None
        self.assertEqual(amount.mantissa, 900.0)
        self.assertEqual(amount.suffix, "B")

    def test_extract_fertilizer_amount_crops_prefers_rightmost_cost_block(self) -> None:
        crop = np.zeros((36, 140, 3), dtype=np.uint8)
        crop[:, :] = [42, 34, 28]
        crop[10:26, 12:32] = [230, 230, 230]
        crop[12:24, 78:110] = [242, 242, 242]
        crop[8:28, 116:132] = [205, 130, 255]

        extracted = extract_fertilizer_amount_crops(crop)

        self.assertTrue(extracted)
        rightmost = extracted[0]
        self.assertLessEqual(rightmost.shape[1], 40)
        self.assertGreaterEqual(rightmost.shape[1], 20)
        self.assertGreater(np.mean(rightmost), 150)

    def test_fertilizer_view_uses_button_references_to_reject_false_open(self) -> None:
        ref_image = np.zeros((80, 160, 3), dtype=np.uint8)
        ref_image[10:40, 90:140] = [210, 210, 210]
        ref_image[18:32, 102:128] = [20, 20, 20]
        button_ref = build_fertilizer_button_reference(ref_image[10:40, 90:140])
        assert button_ref is not None

        field_image = np.zeros((80, 160, 3), dtype=np.uint8)
        field_image[10:40, 10:60] = [245, 245, 245]
        field_capture = WindowCapture(
            window_id=1,
            image_rgb=field_image,
            bounds=PixelRect(0, 0, 160, 80),
            scale_x=1.0,
            scale_y=1.0,
        )
        fertilizer_capture = WindowCapture(
            window_id=1,
            image_rgb=ref_image,
            bounds=PixelRect(0, 0, 160, 80),
            scale_x=1.0,
            scale_y=1.0,
        )
        calibration = GameCalibration(
            fertilizer_cost_1_rect=NormalizedRect(10 / 160, 10 / 80, 40 / 160, 30 / 80),
            fertilizer_purchase_1_rect=NormalizedRect(90 / 160, 10 / 80, 50 / 160, 30 / 80),
            fertilizer_purchase_1_ref=button_ref,
        )

        amount = parse_display_amount("7Q", "white")
        assert amount is not None
        with patch("gaming_idleon.economy.read_fertilizer_offers", return_value=[type("Offer", (), {"row_index": 0, "cost": amount})()]):
            self.assertFalse(is_fertilizer_view(field_capture, calibration, minimum_rows=1))
            self.assertTrue(is_fertilizer_view(fertilizer_capture, calibration, minimum_rows=1))


class GifImportTests(unittest.TestCase):
    def test_load_templates_from_gif_splits_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plant.gif"
            frame_a = Image.new("RGBA", (12, 12), (255, 0, 0, 255))
            frame_b = Image.new("RGBA", (12, 12), (0, 255, 0, 255))
            frame_a.save(path, save_all=True, append_images=[frame_b], duration=80, loop=0)

            templates = load_templates_from_path(path)

            self.assertEqual(len(templates), 2)
            self.assertEqual(templates[0].name, "plant_f000")
            self.assertEqual(templates[1].name, "plant_f001")

    def test_slice_sprite_sheet_gif_splits_each_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sheet.gif"
            frame_a = Image.new("RGBA", (20, 10), (20, 20, 20, 255))
            frame_b = Image.new("RGBA", (20, 10), (25, 25, 25, 255))
            frame_a.paste((255, 0, 0, 255), (2, 2, 4, 8))
            frame_a.paste((255, 0, 0, 255), (0, 4, 8, 6))
            frame_a.paste((0, 0, 255, 255), (12, 2, 14, 8))
            frame_a.paste((0, 0, 255, 255), (10, 4, 18, 6))
            frame_b.paste((0, 255, 0, 255), (2, 2, 4, 8))
            frame_b.paste((0, 255, 0, 255), (0, 4, 8, 6))
            frame_b.paste((255, 255, 0, 255), (12, 2, 14, 8))
            frame_b.paste((255, 255, 0, 255), (10, 4, 18, 6))
            frame_a.save(path, save_all=True, append_images=[frame_b], duration=80, loop=0)

            templates = slice_sprite_sheet(path, SheetImportOptions(rows=1, cols=2))

            self.assertEqual(len(templates), 4)
            self.assertEqual(templates[0].name, "sheet_f000_000")
            self.assertEqual(templates[1].name, "sheet_f000_001")
            self.assertEqual(templates[2].name, "sheet_f001_000")


class PriorityAndSessionTests(unittest.TestCase):
    def test_detection_settings_help_mentions_current_fields(self) -> None:
        help_text = build_detection_settings_help_text()

        self.assertIn("Shape threshold", help_text)
        self.assertIn("Cycle interval (s)", help_text)
        self.assertIn("Click cooldown (s)", help_text)
        self.assertIn("Mouse speed", help_text)
        self.assertIn("Fertilizer recheck (cycles)", help_text)
        self.assertIn("Fertilizer settle (s)", help_text)
        self.assertIn("Fertilizer buy margin", help_text)
        self.assertIn("Sprinkler interval (cycles)", help_text)

    def test_priority_to_tier_and_apply_tiers(self) -> None:
        templates = [
            TemplateImage(name="alpha", source="a.png", image_rgba=np.zeros((8, 8, 4), dtype=np.uint8)),
            TemplateImage(name="beta", source="b.png", image_rgba=np.zeros((8, 8, 4), dtype=np.uint8)),
        ]
        assignments = build_tier_assignments(templates)
        self.assertIn(UNRANKED_TIER, assignments)
        self.assertEqual(len(assignments[UNRANKED_TIER]), 2)

        assignments["S"].append(assignments[UNRANKED_TIER].pop(0))
        assignments["F"].append(assignments[UNRANKED_TIER].pop(0))
        apply_tier_priorities(assignments)

        self.assertEqual(priority_to_tier(templates[0].priority), "S")
        self.assertEqual(priority_to_tier(templates[1].priority), "F")
        self.assertGreaterEqual(templates[0].priority, TIER_BASE_PRIORITIES["S"] - 1)

    def test_fertilizer_row_names_match_expected_labels(self) -> None:
        offers: list[FertilizerOffer] = []
        for index in range(3):
            amount = parse_display_amount("1B", "white")
            assert amount is not None
            offers.append(
                FertilizerOffer(
                    row_index=index,
                    cost=amount,
                    cost_rect=NormalizedRect(0.1, 0.1, 0.1, 0.1),
                    button_rect=NormalizedRect(0.2, 0.1, 0.1, 0.1),
                )
            )

        self.assertEqual([offer.row_name for offer in offers], ["Value", "Speed", "Capacity"])
        self.assertEqual(fertilizer_row_name(0), "Value")
        self.assertEqual(fertilizer_row_name(1), "Speed")
        self.assertEqual(fertilizer_row_name(2), "Capacity")

    def test_restore_templates_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plant.gif"
            frame_a = Image.new("RGBA", (12, 12), (255, 0, 0, 255))
            frame_b = Image.new("RGBA", (12, 12), (0, 255, 0, 255))
            frame_a.save(path, save_all=True, append_images=[frame_b], duration=80, loop=0)

            loaded = load_templates_from_path(path)
            loaded[0].priority = 321
            loaded[0].kind = "plant"
            manifest = serialize_templates([loaded[0]])

            restored = restore_templates_from_manifest(manifest)

            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].name, loaded[0].name)
            self.assertEqual(restored[0].priority, 321)
            self.assertEqual(restored[0].kind, "plant")

    def test_game_calibration_migrates_legacy_fertilizer_panel(self) -> None:
        calibration = GameCalibration.from_dict(
            {
                "field_rect": {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4},
                "bits_rect": {"left": 0.6, "top": 0.1, "width": 0.1, "height": 0.05},
                "fertilizer_button_rect": {"left": 0.8, "top": 0.8, "width": 0.1, "height": 0.1},
                "fertilizer_panel_rect": {"left": 0.2, "top": 0.3, "width": 0.5, "height": 0.3},
            }
        )

        self.assertTrue(calibration.fertilizer_ready())
        self.assertIsNotNone(calibration.fertilizer_cost_1_rect)
        self.assertIsNotNone(calibration.fertilizer_purchase_3_rect)
        assert calibration.fertilizer_cost_1_rect is not None
        assert calibration.fertilizer_cost_3_rect is not None
        self.assertLess(calibration.fertilizer_cost_1_rect.top, calibration.fertilizer_cost_3_rect.top)


class ClickFilterTests(unittest.TestCase):
    def test_window_in_fertilizer_does_not_trust_loose_ocr_before_open(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )

        with patch("gaming_idleon.clicker.is_fertilizer_view", side_effect=[False, True]):
            self.assertFalse(worker._window_in_fertilizer(capture))

        worker._fertilizer_assumed_open = True
        with patch("gaming_idleon.clicker.is_fertilizer_view", side_effect=[False, True]):
            self.assertTrue(worker._window_in_fertilizer(capture))

    def test_filter_clickable_matches_keeps_overlapping_plant_matches(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((120, 120, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 120, 120),
            scale_x=1.0,
            scale_y=1.0,
        )
        field_rect = PixelRect(0, 0, 120, 120)
        matches = [
            MatchResult("high", "memory", 0.90, 5, 50, 50, 20, 20, 1.0),
            MatchResult("low", "memory", 0.82, 4, 52, 51, 20, 20, 1.0),
            MatchResult("far", "memory", 0.80, 3, 90, 90, 18, 18, 1.0),
        ]

        filtered = filter_clickable_matches(
            capture,
            GameCalibration(),
            field_rect,
            matches,
            RunSettings(),
        )

        self.assertEqual([match.template_name for match in filtered], ["high", "low", "far"])

    def test_filter_clickable_matches_does_not_exclude_sprinkler_zone(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((120, 120, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 120, 120),
            scale_x=1.0,
            scale_y=1.0,
        )
        field_rect = PixelRect(0, 0, 120, 120)
        calibration = GameCalibration(sprinkler_rect=NormalizedRect(40 / 120, 40 / 120, 20 / 120, 20 / 120))
        matches = [
            MatchResult("sprinkler_overlap", "memory", 0.90, 5, 50, 50, 20, 20, 1.0),
            MatchResult("safe", "memory", 0.80, 4, 90, 90, 18, 18, 1.0),
        ]

        filtered = filter_clickable_matches(
            capture,
            calibration,
            field_rect,
            matches,
            RunSettings(sprinkler_interval=0),
        )

        self.assertEqual([match.template_name for match in filtered], ["sprinkler_overlap", "safe"])

    def test_filter_clickable_matches_keeps_all_non_import_matches(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((120, 120, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 120, 120),
            scale_x=1.0,
            scale_y=1.0,
        )
        field_rect = PixelRect(0, 0, 120, 120)
        matches = [
            MatchResult("center", "memory", 0.90, 5, 50, 50, 20, 20, 1.0),
            MatchResult("far", "memory", 0.80, 4, 90, 90, 18, 18, 1.0),
        ]

        filtered = filter_clickable_matches(
            capture,
            GameCalibration(),
            field_rect,
            matches,
            RunSettings(threshold=0.48),
        )

        self.assertEqual([match.template_name for match in filtered], ["center", "far"])

    def test_periodic_fertilizer_recheck_runs_without_affordability_trigger(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(fertilizer_check_interval=3),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )
        calls: list[str] = []
        worker._fertilizer_startup_ready = True
        worker._cycle_index = 3

        worker._read_bits = lambda current_capture: None
        worker._force_fertilizer_open_from_unknown_state = lambda current_capture, context: capture
        worker._handle_fertilizer = lambda current_capture, bits_override=None, exit_when_done=True: calls.append("handled")

        worker._maybe_open_fertilizer(capture)

        self.assertEqual(calls, ["handled"])

    def test_fertilizer_recheck_opens_immediately_when_bits_reach_cost(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(fertilizer_check_interval=3),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )
        cost = parse_display_amount("278B", "purple")
        bits = parse_display_amount("402B", "purple")
        assert cost is not None
        assert bits is not None

        worker._fertilizer_startup_ready = True
        worker._known_offers = {1: cost}
        worker._read_confirmed_bits = lambda current_capture: (current_capture, bits)
        calls: list[str] = []
        worker._force_fertilizer_open_from_unknown_state = lambda current_capture, context: capture
        worker._handle_fertilizer = lambda current_capture, bits_override=None, exit_when_done=True: calls.append("handled")

        worker._cycle_index = 3
        worker._maybe_open_fertilizer(capture)

        self.assertEqual(calls, ["handled"])
        self.assertFalse(worker._affordable_recheck_pending)

    def test_handle_fertilizer_logs_when_margin_blocks_otherwise_affordable_offer(self) -> None:
        logs: list[str] = []
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(affordable_margin=1.2),
            log_callback=logs.append,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )
        cost = parse_display_amount("340B", "purple")
        bits = parse_display_amount("397B", "purple")
        assert cost is not None
        assert bits is not None
        offer = FertilizerOffer(
            row_index=1,
            cost=cost,
            cost_rect=NormalizedRect(0.1, 0.1, 0.2, 0.2),
            button_rect=NormalizedRect(0.6, 0.1, 0.2, 0.2),
        )

        worker._read_confirmed_fertilizer_offers = lambda current_capture: (current_capture, [offer])
        worker._read_confirmed_bits = lambda current_capture: (current_capture, bits)
        worker._click_normalized_center = lambda current_capture, rect: True

        with patch("gaming_idleon.clicker.wait_with_cancel", return_value=True):
            with patch("gaming_idleon.clicker.capture_window", return_value=capture):
                worker._handle_fertilizer(capture, exit_when_done=False)

        self.assertFalse(any("Bought Fertilizer Speed for 340B [purple]." in line for line in logs))
        self.assertTrue(
            any("Skipping Fertilizer Speed: 397B [purple] is below the configured 1.2x safety margin for 340B [purple]." in line for line in logs)
        )

    def test_fertilizer_recheck_skips_when_bits_are_far_below_remembered_cost(self) -> None:
        logs: list[str] = []
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(fertilizer_check_interval=3),
            log_callback=logs.append,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )
        cost = parse_display_amount("540B", "purple")
        bits = parse_display_amount("60B", "purple")
        assert cost is not None
        assert bits is not None

        worker._fertilizer_startup_ready = True
        worker._known_offers = {1: cost}
        worker._last_fertilizer_open_cycle = 0
        worker._cycle_index = 3
        worker._read_confirmed_bits = lambda current_capture: (current_capture, bits)
        worker._force_fertilizer_open_from_unknown_state = lambda current_capture, context: (_ for _ in ()).throw(
            AssertionError("Fertilizer should not reopen when bits are still far below cost")
        )

        worker._maybe_open_fertilizer(capture)

        self.assertTrue(any("Skipping Fertilizer recheck" in message for message in logs))

    def test_recheck_failure_disables_future_fertilizer_checks(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(fertilizer_check_interval=3),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )
        worker._fertilizer_startup_ready = True
        worker._cycle_index = 3

        worker._read_bits = lambda current_capture: None
        worker._force_fertilizer_open_from_unknown_state = lambda current_capture, context: None

        worker._maybe_open_fertilizer(capture)

        self.assertTrue(worker._fertilizer_checks_disabled)

    def test_runtime_can_capture_missing_fertilizer_button_refs(self) -> None:
        image = np.zeros((80, 160, 3), dtype=np.uint8)
        image[10:40, 90:140] = [210, 210, 210]
        image[18:32, 102:128] = [20, 20, 20]
        capture = WindowCapture(
            window_id=1,
            image_rgb=image,
            bounds=PixelRect(0, 0, 160, 80),
            scale_x=1.0,
            scale_y=1.0,
        )
        calibration = GameCalibration(
            fertilizer_purchase_1_rect=NormalizedRect(90 / 160, 10 / 80, 50 / 160, 30 / 80),
        )
        worker = ClickWorker(
            window_id=1,
            calibration=calibration,
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )

        worker._capture_missing_fertilizer_button_refs(capture)

        self.assertIsNotNone(calibration.fertilizer_purchase_1_ref)

    def test_aggregate_offer_samples_recovers_row_from_noisy_reads(self) -> None:
        amount_a = parse_display_amount("900B", "purple")
        amount_b = parse_display_amount("901B", "purple")
        wrong = parse_display_amount("10T", "purple")
        assert amount_a is not None
        assert amount_b is not None
        assert wrong is not None
        base_rect = NormalizedRect(0.1, 0.1, 0.1, 0.1)

        samples = [
            [FertilizerOffer(0, wrong, base_rect, base_rect)],
            [FertilizerOffer(0, amount_a, base_rect, base_rect)],
            [FertilizerOffer(0, amount_b, base_rect, base_rect)],
        ]

        offers = aggregate_offer_samples(samples)

        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].cost.suffix, "B")
        self.assertGreaterEqual(offers[0].cost.mantissa, 900.0)

    def test_prime_fertilizer_can_continue_when_prices_are_unstable(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )
        messages: list[str] = []
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=messages.append,
            state_callback=lambda state: None,
        )
        worker._read_bits = lambda current_capture: None
        worker._force_fertilizer_open_from_unknown_state = lambda current_capture, context: capture
        worker._capture_missing_fertilizer_button_refs = lambda current_capture: None
        worker._read_confirmed_fertilizer_offers = lambda current_capture: (capture, [])
        worker._force_fertilizer_closed = lambda current_capture, context: capture
        worker._window_in_fertilizer = lambda current_capture: False

        with patch("gaming_idleon.clicker.capture_window", return_value=capture):
            ok = worker._prime_fertilizer()

        self.assertTrue(ok)
        self.assertTrue(
            any("could not lock onto stable prices" in message for message in messages)
        )
        self.assertTrue(worker._fertilizer_checks_disabled)
        self.assertFalse(worker._fertilizer_startup_ready)

    def test_prime_fertilizer_publishes_preview_frames(self) -> None:
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )
        previews: list[tuple[WindowCapture, object | None]] = []
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
            preview_callback=lambda current_capture, debug_preview: previews.append((current_capture, debug_preview)),
        )
        worker._read_bits = lambda current_capture: None
        worker._force_fertilizer_open_from_unknown_state = lambda current_capture, context: capture
        worker._capture_missing_fertilizer_button_refs = lambda current_capture: None
        worker._read_confirmed_fertilizer_offers = lambda current_capture: (capture, [])
        worker._force_fertilizer_closed = lambda current_capture, context: capture
        worker._window_in_fertilizer = lambda current_capture: False

        with patch("gaming_idleon.clicker.capture_window", return_value=capture):
            ok = worker._prime_fertilizer()

        self.assertTrue(ok)
        self.assertTrue(previews)
        self.assertIs(previews[0][0], capture)

    def test_unstable_startup_blocks_later_fertilizer_rechecks(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(fertilizer_check_interval=5.0),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )

        worker._fertilizer_startup_ready = False
        worker._read_bits = lambda current_capture: None
        worker._force_fertilizer_open_from_unknown_state = lambda current_capture, context: (_ for _ in ()).throw(
            AssertionError("Fertilizer should not reopen after unstable startup")
        )

        with patch("gaming_idleon.clicker.time.monotonic", return_value=10.0):
            worker._maybe_open_fertilizer(capture)

    def test_offer_cost_is_safe_rejects_missing_suffix_against_abbreviated_bits(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        bits = parse_display_amount("318B", "purple")
        suspicious = DisplayAmount(
            raw_text="9008",
            mantissa=9008.0,
            suffix="",
            tier_name="purple",
            tier_index=3,
            rank=(3 * 7),
        )
        assert bits is not None

        self.assertFalse(worker._offer_cost_is_safe(bits, suspicious))

    def test_choose_stable_display_amount_prefers_better_close_bits_read(self) -> None:
        low = parse_display_amount("338B", "purple")
        high = parse_display_amount("341B", "purple")
        assert low is not None
        assert high is not None

        best = choose_stable_display_amount([low, high])

        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.mantissa, 341.0)
        self.assertEqual(best.suffix, "B")

    def test_run_emits_startup_state_after_arming(self) -> None:
        rect = NormalizedRect(0.1, 0.1, 0.1, 0.1)
        calibration = GameCalibration(
            field_rect=rect,
            bits_rect=rect,
            fertilizer_button_rect=rect,
            fertilizer_cost_1_rect=rect,
            fertilizer_purchase_1_rect=rect,
            fertilizer_cost_2_rect=rect,
            fertilizer_purchase_2_rect=rect,
            fertilizer_cost_3_rect=rect,
            fertilizer_purchase_3_rect=rect,
        )
        states: list[str] = []
        logs: list[str] = []
        worker = ClickWorker(
            window_id=1,
            calibration=calibration,
            templates=[],
            settings=RunSettings(start_delay=0.0),
            log_callback=logs.append,
            state_callback=states.append,
        )
        worker._prime_fertilizer = lambda: False

        worker._run()

        self.assertEqual(states[:2], ["arming", "startup"])
        self.assertIn("Running startup Fertilizer checks.", logs)

    def test_fertilizer_settle_delay_has_minimum(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(fertilizer_toggle_delay=0.0),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )

        self.assertEqual(worker._fertilizer_settle_delay(), MIN_FERTILIZER_SETTLE_SECONDS)

    def test_handle_fertilizer_blocks_row_when_purchase_does_not_refresh_cost(self) -> None:
        rect = NormalizedRect(0.1, 0.1, 0.1, 0.1)
        cost = parse_display_amount("278B", "purple")
        bits = parse_display_amount("500B", "purple")
        assert cost is not None
        assert bits is not None
        offer = FertilizerOffer(1, cost, rect, rect)
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 40, 40),
            scale_x=1.0,
            scale_y=1.0,
        )
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(upgrade_delay=0.0, fertilizer_check_interval=3.0),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        worker._read_confirmed_fertilizer_offers = lambda current_capture: (current_capture, [offer])
        worker._read_confirmed_bits = lambda current_capture: (current_capture, bits)
        worker._window_in_fertilizer = lambda current_capture: True

        with patch("gaming_idleon.clicker.capture_window", return_value=capture):
            with patch.object(worker, "_click_normalized_center") as click_mock:
                worker._handle_fertilizer(capture, exit_when_done=False)

        click_mock.assert_called_once()
        self.assertIn(1, worker._blocked_fertilizer_rows)

    def test_should_click_blocks_same_cycle_duplicate_target(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(cooldown_seconds=2.0),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        first = MatchResult("Chemical_f001", "memory", 0.9, 10, 0, 0, 20, 20, 1.0)
        second = MatchResult("Chemical_f002", "memory", 0.9, 9, 0, 0, 20, 20, 1.0)

        self.assertTrue(worker._should_click(first, 500, 600))
        self.assertFalse(worker._should_click(second, 501, 601))

    def test_should_click_allows_distinct_nearby_targets_when_far_enough(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(cooldown_seconds=2.0),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        first = MatchResult("Chemical_f001", "memory", 0.9, 10, 0, 0, 20, 20, 1.0)
        second = MatchResult("Chemical_f002", "memory", 0.9, 9, 0, 0, 20, 20, 1.0)

        self.assertTrue(worker._should_click(first, 500, 600))
        self.assertTrue(worker._should_click(second, 512, 612))

    def test_build_harvest_plan_respects_priority_before_pathing(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((120, 120, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 120, 120),
            scale_x=1.0,
            scale_y=1.0,
        )
        field_rect = PixelRect(0, 0, 120, 120)
        matches = [
            MatchResult("high_far", "memory", 0.90, 10, 90, 10, 20, 20, 1.0),
            MatchResult("high_near", "memory", 0.88, 10, 30, 10, 20, 20, 1.0),
            MatchResult("low_near", "memory", 0.95, 5, 10, 90, 20, 20, 1.0),
            MatchResult("low_far", "memory", 0.89, 5, 60, 90, 20, 20, 1.0),
        ]

        with patch("gaming_idleon.clicker.pyautogui.position", return_value=(0, 0)):
            plan = worker._build_harvest_plan(capture, field_rect, matches)

        self.assertEqual(
            [target.match.template_name for target in plan],
            ["high_near", "high_far", "low_far", "low_near"],
        )

    def test_build_harvest_plan_collapses_only_near_duplicate_targets(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((120, 120, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 120, 120),
            scale_x=1.0,
            scale_y=1.0,
        )
        field_rect = PixelRect(0, 0, 120, 120)
        matches = [
            MatchResult("dup_a", "memory", 0.95, 10, 50, 50, 20, 20, 1.0),
            MatchResult("dup_b", "memory", 0.90, 9, 52, 52, 20, 20, 1.0),
            MatchResult("overlap_ok", "memory", 0.85, 8, 86, 86, 20, 20, 1.0),
        ]

        with patch("gaming_idleon.clicker.pyautogui.position", return_value=(0, 0)):
            plan = worker._build_harvest_plan(capture, field_rect, matches)

        self.assertEqual(
            [target.match.template_name for target in plan],
            ["dup_a", "overlap_ok"],
        )

    def test_build_harvest_plan_rejects_ambiguous_edge_hotspots(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(threshold=0.56),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((140, 140, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 140, 140),
            scale_x=1.0,
            scale_y=1.0,
        )
        field_rect = PixelRect(0, 0, 140, 140)
        matches = [
            MatchResult("Cactus_f003", "memory", 0.69, 10, 131, 110, 20, 20, 1.0),
            MatchResult("Cactus_f005", "memory", 0.68, 10, 130, 118, 20, 20, 1.0),
            MatchResult("Blossom_f002", "memory", 0.67, 10, 128, 112, 20, 20, 1.0),
            MatchResult("Sprouts_f001", "memory", 0.66, 10, 129, 106, 20, 20, 1.0),
        ]

        with patch("gaming_idleon.clicker.pyautogui.position", return_value=(0, 0)):
            plan = worker._build_harvest_plan(capture, field_rect, matches)

        self.assertEqual(plan, [])

    def test_click_screen_point_skips_click_when_stop_is_requested(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )
        worker.stop()

        with patch("gaming_idleon.clicker.pyautogui.moveTo") as move_to_mock:
            with patch("gaming_idleon.clicker.pyautogui.click") as click_mock:
                worker._click_screen_point(100, 200, 20, 20)

        move_to_mock.assert_not_called()
        click_mock.assert_not_called()

    def test_click_screen_point_activates_window_before_clicking(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )

        with patch("gaming_idleon.clicker.window_owner_is_frontmost", return_value=False):
            with patch("gaming_idleon.clicker.activate_window_owner", return_value=True) as activate_mock:
                with patch.object(worker, "_move_mouse_with_cancel", return_value=True):
                    with patch("gaming_idleon.clicker.pyautogui.click") as click_mock:
                        result = worker._click_screen_point(100, 200, 20, 20)

        self.assertTrue(result)
        activate_mock.assert_called_once_with(1)
        click_mock.assert_called_once()

    def test_click_screen_point_skips_activation_when_window_is_already_frontmost(self) -> None:
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=lambda message: None,
            state_callback=lambda state: None,
        )

        with patch("gaming_idleon.clicker.window_owner_is_frontmost", return_value=True):
            with patch.object(worker, "_move_mouse_with_cancel", return_value=True):
                with patch("gaming_idleon.clicker.activate_window_owner") as activate_mock:
                    with patch("gaming_idleon.clicker.pyautogui.click") as click_mock:
                        result = worker._click_screen_point(100, 200, 20, 20)

        self.assertTrue(result)
        activate_mock.assert_not_called()
        click_mock.assert_called_once()

    def test_click_screen_point_skips_click_when_window_activation_fails(self) -> None:
        logs: list[str] = []
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(),
            templates=[],
            settings=RunSettings(),
            log_callback=logs.append,
            state_callback=lambda state: None,
        )

        with patch("gaming_idleon.clicker.window_owner_is_frontmost", return_value=False):
            with patch("gaming_idleon.clicker.activate_window_owner", return_value=False):
                with patch("gaming_idleon.clicker.pyautogui.click") as click_mock:
                    result = worker._click_screen_point(100, 200, 20, 20)

        self.assertFalse(result)
        click_mock.assert_not_called()
        self.assertTrue(any("Could not bring the selected Idleon window to the front" in message for message in logs))

    def test_harvest_cycle_resets_cycle_click_suppression(self) -> None:
        logs: list[str] = []
        worker = ClickWorker(
            window_id=1,
            calibration=GameCalibration(field_rect=NormalizedRect(0.0, 0.0, 1.0, 1.0)),
            templates=[],
            settings=RunSettings(),
            log_callback=logs.append,
            state_callback=lambda state: None,
        )
        capture = WindowCapture(
            window_id=1,
            image_rgb=np.zeros((120, 120, 3), dtype=np.uint8),
            bounds=PixelRect(0, 0, 120, 120),
            scale_x=1.0,
            scale_y=1.0,
        )
        match = MatchResult("plant", "memory", 0.95, 0, 50, 50, 20, 20, 1.0)

        with patch("gaming_idleon.clicker.find_matches", side_effect=[[match], [match]]):
            with patch("gaming_idleon.clicker.filter_clickable_matches", side_effect=lambda *args: args[3]):
                with patch.object(worker, "_click_screen_point", return_value=True):
                    worker._harvest_cycle(capture, [])
                    worker._harvest_cycle(capture, [])

        self.assertEqual(len([line for line in logs if line.startswith("Clicked plant")]), 2)


if __name__ == "__main__":
    unittest.main()
