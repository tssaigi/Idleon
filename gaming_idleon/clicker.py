from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import random
import threading
import time

import pyautogui

from .economy import (
    build_fertilizer_button_reference,
    choose_best_amount,
    is_fertilizer_view,
    read_bits,
    read_fertilizer_offers,
)
from .matcher import find_matches, prepare_templates
from .models import (
    CyclePlanPreview,
    MIN_FERTILIZER_SETTLE_SECONDS,
    DisplayAmount,
    FertilizerOffer,
    GameCalibration,
    MatchResult,
    NormalizedRect,
    PixelRect,
    PreviewTarget,
    RunSettings,
    TemplateImage,
    WindowCapture,
    fertilizer_row_name,
)
from .windowing import activate_window_owner, capture_window, window_owner_is_frontmost


LogCallback = Callable[[str], None]
StateCallback = Callable[[str], None]
PreviewCallback = Callable[[WindowCapture, CyclePlanPreview | None], None]


@dataclass(slots=True)
class PlannedClick:
    match: MatchResult
    image_x: int
    image_y: int
    screen_x: int
    screen_y: int
    global_rect: PixelRect


class ClickWorker:
    def __init__(
        self,
        window_id: int,
        calibration: GameCalibration,
        templates: list[TemplateImage],
        settings: RunSettings,
        log_callback: LogCallback,
        state_callback: StateCallback,
        preview_callback: PreviewCallback | None = None,
    ) -> None:
        self._window_id = window_id
        self._calibration = calibration
        self._templates = templates
        self._settings = settings
        self._log_callback = log_callback
        self._state_callback = state_callback
        self._preview_callback = preview_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cycle_clicked_targets: set[str] = set()
        self._known_offers: dict[int, DisplayAmount] = {}
        self._last_fertilizer_cycle = 0
        self._last_fertilizer_open_cycle = -10_000
        self._last_bits: DisplayAmount | None = None
        self._last_sprinkler_cycle = 0
        self._fertilizer_assumed_open = False
        self._fertilizer_checks_disabled = False
        self._fertilizer_startup_ready = False
        self._affordable_recheck_pending = False
        self._blocked_fertilizer_rows: dict[int, tuple[tuple[int, int], int]] = {}
        self._cycle_index = 0
        self._last_window_activate_attempt = 0.0
        self._last_window_activate_failure_log = 0.0
        self._last_no_match_log = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float = 0.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        if not self._calibration.ready_for_run():
            self._log_callback(
                "Calibration is incomplete. Set field, bits, the Fertilizer toggle button, and all 3 Fertilizer cost/button rows."
            )
            self._state_callback("error")
            return

        plant_templates = [template for template in self._templates if template.kind == "plant"]
        plant_variants = prepare_templates(plant_templates, self._settings)
        self._state_callback("arming")
        if self._settings.start_delay > 0:
            self._log_callback(
                f"Arming for {self._settings.start_delay:.1f}s so you can focus the Idleon window."
            )
            if not wait_with_cancel(self._stop_event, self._settings.start_delay):
                self._state_callback("stopped")
                return
        self._state_callback("startup")
        self._log_callback("Running startup Fertilizer checks.")
        self._ensure_target_window_active(force=True)

        try:
            startup_ok = self._prime_fertilizer()
        except Exception as exc:  # pragma: no cover - OS and game state dependent
            self._log_callback(f"Startup fertilizer check failed: {exc}")
            self._state_callback("error")
            return
        if not startup_ok:
            self._log_callback("Startup checks failed. Not entering harvest mode.")
            self._state_callback("error")
            return

        self._state_callback("running")
        self._log_callback(
            f"Running on selected Idleon window with {len(plant_templates)} plant template(s)."
        )

        while not self._stop_event.is_set():
            cycle_started = time.monotonic()
            try:
                capture = capture_window(self._window_id)
            except Exception as exc:  # pragma: no cover - depends on OS/window state
                self._log_callback(f"Window capture error: {exc}")
                self._state_callback("error")
                return
            started_in_fertilizer = self._window_in_fertilizer(capture)

            if started_in_fertilizer:
                self._publish_preview(capture, None)
                self._handle_fertilizer(capture)
            else:
                self._harvest_cycle(capture, plant_variants)
                self._cycle_index += 1
                self._maybe_open_fertilizer(capture)
                try:
                    fresh_capture = capture_window(self._window_id)
                except Exception as exc:  # pragma: no cover - depends on OS/window state
                    self._log_callback(f"Window capture error after cycle: {exc}")
                    self._state_callback("error")
                    return
                self._publish_preview(fresh_capture, None)
                if self._maybe_click_sprinkler(fresh_capture):
                    if not wait_with_cancel(self._stop_event, self._settings.sprinkler_growth_delay):
                        break
            if started_in_fertilizer:
                self._cycle_index += 1

            elapsed = time.monotonic() - cycle_started
            delay = max(self._settings.scan_interval - elapsed, 0.0)
            if delay > 0 and not wait_with_cancel(self._stop_event, delay):
                break

        self._state_callback("stopped")

    def _prime_fertilizer(self) -> bool:
        if self._fertilizer_checks_disabled:
            return True
        capture = capture_window(self._window_id)
        self._publish_preview(capture)
        capture, confirmed_bits = self._read_confirmed_bits(capture)
        self._last_bits = confirmed_bits
        self._fertilizer_assumed_open = False
        self._fertilizer_startup_ready = False
        capture = self._force_fertilizer_open_from_unknown_state(capture, context="startup")
        if capture is None:
            self._disable_fertilizer_checks("Startup Fertilizer open check failed. Disabling Fertilizer checks for this run.")
            return True
        self._capture_missing_fertilizer_button_refs(capture)
        capture, offers = self._read_confirmed_fertilizer_offers(capture)
        if not offers:
            self._log_callback(
                "Startup Fertilizer check could not lock onto stable prices. Leaving Fertilizer and starting harvest anyway."
            )
            self._disable_fertilizer_checks("Startup Fertilizer prices were unstable. Future Fertilizer checks are disabled for this run.")
            verify_capture = self._force_fertilizer_closed(capture, context="startup")
            if verify_capture is None:
                return True
        else:
            self._fertilizer_startup_ready = True
            self._handle_fertilizer(capture, exit_when_done=True)
            verify_capture = capture_window(self._window_id)
            self._publish_preview(verify_capture)
        if self._window_in_fertilizer(verify_capture):
            self._log_callback("Startup Fertilizer check finished, but Fertilizer still appears open.")
            self._force_fertilizer_closed(verify_capture, context="startup")
            self._disable_fertilizer_checks("Fertilizer did not return cleanly to the field. Future Fertilizer checks are disabled for this run.")
            return True
        self._last_fertilizer_cycle = self._cycle_index
        return True

    def _harvest_cycle(
        self,
        capture: WindowCapture,
        plant_variants: list,
    ) -> None:
        field_rect = capture.rect_from_normalized(self._calibration.field_rect)
        field_rgb = capture.crop(self._calibration.field_rect)
        raw_matches = find_matches(field_rgb, plant_variants, self._settings)
        matches = filter_clickable_matches(
            capture,
            self._calibration,
            field_rect,
            raw_matches,
            self._settings,
        )
        if not matches:
            self._publish_preview(
                capture,
                CyclePlanPreview(planned_targets=[], raw_match_count=len(raw_matches), filtered_match_count=0),
            )
            self._log_no_match_state(raw_matches)
            return

        self._cycle_clicked_targets.clear()
        plan = self._build_harvest_plan(capture, field_rect, matches)
        if not plan:
            self._publish_preview(
                capture,
                CyclePlanPreview(
                    planned_targets=[],
                    raw_match_count=len(raw_matches),
                    filtered_match_count=len(matches),
                ),
            )
            self._log_no_match_state(raw_matches)
            return

        self._publish_preview(
            capture,
            CyclePlanPreview(
                planned_targets=[
                    PreviewTarget(
                        template_name=planned_click.match.template_name,
                        score=planned_click.match.score,
                        priority=planned_click.match.priority,
                        image_x=planned_click.image_x,
                        image_y=planned_click.image_y,
                        box_left=planned_click.global_rect.left,
                        box_top=planned_click.global_rect.top,
                        box_width=planned_click.global_rect.width,
                        box_height=planned_click.global_rect.height,
                    )
                    for planned_click in plan
                ],
                raw_match_count=len(raw_matches),
                filtered_match_count=len(matches),
            ),
        )

        for planned_click in plan:
            if self._stop_event.is_set():
                break
            match = planned_click.match
            screen_x = planned_click.screen_x
            screen_y = planned_click.screen_y
            if not self._should_click(match, screen_x, screen_y):
                continue
            if self._click_screen_point(screen_x, screen_y, match.width, match.height):
                self._log_callback(
                    f"Clicked {match.template_name} at ({screen_x}, {screen_y}) score={match.score:.3f} priority={match.priority}"
                )

    def _build_harvest_plan(
        self,
        capture: WindowCapture,
        field_rect: PixelRect,
        matches: list[MatchResult],
    ) -> list[PlannedClick]:
        clustered_targets = cluster_match_targets(capture, field_rect, matches, self._settings)
        if not clustered_targets:
            return []

        try:
            current_x, current_y = pyautogui.position()
        except Exception:
            current_x, current_y = (0, 0)

        plan: list[PlannedClick] = []
        priorities = sorted({target.match.priority for target in clustered_targets}, reverse=True)
        for priority in priorities:
            pending = [target for target in clustered_targets if target.match.priority == priority]
            while pending:
                next_index = min(
                    range(len(pending)),
                    key=lambda index: target_path_sort_key(pending[index], current_x, current_y),
                )
                next_target = pending.pop(next_index)
                plan.append(next_target)
                current_x, current_y = next_target.screen_x, next_target.screen_y
        return plan

    def _log_no_match_state(
        self,
        raw_matches: list[MatchResult],
    ) -> None:
        now = time.monotonic()
        if now - self._last_no_match_log < 5.0:
            return
        self._last_no_match_log = now
        if raw_matches:
            self._log_callback(
                f"Plant matches were found, but all {len(raw_matches)} candidate(s) were inside Imports."
            )
            return
        self._log_callback("No plant matches found in the calibrated field on this cycle.")

    def _maybe_open_fertilizer(self, capture: WindowCapture) -> None:
        if self._fertilizer_checks_disabled:
            return
        if not self._fertilizer_startup_ready:
            return
        if self._settings.fertilizer_check_interval <= 0:
            return
        interval_cycles = max(int(round(self._settings.fertilizer_check_interval)), 0)
        if interval_cycles <= 0:
            return
        if self._cycle_index - self._last_fertilizer_cycle < interval_cycles:
            return
        self._last_fertilizer_cycle = self._cycle_index

        capture, bits = self._read_confirmed_bits(capture)
        if bits is not None:
            self._last_bits = bits

        active_known_costs = self._active_known_costs()
        if bits is not None and active_known_costs and any(
            self._offer_cost_is_safe(bits, cost)
            and bits.is_at_least(cost, margin=self._settings.affordable_margin)
            for cost in active_known_costs
        ):
            self._affordable_recheck_pending = False
            self._log_callback(
                f"Bits {bits.short_label()} reached a remembered Fertilizer cost. Rechecking Fertilizer."
            )
        else:
            self._affordable_recheck_pending = False
            cheapest_cost = self._cheapest_known_cost()
            hard_refresh_interval = max(interval_cycles * 4, interval_cycles + 1)
            if (
                bits is not None
                and cheapest_cost is not None
                and not self._bits_near_cost(bits, cheapest_cost)
                and (self._cycle_index - self._last_fertilizer_open_cycle) < hard_refresh_interval
            ):
                self._log_callback(
                    f"Skipping Fertilizer recheck: {bits.short_label()} is still well below remembered {cheapest_cost.short_label()}."
                )
                return
            self._log_callback("Running scheduled Fertilizer recheck.")
        capture = self._force_fertilizer_open_from_unknown_state(capture, context="recheck")
        if capture is None:
            self._disable_fertilizer_checks("Could not open Fertilizer during recheck. Disabling further Fertilizer checks for this run.")
            return
        self._capture_missing_fertilizer_button_refs(capture)
        self._handle_fertilizer(capture, bits_override=bits, exit_when_done=True)

    def _handle_fertilizer(
        self,
        capture: WindowCapture,
        bits_override: DisplayAmount | None = None,
        exit_when_done: bool = True,
    ) -> None:
        capture, offers = self._read_confirmed_fertilizer_offers(capture)
        if offers:
            self._remember_fertilizer_offers(offers)
            self._log_fertilizer_offer_summary(offers)
        else:
            self._log_callback("Fertilizer is open, but the 3 calibrated cost boxes did not parse stable values.")
            self._disable_fertilizer_checks("Fertilizer OCR was unstable. Disabling further Fertilizer checks for this run.")

        capture, fresh_bits = self._read_confirmed_bits(capture)
        bits = fresh_bits or bits_override or self._last_bits
        if bits is not None:
            self._last_bits = bits
            self._log_callback(f"Current bits: {bits.short_label()}")

        if offers and bits is not None and not self._stop_event.is_set():
            affordable = [
                offer
                for offer in offers
                if not self._offer_row_is_blocked(offer)
                if self._offer_cost_is_safe(bits, offer.cost)
                if bits.is_at_least(offer.cost, margin=self._settings.affordable_margin)
            ]
            if affordable:
                offer = min(affordable, key=lambda item: item.row_index)
                purchase_signature = amount_signature(offer.cost)
                if not self._click_normalized_center(capture, offer.button_rect):
                    self._log_callback(
                        f"Could not click Fertilizer {offer.row_name}. The selected Idleon window may not be frontmost."
                    )
                    return
                self._log_callback(
                    f"Bought Fertilizer {offer.row_name} for {offer.cost.short_label()}."
                )
                if not wait_with_cancel(self._stop_event, self._settings.upgrade_delay):
                    return

                try:
                    capture = capture_window(self._window_id)
                except Exception as exc:  # pragma: no cover - depends on OS/window state
                    self._log_callback(f"Fertilizer capture failed after purchase: {exc}")
                    return
                self._publish_preview(capture)
                if not self._window_in_fertilizer(capture):
                    capture = self._force_fertilizer_open_from_unknown_state(capture, context="purchase follow-up")
                    if capture is None:
                        return
                    self._capture_missing_fertilizer_button_refs(capture)
                capture, offers = self._read_confirmed_fertilizer_offers(capture)
                if offers:
                    self._remember_fertilizer_offers(offers)
                    self._log_fertilizer_offer_summary(offers)
                else:
                    self._log_callback("Fertilizer offers: no stable costs were visible after the purchase refresh.")
                capture, refreshed_bits = self._read_confirmed_bits(capture)
                bits = refreshed_bits or bits
                if bits is not None:
                    self._last_bits = bits
                if any(
                    item.row_index == offer.row_index and amount_signature(item.cost) == purchase_signature
                    for item in offers
                ):
                    self._mark_fertilizer_row_blocked(offer.row_index, offer.cost)
                    self._log_callback(
                        f"Fertilizer {offer.row_name} still reads the same after the purchase click. Pausing further attempts on that row for a while."
                    )
            else:
                margin_blocked = [
                    offer
                    for offer in offers
                    if not self._offer_row_is_blocked(offer)
                    if self._offer_cost_is_safe(bits, offer.cost)
                    if bits.is_at_least(offer.cost)
                    if not bits.is_at_least(offer.cost, margin=self._settings.affordable_margin)
                ]
                if margin_blocked:
                    offer = min(margin_blocked, key=lambda item: item.row_index)
                    self._log_callback(
                        f"Skipping Fertilizer {offer.row_name}: {bits.short_label()} is below the configured {self._settings.affordable_margin:g}x safety margin for {offer.cost.short_label()}."
                    )

        if exit_when_done:
            closed_capture = self._force_fertilizer_closed(capture, context="recheck")
            if closed_capture is None:
                self._disable_fertilizer_checks("Could not close Fertilizer cleanly. Disabling further Fertilizer checks for this run.")

    def _window_in_fertilizer(self, capture: WindowCapture) -> bool:
        if is_fertilizer_view(capture, self._calibration, minimum_rows=2):
            return True
        if self._fertilizer_assumed_open and is_fertilizer_view(capture, self._calibration, minimum_rows=1):
            return True
        return False

    def _read_bits(self, capture: WindowCapture) -> DisplayAmount | None:
        bits_rgb = capture.crop(self._calibration.bits_rect)
        amount = read_bits(bits_rgb)
        if amount is None:
            return None
        return amount

    def _read_confirmed_bits(
        self,
        capture: WindowCapture,
    ) -> tuple[WindowCapture, DisplayAmount | None]:
        candidates: list[DisplayAmount] = []
        first_bits = self._read_bits(capture)
        if first_bits is not None:
            candidates.append(first_bits)

        latest_capture = capture
        if self._settings.bits_retry_interval > 0:
            for _ in range(1):
                if not wait_with_cancel(self._stop_event, self._settings.bits_retry_interval):
                    break
                try:
                    latest_capture = capture_window(self._window_id)
                except Exception:
                    break
                self._publish_preview(latest_capture)
                retry_bits = self._read_bits(latest_capture)
                if retry_bits is not None:
                    candidates.append(retry_bits)

        return (latest_capture, choose_stable_display_amount(candidates))

    def _toggle_fertilizer(self, capture: WindowCapture) -> None:
        clicked = self._click_normalized_center(capture, self._calibration.fertilizer_button_rect)
        if clicked:
            self._fertilizer_assumed_open = not self._fertilizer_assumed_open

    def _fertilizer_settle_delay(self) -> float:
        return max(self._settings.fertilizer_toggle_delay, MIN_FERTILIZER_SETTLE_SECONDS)

    def _force_fertilizer_open(
        self,
        capture: WindowCapture,
        context: str,
    ) -> WindowCapture | None:
        if self._window_in_fertilizer(capture):
            self._fertilizer_assumed_open = True
            self._last_fertilizer_open_cycle = self._cycle_index
            return capture

        self._log_callback(f"Opening Fertilizer for {context}.")
        for attempt in range(2):
            self._toggle_fertilizer(capture)
            if not wait_with_cancel(self._stop_event, self._fertilizer_settle_delay()):
                return None
            try:
                capture = capture_window(self._window_id)
            except Exception as exc:  # pragma: no cover - depends on OS/window state
                self._log_callback(f"Failed to capture after opening Fertilizer: {exc}")
                return None
            self._publish_preview(capture)
            if is_fertilizer_view(capture, self._calibration, minimum_rows=1):
                self._fertilizer_assumed_open = True
                self._last_fertilizer_open_cycle = self._cycle_index
                return capture
            self._log_callback(f"Fertilizer did not open on attempt {attempt + 1}.")
        return None

    def _force_fertilizer_open_from_unknown_state(
        self,
        capture: WindowCapture,
        context: str,
    ) -> WindowCapture | None:
        self._log_callback(f"Opening Fertilizer for {context}.")
        for attempt in range(3):
            self._toggle_fertilizer(capture)
            if not wait_with_cancel(self._stop_event, self._fertilizer_settle_delay()):
                return None
            try:
                capture = capture_window(self._window_id)
            except Exception as exc:  # pragma: no cover - depends on OS/window state
                self._log_callback(f"Failed to capture after opening Fertilizer: {exc}")
                return None
            self._publish_preview(capture)
            capture, offers = self._read_confirmed_fertilizer_offers(capture)
            if is_fertilizer_view(capture, self._calibration, minimum_rows=1) or len(offers) >= 2:
                self._fertilizer_assumed_open = True
                self._last_fertilizer_open_cycle = self._cycle_index
                return capture
            self._log_callback(f"Fertilizer did not open on attempt {attempt + 1}.")
        return None

    def _force_fertilizer_closed(
        self,
        capture: WindowCapture,
        context: str,
    ) -> WindowCapture | None:
        if not self._window_in_fertilizer(capture) and not self._fertilizer_assumed_open:
            self._fertilizer_assumed_open = False
            return capture

        self._log_callback(f"Closing Fertilizer after {context}.")
        for attempt in range(2):
            self._toggle_fertilizer(capture)
            if not wait_with_cancel(self._stop_event, self._fertilizer_settle_delay()):
                return None
            try:
                capture = capture_window(self._window_id)
            except Exception as exc:  # pragma: no cover - depends on OS/window state
                self._log_callback(f"Failed to capture after closing Fertilizer: {exc}")
                return None
            self._publish_preview(capture)
            if not is_fertilizer_view(capture, self._calibration, minimum_rows=1):
                self._fertilizer_assumed_open = False
                self._log_callback("Leaving Fertilizer to resume harvesting.")
                return capture
            self._log_callback(f"Fertilizer did not close on attempt {attempt + 1}.")
        return None

    def _capture_missing_fertilizer_button_refs(self, capture: WindowCapture) -> None:
        ref_keys = [
            ("fertilizer_purchase_1_rect", "fertilizer_purchase_1_ref"),
            ("fertilizer_purchase_2_rect", "fertilizer_purchase_2_ref"),
            ("fertilizer_purchase_3_rect", "fertilizer_purchase_3_ref"),
        ]
        captured = 0
        for rect_key, ref_key in ref_keys:
            if getattr(self._calibration, ref_key):
                continue
            rect = getattr(self._calibration, rect_key)
            if rect is None:
                continue
            reference = build_fertilizer_button_reference(capture.crop(rect))
            if reference:
                setattr(self._calibration, ref_key, reference)
                captured += 1
        if captured:
            self._log_callback(
                f"Captured {captured} missing Fertilizer button reference snapshot(s) during runtime."
            )

    def _disable_fertilizer_checks(self, reason: str) -> None:
        if self._stop_event.is_set() or self._fertilizer_checks_disabled:
            return
        self._fertilizer_checks_disabled = True
        self._fertilizer_startup_ready = False
        self._known_offers = {}
        self._blocked_fertilizer_rows.clear()
        self._affordable_recheck_pending = False
        self._log_callback(reason)

    def _remember_fertilizer_offers(self, offers: list[FertilizerOffer]) -> None:
        for offer in offers:
            self._known_offers[offer.row_index] = offer.cost
            blocked = self._blocked_fertilizer_rows.get(offer.row_index)
            if blocked is None:
                continue
            if blocked[0] != amount_signature(offer.cost):
                self._blocked_fertilizer_rows.pop(offer.row_index, None)

    def _active_known_costs(self) -> list[DisplayAmount]:
        active: list[DisplayAmount] = []
        for row_index, cost in sorted(self._known_offers.items()):
            if self._known_row_is_blocked(row_index, cost):
                continue
            active.append(cost)
        return active

    def _cheapest_known_cost(self) -> DisplayAmount | None:
        active_costs = self._active_known_costs()
        if not active_costs:
            return None
        return min(active_costs, key=display_amount_sort_key)

    def _bits_near_cost(self, bits: DisplayAmount, cost: DisplayAmount) -> bool:
        if bits.rank != cost.rank:
            return bits.rank > cost.rank
        return bits.mantissa >= (cost.mantissa * 0.75)

    def _mark_fertilizer_row_blocked(self, row_index: int, cost: DisplayAmount) -> None:
        interval_cycles = max(int(round(self._settings.fertilizer_check_interval)), 1)
        unblock_cycle = self._cycle_index + max(interval_cycles * 6, 12)
        self._blocked_fertilizer_rows[row_index] = (amount_signature(cost), unblock_cycle)

    def _known_row_is_blocked(self, row_index: int, cost: DisplayAmount) -> bool:
        blocked = self._blocked_fertilizer_rows.get(row_index)
        if blocked is None:
            return False
        signature, unblock_cycle = blocked
        if signature != amount_signature(cost):
            self._blocked_fertilizer_rows.pop(row_index, None)
            return False
        return self._cycle_index < unblock_cycle

    def _offer_row_is_blocked(self, offer: FertilizerOffer) -> bool:
        return self._known_row_is_blocked(offer.row_index, offer.cost)

    def _offer_cost_is_safe(self, bits: DisplayAmount, cost: DisplayAmount) -> bool:
        if cost.suffix:
            return True
        if bits.suffix and cost.mantissa >= 1000:
            self._log_callback(
                f"Skipping suspicious Fertilizer cost parse {cost.short_label()} because the suffix was missing."
            )
            return False
        return True

    def _publish_preview(
        self,
        capture: WindowCapture,
        debug_preview: CyclePlanPreview | None = None,
    ) -> None:
        if self._preview_callback is None:
            return
        try:
            self._preview_callback(capture, debug_preview)
        except Exception:
            return

    def _log_fertilizer_offer_summary(self, offers: list[FertilizerOffer]) -> None:
        if not offers:
            self._log_callback("Fertilizer offers: none")
            return
        offers_by_row = {offer.row_index: offer for offer in offers}
        parts: list[str] = []
        for row_index, (cost_rect, button_rect) in enumerate(self._calibration.fertilizer_offer_slots()):
            if cost_rect is None or button_rect is None:
                continue
            offer = offers_by_row.get(row_index)
            if offer is None:
                parts.append(f"{fertilizer_row_name(row_index)} unreadable")
            else:
                parts.append(f"{fertilizer_row_name(row_index)} {offer.cost.short_label()}")
        remembered = ", ".join(parts)
        self._log_callback(f"Fertilizer offers: {remembered}")

    def _read_confirmed_fertilizer_offers(
        self,
        capture: WindowCapture,
    ) -> tuple[WindowCapture, list]:
        samples: list[list[FertilizerOffer]] = [read_fertilizer_offers(capture, self._calibration)]
        latest_capture = capture
        expected_rows = sum(
            1
            for cost_rect, button_rect in self._calibration.fertilizer_offer_slots()
            if cost_rect is not None and button_rect is not None
        )

        if self._settings.bits_retry_interval > 0:
            for _ in range(2):
                if len(aggregate_offer_samples(samples)) >= expected_rows:
                    break
                if not wait_with_cancel(self._stop_event, self._settings.bits_retry_interval):
                    break
                try:
                    latest_capture = capture_window(self._window_id)
                except Exception:
                    break
                self._publish_preview(latest_capture)
                samples.append(read_fertilizer_offers(latest_capture, self._calibration))

        aggregated_offers = aggregate_offer_samples(samples)
        return (latest_capture, aggregated_offers)

    def _maybe_click_sprinkler(self, capture: WindowCapture) -> bool:
        if self._calibration.sprinkler_rect is None:
            return False
        if self._settings.sprinkler_interval <= 0:
            return False
        if self._window_in_fertilizer(capture):
            return False
        interval_cycles = max(int(round(self._settings.sprinkler_interval)), 0)
        if interval_cycles <= 0:
            return False
        if self._cycle_index - self._last_sprinkler_cycle < interval_cycles:
            return False
        if not self._click_normalized_center(capture, self._calibration.sprinkler_rect):
            self._log_callback("Could not click the Gaming sprinkler. The selected Idleon window may not be frontmost.")
            return False
        self._last_sprinkler_cycle = self._cycle_index
        self._log_callback("Clicked Gaming sprinkler.")
        return True

    def _click_normalized_center(self, capture: WindowCapture, rect: NormalizedRect) -> bool:
        pixel_rect = capture.rect_from_normalized(rect)
        image_x, image_y = pixel_rect.center
        screen_x, screen_y = capture.image_to_screen(image_x, image_y)
        return self._click_screen_point(screen_x, screen_y, pixel_rect.width, pixel_rect.height)

    def _click_screen_point(self, screen_x: int, screen_y: int, width: int, height: int) -> bool:
        if self._stop_event.is_set():
            return False
        if not self._ensure_target_window_active():
            return False
        target_x = jitter_value(screen_x, width, self._settings.jitter_pixels)
        target_y = jitter_value(screen_y, height, self._settings.jitter_pixels)
        duration = self._mouse_move_duration()
        if not self._move_mouse_with_cancel(target_x, target_y, duration):
            return False
        if self._stop_event.is_set():
            return False
        pyautogui.click()
        return True

    def _move_mouse_with_cancel(self, target_x: int, target_y: int, duration: float) -> bool:
        if duration <= 0.02:
            pyautogui.moveTo(target_x, target_y, duration=duration, tween=pyautogui.easeInOutQuad)
            return not self._stop_event.is_set()

        start_x, start_y = pyautogui.position()
        steps = max(int(duration / 0.02), 1)
        step_delay = duration / steps
        for step in range(1, steps + 1):
            if self._stop_event.is_set():
                return False
            progress = pyautogui.easeInOutQuad(step / steps)
            current_x = round(start_x + ((target_x - start_x) * progress))
            current_y = round(start_y + ((target_y - start_y) * progress))
            pyautogui.moveTo(current_x, current_y, duration=0)
            if step < steps and self._stop_event.wait(step_delay):
                return False
        return not self._stop_event.is_set()

    def _mouse_move_duration(self) -> float:
        duration = random.uniform(
            self._settings.move_duration_min,
            self._settings.move_duration_max,
        )
        return max(duration / max(self._settings.mouse_speed, 0.1), 0.01)

    def _ensure_target_window_active(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and window_owner_is_frontmost(self._window_id):
            return True
        self._last_window_activate_attempt = now
        try:
            activated = activate_window_owner(self._window_id)
        except Exception:
            activated = False
        if not activated and now - self._last_window_activate_failure_log >= 10.0:
            self._last_window_activate_failure_log = now
            self._log_callback(
                "Could not bring the selected Idleon window to the front. Clicks may miss if another app stays focused."
            )
        return activated

    def _should_click(self, match: MatchResult, screen_x: int, screen_y: int) -> bool:
        key = cycle_click_key(screen_x, screen_y)
        if key in self._cycle_clicked_targets:
            return False
        self._cycle_clicked_targets.add(key)
        return True


def jitter_value(center: int, size: int, jitter_pixels: int) -> int:
    if jitter_pixels <= 0:
        return center
    half = max((size // 2) - 2, 0)
    jitter = min(jitter_pixels, half)
    return center + random.randint(-jitter, jitter)


def cooldown_key(template_name: str, x: int, y: int) -> str:
    return f"{template_name}:{round(x / 10)}:{round(y / 10)}"


def cycle_click_key(x: int, y: int) -> str:
    return f"{round(x / 4)}:{round(y / 4)}"


def points_within_distance(
    left_x: int,
    left_y: int,
    right_x: int,
    right_y: int,
    min_distance: int,
) -> bool:
    dx = left_x - right_x
    dy = left_y - right_y
    return (dx * dx) + (dy * dy) <= (min_distance * min_distance)


def target_path_sort_key(
    target: PlannedClick,
    current_x: int,
    current_y: int,
) -> tuple[int, float]:
    screen_x = target.screen_x
    screen_y = target.screen_y
    dx = screen_x - current_x
    dy = screen_y - current_y
    return ((dx * dx) + (dy * dy), -target.match.score)


def wait_with_cancel(stop_event: threading.Event, seconds: float) -> bool:
    deadline = time.monotonic() + max(seconds, 0.0)
    while time.monotonic() < deadline:
        if stop_event.wait(0.05):
            return False
    return not stop_event.is_set()


def filter_clickable_matches(
    capture: WindowCapture,
    calibration: GameCalibration,
    field_rect: PixelRect,
    matches: list[MatchResult],
    settings: RunSettings,
) -> list[MatchResult]:
    imports_px = (
        capture.rect_from_normalized(calibration.imports_rect)
        if calibration.imports_rect is not None
        else None
    )
    accepted: list[MatchResult] = []
    for match in matches:
        global_rect = match_to_global_rect(field_rect, match)
        if imports_px is not None and rects_overlap(global_rect, imports_px):
            continue
        accepted.append(match)
    return accepted


def match_to_global_rect(field_rect: PixelRect, match: MatchResult) -> PixelRect:
    left = field_rect.left + match.center_x - (match.width // 2)
    top = field_rect.top + match.center_y - (match.height // 2)
    return PixelRect(left=left, top=top, width=match.width, height=match.height)


def rects_overlap(left: PixelRect, right: PixelRect) -> bool:
    if left.right <= right.left or right.right <= left.left:
        return False
    if left.bottom <= right.top or right.bottom <= left.top:
        return False
    return True


def rects_conflict(left: PixelRect, right: PixelRect, ratio: float) -> bool:
    if rects_overlap(left, right):
        return True
    left_center = left.center
    right_center = right.center
    dx = left_center[0] - right_center[0]
    dy = left_center[1] - right_center[1]
    min_distance = max(int(max(left.width, left.height, right.width, right.height) * max(ratio, 0.35)), 10)
    return (dx * dx) + (dy * dy) <= (min_distance * min_distance)


def cluster_match_targets(
    capture: WindowCapture,
    field_rect: PixelRect,
    matches: list[MatchResult],
    settings: RunSettings,
) -> list[PlannedClick]:
    clusters: list[list[MatchResult]] = []
    cluster_rects: list[PixelRect] = []

    for match in sorted(matches, key=lambda item: (item.score, item.priority), reverse=True):
        global_rect = match_to_global_rect(field_rect, match)
        assigned_index = None
        for index, cluster_rect in enumerate(cluster_rects):
            if rects_conflict(global_rect, cluster_rect, 1.15):
                assigned_index = index
                break
        if assigned_index is None:
            clusters.append([match])
            cluster_rects.append(global_rect)
        else:
            clusters[assigned_index].append(match)
            cluster_rects[assigned_index] = union_rect(cluster_rects[assigned_index], global_rect)

    targets: list[PlannedClick] = []
    for cluster_index, cluster in enumerate(clusters):
        dominant_family = choose_cluster_family(cluster, settings)
        if dominant_family is None:
            continue
        family_matches = [match for match in cluster if template_family(match.template_name) == dominant_family]
        representative = max(family_matches, key=lambda item: (item.priority, item.score))
        target_rect = expand_rect(
            union_rects([match_to_global_rect(field_rect, match) for match in family_matches]),
            4,
        )
        if is_ambiguous_edge_hotspot(cluster_rects[cluster_index], field_rect, cluster, dominant_family):
            continue
        image_x, image_y = weighted_cluster_center(field_rect, family_matches)
        screen_x, screen_y = capture.image_to_screen(image_x, image_y)
        targets.append(
            PlannedClick(
                match=representative,
                image_x=image_x,
                image_y=image_y,
                screen_x=screen_x,
                screen_y=screen_y,
                global_rect=target_rect,
            )
        )
    return targets


def choose_cluster_family(
    cluster: list[MatchResult],
    settings: RunSettings,
) -> str | None:
    if not cluster:
        return None
    family_counts = Counter(template_family(match.template_name) for match in cluster)
    family_weights = Counter(
        {
            family: sum(max(match.score - settings.threshold, 0.03) for match in cluster if template_family(match.template_name) == family)
            for family in family_counts
        }
    )
    dominant_family, dominant_count = max(
        family_counts.items(),
        key=lambda item: (
            family_weights[item[0]],
            item[1],
            max(match.score for match in cluster if template_family(match.template_name) == item[0]),
        ),
    )
    dominant_matches = [match for match in cluster if template_family(match.template_name) == dominant_family]
    best_score = max(match.score for match in dominant_matches)
    dominant_weight = family_weights[dominant_family]
    total_weight = max(sum(family_weights.values()), 1e-6)
    weight_share = dominant_weight / total_weight
    count_share = dominant_count / max(len(cluster), 1)
    if len(cluster) == 1:
        if best_score >= max(settings.threshold + 0.10, 0.68):
            return dominant_family
        return None
    if dominant_count >= 3 and weight_share >= 0.42:
        return dominant_family
    if dominant_count >= 2 and weight_share >= 0.54 and count_share >= 0.34:
        return dominant_family
    if best_score >= max(settings.threshold + 0.26, 0.84) and weight_share >= 0.40:
        return dominant_family
    return None


def weighted_cluster_center(
    field_rect: PixelRect,
    matches: list[MatchResult],
) -> tuple[int, int]:
    total_weight = sum(max(match.score, 0.01) for match in matches)
    image_x = round(
        sum((field_rect.left + match.center_x) * max(match.score, 0.01) for match in matches) / total_weight
    )
    image_y = round(
        sum((field_rect.top + match.center_y) * max(match.score, 0.01) for match in matches) / total_weight
    )
    return (image_x, image_y)


def template_family(template_name: str) -> str:
    return template_name.split("_", 1)[0]


def union_rect(left: PixelRect, right: PixelRect) -> PixelRect:
    min_left = min(left.left, right.left)
    min_top = min(left.top, right.top)
    max_right = max(left.right, right.right)
    max_bottom = max(left.bottom, right.bottom)
    return PixelRect(
        left=min_left,
        top=min_top,
        width=max_right - min_left,
        height=max_bottom - min_top,
    )


def union_rects(rects: list[PixelRect]) -> PixelRect:
    current = rects[0]
    for rect in rects[1:]:
        current = union_rect(current, rect)
    return current


def expand_rect(rect: PixelRect, padding: int) -> PixelRect:
    return PixelRect(
        left=rect.left - padding,
        top=rect.top - padding,
        width=rect.width + (padding * 2),
        height=rect.height + (padding * 2),
    )


def is_ambiguous_edge_hotspot(
    cluster_rect: PixelRect,
    field_rect: PixelRect,
    cluster: list[MatchResult],
    dominant_family: str,
) -> bool:
    edge_margin = 8
    touches_edge = (
        cluster_rect.left <= field_rect.left + edge_margin
        or cluster_rect.top <= field_rect.top + edge_margin
        or cluster_rect.right >= field_rect.right - edge_margin
        or cluster_rect.bottom >= field_rect.bottom - edge_margin
    )
    if not touches_edge:
        return False

    family_counts = Counter(template_family(match.template_name) for match in cluster)
    dominant_count = family_counts.get(dominant_family, 0)
    if len(family_counts) <= 1:
        return False
    return dominant_count / max(len(cluster), 1) < 0.70


def same_display_amount(left: DisplayAmount, right: DisplayAmount) -> bool:
    if left.rank != right.rank:
        return False
    tolerance = max(left.mantissa, right.mantissa, 1.0) * 0.015
    return abs(left.mantissa - right.mantissa) <= tolerance


def amount_signature(amount: DisplayAmount) -> tuple[int, int]:
    return (amount.rank, int(round(amount.mantissa * 1000)))


def display_amount_sort_key(amount: DisplayAmount) -> tuple[int, float]:
    return (amount.rank, amount.mantissa)


def choose_stable_display_amount(candidates: list[DisplayAmount]) -> DisplayAmount | None:
    if not candidates:
        return None

    groups: list[list[DisplayAmount]] = []
    for candidate in candidates:
        for group in groups:
            if same_display_amount(candidate, group[0]):
                group.append(candidate)
                break
        else:
            groups.append([candidate])

    best_group = max(
        groups,
        key=lambda group: (
            len(group),
            stable_amount_group_key(group),
        ),
    )
    return choose_best_amount(best_group)


def aggregate_offer_samples(samples: list[list[FertilizerOffer]]) -> list[FertilizerOffer]:
    offers_by_row: dict[int, list[FertilizerOffer]] = {}
    for offers in samples:
        for offer in offers:
            offers_by_row.setdefault(offer.row_index, []).append(offer)

    aggregated: list[FertilizerOffer] = []
    for row_index in sorted(offers_by_row):
        best_offer = choose_best_offer(offers_by_row[row_index])
        if best_offer is not None:
            aggregated.append(best_offer)
    return aggregated


def choose_best_offer(offers: list[FertilizerOffer]) -> FertilizerOffer | None:
    if not offers:
        return None

    groups: list[list[FertilizerOffer]] = []
    for offer in offers:
        for group in groups:
            if same_display_amount(offer.cost, group[0].cost):
                group.append(offer)
                break
        else:
            groups.append([offer])

    best_group = max(
        groups,
        key=lambda group: (
            len(group),
            best_offer_amount_key(group),
        ),
    )
    best_amount = choose_best_amount([offer.cost for offer in best_group])
    if best_amount is None:
        return best_group[-1]

    for offer in reversed(best_group):
        if same_display_amount(offer.cost, best_amount):
            return offer
    return best_group[-1]


def best_offer_amount_key(offers: list[FertilizerOffer]) -> tuple[int, int, int, float]:
    best_amount = choose_best_amount([offer.cost for offer in offers])
    if best_amount is None:
        return (0, 0, 0, 0.0)
    return (
        int(best_amount.mantissa > 0),
        mantissa_digit_count(best_amount),
        -best_amount.rank,
        best_amount.mantissa,
    )


def stable_amount_group_key(amounts: list[DisplayAmount]) -> tuple[int, int, int, float]:
    best_amount = choose_best_amount(amounts)
    if best_amount is None:
        return (0, 0, 0, 0.0)
    return (
        int(best_amount.mantissa > 0),
        mantissa_digit_count(best_amount),
        -best_amount.rank,
        best_amount.mantissa,
    )


def mantissa_digit_count(amount: DisplayAmount) -> int:
    if abs(amount.mantissa - round(amount.mantissa)) < 1e-6:
        return len(str(int(round(amount.mantissa))))
    return len("".join(char for char in f"{amount.mantissa:g}" if char.isdigit()))
