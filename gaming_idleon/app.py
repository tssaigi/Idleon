from __future__ import annotations

from dataclasses import dataclass
import queue
import random
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import customtkinter as ctk
import pyautogui
from PIL import Image, ImageDraw, ImageTk

from .clicker import ClickWorker, filter_clickable_matches
from .economy import (
    build_fertilizer_button_reference,
    is_fertilizer_view,
    read_bits,
    read_fertilizer_offers,
)
from .matcher import find_matches, prepare_templates
from .models import (
    CyclePlanPreview,
    MIN_FERTILIZER_SETTLE_SECONDS,
    GameCalibration,
    NormalizedRect,
    PreviewTarget,
    RunSettings,
    SheetImportOptions,
    TemplateImage,
    WindowCapture,
    fertilizer_row_name,
)
from .sprites import load_templates_from_path, slice_sprite_sheet
from .storage import clear_session, load_session, save_session
from .windowing import (
    capture_window,
    capture_window_thumbnail,
    find_window,
    has_screen_capture_access,
    list_windows,
    open_screen_recording_settings,
    request_screen_capture_access,
)


IMAGE_FILE_TYPES = [
    ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.gif"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("GIF", "*.gif"),
    ("Bitmap", "*.bmp"),
    ("All files", "*.*"),
]

BUTTON_DEBOUNCE_SECONDS = 0.08
IDLE_PREVIEW_REFRESH_MS = 900
ACTIVE_PREVIEW_REFRESH_MS = 5000

DETECTION_SECTION_SUMMARIES = {
    "Matching": "How strict the plant matching is and how much scale variation it searches.",
    "Rhythm": "How quickly the bot repeats work and how aggressively it avoids re-clicking the same area.",
    "Automation": "How it revisits Fertilizer and the sprinkler once the loop is running.",
}

DETECTION_SETTINGS_LAYOUT = [
    (
        "Matching",
        "Shape threshold",
        "threshold_var",
        "How sure the shape match needs to be before the bot believes it found a plant.",
    ),
    (
        "Matching",
        "Min scale",
        "min_scale_var",
        "The smallest size version of your sprite that the matcher will try.",
    ),
    (
        "Matching",
        "Max scale",
        "max_scale_var",
        "The largest size version of your sprite that the matcher will try.",
    ),
    (
        "Matching",
        "Scale step",
        "scale_step_var",
        "How finely the matcher walks between min and max scale. Lower is more precise but slower.",
    ),
    (
        "Rhythm",
        "Cycle interval (s)",
        "cycle_interval_var",
        "How long the bot waits after finishing a cycle before it starts the next one. Set 0 for no extra wait.",
    ),
    (
        "Rhythm",
        "Click cooldown (s)",
        "click_cooldown_var",
        "How long the bot avoids clicking the same plant area again, even if a different sprite frame matches there.",
    ),
    (
        "Rhythm",
        "Mouse speed",
        "mouse_speed_var",
        "How fast the cursor moves before each click. Higher is faster.",
    ),
    (
        "Rhythm",
        "Start delay (s)",
        "start_delay_var",
        "How long the bot arms itself before it starts, so you can refocus the game window.",
    ),
    (
        "Rhythm",
        "Jitter px",
        "jitter_var",
        "Small random mouse offset in pixels so the clicks do not land on the exact same point every time.",
    ),
    (
        "Automation",
        "Fertilizer recheck (cycles)",
        "fertilizer_check_var",
        "How many full harvest cycles it waits before trying Fertilizer again. Set 0 to disable Fertilizer rechecks.",
    ),
    (
        "Automation",
        "Fertilizer settle (s)",
        "fertilizer_settle_var",
        "How long the bot keeps Fertilizer open before reading it after a toggle. There is a minimum so OCR has time to settle.",
    ),
    (
        "Automation",
        "Fertilizer buy margin",
        "fertilizer_margin_var",
        "A safety multiplier on Fertilizer prices before the bot buys. For example, 1.2 means it waits until bits are at least 120% of the shown cost.",
    ),
    (
        "Automation",
        "Sprinkler interval (cycles)",
        "sprinkler_interval_var",
        "How many full harvest cycles it waits before clicking the sprinkler again. Set 0 to disable it and treat the sprinkler like a no-click zone.",
    ),
]


def build_detection_settings_help_text() -> str:
    lines = ["What these settings do:", "Timing values are labeled in either seconds or cycles."]
    current_section = None
    for section, label, _, description in DETECTION_SETTINGS_LAYOUT:
        if section != current_section:
            current_section = section
            lines.append(section)
        lines.append(f"{label}: {description}")
    return "\n\n".join(lines)


@dataclass(slots=True)
class CalibrationStep:
    key: str
    title: str
    prompt: str


class SpriteSheetDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc) -> None:
        self.rows_var = tk.IntVar(value=4)
        self.cols_var = tk.IntVar(value=4)
        self.margin_var = tk.IntVar(value=0)
        self.spacing_var = tk.IntVar(value=0)
        self.trim_var = tk.BooleanVar(value=True)
        self.result: SheetImportOptions | None = None
        super().__init__(parent, title="Sprite Sheet Grid")

    def body(self, master: tk.Misc) -> tk.Widget | None:
        tk.Label(master, text="Rows").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        tk.Entry(master, textvariable=self.rows_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        tk.Label(master, text="Columns").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        tk.Entry(master, textvariable=self.cols_var, width=10).grid(
            row=1, column=1, sticky="ew", padx=4, pady=4
        )
        tk.Label(master, text="Outer margin").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        tk.Entry(master, textvariable=self.margin_var, width=10).grid(
            row=2, column=1, sticky="ew", padx=4, pady=4
        )
        tk.Label(master, text="Cell spacing").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        tk.Entry(master, textvariable=self.spacing_var, width=10).grid(
            row=3, column=1, sticky="ew", padx=4, pady=4
        )
        tk.Checkbutton(master, text="Trim empty borders", variable=self.trim_var).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            padx=4,
            pady=4,
        )
        return None

    def validate(self) -> bool:
        return self.rows_var.get() > 0 and self.cols_var.get() > 0

    def apply(self) -> None:
        self.result = SheetImportOptions(
            rows=self.rows_var.get(),
            cols=self.cols_var.get(),
            margin=max(self.margin_var.get(), 0),
            spacing=max(self.spacing_var.get(), 0),
            trim_empty=self.trim_var.get(),
        )


class WindowPickerDialog(ctk.CTkToplevel):
    def __init__(self, master: ctk.CTk) -> None:
        super().__init__(master)
        self.title("Choose Idleon Window")
        self.geometry("1040x760")
        self.result: int | None = None
        self._images: list[ctk.CTkImage] = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="#F4EAD6", corner_radius=18)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=18)
        ctk.CTkLabel(
            header,
            text="Choose the Idleon window to control",
            font=ctk.CTkFont("Avenir Next", 24, "bold"),
            text_color="#26412F",
        ).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            header,
            text="This behaves like a screen-share picker: pick one live window, then all scanning and clicking stay inside it.",
            font=ctk.CTkFont("Avenir Next", 14),
            text_color="#5D6A60",
        ).pack(anchor="w", padx=18, pady=(0, 16))

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#FBF7EF", corner_radius=18)
        self.scroll.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.scroll.grid_columnconfigure(0, weight=1)
        self.scroll.grid_columnconfigure(1, weight=1)

        if not has_screen_capture_access():
            card = ctk.CTkFrame(self.scroll, fg_color="#FFFFFF", corner_radius=18)
            card.grid(row=0, column=0, columnspan=2, padx=12, pady=12, sticky="ew")
            ctk.CTkLabel(
                card,
                text="Screen Recording permission is missing.",
                font=ctk.CTkFont("Avenir Next", 22, "bold"),
                text_color="#2D3D34",
            ).pack(anchor="w", padx=16, pady=(16, 8))
            ctk.CTkLabel(
                card,
                text=(
                    "macOS will return blank previews for other apps until you allow Screen Recording "
                    "for the terminal or app that launched GamingIdleon."
                ),
                font=ctk.CTkFont("Avenir Next", 14),
                text_color="#5D6A60",
                justify="left",
                wraplength=860,
            ).pack(anchor="w", padx=16, pady=(0, 14))
            ctk.CTkButton(
                card,
                text="Open Screen Recording Settings",
                fg_color="#2F6B43",
                hover_color="#245236",
                command=open_screen_recording_settings,
            ).pack(anchor="w", padx=16, pady=(0, 16))
            self.transient(master)
            self.grab_set()
            return

        windows = list_windows()
        if not windows:
            ctk.CTkLabel(self.scroll, text="No shareable windows were found.").grid(
                row=0, column=0, padx=12, pady=12, sticky="w"
            )
        else:
            for index, window in enumerate(windows):
                thumb = capture_window_thumbnail(window.window_id)
                card = ctk.CTkFrame(self.scroll, fg_color="#FFFFFF", corner_radius=16)
                card.grid(row=index // 2, column=index % 2, padx=12, pady=12, sticky="nsew")

                if thumb is not None:
                    image = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)
                    self._images.append(image)
                    preview = ctk.CTkButton(
                        card,
                        text="",
                        image=image,
                        width=thumb.size[0] + 12,
                        height=thumb.size[1] + 12,
                        fg_color="#EEF3EB",
                        hover_color="#DFEBDD",
                        command=lambda win_id=window.window_id: self._choose(win_id),
                    )
                    preview.pack(fill="both", padx=12, pady=(12, 8))

                ctk.CTkButton(
                    card,
                    text=window.label,
                    font=ctk.CTkFont("Avenir Next", 16, "bold"),
                    anchor="w",
                    fg_color="#2F6B43",
                    hover_color="#245236",
                    command=lambda win_id=window.window_id: self._choose(win_id),
                ).pack(fill="x", padx=12, pady=(0, 8))
                ctk.CTkLabel(
                    card,
                    text=f"{window.bounds.width} x {window.bounds.height}",
                    font=ctk.CTkFont("Avenir Next", 13),
                    text_color="#6A756D",
                ).pack(anchor="w", padx=12, pady=(0, 12))

        self.transient(master)
        self.grab_set()

    def _choose(self, window_id: int) -> None:
        self.result = window_id
        self.destroy()


class RegionSelectorDialog(ctk.CTkToplevel):
    def __init__(
        self,
        master: ctk.CTk,
        window_id: int,
        steps: list[CalibrationStep],
        existing: dict[str, NormalizedRect | None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Live Calibration")
        self.geometry("1260x940")
        self.result_map: dict[str, NormalizedRect] = {}
        self._window_id = window_id
        self._steps = steps
        self._step_index = 0
        self._existing = existing or {}
        self._capture_after_id: str | None = None
        self._latest_image: Image.Image | None = None
        self._display_image = Image.new("RGB", (960, 540), "#111111")
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._start: tuple[int, int] | None = None
        self._rect_id: int | None = None
        self._current_rect: tuple[int, int, int, int] | None = None
        self._image_item: int | None = None
        self._overlay_ids: list[int] = []
        self._updating = True
        self._capture_inflight = False
        self._capture_result_queue: queue.Queue[Image.Image | None] = queue.Queue()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="#F4EAD6", corner_radius=18)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=18)
        self.title_var = tk.StringVar(value=self._steps[0].title)
        self.prompt_var = tk.StringVar(value=self._steps[0].prompt)
        ctk.CTkLabel(
            header,
            textvariable=self.title_var,
            font=ctk.CTkFont("Avenir Next", 22, "bold"),
            text_color="#26412F",
        ).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            header,
            textvariable=self.prompt_var,
            font=ctk.CTkFont("Avenir Next", 14),
            text_color="#5D6A60",
            justify="left",
            wraplength=1100,
        ).pack(anchor="w", padx=18, pady=(0, 16))

        canvas_frame = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=18)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0, background="#111111")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self._render_image()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        self.step_var = tk.StringVar(value=self._step_status())
        ctk.CTkLabel(
            footer,
            textvariable=self.step_var,
            font=ctk.CTkFont("Avenir Next", 13, "bold"),
            text_color="#415047",
        ).pack(side="left", padx=(0, 14))
        ctk.CTkButton(
            footer,
            text="Reset",
            fg_color="#E7E4DB",
            hover_color="#D7D1C1",
            text_color="#33423A",
            command=self._reset,
        ).pack(side="left")
        ctk.CTkButton(
            footer,
            text="Cancel",
            fg_color="#C95E58",
            hover_color="#AF4B45",
            command=self._cancel,
        ).pack(side="right")
        self.use_button = ctk.CTkButton(
            footer,
            text="Save Region",
            fg_color="#2F6B43",
            hover_color="#245236",
            command=self._accept,
        )
        self.use_button.pack(side="right", padx=(0, 12))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.transient(master)
        self.grab_set()
        self._schedule_capture()

    def _render_image(self) -> None:
        max_width = 1160
        max_height = 720
        self._display_image = self._latest_image.copy() if self._latest_image else self._display_image.copy()
        self._display_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        source_width = max((self._latest_image.width if self._latest_image else self._display_image.width), 1)
        self._scale = self._display_image.width / source_width
        self._photo = ImageTk.PhotoImage(self._display_image)
        self.canvas.config(width=self._display_image.width, height=self._display_image.height)
        if self._image_item is None:
            self._image_item = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfigure(self._image_item, image=self._photo)
        self._redraw_overlays()

    def _on_press(self, event: tk.Event) -> None:
        self._start = (int(event.x), int(event.y))
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#7FE2A4",
            width=3,
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self._start is None or self._rect_id is None:
            return
        self.canvas.coords(self._rect_id, self._start[0], self._start[1], event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self._start is None:
            return
        x0, y0 = self._start
        x1, y1 = int(event.x), int(event.y)
        left, right = sorted([x0, x1])
        top, bottom = sorted([y0, y1])
        self._current_rect = (left, top, right, bottom)

    def _reset(self) -> None:
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = None
        self._current_rect = None
        self._start = None
        self._redraw_overlays()

    def _accept(self) -> None:
        if self._current_rect is None:
            messagebox.showinfo("No region", "Draw a rectangle first.")
            return
        left, top, right, bottom = self._current_rect
        if right - left < 8 or bottom - top < 8:
            messagebox.showinfo("Too small", "Choose a larger region.")
            return
        scale = max(self._scale, 1e-6)
        source_width = max((self._latest_image.width if self._latest_image else self._display_image.width), 1)
        source_height = max((self._latest_image.height if self._latest_image else self._display_image.height), 1)
        rect = NormalizedRect(
            left=(left / scale) / source_width,
            top=(top / scale) / source_height,
            width=((right - left) / scale) / source_width,
            height=((bottom - top) / scale) / source_height,
        ).clamp()
        step = self._steps[self._step_index]
        self.result_map[step.key] = rect
        self._existing[step.key] = rect
        self._current_rect = None
        self._start = None
        self._step_index += 1
        if self._step_index >= len(self._steps):
            self._finish()
            return
        self.title_var.set(self._steps[self._step_index].title)
        self.prompt_var.set(self._steps[self._step_index].prompt)
        self.step_var.set(self._step_status())
        self._redraw_overlays()

    def _schedule_capture(self) -> None:
        self._capture_after_id = self.after(220, self._refresh_capture)

    def _refresh_capture(self) -> None:
        if not self._updating:
            return
        self._drain_capture_queue()
        if not self._capture_inflight:
            self._capture_inflight = True
            threading.Thread(target=self._load_capture_async, daemon=True).start()
        self._schedule_capture()

    def _load_capture_async(self) -> None:
        image: Image.Image | None = None
        try:
            capture = capture_window(self._window_id)
            image = Image.fromarray(capture.image_rgb)
        except Exception:
            image = None
        self._capture_result_queue.put(image)

    def _drain_capture_queue(self) -> None:
        drained = False
        latest_image: Image.Image | None = None
        while True:
            try:
                latest_image = self._capture_result_queue.get_nowait()
                drained = True
            except queue.Empty:
                break
        if not drained:
            return
        self._capture_inflight = False
        if not self._updating or latest_image is None:
            return
        self._latest_image = latest_image
        self._render_image()

    def _redraw_overlays(self) -> None:
        for overlay_id in self._overlay_ids:
            self.canvas.delete(overlay_id)
        self._overlay_ids.clear()

        if self._latest_image is None:
            return

        palette = ["#7FE2A4", "#FFB45C", "#76B8FF", "#D58CFF"]
        for index, step in enumerate(self._steps):
            rect = self.result_map.get(step.key) or self._existing.get(step.key)
            if rect is None:
                continue
            left = rect.left * self._latest_image.width * self._scale
            top = rect.top * self._latest_image.height * self._scale
            right = (rect.left + rect.width) * self._latest_image.width * self._scale
            bottom = (rect.top + rect.height) * self._latest_image.height * self._scale
            color = palette[index % len(palette)]
            self._overlay_ids.append(
                self.canvas.create_rectangle(left, top, right, bottom, outline=color, width=2)
            )
            self._overlay_ids.append(
                self.canvas.create_text(
                    left + 8,
                    top + 8,
                    anchor="nw",
                    text=step.title,
                    fill=color,
                    font=("Avenir Next", 12, "bold"),
                )
            )

        if self._current_rect is not None:
            left, top, right, bottom = self._current_rect
            self._overlay_ids.append(
                self.canvas.create_rectangle(left, top, right, bottom, outline="#7FE2A4", width=3)
            )

    def _step_status(self) -> str:
        return f"Step {self._step_index + 1} of {len(self._steps)}"

    def _finish(self) -> None:
        self._updating = False
        if self._capture_after_id is not None:
            self.after_cancel(self._capture_after_id)
            self._capture_after_id = None
        self.destroy()

    def _cancel(self) -> None:
        self.result_map = {}
        self._finish()


class PriorityDialog(ctk.CTkToplevel):
    def __init__(self, master: ctk.CTk, templates: list[TemplateImage]) -> None:
        super().__init__(master)
        self.title("Set Sprite Priority")
        self.geometry("1040x820")
        self._templates = templates
        self._images: list[ctk.CTkImage] = []
        self._vars: list[tk.StringVar] = []
        self._mode_var = tk.StringVar(value="value")
        self._value_frame: ctk.CTkScrollableFrame | None = None
        self._tier_frame: ctk.CTkScrollableFrame | None = None
        self._tier_zone_frames: dict[str, ctk.CTkFrame] = {}
        self._tier_drop_colors: dict[str, str] = {}
        self._tier_assignments: dict[str, list[TemplateImage]] = build_tier_assignments(self._templates)
        self._drag_template: TemplateImage | None = None
        self._drag_widget: tk.Widget | None = None
        self._hover_tier: str | None = None
        self.saved = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="#F4EAD6", corner_radius=18)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=18)
        ctk.CTkLabel(
            header,
            text="Set Sprite Priority",
            font=ctk.CTkFont("Avenir Next", 24, "bold"),
            text_color="#26412F",
        ).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            header,
            text="Higher priority sprites are clicked before lower priority sprites, even when match scores are similar.",
            font=ctk.CTkFont("Avenir Next", 14),
            text_color="#5D6A60",
            justify="left",
            wraplength=860,
        ).pack(anchor="w", padx=18, pady=(0, 16))
        ctk.CTkLabel(
            header,
            text="Tier mode starts new sprites as Unranked. Drag images into S-F when you want them in the click order.",
            font=ctk.CTkFont("Avenir Next", 13),
            text_color="#6C776F",
            justify="left",
            wraplength=860,
        ).pack(anchor="w", padx=18, pady=(0, 14))
        mode_switch = ctk.CTkSegmentedButton(
            header,
            values=["Value", "Tier List"],
            variable=self._mode_var,
            command=self._switch_mode,
            selected_color="#2F6B43",
            selected_hover_color="#245236",
            unselected_color="#E7E4DB",
            unselected_hover_color="#D7D1C1",
        )
        mode_switch.pack(anchor="w", padx=18, pady=(0, 16))
        mode_switch.set("Value")

        self._build_value_mode()
        self._build_tier_mode()
        self._show_mode("value")

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        ctk.CTkButton(
            footer,
            text="Cancel",
            fg_color="#C95E58",
            hover_color="#AF4B45",
            command=self.destroy,
        ).pack(side="right")
        ctk.CTkButton(
            footer,
            text="Save Priorities",
            fg_color="#2F6B43",
            hover_color="#245236",
            command=self._save,
        ).pack(side="right", padx=(0, 12))

        self.transient(master)
        self.grab_set()

    def _build_value_mode(self) -> None:
        self._value_frame = ctk.CTkScrollableFrame(self, fg_color="#FBF7EF", corner_radius=18)
        self._value_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self._value_frame.grid_columnconfigure(1, weight=1)

        for row_index, template in enumerate(self._templates):
            card = ctk.CTkFrame(self._value_frame, fg_color="#FFFFFF", corner_radius=16)
            card.grid(row=row_index, column=0, sticky="ew", padx=12, pady=8)
            card.grid_columnconfigure(1, weight=1)

            image = make_template_preview(template, size=(76, 76))
            self._images.append(image)
            ctk.CTkLabel(card, text="", image=image, width=76, height=76).grid(
                row=0, column=0, rowspan=2, padx=14, pady=12
            )
            ctk.CTkLabel(
                card,
                text=template.name,
                font=ctk.CTkFont("Avenir Next", 16, "bold"),
                text_color="#2D3D34",
            ).grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(12, 2))
            ctk.CTkLabel(
                card,
                text=f"{Path(template.source).name}  |  {template.width}x{template.height}",
                font=ctk.CTkFont("Avenir Next", 13),
                text_color="#6A756D",
            ).grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(0, 12))

            value_var = tk.StringVar(value=str(template.priority))
            self._vars.append(value_var)
            ctk.CTkLabel(
                card,
                text="Priority",
                font=ctk.CTkFont("Avenir Next", 13, "bold"),
                text_color="#415047",
            ).grid(row=0, column=2, sticky="e", padx=(0, 8), pady=(12, 2))
            ctk.CTkEntry(card, textvariable=value_var, width=90).grid(
                row=1, column=2, sticky="e", padx=(0, 14), pady=(0, 12)
            )

    def _build_tier_mode(self) -> None:
        self._tier_frame = ctk.CTkScrollableFrame(self, fg_color="#FBF7EF", corner_radius=18)
        self._tier_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self._render_tier_mode()

    def _render_tier_mode(self) -> None:
        if self._tier_frame is None:
            return
        for child in self._tier_frame.winfo_children():
            child.destroy()
        self._tier_zone_frames.clear()
        self._tier_drop_colors.clear()

        for row_index, tier in enumerate(TIER_ORDER):
            row = ctk.CTkFrame(self._tier_frame, fg_color="#F4EFE4", corner_radius=12)
            row.grid(row=row_index, column=0, sticky="ew", padx=10, pady=4)
            row.grid_columnconfigure(1, weight=1)

            badge = ctk.CTkFrame(row, fg_color=TIER_COLORS[tier], corner_radius=10, width=96, height=96)
            badge.grid(row=0, column=0, sticky="ns", padx=(0, 4))
            badge.grid_propagate(False)
            ctk.CTkLabel(
                badge,
                text=tier,
                font=ctk.CTkFont("Avenir Next", 30, "bold"),
                text_color="#FFFFFF",
            ).place(relx=0.5, rely=0.42, anchor="center")
            ctk.CTkLabel(
                badge,
                text=f"{len(self._tier_assignments[tier])}",
                font=ctk.CTkFont("Avenir Next", 12, "bold"),
                text_color="#FFF4EA",
            ).place(relx=0.5, rely=0.78, anchor="center")

            zone = ctk.CTkFrame(row, fg_color="#FFFFFF", corner_radius=10, border_width=1, border_color="#D8D0BF")
            zone.grid(row=0, column=1, sticky="ew", padx=(0, 0))
            for grid_col in range(6):
                zone.grid_columnconfigure(grid_col, weight=0)
            self._tier_zone_frames[tier] = zone
            self._tier_drop_colors[tier] = "#FFFFFF"
            self._bind_drag_surface(zone)

            self._render_tier_zone(zone, tier, empty_text="Drop sprites here")

        bank_card = ctk.CTkFrame(self._tier_frame, fg_color="#FFFDF7", corner_radius=18)
        bank_card.grid(row=len(TIER_ORDER), column=0, sticky="ew", padx=10, pady=(14, 8))
        bank_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            bank_card,
            text="Sprite Bank",
            font=ctk.CTkFont("Avenir Next", 20, "bold"),
            text_color="#24362C",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 2))
        ctk.CTkLabel(
            bank_card,
            text="Unranked sprites live here until you drag them into a tier.",
            font=ctk.CTkFont("Avenir Next", 12),
            text_color="#65736A",
        ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 10))

        bank_zone = ctk.CTkFrame(bank_card, fg_color="#F8F4EB", corner_radius=16, border_width=1, border_color="#DDD3C2")
        bank_zone.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        for grid_col in range(6):
            bank_zone.grid_columnconfigure(grid_col, weight=0)
        self._tier_zone_frames[UNRANKED_TIER] = bank_zone
        self._tier_drop_colors[UNRANKED_TIER] = "#F8F4EB"
        self._bind_drag_surface(bank_zone)
        self._render_tier_zone(bank_zone, UNRANKED_TIER, empty_text="New sprites appear here")

    def _render_tier_zone(self, zone: ctk.CTkFrame, tier: str, empty_text: str) -> None:
        for child in zone.winfo_children():
            child.destroy()
        templates = self._tier_assignments[tier]
        if not templates:
            placeholder = ctk.CTkLabel(
                zone,
                text=empty_text,
                font=ctk.CTkFont("Avenir Next", 13),
                text_color="#8A8F85",
                height=54,
            )
            placeholder.grid(row=0, column=0, padx=14, pady=12, sticky="w")
            self._bind_drag_surface(placeholder)
            return

        for index, template in enumerate(templates):
            image = make_template_preview(template, size=(74, 74))
            self._images.append(image)
            label = ctk.CTkLabel(
                zone,
                text="",
                image=image,
                width=74,
                height=74,
                fg_color="#EEE6D6" if tier == UNRANKED_TIER else "#F2EDE2",
                corner_radius=8,
                cursor="hand2",
            )
            grid_row = index // 6
            grid_col = index % 6
            label.grid(row=grid_row, column=grid_col, padx=7, pady=7, sticky="w")
            label.bind("<ButtonPress-1>", lambda event, item=template: self._start_drag(event, item), add="+")
            label.bind("<B1-Motion>", self._drag_motion, add="+")
            label.bind("<ButtonRelease-1>", self._finish_drag, add="+")

    def _bind_drag_surface(self, widget: tk.Widget) -> None:
        widget.bind("<B1-Motion>", self._drag_motion, add="+")
        widget.bind("<ButtonRelease-1>", self._finish_drag, add="+")

    def _switch_mode(self, label: str) -> None:
        mode = "value" if label == "Value" else "tier"
        self._show_mode(mode)

    def _show_mode(self, mode: str) -> None:
        if self._value_frame is None or self._tier_frame is None:
            return
        if mode == "value":
            self._tier_frame.grid_remove()
            self._value_frame.grid()
        else:
            self._value_frame.grid_remove()
            self._tier_frame.grid()

    def _start_drag(self, event: tk.Event, template: TemplateImage) -> None:
        self._drag_template = template
        self._drag_widget = event.widget
        try:
            event.widget.grab_set()
        except tk.TclError:
            pass
        if hasattr(event.widget, "configure"):
            event.widget.configure(fg_color="#DCCB9A")

    def _drag_motion(self, event: tk.Event) -> None:
        if self._drag_template is None:
            return
        self._set_hover_tier(self._tier_from_screen_position(event.x_root, event.y_root))

    def _finish_drag(self, event: tk.Event) -> None:
        if self._drag_template is None:
            return
        target_tier = self._tier_from_screen_position(event.x_root, event.y_root)
        template = self._drag_template
        drag_widget = self._drag_widget
        self._drag_template = None
        self._drag_widget = None
        self._set_hover_tier(None)
        if drag_widget is not None and hasattr(drag_widget, "configure"):
            try:
                drag_widget.configure(fg_color="#EEE6D6" if priority_to_tier(template.priority) == UNRANKED_TIER else "#F2EDE2")
            except tk.TclError:
                pass
        if drag_widget is not None:
            try:
                drag_widget.grab_release()
            except tk.TclError:
                pass
        if target_tier is None:
            return

        for tier, templates in self._tier_assignments.items():
            if template in templates:
                templates.remove(template)
                break
        self._tier_assignments[target_tier].append(template)
        self._render_tier_mode()

    def _tier_from_screen_position(self, x_root: int, y_root: int) -> str | None:
        for tier, frame in self._tier_zone_frames.items():
            left = frame.winfo_rootx()
            top = frame.winfo_rooty()
            right = left + frame.winfo_width()
            bottom = top + frame.winfo_height()
            if left <= x_root <= right and top <= y_root <= bottom:
                return tier
        return None

    def _set_hover_tier(self, tier: str | None) -> None:
        if tier == self._hover_tier:
            return
        self._hover_tier = tier
        for key, frame in self._tier_zone_frames.items():
            base_color = self._tier_drop_colors.get(key, "#FFFFFF")
            hover_color = lighten_hex(base_color, 0.10)
            frame.configure(fg_color=hover_color if key == tier else base_color, border_color="#C0B08E" if key == tier else "#D8D0BF")

    def _save(self) -> None:
        if self._mode_var.get().lower().startswith("value"):
            for template, value_var in zip(self._templates, self._vars):
                try:
                    template.priority = int(value_var.get().strip() or "0")
                except ValueError:
                    messagebox.showerror("Invalid Priority", f"Priority for {template.name} must be an integer.")
                    return
        else:
            apply_tier_priorities(self._tier_assignments)
        self.saved = True
        self.destroy()


class MainWindow:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("GamingIdleon")
        self.root.geometry("1440x920")
        self.root.minsize(1220, 800)

        session = load_session()
        restore_last_session = False
        if session_has_restorable_data(session):
            restore_last_session = messagebox.askyesno(
                "Restore Last Session?",
                (
                    "Restore last session?\n\n"
                    "Yes keeps the last field calibration, Fertilizer calibration, window selection, "
                    "loaded templates, and sprinkler region.\n"
                    "No resets the app to a fresh start."
                ),
            )

        if restore_last_session:
            self.settings = RunSettings.from_dict(session.get("settings", {}))
            self.calibration = GameCalibration.from_dict(session.get("calibration"))
            self.templates = restore_templates_from_manifest(session.get("templates", []))
            self.selected_window_id = session.get("selected_window_id")
            if self.selected_window_id is not None and find_window(self.selected_window_id) is None:
                self.selected_window_id = None
        else:
            if session:
                clear_session()
            self.settings = RunSettings()
            self.calibration = GameCalibration()
            self.templates = []
            self.selected_window_id = None

        self.worker: ClickWorker | None = None

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.state_queue: queue.Queue[str] = queue.Queue()
        self.preview_queue: queue.Queue[tuple[int, object | None, Image.Image | None]] = queue.Queue()
        self.worker_preview_queue: queue.Queue[tuple[int, WindowCapture, CyclePlanPreview | None]] = queue.Queue()
        self._window_preview: ImageTk.PhotoImage | None = None
        self._window_preview_debug: CyclePlanPreview | None = None
        self._preview_after_id: str | None = None
        self._preview_refresh_inflight = False
        self._scan_once_inflight = False
        self._stop_inflight = False
        self._button_press_times: dict[str, float] = {}

        self.threshold_var = tk.StringVar(value=f"{self.settings.threshold:g}")
        self.min_scale_var = tk.StringVar(value=f"{self.settings.min_scale:g}")
        self.max_scale_var = tk.StringVar(value=f"{self.settings.max_scale:g}")
        self.scale_step_var = tk.StringVar(value=f"{self.settings.scale_step:g}")
        self.cycle_interval_var = tk.StringVar(value=f"{self.settings.scan_interval:g}")
        self.click_cooldown_var = tk.StringVar(value=f"{self.settings.cooldown_seconds:g}")
        self.mouse_speed_var = tk.StringVar(value=f"{self.settings.mouse_speed:g}")
        self.start_delay_var = tk.StringVar(value=f"{self.settings.start_delay:g}")
        self.fertilizer_check_var = tk.StringVar(value=str(int(round(self.settings.fertilizer_check_interval))))
        self.fertilizer_settle_var = tk.StringVar(value=f"{self.settings.fertilizer_toggle_delay:g}")
        self.fertilizer_margin_var = tk.StringVar(value=f"{self.settings.affordable_margin:g}")
        self.sprinkler_interval_var = tk.StringVar(value=str(int(round(self.settings.sprinkler_interval))))
        self.jitter_var = tk.StringVar(value=str(self.settings.jitter_pixels))

        self.window_label_var = tk.StringVar(value="No window selected")
        self.state_var = tk.StringVar(value="Idle")
        self.calibration_var = tk.StringVar(value=self._calibration_summary())

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._drain_queues)
        self.root.after(150, self._ensure_screen_recording_access)
        self.root.after(500, self._auto_pick_window)
        self.root.after(700, self._schedule_window_preview_refresh)

    def _build_ui(self) -> None:
        shell = ctk.CTkFrame(self.root, fg_color="#EEE7D8", corner_radius=0)
        shell.pack(fill="both", expand=True)
        shell.grid_columnconfigure(0, weight=0)
        shell.grid_columnconfigure(1, weight=1)
        shell.grid_rowconfigure(0, weight=1)

        rail = ctk.CTkFrame(shell, width=360, fg_color="#F7F1E4", corner_radius=0)
        rail.grid(row=0, column=0, sticky="nsw")
        rail.grid_propagate(False)

        main = ctk.CTkFrame(shell, fg_color="#F4F2EC", corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        hero = ctk.CTkFrame(rail, fg_color="#FFFFFF", corner_radius=18, border_width=1, border_color="#DDD8CC")
        hero.pack(fill="x", padx=18, pady=(18, 12))
        ctk.CTkLabel(
            hero,
            text="GamingIdleon",
            font=ctk.CTkFont("Avenir Next", 27, "bold"),
            text_color="#1F2A23",
        ).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            hero,
            text="Minimal Idleon Gaming automation with one-window capture, OCR, and Fertilizer routing.",
            font=ctk.CTkFont("Avenir Next", 14),
            text_color="#667066",
            justify="left",
            wraplength=300,
        ).pack(anchor="w", padx=18, pady=(0, 14))
        ctk.CTkLabel(
            hero,
            textvariable=self.state_var,
            font=ctk.CTkFont("Avenir Next", 14, "bold"),
            text_color="#244533",
            fg_color="#E7F0E8",
            corner_radius=999,
            padx=14,
            pady=6,
        ).pack(anchor="w", padx=18, pady=(0, 16))

        action_card = ctk.CTkFrame(rail, fg_color="#FFFFFF", corner_radius=18, border_width=1, border_color="#E2DDD1")
        action_card.pack(fill="x", padx=18, pady=(0, 12))
        buttons = [
            ("Choose Window", self.pick_window, "#2C4D3A"),
            ("Calibrate Field", self.calibrate_field_regions, "#ECE7DC"),
            ("Add Sprinkler", self.calibrate_sprinkler, "#ECE7DC"),
            ("Calibrate Fertilizer", self.calibrate_fertilizer_panel, "#ECE7DC"),
            ("Add Images", self.add_images, "#ECE7DC"),
            ("Set Priority", self.set_priority, "#ECE7DC"),
            ("Scan Once", self.scan_once, "#EDF3EC"),
            ("Start", self.start_clicking, "#2F6B43"),
            ("Stop", self.stop_clicking, "#C95E58"),
        ]
        for index, (label, command, color) in enumerate(buttons):
            is_neutral = color in {"#ECE7DC", "#EDF3EC"}
            self._make_responsive_button(
                action_card,
                action_key=f"main:{label}",
                text=label,
                command=command,
                fg_color=color,
                hover_color=darken_hex(color),
                text_color="#2F3A33" if is_neutral else "#F7F6F1",
                border_width=1 if is_neutral else 0,
                border_color="#D8D1C2" if is_neutral else None,
                font=ctk.CTkFont("Avenir Next", 14, "bold"),
                height=38,
            ).pack(fill="x", padx=14, pady=(14 if index == 0 else 0, 10))

        template_card = ctk.CTkFrame(rail, fg_color="#FFFFFF", corner_radius=18, border_width=1, border_color="#E2DDD1")
        template_card.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        ctk.CTkLabel(
            template_card,
            text="Loaded Templates",
            font=ctk.CTkFont("Avenir Next", 19, "bold"),
            text_color="#2D3D34",
        ).pack(anchor="w", padx=16, pady=(14, 8))
        self.template_list = tk.Listbox(
            template_card,
            height=14,
            bg="#FCFAF5",
            fg="#2A352D",
            highlightthickness=0,
            borderwidth=0,
            selectbackground="#B6D9BF",
            selectforeground="#183122",
            font=("Avenir Next", 13),
        )
        self.template_list.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        footer = ctk.CTkFrame(template_card, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(0, 16))
        self._make_responsive_button(
            footer,
            action_key="templates:remove",
            text="Remove",
            width=100,
            fg_color="#E7E4DB",
            hover_color="#D7D1C1",
            text_color="#33423A",
            command=self.remove_selected,
        ).pack(side="left")
        self._make_responsive_button(
            footer,
            action_key="templates:reset",
            text="Reset",
            width=100,
            fg_color="#E7E4DB",
            hover_color="#D7D1C1",
            text_color="#33423A",
            command=self.reset_templates,
        ).pack(side="left", padx=(10, 0))

        content = ctk.CTkScrollableFrame(
            main,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color="#D6D1C4",
            scrollbar_button_hover_color="#C7C0B0",
        )
        content.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        content.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(content, fg_color="transparent", corner_radius=0)
        top.pack(fill="x", pady=(0, 12))
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)

        window_card = ctk.CTkFrame(top, fg_color="#FFFFFF", corner_radius=18, border_width=1, border_color="#E2DDD1")
        window_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ctk.CTkLabel(
            window_card,
            text="Selected Window",
            font=ctk.CTkFont("Avenir Next", 20, "bold"),
            text_color="#2D3D34",
        ).pack(anchor="w", padx=16, pady=(14, 6))
        self.window_preview_label = ctk.CTkLabel(
            window_card,
            text="Choose the Idleon game window",
            font=ctk.CTkFont("Avenir Next", 14),
            text_color="#5E6B63",
            width=360,
            height=220,
            fg_color="#F4F2EC",
            corner_radius=16,
        )
        self.window_preview_label.pack(fill="x", padx=16, pady=(0, 10))
        ctk.CTkLabel(
            window_card,
            textvariable=self.window_label_var,
            font=ctk.CTkFont("Avenir Next", 15),
            text_color="#39463E",
        ).pack(anchor="w", padx=16, pady=(0, 16))

        calibration_card = ctk.CTkFrame(top, fg_color="#FFFFFF", corner_radius=18, border_width=1, border_color="#E2DDD1")
        calibration_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ctk.CTkLabel(
            calibration_card,
            text="Calibration",
            font=ctk.CTkFont("Avenir Next", 20, "bold"),
            text_color="#2D3D34",
        ).pack(anchor="w", padx=16, pady=(14, 6))
        ctk.CTkLabel(
            calibration_card,
            textvariable=self.calibration_var,
            font=ctk.CTkFont("Avenir Next", 14),
            text_color="#39463E",
            justify="left",
            wraplength=520,
        ).pack(anchor="w", padx=16, pady=(0, 12))
        ctk.CTkLabel(
            calibration_card,
            text=(
                "Field view: draw the plant area, bits number, Fertilizer toggle button, and Imports no-click zone. "
                "Add the optional sprinkler separately. Fertilizer view: draw 6 exact boxes: cost 1, buy 1, cost 2, buy 2, cost 3, buy 3."
            ),
            font=ctk.CTkFont("Avenir Next", 13),
            text_color="#6C776F",
            justify="left",
            wraplength=520,
        ).pack(anchor="w", padx=16, pady=(0, 16))

        settings_card = ctk.CTkFrame(content, fg_color="#FFFFFF", corner_radius=18, border_width=1, border_color="#E2DDD1")
        settings_card.pack(fill="x", pady=(0, 12))
        settings_card.grid_columnconfigure(0, weight=1)
        settings_header = ctk.CTkFrame(settings_card, fg_color="transparent")
        settings_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))
        settings_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            settings_header,
            text="Detection Settings",
            font=ctk.CTkFont("Avenir Next", 20, "bold"),
            text_color="#2D3D34",
        ).grid(row=0, column=0, sticky="w")
        self._make_responsive_button(
            settings_header,
            action_key="settings:help",
            text="?",
            width=34,
            height=34,
            fg_color="#E7E4DB",
            hover_color="#D7D1C1",
            text_color="#33423A",
            font=ctk.CTkFont("Avenir Next", 18, "bold"),
            command=self.show_detection_settings_help,
        ).grid(row=0, column=1, padx=(0, 10))
        self._make_responsive_button(
            settings_header,
            action_key="settings:set",
            text="Set",
            width=84,
            height=34,
            fg_color="#2F6B43",
            hover_color="#245236",
            font=ctk.CTkFont("Avenir Next", 14, "bold"),
            command=self.apply_settings,
        ).grid(row=0, column=2)
        ctk.CTkLabel(
            settings_card,
            text="Everything is grouped by what it changes, so matching, loop timing, and automation are easier to tune at a glance.",
            font=ctk.CTkFont("Avenir Next", 13),
            text_color="#667066",
            justify="left",
            wraplength=1080,
        ).grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 14))

        settings_sections = ctk.CTkFrame(settings_card, fg_color="transparent")
        settings_sections.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        for column in range(3):
            settings_sections.grid_columnconfigure(column, weight=1, uniform="settings")

        settings_by_section: dict[str, list[tuple[str, str, str]]] = {}
        for section, label, variable_name, description in DETECTION_SETTINGS_LAYOUT:
            settings_by_section.setdefault(section, []).append((label, variable_name, description))

        for column, section in enumerate(DETECTION_SECTION_SUMMARIES):
            section_card = ctk.CTkFrame(
                settings_sections,
                fg_color="#F8F6F0",
                corner_radius=18,
                border_width=1,
                border_color="#E2DDD1",
            )
            section_card.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else 6, 0 if column == 2 else 6),
            )
            ctk.CTkLabel(
                section_card,
                text=section,
                font=ctk.CTkFont("Avenir Next", 18, "bold"),
                text_color="#2A372F",
            ).pack(anchor="w", padx=16, pady=(14, 4))
            ctk.CTkLabel(
                section_card,
                text=DETECTION_SECTION_SUMMARIES[section],
                font=ctk.CTkFont("Avenir Next", 12),
                text_color="#6A7368",
                justify="left",
                wraplength=320,
            ).pack(anchor="w", padx=16, pady=(0, 12))
            for label, variable_name, description in settings_by_section.get(section, []):
                self._add_setting_card(
                    section_card,
                    label,
                    getattr(self, variable_name),
                    description,
                )

        log_card = ctk.CTkFrame(content, fg_color="#FFFFFF", corner_radius=18, border_width=1, border_color="#E2DDD1")
        log_card.pack(fill="both", expand=True, pady=(0, 0))
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            log_card,
            text="Activity",
            font=ctk.CTkFont("Avenir Next", 20, "bold"),
            text_color="#2D3D34",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))
        self.log_text = ctk.CTkTextbox(
            log_card,
            fg_color="#F8F6F0",
            text_color="#28342D",
            font=ctk.CTkFont("Avenir Next", 13),
            height=360,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.log_text.configure(state="disabled")

        self._refresh_templates()
        self._refresh_window_preview()

    def _add_setting_card(
        self,
        parent: ctk.CTkFrame,
        label: str,
        variable: tk.Variable,
        description: str,
    ) -> None:
        card = ctk.CTkFrame(
            parent,
            fg_color="#FFFFFF",
            corner_radius=14,
            border_width=1,
            border_color="#E2DDD1",
        )
        card.pack(fill="x", padx=12, pady=(0, 10))
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=label,
            font=ctk.CTkFont("Avenir Next", 13, "bold"),
            text_color="#33423A",
        ).grid(row=0, column=0, sticky="w", padx=(14, 10), pady=(12, 4))
        ctk.CTkEntry(
            card,
            textvariable=variable,
            width=96,
            fg_color="#F6F2EA",
            border_color="#D7CCBA",
            text_color="#24322A",
        ).grid(row=0, column=1, sticky="e", padx=(0, 14), pady=(10, 4))
        ctk.CTkLabel(
            card,
            text=description,
            font=ctk.CTkFont("Avenir Next", 11),
            text_color="#6C776F",
            justify="left",
            wraplength=250,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 12))

    def _make_responsive_button(
        self,
        parent,
        action_key: str,
        command,
        **kwargs,
    ) -> ctk.CTkButton:
        button = ctk.CTkButton(
            parent,
            command=lambda key=action_key, callback=command: self._invoke_button_action(key, callback),
            **kwargs,
        )
        button.bind(
            "<ButtonPress-1>",
            lambda _event, key=action_key, callback=command: self._invoke_button_action(key, callback),
            add="+",
        )
        return button

    def _invoke_button_action(self, action_key: str, command) -> None:
        now = time.monotonic()
        last_time = self._button_press_times.get(action_key, 0.0)
        if now - last_time < BUTTON_DEBOUNCE_SECONDS:
            return
        self._button_press_times[action_key] = now
        self.root.after(1, command)

    def _auto_pick_window(self) -> None:
        if self.selected_window_id is None and has_screen_capture_access():
            self.pick_window()

    def pick_window(self) -> None:
        if not self._ensure_screen_recording_access():
            return
        dialog = WindowPickerDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.result is None:
            return
        self.selected_window_id = dialog.result
        info = find_window(self.selected_window_id)
        self.window_label_var.set(info.label if info else f"Window #{self.selected_window_id}")
        self._refresh_window_preview()
        self._save_session()
        self.log(f"Selected window: {self.window_label_var.get()}")

    def calibrate_field_regions(self) -> None:
        if self.selected_window_id is None:
            messagebox.showinfo("No window", "Choose the Idleon window first.")
            return
        self.calibration.field_rect = None
        self.calibration.bits_rect = None
        self.calibration.fertilizer_button_rect = None
        self.calibration.imports_rect = None
        self._after_calibration_change()

        steps = [
            CalibrationStep(
                key="field_rect",
                title="Garden Field",
                prompt="Draw a box around the whole plant area. Exclude the top-right bits, side buttons, and Fertilizer panel.",
            ),
            CalibrationStep(
                key="bits_rect",
                title="Bits Counter",
                prompt=(
                    "Draw a box around the bits number and the colored bits icon immediately to its right, "
                    "near the top-right and left of the Log Book button."
                ),
            ),
            CalibrationStep(
                key="fertilizer_button_rect",
                title="Fertilizer Button",
                prompt="Draw a box around the Fertilizer button or tab that opens the upgrade screen.",
            ),
            CalibrationStep(
                key="imports_rect",
                title="Imports",
                prompt=(
                    "Draw a box around the Imports area or any field subsection the bot should never click. "
                    "Matches inside this rectangle are ignored."
                ),
            ),
        ]
        dialog = RegionSelectorDialog(self.root, self.selected_window_id, steps, existing={})
        self.root.wait_window(dialog)
        if not dialog.result_map:
            return
        for step in steps:
            rect = dialog.result_map.get(step.key)
            if rect is not None:
                setattr(self.calibration, step.key, rect)
        self._after_calibration_change()
        self.log("Saved field calibration regions.")

    def calibrate_fertilizer_panel(self) -> None:
        if self.selected_window_id is None:
            messagebox.showinfo("No window", "Choose the Idleon window first.")
            return
        messagebox.showinfo(
            "Open Fertilizer",
            (
                "Open the Fertilizer screen in Idleon first. "
                "The next window will ask for 6 exact boxes: cost 1, purchase button 1, cost 2, purchase button 2, cost 3, purchase button 3."
            ),
        )
        capture = self._require_capture()
        if capture is None:
            return
        steps = [
            CalibrationStep(
                key="fertilizer_cost_1_rect",
                title="Cost 1",
                prompt="Draw a box around the first visible Fertilizer cost and the bits icon immediately to its right.",
            ),
            CalibrationStep(
                key="fertilizer_purchase_1_rect",
                title="Purchase 1",
                prompt="Draw a box around the first row purchase button only.",
            ),
            CalibrationStep(
                key="fertilizer_cost_2_rect",
                title="Cost 2",
                prompt="Draw a box around the second visible Fertilizer cost and the bits icon immediately to its right.",
            ),
            CalibrationStep(
                key="fertilizer_purchase_2_rect",
                title="Purchase 2",
                prompt="Draw a box around the second row purchase button only.",
            ),
            CalibrationStep(
                key="fertilizer_cost_3_rect",
                title="Cost 3",
                prompt="Draw a box around the third visible Fertilizer cost and the bits icon immediately to its right.",
            ),
            CalibrationStep(
                key="fertilizer_purchase_3_rect",
                title="Purchase 3",
                prompt="Draw a box around the third row purchase button only.",
            ),
        ]
        dialog = RegionSelectorDialog(
            self.root,
            self.selected_window_id,
            steps,
            existing={
                "fertilizer_cost_1_rect": self.calibration.fertilizer_cost_1_rect,
                "fertilizer_purchase_1_rect": self.calibration.fertilizer_purchase_1_rect,
                "fertilizer_cost_2_rect": self.calibration.fertilizer_cost_2_rect,
                "fertilizer_purchase_2_rect": self.calibration.fertilizer_purchase_2_rect,
                "fertilizer_cost_3_rect": self.calibration.fertilizer_cost_3_rect,
                "fertilizer_purchase_3_rect": self.calibration.fertilizer_purchase_3_rect,
            },
        )
        self.root.wait_window(dialog)
        if not dialog.result_map:
            return

        for step in steps:
            rect = dialog.result_map.get(step.key)
            if rect is not None:
                setattr(self.calibration, step.key, rect)
        self._after_calibration_change()

        capture = self._require_capture()
        if capture is None:
            return
        button_ref_map = {
            "fertilizer_purchase_1_ref": ("fertilizer_purchase_1_rect", "Purchase 1"),
            "fertilizer_purchase_2_ref": ("fertilizer_purchase_2_rect", "Purchase 2"),
            "fertilizer_purchase_3_ref": ("fertilizer_purchase_3_rect", "Purchase 3"),
        }
        stored_refs = 0
        for ref_key, (rect_key, label) in button_ref_map.items():
            rect = getattr(self.calibration, rect_key)
            if rect is None:
                continue
            reference = build_fertilizer_button_reference(capture.crop(rect))
            setattr(self.calibration, ref_key, reference)
            if reference:
                stored_refs += 1
        offers = read_fertilizer_offers(capture, self.calibration)
        if offers:
            summary = ", ".join(
                f"{fertilizer_row_name(offer.row_index)} {offer.cost.short_label()}" for offer in offers
            )
            self.log(f"Fertilizer OCR check: {summary}")
        else:
            self.log("Fertilizer regions saved, but OCR did not parse stable values from the 3 cost boxes yet.")
        if stored_refs:
            self.log(f"Captured {stored_refs} Fertilizer button reference snapshot(s) for stricter screen detection.")

    def calibrate_sprinkler(self) -> None:
        if self.selected_window_id is None:
            messagebox.showinfo("No window", "Choose the Idleon window first.")
            return
        dialog = RegionSelectorDialog(
            self.root,
            self.selected_window_id,
            [
                CalibrationStep(
                    key="sprinkler_rect",
                    title="Gaming Sprinkler",
                    prompt=(
                        "Draw a box around the Gaming sprinkler import. In IdleOn Gaming, the sprinkler "
                        "instantly grows sprouts and recharges over time, so this click target is used on a timer."
                    ),
                )
            ],
            existing={"sprinkler_rect": self.calibration.sprinkler_rect},
        )
        self.root.wait_window(dialog)
        rect = dialog.result_map.get("sprinkler_rect")
        if rect is None:
            return
        self.calibration.sprinkler_rect = rect
        self._after_calibration_change()
        self.log("Saved sprinkler region.")

    def add_images(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Choose plant images",
            filetypes=IMAGE_FILE_TYPES,
        )
        if not paths:
            return
        added = 0
        for path in paths:
            try:
                imported = load_templates_from_path(path)
            except Exception as exc:
                self.log(f"Failed to load {path}: {exc}")
                continue
            for template in imported:
                template.kind = "plant"
                self.templates.append(make_unique_template(template, self.templates))
                added += 1
        self._refresh_templates()
        self.log(f"Loaded {added} template image(s).")

    def add_squirrel(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Choose squirrel image(s)",
            filetypes=IMAGE_FILE_TYPES,
        )
        if not paths:
            return
        added = 0
        for path in paths:
            try:
                imported = load_templates_from_path(path)
            except Exception as exc:
                self.log(f"Failed to load squirrel {path}: {exc}")
                continue
            for template in imported:
                template.kind = "squirrel"
                template.priority = -99
                self.templates.append(make_unique_template(template, self.templates))
                added += 1
        self._refresh_templates()
        self.log(f"Loaded {added} squirrel template(s).")

    def set_priority(self) -> None:
        plant_templates = [template for template in self.templates if template.kind == "plant"]
        if not plant_templates:
            messagebox.showinfo("No plant sprites", "Load plant images first.")
            return
        dialog = PriorityDialog(self.root, plant_templates)
        self.root.wait_window(dialog)
        if not dialog.saved:
            return
        self._refresh_templates()
        self.log("Updated sprite priorities.")

    def remove_selected(self) -> None:
        indices = list(self.template_list.curselection())
        if not indices:
            return
        for index in reversed(indices):
            self.templates.pop(index)
        self._refresh_templates()
        self.log(f"Removed {len(indices)} template(s).")

    def reset_templates(self) -> None:
        if not self.templates:
            self.log("No templates to reset.")
            return
        if not messagebox.askyesno(
            "Reset Loaded Templates",
            "Remove all loaded templates from the app?\n\nThis does not delete any files on disk.",
        ):
            return
        self.stop_clicking()
        removed = len(self.templates)
        self.templates.clear()
        self._refresh_templates()
        self.log(f"Reset {removed} loaded template(s).")

    def clear_templates(self) -> None:
        self.reset_templates()

    def scan_once(self) -> None:
        if self._scan_once_inflight:
            self.log("Scan Once is already running.")
            return
        if not self._ensure_screen_recording_access():
            return
        settings = self.apply_settings(log_message=False)
        if self.selected_window_id is None:
            messagebox.showinfo("No window", "Choose the Idleon window first.")
            return
        templates = list(self.templates)
        calibration = self.calibration
        self._scan_once_inflight = True
        self.log("Running Scan Once.")
        threading.Thread(
            target=self._scan_once_worker,
            args=(self.selected_window_id, templates, calibration, settings),
            daemon=True,
        ).start()

    def start_clicking(self) -> None:
        if self.worker is not None and not self.worker.running:
            self._finalize_worker_stop(log_message=False)
        if self._stop_inflight:
            return
        if self.selected_window_id is None:
            messagebox.showinfo("No window", "Choose the Idleon window first.")
            return
        if not any(template.kind == "plant" for template in self.templates):
            messagebox.showinfo("No plant templates", "Load at least one plant image first.")
            return
        if not self.calibration.ready_for_run():
            messagebox.showinfo(
                "Incomplete calibration",
                "Calibrate the field regions, Fertilizer toggle button, and all 3 Fertilizer cost/button rows first.",
            )
            return
        if self.worker is not None and self.worker.running:
            return

        settings = self.apply_settings(log_message=False)
        self.worker = ClickWorker(
            window_id=self.selected_window_id,
            calibration=self.calibration,
            templates=list(self.templates),
            settings=settings,
            log_callback=lambda message: self.log_queue.put(message),
            state_callback=lambda state: self.state_queue.put(state),
            preview_callback=lambda capture, debug_preview, window_id=self.selected_window_id: self.worker_preview_queue.put(
                (window_id, capture, debug_preview)
            ),
        )
        self.worker.start()
        self.state_var.set("Starting")
        self._save_session()

    def stop_clicking(self) -> None:
        if self.worker is None:
            return
        if not self.worker.running:
            self._finalize_worker_stop(log_message=False)
            return
        if self._stop_inflight:
            return
        self._stop_inflight = True
        self.worker.stop()
        self.state_var.set("Stopping")
        self.log("Stop requested. Waiting for the current action to finish.")

    def collect_settings(self) -> RunSettings:
        fertilizer_settle_delay = max(
            self._safe_float(self.fertilizer_settle_var, self.settings.fertilizer_toggle_delay),
            MIN_FERTILIZER_SETTLE_SECONDS,
        )
        fertilizer_buy_margin = max(
            self._safe_float(self.fertilizer_margin_var, self.settings.affordable_margin),
            1.0,
        )
        self.fertilizer_settle_var.set(f"{fertilizer_settle_delay:g}")
        self.fertilizer_margin_var.set(f"{fertilizer_buy_margin:g}")
        self.settings = RunSettings(
            threshold=self._safe_float(self.threshold_var, self.settings.threshold),
            min_scale=self._safe_float(self.min_scale_var, self.settings.min_scale),
            max_scale=self._safe_float(self.max_scale_var, self.settings.max_scale),
            scale_step=self._safe_float(self.scale_step_var, self.settings.scale_step),
            max_matches_per_template=0,
            max_clicks_per_cycle=0,
            scan_interval=max(self._safe_float(self.cycle_interval_var, self.settings.scan_interval), 0.0),
            cooldown_seconds=max(
                self._safe_float(self.click_cooldown_var, self.settings.cooldown_seconds),
                0.0,
            ),
            start_delay=max(self._safe_float(self.start_delay_var, self.settings.start_delay), 0.0),
            mouse_speed=max(self._safe_float(self.mouse_speed_var, self.settings.mouse_speed), 0.1),
            fertilizer_check_interval=max(
                self._safe_int(self.fertilizer_check_var, int(round(self.settings.fertilizer_check_interval))),
                0.0,
            ),
            fertilizer_toggle_delay=fertilizer_settle_delay,
            affordable_margin=fertilizer_buy_margin,
            sprinkler_interval=max(
                self._safe_int(self.sprinkler_interval_var, int(round(self.settings.sprinkler_interval))),
                0.0,
            ),
            jitter_pixels=max(self._safe_int(self.jitter_var, self.settings.jitter_pixels), 0),
        )
        return self.settings

    def apply_settings(self, log_message: bool = True) -> RunSettings:
        settings = self.collect_settings()
        self._save_session()
        if log_message:
            self.log("Applied detection settings.")
            if self.worker is not None and self.worker.running:
                self.log("A running worker keeps its current detection snapshot. Stop and Start to fully refresh it.")
        return settings

    def show_detection_settings_help(self) -> None:
        messagebox.showinfo("Detection Settings Help", build_detection_settings_help_text())

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_queues(self) -> None:
        latest_worker_preview: tuple[int, WindowCapture, CyclePlanPreview | None] | None = None
        while True:
            try:
                latest_worker_preview = self.worker_preview_queue.get_nowait()
            except queue.Empty:
                break
        if latest_worker_preview is not None:
            self._apply_worker_preview(*latest_worker_preview)

        while True:
            try:
                window_id, info, thumb = self.preview_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_window_preview_async(window_id, info, thumb)

        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log(message)

        while True:
            try:
                state = self.state_queue.get_nowait()
            except queue.Empty:
                break
            if state == "scan_once_done":
                self._finish_scan_once()
                continue
            if state == "stopped":
                self._finalize_worker_stop(log_message=self._stop_inflight)
                continue
            self.state_var.set(state.capitalize())
            if state == "error":
                self._stop_inflight = False
                self.worker = None

        self.root.after(120, self._drain_queues)

    def _apply_worker_preview(
        self,
        window_id: int,
        capture: WindowCapture,
        debug_preview: CyclePlanPreview | None,
    ) -> None:
        if self.selected_window_id != window_id:
            return
        effective_debug_preview = debug_preview
        if effective_debug_preview is None and self.worker is not None and self.worker.running:
            effective_debug_preview = self._window_preview_debug
        self._window_preview_debug = effective_debug_preview
        thumb = self._build_window_preview_image(
            capture,
            max_size=(360, 220),
            debug_preview=effective_debug_preview,
        )
        self._set_window_preview_image(thumb)

    def _refresh_templates(self) -> None:
        self.template_list.delete(0, "end")
        for template in self.templates:
            self.template_list.insert(
                "end",
                f"{template.kind[:1].upper()}  P{template.priority:>2}  {template.name} ({template.width}x{template.height})",
            )

    def _refresh_window_preview(self) -> None:
        if self.selected_window_id is None:
            self._set_window_preview_text("Choose the Idleon game window")
            return
        if self.worker is not None and self.worker.running:
            return
        if self._worker_blocks_preview():
            return
        if not has_screen_capture_access():
            self._set_window_preview_text(
                (
                    "Screen Recording permission is missing.\n"
                    "Open System Settings and allow it, then restart GamingIdleon."
                )
            )
            return
        if self._preview_refresh_inflight:
            return
        self._preview_refresh_inflight = True
        threading.Thread(
            target=self._load_window_preview_async,
            args=(self.selected_window_id,),
            daemon=True,
        ).start()

    def _schedule_window_preview_refresh(self) -> None:
        self._refresh_window_preview()
        self._preview_after_id = self.root.after(self._preview_refresh_delay_ms(), self._schedule_window_preview_refresh)

    def _worker_blocks_preview(self) -> bool:
        worker = self.worker
        if worker is None:
            return False
        if self._stop_inflight:
            return True
        if not worker.running:
            return False
        return self.state_var.get().strip().lower() in {"arming", "startup", "starting"}

    def _preview_refresh_delay_ms(self) -> int:
        worker = self.worker
        if worker is not None and worker.running:
            return ACTIVE_PREVIEW_REFRESH_MS
        return IDLE_PREVIEW_REFRESH_MS

    def _load_window_preview_async(self, window_id: int) -> None:
        info = find_window(window_id)
        thumb = None
        if info is not None:
            try:
                capture = capture_window(window_id)
            except Exception:
                thumb = capture_window_thumbnail(window_id, max_size=(360, 220))
            else:
                thumb = self._build_window_preview_image(capture, max_size=(360, 220))
        self.preview_queue.put((window_id, info, thumb))

    def _apply_window_preview_async(self, window_id: int, info, thumb: Image.Image | None) -> None:
        self._preview_refresh_inflight = False
        if self.selected_window_id != window_id:
            return
        if info is None:
            self.window_label_var.set(f"Window #{window_id} is no longer available")
            self._set_window_preview_text("Selected window is no longer available.")
            return
        self.window_label_var.set(info.label)
        if thumb is None:
            self._set_window_preview_text(
                (
                    "Window preview unavailable.\n"
                    "If other apps appear blank, grant Screen Recording and relaunch."
                )
            )
            return
        if self.worker is not None and self.worker.running:
            return
        self._window_preview_debug = None
        self._set_window_preview_image(thumb)

    def _set_window_preview_text(self, text: str) -> None:
        self._window_preview = None
        self._window_preview_debug = None
        self.window_preview_label._text = text
        self.window_preview_label._label.configure(image=None, text=text)

    def _set_window_preview_image(self, image: Image.Image) -> None:
        self._window_preview = ImageTk.PhotoImage(image)
        self.window_preview_label._text = ""
        self.window_preview_label._label.configure(image=self._window_preview, text="")

    def _build_window_preview_image(
        self,
        capture: WindowCapture,
        max_size: tuple[int, int],
        debug_preview: CyclePlanPreview | None = None,
    ) -> Image.Image:
        image = Image.fromarray(capture.image_rgb).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        region_specs = [
            ("Field", self.calibration.field_rect, "#56B870"),
            ("Bits", self.calibration.bits_rect, "#F2C14E"),
            ("Fertilizer", self.calibration.fertilizer_button_rect, "#B36AE2"),
            ("F Cost 1", self.calibration.fertilizer_cost_1_rect, "#6EC6FF"),
            ("F Buy 1", self.calibration.fertilizer_purchase_1_rect, "#3F8EFC"),
            ("F Cost 2", self.calibration.fertilizer_cost_2_rect, "#6EC6FF"),
            ("F Buy 2", self.calibration.fertilizer_purchase_2_rect, "#3F8EFC"),
            ("F Cost 3", self.calibration.fertilizer_cost_3_rect, "#6EC6FF"),
            ("F Buy 3", self.calibration.fertilizer_purchase_3_rect, "#3F8EFC"),
            ("Sprinkler", self.calibration.sprinkler_rect, "#FF8A65"),
            ("Imports", self.calibration.imports_rect, "#E75A5A"),
        ]

        for label, rect, color in region_specs:
            if rect is None:
                continue
            pixel_rect = capture.rect_from_normalized(rect)
            draw.rectangle(
                [pixel_rect.left, pixel_rect.top, pixel_rect.right, pixel_rect.bottom],
                outline=color,
                width=3,
            )
            draw.rectangle(
                [pixel_rect.left, max(pixel_rect.top - 20, 0), pixel_rect.left + 88, pixel_rect.top],
                fill=hex_to_rgba(color, 192),
            )
            draw.text(
                (pixel_rect.left + 4, max(pixel_rect.top - 18, 2)),
                label,
                fill="#FFFFFF",
            )

        if debug_preview is not None:
            self._draw_cycle_debug_overlay(draw, debug_preview)

        plant_count = sum(1 for template in self.templates if template.kind == "plant")
        badge_lines = [
            f"Plants: {plant_count}",
        ]
        if debug_preview is not None:
            badge_lines.append(f"Recognized: {debug_preview.filtered_match_count}")
            badge_lines.append(f"Planned: {len(debug_preview.planned_targets)}")
        if self.calibration.field_rect is not None:
            badge_lines.append("Regions shown live")
        badge_height = 18 + (18 * len(badge_lines))
        draw.rounded_rectangle(
            [12, 12, 152, 12 + badge_height],
            radius=12,
            fill=(20, 30, 24, 176),
        )
        for index, line in enumerate(badge_lines):
            draw.text((22, 22 + (index * 18)), line, fill="#F6F4EE")

        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

    def _draw_cycle_debug_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        debug_preview: CyclePlanPreview,
    ) -> None:
        if not debug_preview.planned_targets:
            return

        points = [(target.image_x, target.image_y) for target in debug_preview.planned_targets]
        if len(points) >= 2:
            draw.line(points, fill=(245, 110, 68, 210), width=3)

        for index, target in enumerate(debug_preview.planned_targets, start=1):
            draw.rectangle(
                [
                    target.box_left,
                    target.box_top,
                    target.box_left + target.box_width,
                    target.box_top + target.box_height,
                ],
                outline=(115, 16, 24, 235),
                width=3,
            )
            radius = 8
            left = target.image_x - radius
            top = target.image_y - radius
            right = target.image_x + radius
            bottom = target.image_y + radius
            fill = (74, 163, 111, 230) if index == 1 else (41, 121, 255, 220)
            draw.ellipse([left, top, right, bottom], fill=fill, outline=(255, 255, 255, 230), width=2)
            draw.text((target.image_x - 4, target.image_y - 6), str(index), fill="#FFFFFF")

            label = debug_label_for_target(target)
            label_left = max(target.image_x + 10, 0)
            label_top = max(target.image_y - 9, 0)
            text_width = max(48, min(116, 8 + (len(label) * 6)))
            draw.rounded_rectangle(
                [label_left, label_top, label_left + text_width, label_top + 18],
                radius=8,
                fill=(16, 24, 20, 180),
            )
            draw.text((label_left + 6, label_top + 3), label, fill="#F6F4EE")

    def _calibration_summary(self) -> str:
        checks = [
            ("Garden field", self.calibration.field_rect),
            ("Bits counter", self.calibration.bits_rect),
            ("Fertilizer button", self.calibration.fertilizer_button_rect),
            ("Fertilizer cost 1", self.calibration.fertilizer_cost_1_rect),
            ("Fertilizer buy 1", self.calibration.fertilizer_purchase_1_rect),
            ("Fertilizer cost 2", self.calibration.fertilizer_cost_2_rect),
            ("Fertilizer buy 2", self.calibration.fertilizer_purchase_2_rect),
            ("Fertilizer cost 3", self.calibration.fertilizer_cost_3_rect),
            ("Fertilizer buy 3", self.calibration.fertilizer_purchase_3_rect),
            ("Sprinkler", self.calibration.sprinkler_rect),
            ("Imports", self.calibration.imports_rect),
        ]
        return "\n".join(
            f"{'Ready' if rect is not None else 'Missing'}  {label}"
            for label, rect in checks
        )

    def _after_calibration_change(self) -> None:
        self.calibration_var.set(self._calibration_summary())
        self._save_session()

    def _require_capture(self):
        if not self._ensure_screen_recording_access():
            return None
        if self.selected_window_id is None:
            messagebox.showinfo("No window", "Choose the Idleon window first.")
            return None
        try:
            capture = capture_window(self.selected_window_id)
        except Exception as exc:
            messagebox.showerror("Capture failed", str(exc))
            return None
        return capture

    def _scan_once_worker(
        self,
        window_id: int,
        templates: list[TemplateImage],
        calibration: GameCalibration,
        settings: RunSettings,
    ) -> None:
        try:
            capture = capture_window(window_id)
        except Exception as exc:
            self.log_queue.put(f"Scan Once capture failed: {exc}")
            self.state_queue.put("scan_once_done")
            return

        plant_templates = [template for template in templates if template.kind == "plant"]
        plant_variants = prepare_templates(plant_templates, settings) if plant_templates else []

        if calibration.field_rect is not None and plant_variants:
            field_rect = capture.rect_from_normalized(calibration.field_rect)
            field_rgb = capture.crop(calibration.field_rect)
            matches = find_matches(field_rgb, plant_variants, settings)
            matches = filter_clickable_matches(
                capture,
                calibration,
                field_rect,
                matches,
                settings,
            )
            if not matches:
                self.log_queue.put("Scan Once: no plant matches above the current shape threshold.")
            else:
                for match in matches[:20]:
                    self.log_queue.put(
                        f"Scan Once field: {match.template_name} at "
                        f"({field_rect.left + match.center_x}, {field_rect.top + match.center_y}) "
                        f"score={match.score:.3f} priority={match.priority}"
                    )
        elif calibration.field_rect is not None:
            self.log_queue.put("Scan Once field: no templates loaded, so plant matching was skipped.")

        if calibration.bits_rect is not None:
            amount = read_bits(capture.crop(calibration.bits_rect))
            if amount is not None:
                self.log_queue.put(f"Scan Once bits: {amount.short_label()}")
            else:
                self.log_queue.put("Scan Once bits: no readable amount found.")

        if calibration.fertilizer_button_rect is not None and calibration.fertilizer_ready():
            self.log_queue.put("Scan Once fertilizer: opening Fertilizer for a strict price check.")
            capture = self._force_fertilizer_open_from_unknown_state_for_scan(
                window_id=window_id,
                capture=capture,
                calibration=calibration,
                settings=settings,
            )
            if capture is None:
                self.log_queue.put("Scan Once fertilizer: could not reliably open Fertilizer.")
                self.state_queue.put("scan_once_done")
                return

            capture, offers = self._read_confirmed_fertilizer_offers_for_scan(
                window_id,
                capture,
                calibration,
                settings,
            )
            if offers:
                self.log_queue.put(
                    "Scan Once fertilizer: "
                    + ", ".join(
                        f"{fertilizer_row_name(offer.row_index)} {offer.cost.short_label()}"
                        for offer in offers
                    )
                )
            else:
                self.log_queue.put("Scan Once fertilizer: no stable upgrade costs were parsed from the 3 cost boxes.")

            capture = self._force_fertilizer_state_for_scan(
                window_id=window_id,
                capture=capture,
                calibration=calibration,
                settings=settings,
                should_be_open=False,
            )
            if capture is None:
                self.log_queue.put("Scan Once fertilizer: Fertilizer did not close cleanly.")
        else:
            self.log_queue.put("Scan Once fertilizer: calibration is incomplete, so Fertilizer scan was skipped.")

        self.state_queue.put("scan_once_done")

    def _force_fertilizer_open_from_unknown_state_for_scan(
        self,
        window_id: int,
        capture: WindowCapture,
        calibration: GameCalibration,
        settings: RunSettings,
    ) -> WindowCapture | None:
        for _ in range(3):
            self._click_normalized_rect(capture, calibration.fertilizer_button_rect, settings)
            time.sleep(max(settings.fertilizer_toggle_delay, MIN_FERTILIZER_SETTLE_SECONDS))
            try:
                capture = capture_window(window_id)
            except Exception as exc:
                self.log_queue.put(f"Scan Once Fertilizer open capture failed: {exc}")
                return None
            capture, offers = self._read_confirmed_fertilizer_offers_for_scan(
                window_id,
                capture,
                calibration,
                settings,
            )
            if is_fertilizer_view(capture, calibration, minimum_rows=1) or len(offers) >= 2:
                return capture
        return None

    def _force_fertilizer_state_for_scan(
        self,
        window_id: int,
        capture: WindowCapture,
        calibration: GameCalibration,
        settings: RunSettings,
        should_be_open: bool,
    ) -> WindowCapture | None:
        loose_state = is_fertilizer_view(capture, calibration, minimum_rows=1)
        if not should_be_open and not loose_state:
            return capture

        action = "open" if should_be_open else "close"
        for _ in range(2):
            self._click_normalized_rect(capture, calibration.fertilizer_button_rect, settings)
            time.sleep(max(settings.fertilizer_toggle_delay, MIN_FERTILIZER_SETTLE_SECONDS))
            try:
                capture = capture_window(window_id)
            except Exception as exc:
                self.log_queue.put(f"Scan Once Fertilizer {action} capture failed: {exc}")
                return None
            current_state = is_fertilizer_view(capture, calibration, minimum_rows=1)
            if current_state == should_be_open:
                if not should_be_open:
                    self.log_queue.put("Scan Once fertilizer: closed Fertilizer and returned to the field.")
                return capture
        if not should_be_open:
            self._click_normalized_rect(capture, calibration.fertilizer_button_rect, settings)
            time.sleep(max(settings.fertilizer_toggle_delay, MIN_FERTILIZER_SETTLE_SECONDS))
            try:
                capture = capture_window(window_id)
            except Exception as exc:
                self.log_queue.put(f"Scan Once Fertilizer forced close capture failed: {exc}")
                return None
            self.log_queue.put("Scan Once fertilizer: forced one extra Fertilizer toggle to make sure it closed.")
            return capture
        return None

    def _read_confirmed_fertilizer_offers_for_scan(
        self,
        window_id: int,
        capture: WindowCapture,
        calibration: GameCalibration,
        settings: RunSettings,
    ) -> tuple[WindowCapture, list]:
        primary_offers = read_fertilizer_offers(capture, calibration)
        if not primary_offers:
            return (capture, [])
        time.sleep(max(settings.bits_retry_interval, 0.0))
        try:
            confirmed_capture = capture_window(window_id)
        except Exception as exc:
            self.log_queue.put(f"Scan Once Fertilizer confirmation capture failed: {exc}")
            return (capture, primary_offers)
        confirmed_offers = read_fertilizer_offers(confirmed_capture, calibration)
        confirmed_by_row = {offer.row_index: offer for offer in confirmed_offers}
        stable_offers = [
            confirmed_by_row[offer.row_index]
            for offer in primary_offers
            if offer.row_index in confirmed_by_row
            and confirmed_by_row[offer.row_index].cost.short_label() == offer.cost.short_label()
        ]
        return (confirmed_capture, stable_offers)

    def _click_normalized_rect(
        self,
        capture: WindowCapture,
        rect: NormalizedRect,
        settings: RunSettings,
    ) -> None:
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        pixel_rect = capture.rect_from_normalized(rect)
        image_x, image_y = pixel_rect.center
        screen_x, screen_y = capture.image_to_screen(image_x, image_y)
        jitter = max(int(settings.jitter_pixels), 0)
        offset_x = random.randint(-jitter, jitter) if jitter else 0
        offset_y = random.randint(-jitter, jitter) if jitter else 0
        pyautogui.moveTo(
            screen_x + offset_x,
            screen_y + offset_y,
            duration=self._mouse_move_duration(settings),
            tween=pyautogui.easeInOutQuad,
        )
        pyautogui.click()

    def _mouse_move_duration(self, settings: RunSettings) -> float:
        base_duration = min(max(settings.move_duration_min, 0.03), 0.12)
        speed = max(settings.mouse_speed, 0.1)
        return max(base_duration / speed, 0.01)

    def _finish_scan_once(self) -> None:
        self._scan_once_inflight = False
        self.log("Scan Once complete.")

    def _finalize_worker_stop(self, log_message: bool) -> None:
        worker = self.worker
        self._stop_inflight = False
        if worker is not None:
            worker.join(timeout=0.0)
        self.worker = None
        self.state_var.set("Stopped")
        if log_message:
            self.log("Stopped worker.")

    def _save_session(self) -> None:
        save_session(
            {
                "settings": self.settings.to_dict(),
                "calibration": self.calibration.to_dict(),
                "templates": serialize_templates(self.templates),
                "selected_window_id": self.selected_window_id,
            }
        )

    def _safe_float(self, variable: tk.Variable, fallback: float) -> float:
        try:
            return float(variable.get())
        except (tk.TclError, ValueError):
            try:
                variable.set(f"{fallback:g}")
            except tk.TclError:
                pass
            return float(fallback)

    def _safe_int(self, variable: tk.Variable, fallback: int) -> int:
        try:
            return int(variable.get())
        except (tk.TclError, ValueError):
            try:
                variable.set(str(fallback))
            except tk.TclError:
                pass
            return int(fallback)

    def _ensure_screen_recording_access(self) -> bool:
        if has_screen_capture_access():
            return True

        should_request = messagebox.askyesno(
            "Screen Recording Needed",
            (
                "macOS is blocking window capture for other apps.\n\n"
                "GamingIdleon can only preview and control external windows after "
                "Screen Recording is granted. Request permission now?"
            ),
        )
        if should_request:
            granted = request_screen_capture_access()
            if granted and has_screen_capture_access():
                messagebox.showinfo(
                    "Permission Granted",
                    "Screen Recording is enabled. If previews still look blank, relaunch GamingIdleon once.",
                )
                self._refresh_window_preview()
                return True

        open_settings = messagebox.askyesno(
            "Screen Recording Still Missing",
            (
                "Open System Settings -> Privacy & Security -> Screen Recording now?\n\n"
                "After enabling it for the app or terminal that launched GamingIdleon, quit and reopen GamingIdleon."
            ),
        )
        if open_settings:
            open_screen_recording_settings()
        self._refresh_window_preview()
        return False

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.running:
            self.stop_clicking()
            self.root.after(100, self._on_close)
            return
        if self._preview_after_id is not None:
            self.root.after_cancel(self._preview_after_id)
            self._preview_after_id = None
        self._save_session()
        self.root.destroy()


def make_unique_template(template: TemplateImage, existing: list[TemplateImage]) -> TemplateImage:
    names = {item.name for item in existing}
    if template.name not in names:
        return template
    counter = 2
    base_name = template.name
    while f"{base_name}_{counter}" in names:
        counter += 1
    return TemplateImage(
        name=f"{base_name}_{counter}",
        source=template.source,
        image_rgba=template.image_rgba,
        priority=template.priority,
        kind=template.kind,
    )


UNRANKED_TIER = "Unranked"
TIER_ORDER = ["S", "A", "B", "C", "D", "E", "F"]
TIER_DISPLAY_ORDER = [UNRANKED_TIER, *TIER_ORDER]
TIER_COLORS = {
    UNRANKED_TIER: "#8A8578",
    "S": "#D94E3D",
    "A": "#E57B39",
    "B": "#E3A83A",
    "C": "#A0B64A",
    "D": "#5EAF61",
    "E": "#5198BF",
    "F": "#7A74C9",
}
TIER_DESCRIPTIONS = {
    UNRANKED_TIER: "New sprites start here until you drag them into a tier.",
    "S": "Click these first whenever they are visible.",
    "A": "Very strong picks, just below S.",
    "B": "Solid mid-high priority.",
    "C": "Middle of the pack.",
    "D": "Low priority.",
    "E": "Very low priority.",
    "F": "Only click these after everything else.",
}
TIER_BASE_PRIORITIES = {
    "S": 700,
    "A": 600,
    "B": 500,
    "C": 400,
    "D": 300,
    "E": 200,
    "F": 100,
}


def priority_to_tier(priority: int) -> str:
    if priority <= 0:
        return UNRANKED_TIER
    if priority >= 650:
        return "S"
    if priority >= 550:
        return "A"
    if priority >= 450:
        return "B"
    if priority >= 350:
        return "C"
    if priority >= 250:
        return "D"
    if priority >= 150:
        return "E"
    return "F"


def build_tier_assignments(templates: list[TemplateImage]) -> dict[str, list[TemplateImage]]:
    assignments = {tier: [] for tier in TIER_DISPLAY_ORDER}
    for template in sorted(templates, key=lambda item: (-item.priority, item.name)):
        assignments[priority_to_tier(template.priority)].append(template)
    return assignments


def apply_tier_priorities(assignments: dict[str, list[TemplateImage]]) -> None:
    for template in assignments.get(UNRANKED_TIER, []):
        template.priority = 0
    for tier in TIER_ORDER:
        for index, template in enumerate(assignments.get(tier, [])):
            template.priority = TIER_BASE_PRIORITIES[tier] - index


def session_has_restorable_data(session: dict) -> bool:
    return bool(
        session.get("calibration")
        or session.get("templates")
        or session.get("selected_window_id")
        or session.get("settings")
    )


def serialize_templates(templates: list[TemplateImage]) -> list[dict[str, object]]:
    return [
        {
            "name": template.name,
            "source": template.source,
            "priority": template.priority,
            "kind": template.kind,
        }
        for template in templates
    ]


def restore_templates_from_manifest(manifest: list[dict[str, object]]) -> list[TemplateImage]:
    restored: list[TemplateImage] = []
    source_cache: dict[str, list[TemplateImage]] = {}
    used_names: set[str] = set()

    for item in manifest:
        source = str(item.get("source", ""))
        if not source:
            continue
        if source not in source_cache:
            try:
                source_cache[source] = load_templates_from_path(source)
            except Exception:
                source_cache[source] = []

        candidates = source_cache[source]
        saved_name = str(item.get("name", ""))
        candidate = next((template for template in candidates if template.name == saved_name), None)
        if candidate is None:
            base_name = saved_name.rsplit("_", 1)[0] if "_" in saved_name else saved_name
            candidate = next(
                (
                    template
                    for template in candidates
                    if template.name == base_name or template.name.startswith(base_name + "_f")
                ),
                None,
            )
        if candidate is None:
            continue

        restored_template = TemplateImage(
            name=saved_name or candidate.name,
            source=candidate.source,
            image_rgba=candidate.image_rgba,
            priority=int(item.get("priority", 0)),
            kind=str(item.get("kind", "plant")),
        )
        restored_template = make_unique_template(restored_template, restored)
        if restored_template.name in used_names:
            continue
        used_names.add(restored_template.name)
        restored.append(restored_template)
    return restored


def make_template_preview(template: TemplateImage, size: tuple[int, int] = (84, 84)) -> ctk.CTkImage:
    image = Image.fromarray(template.image_rgba)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new("RGBA", size, (245, 241, 232, 255))
    left = (size[0] - image.width) // 2
    top = (size[1] - image.height) // 2
    background.alpha_composite(image, dest=(left, top))
    return ctk.CTkImage(light_image=background, dark_image=background, size=size)


def darken_hex(color: str) -> str:
    color = color.lstrip("#")
    rgb = [max(int(color[index:index + 2], 16) - 18, 0) for index in (0, 2, 4)]
    return "#" + "".join(f"{value:02X}" for value in rgb)


def lighten_hex(color: str, amount: float = 0.10) -> str:
    color = color.lstrip("#")
    rgb = [int(color[index:index + 2], 16) for index in (0, 2, 4)]
    lighter = [min(int(value + ((255 - value) * amount)), 255) for value in rgb]
    return "#" + "".join(f"{value:02X}" for value in lighter)


def debug_label_for_target(target: PreviewTarget) -> str:
    species = target.template_name.split("_", 1)[0]
    if len(species) > 8:
        species = species[:8]
    return f"{species} {target.score:.2f}"


def hex_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        return (43, 74, 55, alpha)
    return (
        int(color[0:2], 16),
        int(color[2:4], 16),
        int(color[4:6], 16),
        alpha,
    )


def run() -> None:
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")
    root = ctk.CTk()
    MainWindow(root)
    root.mainloop()
