from __future__ import annotations

import subprocess
from typing import Any

import numpy as np
import Quartz
from PIL import Image

try:
    import AppKit
except Exception:  # pragma: no cover - optional macOS activation backend
    AppKit = None

from .models import PixelRect, WindowCapture, WindowInfo


EXCLUDED_OWNERS = {
    "Dock",
    "Window Server",
    "Control Center",
    "Notification Center",
}


def list_windows() -> list[WindowInfo]:
    options = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements
    )
    window_entries = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []

    windows: list[WindowInfo] = []
    for entry in window_entries:
        owner_name = str(entry.get("kCGWindowOwnerName", "")).strip()
        title = str(entry.get("kCGWindowName", "")).strip()
        layer = int(entry.get("kCGWindowLayer", 0) or 0)
        alpha = float(entry.get("kCGWindowAlpha", 1.0) or 1.0)
        bounds = parse_bounds(entry.get("kCGWindowBounds"))
        if layer != 0 or alpha <= 0.0:
            continue
        if bounds.width < 240 or bounds.height < 160:
            continue
        if owner_name in EXCLUDED_OWNERS:
            continue
        if not owner_name and not title:
            continue

        windows.append(
            WindowInfo(
                window_id=int(entry["kCGWindowNumber"]),
                owner_name=owner_name or "Unknown App",
                title=title,
                bounds=bounds,
                thumbnail_size=(bounds.width, bounds.height),
                owner_pid=int(entry.get("kCGWindowOwnerPID", 0) or 0) or None,
            )
        )

    windows.sort(key=lambda window: (window.owner_name.lower(), window.title.lower()))
    return windows


def capture_window(window_id: int) -> WindowCapture:
    if not has_screen_capture_access():
        raise RuntimeError(
            "Screen Recording permission is not granted. macOS will return blank images for other apps until you allow it."
        )

    info = find_window(window_id)
    if info is None:
        raise RuntimeError("Selected window is no longer available.")

    image = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
        Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution,
    )
    if image is None:
        raise RuntimeError("Quartz could not capture the selected window.")

    pil_image = cg_image_to_pil(image).convert("RGB")
    rgb = np.array(pil_image)
    scale_x = pil_image.width / max(info.bounds.width, 1)
    scale_y = pil_image.height / max(info.bounds.height, 1)
    return WindowCapture(
        window_id=window_id,
        image_rgb=rgb,
        bounds=info.bounds,
        scale_x=scale_x,
        scale_y=scale_y,
    )


def capture_window_thumbnail(window_id: int, max_size: tuple[int, int] = (320, 200)) -> Image.Image | None:
    try:
        capture = capture_window(window_id)
    except Exception:
        return None

    image = Image.fromarray(capture.image_rgb)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def find_window(window_id: int) -> WindowInfo | None:
    for window in list_windows():
        if window.window_id == window_id:
            return window
    return None


def activate_window_owner(window_id: int) -> bool:
    if AppKit is None:
        return False
    window = find_window(window_id)
    if window is None or window.owner_pid is None:
        return False
    application = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(window.owner_pid)
    if application is None:
        return False
    options = AppKit.NSApplicationActivateAllWindows | AppKit.NSApplicationActivateIgnoringOtherApps
    return bool(application.activateWithOptions_(options))


def window_owner_is_frontmost(window_id: int) -> bool:
    if AppKit is None:
        return False
    window = find_window(window_id)
    if window is None or window.owner_pid is None:
        return False
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    frontmost = workspace.frontmostApplication()
    if frontmost is None:
        return False
    try:
        return int(frontmost.processIdentifier()) == int(window.owner_pid)
    except Exception:
        return False


def has_screen_capture_access() -> bool:
    preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
    if preflight is None:
        return True
    return bool(preflight())


def request_screen_capture_access() -> bool:
    request = getattr(Quartz, "CGRequestScreenCaptureAccess", None)
    if request is None:
        return has_screen_capture_access()
    return bool(request())


def open_screen_recording_settings() -> None:
    subprocess.Popen(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
        ]
    )


def parse_bounds(raw_bounds: Any) -> PixelRect:
    if not raw_bounds:
        return PixelRect(0, 0, 0, 0)
    return PixelRect(
        left=int(round(raw_bounds.get("X", 0))),
        top=int(round(raw_bounds.get("Y", 0))),
        width=int(round(raw_bounds.get("Width", 0))),
        height=int(round(raw_bounds.get("Height", 0))),
    )


def cg_image_to_pil(cg_image: Quartz.CGImageRef) -> Image.Image:
    width = Quartz.CGImageGetWidth(cg_image)
    height = Quartz.CGImageGetHeight(cg_image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
    provider = Quartz.CGImageGetDataProvider(cg_image)
    data = Quartz.CGDataProviderCopyData(provider)
    image = Image.frombuffer(
        "RGBA",
        (width, height),
        bytes(data),
        "raw",
        "BGRA",
        bytes_per_row,
        1,
    )
    return image.copy()
