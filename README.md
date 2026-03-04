# GamingIdleon

Idleon Gaming harvester for macOS. It controls one selected game window, clicks plant matches by shape instead of color, reads your bits counter, checks Fertilizer costs, and revisits Fertilizer when one of the remembered upgrades becomes affordable.

## What it does

- Opens with a window picker so you choose the exact Idleon window to target.
- Prompts on launch with `Restore last session?` so you can keep or clear the previous calibration and selections.
- Keeps the selected-window preview updating live in the main UI.
- Loads standalone plant images and scans each GIF frame as its own template.
- Loads squirrel images or GIFs as exclusion templates so the bot learns what not to click.
- Lets you review loaded plant sprites and assign click priority manually with either numeric values or an S-to-F tier list.
- Matches plants by grayscale edges and shape, not exact color.
- Tolerates minor plant variation such as a few random extra pixels and slight rotation.
- Suppresses overlapping cross-template matches so one plant does not get "recognized" as several different sprites at once.
- Reads the bits counter with a Paddle-based RapidOCR `en_PP-OCRv5` recognizer first, then Apple Vision OCR, then Tesseract, while also tracking the visible color tier from the bits icon to the right of the amount.
- Opens Fertilizer at startup, reads the 3 visible costs from separately calibrated boxes plus their bits icons, remembers them, and buys affordable upgrades later.
- Supports the Gaming sprinkler on a timer after you calibrate it.
- Keeps all capture and mouse clicks inside the selected Idleon window.
- The Detection Settings card is grouped into `Matching`, `Rhythm`, and `Automation` sections, with a `Set` button to apply edits and a `?` button that explains each current setting in plain language.

## First run

```bash
cd /Volumes/Sovereignty/GamingIdleon
./run_app.command
```

The launcher creates the local `.venv` if needed and reinstalls dependencies whenever `requirements.txt` changes.

## Required setup in the app

1. Choose the Idleon window from the popup picker.
2. Click `Calibrate Field` while the normal Gaming field is visible.
3. In the live calibration window, redraw the garden field, bits counter including the bits icon on the right, Fertilizer button, and Imports no-click region from scratch.
4. If you use the Gaming sprinkler, click `Add Sprinkler` and draw its click region.
5. Open Fertilizer in Idleon.
6. Click `Calibrate Fertilizer`.
7. In the live calibration window, draw 6 exact regions in order: `Cost 1`, `Purchase 1`, `Cost 2`, `Purchase 2`, `Cost 3`, `Purchase 3`.
8. Add plant images with `Add Images`. GIFs are supported.
9. Add squirrel images with `Add Squirrel` if squirrels appear in your garden.
10. If needed, click `Set Priority` to raise important plant sprites above others.
In `Value` mode you set exact numbers. In `Tier List` mode you drag sprite images between `S` and `F`, and the app converts those tiers into click priority values.
11. Set `Sprinkler interval (s)` if you want timed sprinkler clicks. Time-based settings in the UI are all seconds.
12. Use `Scan Once` before `Start`. It reads the field and bits first, opens Fertilizer if needed, rechecks the 3 calibrated costs, and clicks the Fertilizer toggle again to close it.

## Notes on matching and upgrades

- Plant detection is shape-first. It ignores most color information on purpose.
- Squirrel matches and anything inside the Imports rectangle are treated as no-click areas.
- Higher priority sprites are clicked before lower priority sprites.
- A cycle now tries to click every visible plant match that survives filtering, instead of stopping at a per-cycle click cap.
- `Restore last session? -> Yes` reloads the last window selection, calibration, plant templates, squirrel templates, priorities, and sprinkler region.
- `Restore last session? -> No` clears the saved session and starts fresh.
- Bits and Fertilizer costs are compared by OCR text plus the color of the bits icon beside the amount. The current build uses a Paddle-based RapidOCR `en_PP-OCRv5` recognizer first, then Apple Vision OCR on macOS, and keeps Tesseract as the final fallback.
- Fertilizer buying is top-to-bottom priority among the currently affordable rows.
- Fertilizer costs are only trusted when the same row parses consistently across two quick reads, which reduces bad buys from OCR noise.
- The bot verifies whether Fertilizer is open or closed using the 3 calibrated cost boxes and will click the Fertilizer toggle again to exit before returning to harvesting.
- The sprinkler interval is disabled at `0` and enabled at any positive number of seconds.
- Move the mouse to a screen corner to trigger PyAutoGUI fail-safe if you need an immediate stop.

## macOS permissions

This app needs both:

- `System Settings -> Privacy & Security -> Screen Recording`
- `System Settings -> Privacy & Security -> Accessibility`

Without those permissions, capture or clicking will fail.

If the window picker shows blank previews for other apps but not for GamingIdleon itself, Screen Recording is still blocked by macOS. The current build now detects that case, offers to open the correct settings page, and tells you to relaunch after enabling it.

## Project layout

- `gaming_idleon/app.py`: polished desktop UI, window picker, calibration flow
- `gaming_idleon/windowing.py`: Quartz window listing and window capture
- `gaming_idleon/matcher.py`: grayscale edge-based plant matching
- `gaming_idleon/economy.py`: bits/fertilizer OCR and color-tier parsing
- `gaming_idleon/clicker.py`: harvesting loop and Fertilizer automation
- `tests/`: synthetic unit tests
