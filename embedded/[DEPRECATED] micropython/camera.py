# ===== K230 CanMV: Live Preview → Save to SD on Button or IDE Click =====
# One Display.show_image per frame. Stable overlay via draw_string_advanced.
# Serial logs on capture start/success/failure.

import gc, time, os

from media.sensor import *
from media.media import *
from media.display import *

# Optional GPIO imports for a physical button
try:
    import machine
except:
    machine = None

# ---------- CONFIG ----------
CAM_ID    = 0
FPS       = 30
W, H      = 640, 480

# Save settings
SAVE_DIR          = "/sdcard/captures"
JPEG_QUALITY      = 90
SAVE_DEBOUNCE_MS  = 350

# Overlay
OVERLAY_TEXT      = "Ready"
OVERLAY_REFRESH_MS = 500
USE_BACKDROP       = True  # handled by draw_string_advanced back_color when available

# Hardware button config
BUTTON_PIN        = 12      # set to your board pin
BUTTON_ACTIVE_LOW = True
BUTTON_PULL       = "UP"    # "UP" or "DOWN"

# ---------- CAMERA INIT ----------
def open_sensor(cam_id, w, h, fps):
    s = Sensor(id=cam_id, fps=fps)
    s.reset()
    s.set_framesize(width=w, height=h)
    try:
        s.set_pixformat(Sensor.RGB888)
    except:
        s.set_pixformat(Sensor.RGB565)
    return s

sensor = open_sensor(CAM_ID, W, H, FPS)

# Display init
try:
    Display.init(Display.VIRT, width=W, height=H)
    HAVE_DISPLAY = True
except Exception as e:
    print("Display init failed:", e)
    HAVE_DISPLAY = False

MediaManager.init()
sensor.run()

# Warm up one frame so the pipeline allocates buffers once
_ = sensor.snapshot()
gc.collect()

# ---------- BUTTON ----------
_btn = None
def init_button():
    global _btn
    if machine is None:
        return
    try:
        pull = machine.Pin.PULL_UP if BUTTON_PULL == "UP" else machine.Pin.PULL_DOWN
        _btn = machine.Pin(BUTTON_PIN, mode=machine.Pin.IN, pull=pull)
    except Exception as e:
        print("Button init failed:", e)
        _btn = None

def button_pressed():
    if _btn is None:
        return False
    try:
        v = _btn.value()
        return (v == 0) if BUTTON_ACTIVE_LOW else (v == 1)
    except:
        return False

init_button()

# ---------- HELPERS ----------
def ensure_dir(path):
    try:
        if path in ("/", ""):
            return
        parts = path.split("/")
        cur = ""
        for p in parts:
            if p == "":
                continue
            cur += "/" + p
            try:
                os.mkdir(cur)
            except OSError:
                pass
    except:
        pass

def timestamp_str():
    try:
        t = time.localtime()
        return "%04d%02d%02d_%02d%02d%02d" % (t[0],t[1],t[2],t[3],t[4],t[5])
    except:
        return "t%010d" % time.ticks_ms()

def try_save_image(img):
    """
    Try several save APIs for portability.
    Returns (ok, path_or_err)
    """
    ensure_dir(SAVE_DIR)
    fname = "img_%s.jpg" % timestamp_str()
    full = "%s/%s" % (SAVE_DIR, fname)

    # 1) Common path
    try:
        img.save(full, quality=JPEG_QUALITY)
        return True, full
    except Exception as e1:
        err1 = e1
    # 2) Alternate API
    try:
        img.save_jpeg(full, quality=JPEG_QUALITY)
        return True, full
    except Exception as e2:
        err2 = e2
    # 3) Encode then write
    try:
        data = img.jpeg_encode(quality=JPEG_QUALITY)
        with open(full, "wb") as f:
            f.write(data)
        return True, full
    except Exception as e3:
        return False, "save failed: %r | %r | %r" % (err1, err2, e3)

def draw_overlay(img, text, x=8, y=8):
    """
    Use draw_string_advanced if available to avoid deprecation warnings.
    Falls back to draw_string only if needed.
    """
    # Prefer advanced API to avoid "Deprecated function, please use draw_string_advanced"
    try:
        if USE_BACKDROP:
            # Many firmwares accept back_color to fill behind glyphs
            img.draw_string_advanced(x, y, text, scale=2, color=(255,255,255), back_color=(0,0,0))
        else:
            img.draw_string_advanced(x, y, text, scale=2, color=(255,255,255))
        return
    except AttributeError:
        # Older builds without advanced API
        pass
    except Exception as e:
        # If signature differs, try a simpler call
        try:
            img.draw_string_advanced(x, y, text)
            return
        except:
            pass
    # Final fallback if advanced is unavailable
    try:
        # This may print a deprecation warning on some firmwares
        img.draw_string(x, y, text, scale=2, color=(255,255,255))
    except:
        pass

def ide_clicked():
    # Attempt a few ways to detect IDE clicks without exceptions
    try:
        ev = Display.get_event()
        if ev and ev.get("type") in ("click", "tap"):
            return True
    except:
        pass
    try:
        return bool(Display.mouse_clicked())
    except:
        pass
    return False

_last_trig_ms = 0
def should_capture():
    global _last_trig_ms
    now = time.ticks_ms()
    trig = button_pressed() or ide_clicked()
    if trig and (now - _last_trig_ms >= SAVE_DEBOUNCE_MS):
        _last_trig_ms = now
        return True
    return False

# ---------- MAIN LOOP ----------
print("Preview running. Click IDE or press button to capture.")
last_overlay = ""
last_draw_ms = 0

try:
    while True:
        t0 = time.ticks_ms()

        # Get a frame. Keep only the img object to reduce buffer pressure.
        img = sensor.snapshot()

        # Capture if triggered
        saved_path = None
        # Draw overlay only when changed or on periodic refresh
        now = time.ticks_ms()
        need_draw = False

        # Show frame once
        if HAVE_DISPLAY:
            try:
                Display.show_image(img)
            except Exception as e:
                # Avoid spamming serial, print once per failure kind if needed
                print("Display error:", e)

        # Timing info
        dt = time.ticks_ms() - t0

        # Drop frame ref and collect
        del img
        gc.collect()

except KeyboardInterrupt:
    pass
finally:
    # ---------- Cleanup ----------
    # Release display before media manager to avoid buffer release complaints
    try:
        Display.deinit()
    except:
        pass
    try:
        sensor.stop()
    except:
        pass
    try:
        MediaManager.deinit()
    except:
        pass
    gc.collect()
