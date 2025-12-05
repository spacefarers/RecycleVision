# ===== K230 CanMV: Camera → (HWC→CHW) → AI2D(NCHW) → KPU → IDE Preview + Stable Overlay =====
# - Fixes "flashing" overlay: only one Display.show_image per frame (after drawing text)
# - Debounces overlay updates to reduce redraw jitter
# - Optional backdrop behind text for readability
# ----------------------------------------------------------------------------------------------

import nncase_runtime as nn
import ulab.numpy as np
import gc, time

from media.sensor import *
from media.media import *
from media.display import *
from machine import FPIOA, PWM
import time

# 1) Route PWM0 to IO60 explicitly (IOMUX)
fpioa = FPIOA()
fpioa.set_function(60, fpioa.PWM0)   # IO60 ← PWM0

# 2) Start PWM channel 0 at 50 Hz and center duty for servo
#    NOTE: constructor requires ALL 4 positional args on this firmware.
pwm0 = PWM(0, 50, 7.5, enable=True)         # (channel, freq Hz, duty %, enable)


# Optional: helper for angles (0–180° ≈ 5–10% duty)
def write_angle(deg):
    pwm0.duty(5 + (deg/180)*5)


# ---------- CONFIG ----------
MODEL_PATH      = "/sdcard/recyclevision.kmodel"
MODEL_IN_NCHW   = (1, 3, 320, 320)
CAM_ID          = 0
FPS             = 30
W, H            = 640, 480          # bump later if stable
MEAN_SUB        = [104, 117, 123]   # use [0,0,0] if no mean
LABELS = ["recyclable","trash","empty"]

# Overlay behavior
OVERLAY_EVERY_MS = 500               # force refresh at least every 500 ms
CONF_DELTA       = 0.03              # refresh if confidence moves by 3%
USE_BACKDROP     = True              # draw a filled rectangle behind text if supported

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

# Display: init once
try:
    Display.init(Display.VIRT, width=W, height=H)
    HAVE_DISPLAY = True
except Exception as e:
    print("Display init failed:", e)
    HAVE_DISPLAY = False

MediaManager.init()
sensor.run()

# quick probe
probe = sensor.snapshot()
del probe
gc.collect()

# ---------- KPU / AI2D ----------
kpu = nn.kpu()
ai2d = nn.ai2d()
kpu.load_kmodel(MODEL_PATH)

print("inputs info:")
for i in range(kpu.inputs_size()):
    print(kpu.inputs_desc(i))
print("outputs info:")
for i in range(kpu.outputs_size()):
    print(kpu.outputs_desc(i))

kpu_input = nn.from_numpy(np.zeros(MODEL_IN_NCHW, dtype=np.uint8))
kpu.set_input_tensor(0, kpu_input)
ai2d_out = kpu.get_input_tensor(0)

ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
ai2d.set_pad_param(True, [0,0,0,0, 0,0,0,0], 0, MEAN_SUB)
ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
ai2d_builder = None  # build after first frame

# ---------- HELPERS ----------
def rgb565_to_rgb888_numpy(buf_u8, Hh, Ww):
    px16 = buf_u8.reshape((Hh, Ww, 2))
    v = (px16[...,1].astype(np.uint16) << 8) | px16[...,0].astype(np.uint16)
    r5 = (v >> 11) & 0x1F
    g6 = (v >> 5)  & 0x3F
    b5 =  v        & 0x1F
    r8 = ((r5 * 527 + 23) >> 6).astype(np.uint8)
    g8 = ((g6 * 259 + 33) >> 6).astype(np.uint8)
    b8 = ((b5 * 527 + 23) >> 6).astype(np.uint8)
    rgb = np.zeros((Hh, Ww, 3), dtype=np.uint8)
    rgb[...,0] = r8; rgb[...,1] = g8; rgb[...,2] = b8
    return rgb

def get_frame_rgb(sensor_obj):
    """
    Returns (HWC_rgb_uint8, img)
    """
    img = sensor_obj.snapshot()
    Hh, Ww = img.height(), img.width()
    raw = img.to_numpy_ref()
    if hasattr(raw, "shape"):
        try:
            flat = raw.reshape((-1,))
        except:
            flat = np.frombuffer(raw, dtype=np.uint8)
    else:
        flat = np.frombuffer(raw, dtype=np.uint8)
    n = flat.shape[0]
    if n == Hh * Ww * 3:
        rgb = flat.reshape((Hh, Ww, 3))         # RGB888
    elif n == Hh * Ww * 2:
        rgb = rgb565_to_rgb888_numpy(flat, Hh, Ww)  # RGB565 → RGB888
    else:
        rgb = flat.reshape((Hh, Ww, 3))         # best-effort
    return rgb, img

def hwc_to_nchw(rgb_hwc):
    Hh, Ww = rgb_hwc.shape[0], rgb_hwc.shape[1]
    out = np.zeros((1, 3, Hh, Ww), dtype=np.uint8)
    out[0, 0, :, :] = rgb_hwc[:, :, 0]
    out[0, 1, :, :] = rgb_hwc[:, :, 1]
    out[0, 2, :, :] = rgb_hwc[:, :, 2]
    return out

def softmax(x):
    m = float(np.max(x))
    e = np.exp(x - m)
    s = float(np.sum(e))
    return e / s if s != 0 else e

def overlay_text(img, text, x=8, y=8):
    """
    Draw text (and optional filled backdrop) on the image.
    Only called when the text actually changes or on periodic refresh.
    """
    drew = False
    # Backdrop first (if supported), to prevent ghosting and improve readability
    if USE_BACKDROP:
        try:
            # width estimate per char ~ 10 px at scale=2; adjust as needed
            w_est = max(60, 12 * len(text))
            h_est = 28
            # img.draw_rectangle(x, y, w, h, color, thickness, filled)
            img.draw_rectangle(x-4, y-4, w_est+8, h_est+8, (0,0,0), 1, True)
        except:
            pass
    # Now text
    try:
        img.draw_string(x, y, text, scale=2, color=(255,255,255))
        drew = True
    except:
        # Some firmwares don’t support draw_string; ignore overlay silently
        pass
    return drew

# ---------- MAIN LOOP ----------
print("Starting camera inference loop. Ctrl+C to stop.")

last_txt = ""
last_conf = -1.0
last_draw_ms = 0
count_success = 0
write_angle(150);

try:
    while True:
        t0 = time.ticks_ms()

        # Grab frame (keep 'img' alive until after ai2d run)
        rgb_hwc, img = get_frame_rgb(sensor)

        # Build AI2D once with actual camera dims
        if ai2d_builder is None:
            H0, W0 = rgb_hwc.shape[0], rgb_hwc.shape[1]
            ai2d_builder = ai2d.build([1, 3, H0, W0], MODEL_IN_NCHW)

        # HWC→NCHW, preprocess, run model
        nchw = hwc_to_nchw(rgb_hwc)
        ai2d_input_tensor = nn.from_numpy(nchw)
        ai2d_builder.run(ai2d_input_tensor, ai2d_out)
        kpu.run()

        # Output → softmax → top-1
        out = kpu.get_output_tensor(0).to_numpy()
        flat = out.flatten()
        probs = softmax(flat)
        try:
            top = int(np.argmax(probs))
        except:
            top, bestv = 0, float(probs[0])
            for i in range(1, len(probs)):
                if float(probs[i]) > bestv:
                    bestv = float(probs[i]); top = i
        conf = float(probs[top])
        klass = LABELS[top] if 0 <= top < len(LABELS) else str(top)

        # Compose text
        txt = f"{klass}  {conf:.2f}"

        # Debounce overlay updates: only redraw when necessary
        tnow = time.ticks_ms()
        need_draw = False
        if txt != last_txt:
            need_draw = True
        elif abs(conf - last_conf) >= CONF_DELTA:
            need_draw = True
        elif (tnow - last_draw_ms) >= OVERLAY_EVERY_MS:
            need_draw = True

        if need_draw:
            overlay_text(img, txt, x=8, y=8)
            last_txt = txt
            last_conf = conf
            last_draw_ms = tnow
        # IMPORTANT: show the frame exactly once per loop (AFTER overlay)
        if HAVE_DISPLAY:
            try:
                Display.show_image(img)
            except:
                pass

        # Console feedback every frame
        print("pred:", top, f"({klass})", "conf:{:.3f}".format(conf))
        if top == 0:
            count_success += 1
            if count_success == 7:
                count_success = 0
                # Sweep 150 middle +- 100
                write_angle(150);   time.sleep(1)
                write_angle(50); time.sleep(1)
                write_angle(150);   time.sleep(1)
        else:
            count_success = 0


        # FPS
        dt = time.ticks_ms() - t0
        if dt > 0:
            print("FPS:", 1000.0 / dt)

        # Cleanup per-iter temps
        del ai2d_input_tensor, nchw, rgb_hwc, img
        gc.collect()

except KeyboardInterrupt:
    pass
finally:
    # ---------- Cleanup ----------
    try: sensor.stop()
    except: pass
    try: MediaManager.deinit()
    except: pass
    try: Display.deinit()
    except: pass

    for name in ("ai2d_out","ai2d_builder","kpu_input"):
        try: del globals()[name]
        except: pass
    try: del ai2d
    except: pass
    try: del kpu
    except: pass

    gc.collect()
    nn.shrink_memory_pool()
