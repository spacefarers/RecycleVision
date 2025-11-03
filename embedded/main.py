# ===== Combined: Servo Control + Model Inference with Decision Logic =====
# Target: K230 CanMV-style runtime (nncase_runtime, media.sensor/display, machine.PWM)
#
# Behavior:
# - The model outputs three classes in this order: [empty, recyclable, trash].
# - If the top class is 'empty': do nothing (no servo action).
# - If P(empty) stays below 0.5 for at least 1.0 second continuously,
#   choose max between P(recyclable) and P(trash) and actuate servo to deposit.
# - After an actuation, the system re-arms only after P(empty) has recovered
#   to >= 0.5 for 0.5s to avoid repeated triggers.
#
# Notes:
# - This file combines the structure of your test2.py loop with the servo setup from test.py.
# - It includes a minimal, robust state machine for timing-based decisions.
# - You can adjust the servo angles for your hardware.

import gc, time

# ----- Servo setup (from test.py) -----
try:
    from machine import FPIOA, PWM
    SERVO_IO = 60
    PWM_CH = 0
    SERVO_FREQ = 50
    ANG_CENTER = 150
    ANG_RECYCLE = 50
    ANG_TRASH = 250

    fpioa = FPIOA()
    fpioa.set_function(SERVO_IO, fpioa.PWM0)
    pwm0 = PWM(PWM_CH, SERVO_FREQ, 7.5, enable=True)

    def servo_write_angle(deg):
        pwm0.duty(5 + (deg/180)*5)
    def servo_center():
        servo_write_angle(ANG_CENTER)
    def servo_to_recycle():
        servo_write_angle(ANG_RECYCLE)
    def servo_to_trash():
        servo_write_angle(ANG_TRASH)
    SERVO_OK = True
except Exception:
    SERVO_OK = False
    def servo_write_angle(_): pass
    def servo_center(): pass
    def servo_to_recycle(): pass
    def servo_to_trash(): pass

servo_center()

# ----- Model / Camera setup (based on test2.py) -----
try:
    import nncase_runtime as nn
    import ulab.numpy as np
    from media.sensor import Sensor
    from media.media import MediaManager
    from media.display import Display
except Exception as e:
    raise RuntimeError("This script must run on the K230 CanMV runtime with nncase_runtime and media modules available.") from e

MODEL_PATH = "/sdcard/models/recyclevision_3cls.kmodel"
MODEL_IN_NCHW = (1, 3, 320, 320)
LABELS = ["empty", "recyclable", "trash"]

CAM_ID = 0
FPS = 30
W, H = 640, 480
MEAN_SUB = [0, 0, 0]

def open_sensor(cam_id, w, h, fps):
    s = Sensor(id=cam_id, fps=fps)
    s.reset()
    s.set_framesize(width=w, height=h)
    try:
        s.set_pixformat(Sensor.RGB888)
    except:
        s.set_pixformat(Sensor.RGB565)
    return s

def rgb565_to_rgb888_numpy(buf_u8, Hh, Ww):
    px16 = buf_u8.reshape((Hh, Ww, 2))
    v = (px16[...,1].astype(np.uint16) << 8) | px16[...,0].astype(np.uint16)
    r5 = (v >> 11) & 0x1F
    g6 = (v >> 5) & 0x3F
    b5 = v & 0x1F
    r8 = ((r5 * 527 + 23) >> 6).astype(np.uint8)
    g8 = ((g6 * 259 + 33) >> 6).astype(np.uint8)
    b8 = ((b5 * 527 + 23) >> 6).astype(np.uint8)
    rgb = np.zeros((Hh, Ww, 3), dtype=np.uint8)
    rgb[...,0] = r8
    rgb[...,1] = g8
    rgb[...,2] = b8
    return rgb

ai2d = nn.ai2d()
kpu = nn.kpu()
kpu.load_model(MODEL_PATH)

kpu_input = nn.from_numpy(np.zeros(MODEL_IN_NCHW, dtype=np.uint8))
kpu.set_input_tensor(0, kpu_input)
ai2d_out = kpu.get_input_tensor(0)

ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
ai2d.set_pad_param(True, [0,0,0,0, 0,0,0,0], 0, MEAN_SUB)
ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
ai2d_builder = None

sensor = open_sensor(CAM_ID, W, H, FPS)

HAVE_DISPLAY = False
try:
    Display.init(Display.VIRT, width=W, height=H)
    HAVE_DISPLAY = True
except:
    HAVE_DISPLAY = False

def draw_text(img, text, x=8, y=8):
    try:
        w_est = max(60, 10 * len(text))
        h_est = 24
        img.draw_rectangle(x-3, y-3, w_est+6, h_est+6, (0,0,0), 1, True)
    except:
        pass
    try:
        img.draw_string(x, y, text, scale=2, color=(255,255,255))
    except:
        pass

EMPTY_THRESH = 0.5
BELOW_FOR_MS = 1000
REARM_MS = 500

below_start_ms = None
rearm_start_ms = None
armed = True
last_actuation_ms = 0

def decide_and_act(p_empty, p_recycle, p_trash, now_ms):
    global below_start_ms, rearm_start_ms, armed, last_actuation_ms

    if p_empty >= EMPTY_THRESH:
        if rearm_start_ms is None:
            rearm_start_ms = now_ms
        if not armed and (now_ms - rearm_start_ms) >= REARM_MS:
            armed = True
        below_start_ms = None
    else:
        rearm_start_ms = None
        if below_start_ms is None:
            below_start_ms = now_ms
        if armed and (now_ms - below_start_ms) >= BELOW_FOR_MS:
            if p_recycle >= p_trash:
                if SERVO_OK:
                    servo_to_recycle()
                    time.sleep(0.6)
                    servo_center()
                action = "recyclable"
            else:
                if SERVO_OK:
                    servo_to_trash()
                    time.sleep(0.6)
                    servo_center()
                action = "trash"
            last_actuation_ms = now_ms
            armed = False
            below_start_ms = None
            return action
    return None

print("Starting loop. Ctrl+C to stop.")

try:
    while True:
        t0 = time.ticks_ms()
        img = sensor.snapshot()
        try:
            rgb_hwc = img.to_numpy()
        except:
            buf = img.to_bytes()
            rgb_hwc = rgb565_to_rgb888_numpy(np.frombuffer(buf, dtype=np.uint8), H, W)

        nchw = np.transpose(rgb_hwc, (2,0,1))
        nchw = nchw.reshape((1, 3, H, W))

        if ai2d_builder is None:
            ai2d_builder = ai2d.build([1,3,H,W], MODEL_IN_NCHW)

        ai2d_input_tensor = nn.from_numpy(nchw)
        ai2d_builder.run(ai2d_input_tensor, ai2d_out)
        kpu.run(ai2d_out)
        out_tensor = kpu.get_output_tensor(0)
        out = out_tensor.to_numpy().reshape(-1)

        def softmax(x):
            x = x - np.max(x)
            ex = np.exp(x)
            return ex / np.sum(ex)
        probs = softmax(out)

        p_empty, p_recycle, p_trash = float(probs[0]), float(probs[1]), float(probs[2])
        top_idx = int(np.argmax(probs))
        top_label = LABELS[top_idx]
        conf = float(probs[top_idx])

        action = decide_and_act(p_empty, p_recycle, p_trash, t0)

        if HAVE_DISPLAY:
            txt = "Top: %s  conf: %.2f  empty: %.2f  rec: %.2f  trash: %.2f" % (
                top_label, conf, p_empty, p_recycle, p_trash
            )
            draw_text(img, txt, 6, 6)
            Display.show_image(img)

        print("pred:", top_label, "conf: %.3f" % conf,
              "| pe=%.2f pr=%.2f pt=%.2f" % (p_empty, p_recycle, p_trash),
              "| action:", action if action else "none")

        del ai2d_input_tensor, nchw, rgb_hwc, img, out_tensor
        gc.collect()
        dt = time.ticks_ms() - t0
        if dt > 0:
            print("FPS:", 1000.0 / dt)

except KeyboardInterrupt:
    pass
finally:
    try: sensor.stop()
    except: pass
    try: MediaManager.deinit()
    except: pass
    try:
        if HAVE_DISPLAY:
            Display.deinit()
    except: pass
    for name in ("ai2d_out","ai2d_builder","kpu_input"):
        try: del globals()[name]
        except: pass
    try: del ai2d
    except: pass
    try: del kpu
    except: pass
    gc.collect()
    try: nn.shrink_memory_pool()
    except: pass
