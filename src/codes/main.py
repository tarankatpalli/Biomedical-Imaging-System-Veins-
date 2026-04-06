import cv2
import numpy as np
from scipy import signal
from scipy.fftpack import fftfreq
from datetime import datetime
import os
import time
import subprocess
import signal as py_signal
import torch
import threading

import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

from train.model import VeinCNN

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
BUFFER_SECONDS = 10
BUFFER_SIZE = int(CAM_FPS * BUFFER_SECONDS)

MIN_BPM = 42
MAX_BPM = 180

SUMMARY_INTERVAL = 60
BOTTOM_PANEL_HEIGHT = 120

ENABLE_LOGGING = True
ENABLE_PLOTTING = True
LOG_INTERVAL = 1.0

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "rppg_log.csv")

CNN_WEIGHTS = "/home/taran/vein-t/weights/cnn_veins.pth"
CNN_IMG_SIZE = 256
CNN_THRESHOLD = 0.5

roi_selected = False
roi_coords = (0, 0, 0, 0)
dragging = False

signal_buffer = []
final_filtered_signal = None
last_bpm = None

latest_summary_text = "System initializing..."
last_summary_time = 0

log_buffer = []
last_log_time = 0

plot_buffer = deque(maxlen=BUFFER_SIZE)

LIBCAM_FIFO = "/tmp/libcam_pipe.mjpeg"
libcam_process = None

if ENABLE_LOGGING and not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["timestamp", "bpm", "roi_active"]).to_csv(
        CSV_PATH, index=False
    )

cnn_device = torch.device("cpu")
cnn_model = VeinCNN().to(cnn_device)
cnn_model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=cnn_device))
cnn_model.eval()
print("[CNN] Model loaded")


def on_mouse(event, x, y, flags, param):
    global roi_selected, roi_coords, dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        roi_selected = False
        roi_coords = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        x1, y1, _, _ = roi_coords
        roi_coords = (x1, y1, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        x1, y1, x2, y2 = roi_coords
        roi_coords = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
        roi_selected = abs(x2-x1) > 5 and abs(y2-y1) > 5

def process_signal(sig_buf, fps):
    if len(sig_buf) < 3:
        return None, None
    detr = signal.detrend(sig_buf)
    std = np.std(detr)
    if std == 0:
        return None, None
    norm = detr / std

    min_f = MIN_BPM / 60
    max_f = MAX_BPM / 60
    nyq = 0.5 * fps

    b, a = signal.butter(
        2, [min_f/nyq, max_f/nyq], btype="band"
    )
    filtered = signal.filtfilt(b, a, norm)

    freqs = fftfreq(len(filtered), 1/fps)
    fft_vals = np.abs(np.fft.fft(filtered))
    idx = np.where((freqs > min_f) & (freqs < max_f))
    if len(idx[0]) == 0:
        return None, filtered

    peak = idx[0][np.argmax(fft_vals[idx])]
    return freqs[peak]*60, filtered

def vein_process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.5, (8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.medianBlur(enhanced, 5)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 21, 9
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    cnn_in = cv2.resize(gray, (CNN_IMG_SIZE, CNN_IMG_SIZE)) / 255.0
    x = torch.tensor(cnn_in).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = torch.sigmoid(cnn_model(x))

    mask = (pred.squeeze().numpy() > CNN_THRESHOLD).astype(np.uint8)
    mask = cv2.resize(mask, gray.shape[::-1])

    traced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        if cv2.arcLength(cnt, True) > 50:
            temp = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(temp, [cnt], -1, 255, -1)
            if np.sum(temp & (mask*255)) > 30:
                cv2.drawContours(traced, [cnt], -1, (0,255,0), 1)

    return traced

def slm_loop():
    global latest_summary_text, last_summary_time
    while True:
        if time.time() - last_summary_time >= SUMMARY_INTERVAL:
            bpm_txt = f"{int(last_bpm)} BPM" if last_bpm else "not available"
            roi_txt = "active" if roi_selected else "not selected"

            prompt = f"""
You are a system explanation assistant.

Rules:
- Do NOT give medical advice
- Do NOT interpret health meaning
- ONLY explain what the system is doing
- Max 4 sentences

System status:
- Pulse estimate: {bpm_txt}
- ROI: {roi_txt}
- Vein visualization: active
- Camera processing: normal

Explain what is happening.
"""
            try:
                res = subprocess.run(
                    ["ollama", "run", "phi3.5"],
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=20
                )
                latest_summary_text = res.stdout.strip()
            except:
                latest_summary_text = "System is processing visual and signal data."

            last_summary_time = time.time()
        time.sleep(1)

def start_libcamera_stream():
    global libcam_process
    if os.path.exists(LIBCAM_FIFO):
        os.remove(LIBCAM_FIFO)
    os.mkfifo(LIBCAM_FIFO)
    cmd = f"libcamera-vid -t 0 --codec mjpeg -o {LIBCAM_FIFO}"
    libcam_process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, preexec_fn=os.setsid
    )
    time.sleep(0.5)
    return cv2.VideoCapture(LIBCAM_FIFO)

def stop_libcamera_stream():
    if libcam_process:
        os.killpg(os.getpgid(libcam_process.pid), py_signal.SIGTERM)
    if os.path.exists(LIBCAM_FIFO):
        os.remove(LIBCAM_FIFO)

def main():
    global last_bpm, final_filtered_signal, last_log_time

    threading.Thread(target=slm_loop, daemon=True).start()

    if ENABLE_PLOTTING:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,3))
        line, = ax.plot([], [])
        ax.set_title("Filtered rPPG Signal")

    cap = start_libcamera_stream()
    cv2.namedWindow("rPPG + Vein Detection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("rPPG + Vein Detection", on_mouse)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            vein = vein_process_frame(frame)
            display = frame.copy()

            if roi_selected:
                x1,y1,x2,y2 = roi_coords
                cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
                roi = display[y1:y2, x1:x2]
                if roi.size:
                    signal_buffer.append(np.mean(roi[:,:,1]))
                    if len(signal_buffer) > BUFFER_SIZE:
                        signal_buffer.pop(0)
                    bpm, filtered = process_signal(
                        np.array(signal_buffer), CAM_FPS
                    )
                    if bpm:
                        last_bpm = bpm
                        final_filtered_signal = filtered
                        plot_buffer.clear()
                        plot_buffer.extend(filtered.tolist())

            if last_bpm:
                cv2.putText(
                    display, f"HR: {int(last_bpm)} BPM",
                    (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2
                )

            if ENABLE_LOGGING and time.time() - last_log_time >= LOG_INTERVAL:
                log_buffer.append({
                    "timestamp": datetime.now().isoformat(),
                    "bpm": float(last_bpm) if last_bpm else None,
                    "roi_active": roi_selected
                })
                last_log_time = time.time()
                if len(log_buffer) >= 10:
                    pd.DataFrame(log_buffer).to_csv(
                        CSV_PATH, mode="a",
                        header=False, index=False
                    )
                    log_buffer.clear()

            if ENABLE_PLOTTING and len(plot_buffer) > 10:
                line.set_data(range(len(plot_buffer)), plot_buffer)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)

            sep = np.ones((CAM_HEIGHT,6,3), np.uint8)*50
            combined = np.hstack([display, sep, vein])

            bottom = np.zeros(
                (BOTTOM_PANEL_HEIGHT, combined.shape[1], 3),
                np.uint8
            )
            y = 30
            for line_txt in latest_summary_text.split("\n"):
                cv2.putText(
                    bottom, line_txt.strip(),
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200,200,200), 1
                )
                y += 25

            cv2.imshow(
                "rPPG + Vein Detection",
                np.vstack([combined, bottom])
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        stop_libcamera_stream()
        if ENABLE_LOGGING and log_buffer:
            pd.DataFrame(log_buffer).to_csv(
                CSV_PATH, mode="a",
                header=False, index=False
            )
        if ENABLE_PLOTTING:
            plt.ioff()
            plt.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
