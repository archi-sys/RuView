"""
AuraTrack Firebase Bridge
=========================
Combines:
  1. ESP32 RSSI over UDP → real vital signs via bandpass filtering
  2. Webcam + MediaPipe → real 17-keypoint skeleton
  3. Sends fused data to:
     - Rust server WebSocket → Observatory 3D view (local)
     - Firebase Realtime Database → cloud storage + remote access

Firebase structure:
  auratrack/
    live/          ← latest frame (overwritten every 100ms)
      presence
      motion_level
      persons_count
      breathing_bpm
      heart_rate_bpm
      confidence
      rssi
      timestamp
    history/       ← last 100 frames (rolling)
      {timestamp}/
        ...
    alerts/        ← triggered alerts
      {id}/
        type
        message
        timestamp

Usage:
  python auratrack_bridge.py
"""

import asyncio
import json
import socket
import time
import threading
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
import cv2
import mediapipe as mp
import websockets
import requests

# ── Firebase Configuration ────────────────────────────────────────────────────
FIREBASE_CONFIG = {
    "apiKey":            "AIzaSy8pUZnTVMTc6toRnMjGzHywgvFHnotVd7U",
    "authDomain":        "auratrack-b29d0.firebaseapp.com",
    "databaseURL":       "https://auratrack-b29d0-default-rtdb.firebaseio.com",
    "projectId":         "auratrack-b29d0",
    "storageBucket":     "auratrack-b29d0.firebasestorage.app",
    "messagingSenderId": "266782084638",
    "appId":             "1:266782084638:web:55545de68fc0ed9350e80f",
}

FIREBASE_DB_URL = FIREBASE_CONFIG["databaseURL"]

# ── Configuration ─────────────────────────────────────────────────────────────
UDP_PORT         = 5006
SERVER_WS_URL    = "ws://localhost:3001/ws/sensing"
SAMPLE_RATE      = 10.0
BUFFER_SIZE      = 200
CAMERA_INDEX     = 0
SHOW_CAMERA      = False

# Firebase push intervals
FIREBASE_LIVE_INTERVAL    = 2.0   # push live data every 2 seconds
FIREBASE_HISTORY_INTERVAL = 10.0  # push to history every 10 seconds

# Alert thresholds
ALERT_NO_PRESENCE_MINUTES = 30    # alert if no presence for 30 mins
ALERT_HIGH_HEART_RATE     = 110   # alert if HR > 110 BPM
ALERT_LOW_HEART_RATE      = 45    # alert if HR < 45 BPM

# ── Bandpass filter ───────────────────────────────────────────────────────────
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut/nyq, min(highcut/nyq, 0.99)], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs):
    if len(data) < 20:
        return np.zeros(len(data))
    b, a = butter_bandpass(lowcut, highcut, fs)
    try:
        return filtfilt(b, a, data)
    except:
        return np.zeros(len(data))

def estimate_bpm(signal, fs, low_hz, high_hz):
    if len(signal) < 20:
        return 0.0, 0.0
    fft = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0, 0.0
    band_power = fft[mask]
    band_freqs = freqs[mask]
    peak_idx = np.argmax(band_power)
    peak_freq = band_freqs[peak_idx]
    confidence = min(band_power[peak_idx] / (np.mean(fft) * 3.0 + 1e-9), 1.0)
    return float(peak_freq * 60.0), float(confidence)

# ── MediaPipe keypoint mapping ────────────────────────────────────────────────
KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
MP_TO_COCO = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]

# ── Shared state ──────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.rssi_buffer     = deque(maxlen=BUFFER_SIZE)
        self.rssi_timestamps = deque(maxlen=BUFFER_SIZE)
        self.last_rssi       = -60.0
        self.last_udp_time   = 0.0
        self.breathing_bpm   = 0.0
        self.heart_rate_bpm  = 0.0
        self.breathing_conf  = 0.0
        self.heart_conf      = 0.0
        self.motion_score    = 0.0
        self.presence        = False
        self.signal_quality  = 0.0
        self.keypoints       = []
        self.pose_confidence = 0.0
        self.camera_active   = False
        self.tick            = 0
        self.last_presence_time = time.time()
        self.alert_sent      = {}

state = SharedState()

# ── Firebase REST API helpers ─────────────────────────────────────────────────
def firebase_put(path, data):
    """Write data to Firebase path (overwrites)."""
    url = f"{FIREBASE_DB_URL}/{path}.json"
    try:
        r = requests.put(url, json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"[Firebase] PUT error: {e}")
        return False

def firebase_post(path, data):
    """Push data to Firebase path (creates unique key)."""
    url = f"{FIREBASE_DB_URL}/{path}.json"
    try:
        r = requests.post(url, json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"[Firebase] POST error: {e}")
        return False

def firebase_delete_old_history():
    """Keep only last 100 history entries."""
    url = f"{FIREBASE_DB_URL}/auratrack/history.json?orderBy=\"timestamp\"&limitToFirst=1"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 100:
                # Delete oldest entry
                oldest_key = list(data.keys())[0]
                requests.delete(
                    f"{FIREBASE_DB_URL}/auratrack/history/{oldest_key}.json",
                    timeout=5
                )
    except:
        pass

# ── Alert system ──────────────────────────────────────────────────────────────
def check_and_send_alerts():
    with state.lock:
        presence      = state.presence
        heart_rate    = state.heart_rate_bpm
        last_pres     = state.last_presence_time
        alert_sent    = state.alert_sent

    now = time.time()

    # Alert: no presence for too long
    if presence:
        with state.lock:
            state.last_presence_time = now
        # Clear no-presence alert
        with state.lock:
            state.alert_sent.pop("no_presence", None)
    else:
        minutes_absent = (now - last_pres) / 60.0
        if minutes_absent > ALERT_NO_PRESENCE_MINUTES:
            alert_key = "no_presence"
            if alert_key not in alert_sent:
                alert = {
                    "type": "no_presence",
                    "message": f"No presence detected for {int(minutes_absent)} minutes",
                    "timestamp": now,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                firebase_post("auratrack/alerts", alert)
                print(f"[Alert] {alert['message']}")
                with state.lock:
                    state.alert_sent[alert_key] = now

    # Alert: abnormal heart rate
    if heart_rate > 0:
        if heart_rate > ALERT_HIGH_HEART_RATE:
            alert_key = "high_hr"
            if alert_key not in alert_sent:
                alert = {
                    "type": "high_heart_rate",
                    "message": f"High heart rate detected: {heart_rate:.0f} BPM",
                    "timestamp": now,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "value": heart_rate,
                }
                firebase_post("auratrack/alerts", alert)
                print(f"[Alert] {alert['message']}")
                with state.lock:
                    state.alert_sent[alert_key] = now
        else:
            with state.lock:
                state.alert_sent.pop("high_hr", None)

        if heart_rate < ALERT_LOW_HEART_RATE:
            alert_key = "low_hr"
            if alert_key not in alert_sent:
                alert = {
                    "type": "low_heart_rate",
                    "message": f"Low heart rate detected: {heart_rate:.0f} BPM",
                    "timestamp": now,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "value": heart_rate,
                }
                firebase_post("auratrack/alerts", alert)
                print(f"[Alert] {alert['message']}")
                with state.lock:
                    state.alert_sent[alert_key] = now
        else:
            with state.lock:
                state.alert_sent.pop("low_hr", None)

# ── Firebase Push Thread ──────────────────────────────────────────────────────
def firebase_thread():
    """Continuously push data to Firebase."""
    print("[Firebase] Starting Firebase sync thread")

    # Write initial status
    firebase_put("auratrack/status", {
        "online": True,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0",
        "device": "ESP32 + Webcam",
    })
    print("[Firebase] Connected! Data will appear at:")
    print(f"  {FIREBASE_DB_URL}/auratrack")

    last_live_push    = 0.0
    last_history_push = 0.0

    while True:
        time.sleep(0.5)
        now = time.time()

        with state.lock:
            br_bpm    = state.breathing_bpm
            hr_bpm    = state.heart_rate_bpm
            br_conf   = state.breathing_conf
            hr_conf   = state.heart_conf
            motion    = state.motion_score
            presence  = state.presence
            sig_qual  = state.signal_quality
            rssi      = state.last_rssi
            pose_conf = state.pose_confidence
            tick      = state.tick
            n_persons = len(state.keypoints) > 0 and pose_conf > 0.3

        motion_level = (
            "active"         if motion > 0.6 else
            "present_moving" if motion > 0.2 else
            "present_still"  if presence    else
            "absent"
        )

        # ── Push live data ──────────────────────────────────────────────
        if now - last_live_push >= FIREBASE_LIVE_INTERVAL:
            last_live_push = now
            live_data = {
                "presence":        presence,
                "motion_level":    motion_level,
                "persons_count":   1 if n_persons else 0,
                "breathing_bpm":   round(br_bpm, 1),
                "heart_rate_bpm":  round(hr_bpm, 1),
                "breathing_conf":  round(br_conf, 2),
                "heart_conf":      round(hr_conf, 2),
                "motion_score":    round(motion, 3),
                "signal_quality":  round(sig_qual, 2),
                "rssi_dbm":        round(rssi, 1),
                "confidence":      round(pose_conf, 2),
                "tick":            tick,
                "timestamp":       now,
                "timestamp_str":   time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if firebase_put("auratrack/live", live_data):
                print(f"[Firebase] Live → presence={presence} "
                      f"BR={br_bpm:.1f} HR={hr_bpm:.1f} "
                      f"RSSI={rssi:.0f}")

        # ── Push history ────────────────────────────────────────────────
        if now - last_history_push >= FIREBASE_HISTORY_INTERVAL:
            last_history_push = now
            history_entry = {
                "presence":       presence,
                "motion_level":   motion_level,
                "breathing_bpm":  round(br_bpm, 1),
                "heart_rate_bpm": round(hr_bpm, 1),
                "motion_score":   round(motion, 3),
                "rssi_dbm":       round(rssi, 1),
                "timestamp":      now,
                "timestamp_str":  time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            firebase_post("auratrack/history", history_entry)
            firebase_delete_old_history()

        # ── Check alerts ────────────────────────────────────────────────
        check_and_send_alerts()

# ── UDP Listener ──────────────────────────────────────────────────────────────
def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)
    print(f"[UDP] Listening on port {UDP_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            now  = time.time()
            rssi = None

            if len(data) >= 20:
                magic = int.from_bytes(data[0:4], 'little')
                if magic == 0xC5110001:
                    rssi = float(data[14] if data[14] < 128 else data[14] - 256)

            if rssi is None:
                try:
                    j    = json.loads(data.decode('utf-8'))
                    rssi = float(j.get('mean_rssi') or j.get('rssi') or -60)
                except:
                    pass

            if rssi is not None:
                with state.lock:
                    state.rssi_buffer.append(rssi)
                    state.rssi_timestamps.append(now)
                    state.last_rssi    = rssi
                    state.last_udp_time = now

        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP] Error: {e}")

# ── Vital Signs Processor ─────────────────────────────────────────────────────
def compute_vitals():
    print("[Vitals] Processor started")
    while True:
        time.sleep(0.5)
        with state.lock:
            if len(state.rssi_buffer) < 20:
                continue
            rssi_array = np.array(list(state.rssi_buffer), dtype=float)
            times      = np.array(list(state.rssi_timestamps))

        actual_fs = float(np.clip(
            (len(times)-1) / (times[-1]-times[0]+1e-9), 1.0, 20.0
        )) if len(times) > 1 else SAMPLE_RATE

        rssi_d   = rssi_array - np.mean(rssi_array)
        variance = float(np.var(rssi_d))
        motion   = float(np.clip(variance / 5.0, 0.0, 1.0))
        presence = bool(np.mean(rssi_array) > -80 and variance > 0.1)
        sig_q    = float(np.clip((np.mean(rssi_array) + 100) / 60.0, 0.0, 1.0))

        br_sig  = bandpass_filter(rssi_d, 0.1, 0.5, actual_fs)
        hr_sig  = bandpass_filter(rssi_d, 0.8, 2.0, actual_fs)
        br_bpm, br_conf = estimate_bpm(br_sig, actual_fs, 0.1, 0.5)
        hr_bpm, hr_conf = estimate_bpm(hr_sig, actual_fs, 0.8, 2.0)

        br_bpm = float(np.clip(br_bpm, 6.0, 30.0))  if br_conf > 0.1 else 0.0
        hr_bpm = float(np.clip(hr_bpm, 40.0, 120.0)) if hr_conf > 0.1 else 0.0

        with state.lock:
            state.breathing_bpm  = br_bpm
            state.heart_rate_bpm = hr_bpm
            state.breathing_conf = br_conf
            state.heart_conf     = hr_conf
            state.motion_score   = motion
            state.presence       = presence
            state.signal_quality = sig_q

# ── Camera Thread ─────────────────────────────────────────────────────────────
def camera_thread():
    mp_pose = mp.solutions.pose
    cap     = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[Camera] Failed to open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[Camera] Webcam opened (index {CAMERA_INDEX})")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.033)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)

            keypoints = []
            pose_conf = 0.0

            if results.pose_landmarks:
                landmarks   = results.pose_landmarks.landmark
                confidences = []
                for coco_idx, mp_idx in enumerate(MP_TO_COCO):
                    lm = landmarks[mp_idx]
                    keypoints.append({
                        "name":       KEYPOINT_NAMES[coco_idx],
                        "x":          float(lm.x * 640),
                        "y":          float(lm.y * 480),
                        "z":          float(lm.z * 100),
                        "confidence": float(lm.visibility),
                    })
                    confidences.append(lm.visibility)
                pose_conf = float(np.mean(confidences))

            with state.lock:
                state.keypoints       = keypoints
                state.pose_confidence = pose_conf
                state.camera_active   = True

    cap.release()

# ── WebSocket Sender ──────────────────────────────────────────────────────────
async def websocket_sender():
    print(f"[WS] Connecting to {SERVER_WS_URL}")
    while True:
        try:
            async with websockets.connect(SERVER_WS_URL, ping_interval=20) as ws:
                print("[WS] Connected!")
                while True:
                    await asyncio.sleep(0.1)
                    with state.lock:
                        br_bpm    = state.breathing_bpm
                        hr_bpm    = state.heart_rate_bpm
                        br_conf   = state.breathing_conf
                        hr_conf   = state.heart_conf
                        motion    = state.motion_score
                        presence  = state.presence
                        sig_qual  = state.signal_quality
                        rssi      = state.last_rssi
                        keypoints = list(state.keypoints)
                        pose_conf = state.pose_confidence
                        state.tick += 1
                        tick = state.tick

                    motion_level = (
                        "active"         if motion > 0.6 else
                        "present_moving" if motion > 0.2 else
                        "present_still"  if presence    else
                        "absent"
                    )
                    confidence = float(np.clip(
                        0.3 + sig_qual*0.3 + pose_conf*0.4, 0.0, 1.0
                    ))

                    persons = []
                    if keypoints and pose_conf > 0.3:
                        persons = [{
                            "id": 1,
                            "confidence": float(pose_conf),
                            "keypoints":  keypoints,
                            "bbox": {
                                "x":      float(min(k["x"] for k in keypoints)-10),
                                "y":      float(min(k["y"] for k in keypoints)-10),
                                "width":  float(max(k["x"] for k in keypoints)-min(k["x"] for k in keypoints)+20),
                                "height": float(max(k["y"] for k in keypoints)-min(k["y"] for k in keypoints)+20),
                            },
                            "zone":         "zone_1",
                            "position":     [0.0, 0.0, 0.0],
                            "motion_score": int(motion * 100),
                            "pose":         "standing",
                        }]

                    msg = {
                        "type":      "sensing_update",
                        "timestamp": time.time(),
                        "source":    "auratrack_bridge",
                        "tick":      tick,
                        "nodes": [{
                            "node_id":         1,
                            "rssi_dbm":        rssi,
                            "position":        [0.0, 0.0, 0.0],
                            "amplitude":       [abs(rssi)] * 56,
                            "subcarrier_count": 56,
                        }],
                        "features": {
                            "mean_rssi":           rssi,
                            "variance":            float(motion * 10),
                            "motion_band_power":   float(motion * 15),
                            "breathing_band_power": float(br_conf * 5),
                            "dominant_freq_hz":    float(br_bpm / 60.0),
                            "change_points":       int(motion * 20),
                            "spectral_power":      float(abs(rssi) * 2),
                        },
                        "classification": {
                            "motion_level": motion_level,
                            "presence":     presence,
                            "confidence":   confidence,
                        },
                        "vital_signs": {
                            "breathing_rate_bpm":   br_bpm  if br_bpm  > 0 else None,
                            "heart_rate_bpm":       hr_bpm  if hr_bpm  > 0 else None,
                            "breathing_confidence": br_conf,
                            "heartbeat_confidence": hr_conf,
                            "signal_quality":       sig_qual,
                        },
                        "signal_field": {
                            "grid_size": [20, 1, 20],
                            "values": [
                                float(np.clip(
                                    motion * np.exp(-((i//20-10)**2+(i%20-10)**2)/30.0),
                                    0, 1
                                )) for i in range(400)
                            ],
                        },
                        "persons":           persons,
                        "estimated_persons": len(persons),
                        "pose_source":       "mediapipe_camera",
                    }
                    await ws.send(json.dumps(msg))

        except Exception as e:
            print(f"[WS] Disconnected: {e}, retrying in 3s...")
            await asyncio.sleep(3)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AuraTrack Bridge v1.0")
    print("  ESP32 + Webcam + Firebase Cloud")
    print("=" * 60)
    print(f"  UDP port  : {UDP_PORT}")
    print(f"  WebSocket : {SERVER_WS_URL}")
    print(f"  Firebase  : {FIREBASE_DB_URL}")
    print(f"  Camera    : index {CAMERA_INDEX}")
    print("=" * 60)
    print()
    print("  Live dashboard:")
    print(f"  {FIREBASE_DB_URL}/auratrack/live.json")
    print()
    print("  NOTE: Change ESP32 SERVER_PORT to 5006!")
    print()

    # Start threads
    threading.Thread(target=udp_listener,  daemon=True).start()
    threading.Thread(target=compute_vitals, daemon=True).start()
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=firebase_thread, daemon=True).start()

    print("[Main] All threads started!")
    asyncio.run(websocket_sender())

if __name__ == "__main__":
    main()
