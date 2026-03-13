"""
AuraTrack Bridge v2.0 — with Privacy Mode
==========================================
Privacy mode ON  → sends 3-5 fake persons with random vitals
Privacy mode OFF → sends real ESP32 data as normal

Ports:
  5006 → ESP32 #1 (main sensing node)
  5007 → ESP32 #2 (privacy toggle node)
"""

import asyncio
import json
import socket
import time
import threading
import random
import math
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
import cv2
import mediapipe as mp
import websockets
import requests

# ── Firebase ───────────────────────────────────────────────────────────────
FIREBASE_DB_URL = "https://auratrack-b29d0-default-rtdb.firebaseio.com"

# ── Config ─────────────────────────────────────────────────────────────────
UDP_SENSING_PORT  = 5006
UDP_PRIVACY_PORT  = 5007
SERVER_WS_URL     = "ws://localhost:3001/ws/sensing"
SAMPLE_RATE       = 10.0
BUFFER_SIZE       = 200
CAMERA_INDEX      = 0

FIREBASE_LIVE_INTERVAL    = 2.0
FIREBASE_HISTORY_INTERVAL = 10.0

# ── Privacy fake data config ───────────────────────────────────────────────
FAKE_PERSON_COUNT_MIN = 3
FAKE_PERSON_COUNT_MAX = 5

FAKE_NAMES = [
    "Person A", "Person B", "Person C", "Person D", "Person E"
]

# Fake vitals ranges — realistic-looking but randomized
FAKE_BR_RANGE  = (10, 22)    # breathing BPM
FAKE_HR_RANGE  = (55, 105)   # heart rate BPM
FAKE_RSSI_RANGE = (-45, -80) # signal strength

# Fake keypoint zones — spread people across the room
FAKE_ZONES = [
    {"x_offset": 100, "y_offset": 50},
    {"x_offset": 300, "y_offset": 60},
    {"x_offset": 500, "y_offset": 45},
    {"x_offset": 200, "y_offset": 200},
    {"x_offset": 420, "y_offset": 190},
]

KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# ── Shared state ───────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock             = threading.Lock()
        self.rssi_buffer      = deque(maxlen=BUFFER_SIZE)
        self.rssi_timestamps  = deque(maxlen=BUFFER_SIZE)
        self.last_rssi        = -60.0
        self.last_udp_time    = 0.0
        self.breathing_bpm    = 0.0
        self.heart_rate_bpm   = 0.0
        self.breathing_conf   = 0.0
        self.heart_conf       = 0.0
        self.motion_score     = 0.0
        self.presence         = False
        self.signal_quality   = 0.0
        self.keypoints        = []
        self.pose_confidence  = 0.0
        self.camera_active    = False
        self.tick             = 0
        # Privacy mode
        self.privacy_mode     = False
        self.privacy_changed  = False
        self.fake_persons     = []
        self.fake_tick        = 0

state = SharedState()

# ── Bandpass helpers ───────────────────────────────────────────────────────
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
    peak_idx   = np.argmax(band_power)
    peak_freq  = band_freqs[peak_idx]
    confidence = min(band_power[peak_idx] / (np.mean(fft) * 3.0 + 1e-9), 1.0)
    return float(peak_freq * 60.0), float(confidence)

# ── Firebase helpers ───────────────────────────────────────────────────────
def firebase_put(path, data):
    url = f"{FIREBASE_DB_URL}/{path}.json"
    try:
        r = requests.put(url, json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"[Firebase] PUT error: {e}")
        return False

def firebase_post(path, data):
    url = f"{FIREBASE_DB_URL}/{path}.json"
    try:
        r = requests.post(url, json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"[Firebase] POST error: {e}")
        return False

# ── Privacy fake data generator ────────────────────────────────────────────
def make_fake_skeleton(zone, person_id, t):
    """Generate a fake human skeleton with animated movement."""
    ox = zone["x_offset"]
    oy = zone["y_offset"]

    # Slow random sway
    sway_x = 8.0 * math.sin(t * 0.3 + person_id * 1.7)
    sway_y = 4.0 * math.sin(t * 0.5 + person_id * 2.3)

    # Body proportions (normalized 640x480 space)
    body = [
        # nose
        (ox + sway_x,        oy + sway_y),
        # eyes
        (ox - 8 + sway_x,    oy - 5 + sway_y),
        (ox + 8 + sway_x,    oy - 5 + sway_y),
        # ears
        (ox - 14 + sway_x,   oy + sway_y),
        (ox + 14 + sway_x,   oy + sway_y),
        # shoulders
        (ox - 30 + sway_x,   oy + 40 + sway_y),
        (ox + 30 + sway_x,   oy + 40 + sway_y),
        # elbows (arms swing slightly)
        (ox - 40 + sway_x + 5*math.sin(t*0.8 + person_id), oy + 80 + sway_y),
        (ox + 40 + sway_x - 5*math.sin(t*0.8 + person_id), oy + 80 + sway_y),
        # wrists
        (ox - 45 + sway_x + 8*math.sin(t*0.8 + person_id), oy + 120 + sway_y),
        (ox + 45 + sway_x - 8*math.sin(t*0.8 + person_id), oy + 120 + sway_y),
        # hips
        (ox - 20 + sway_x,   oy + 140 + sway_y),
        (ox + 20 + sway_x,   oy + 140 + sway_y),
        # knees
        (ox - 22 + sway_x,   oy + 200 + sway_y),
        (ox + 22 + sway_x,   oy + 200 + sway_y),
        # ankles
        (ox - 22 + sway_x,   oy + 260 + sway_y),
        (ox + 22 + sway_x,   oy + 260 + sway_y),
    ]

    keypoints = []
    for i, (x, y) in enumerate(body):
        keypoints.append({
            "name":       KEYPOINT_NAMES[i],
            "x":          float(x),
            "y":          float(y),
            "z":          float(random.uniform(-10, 10)),
            "confidence": float(random.uniform(0.7, 0.99)),
        })
    return keypoints

def generate_fake_persons(t):
    """Generate N fake persons with random vitals + animated skeletons."""
    n = random.randint(FAKE_PERSON_COUNT_MIN, FAKE_PERSON_COUNT_MAX)
    persons = []
    zones   = random.sample(FAKE_ZONES, min(n, len(FAKE_ZONES)))

    for i in range(n):
        zone      = zones[i % len(zones)]
        keypoints = make_fake_skeleton(zone, i, t)
        xs = [k["x"] for k in keypoints]
        ys = [k["y"] for k in keypoints]

        persons.append({
            "id":           i + 1,
            "confidence":   round(random.uniform(0.72, 0.97), 2),
            "keypoints":    keypoints,
            "bbox": {
                "x":      float(min(xs) - 10),
                "y":      float(min(ys) - 10),
                "width":  float(max(xs) - min(xs) + 20),
                "height": float(max(ys) - min(ys) + 20),
            },
            "zone":         f"zone_{i+1}",
            "position":     [float(zone["x_offset"]/100 - 3), 0.0,
                             float(zone["y_offset"]/100 - 2)],
            "motion_score": random.randint(20, 80),
            "pose":         random.choice(["standing", "sitting", "walking"]),
            "vital_signs": {
                "breathing_rate_bpm":   round(random.uniform(*FAKE_BR_RANGE), 1),
                "heart_rate_bpm":       round(random.uniform(*FAKE_HR_RANGE), 1),
                "breathing_confidence": round(random.uniform(0.6, 0.95), 2),
                "heartbeat_confidence": round(random.uniform(0.6, 0.95), 2),
            },
        })
    return persons

# ── UDP Listeners ──────────────────────────────────────────────────────────
def udp_sensing_listener():
    """Listen to ESP32 #1 on port 5006."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_SENSING_PORT))
    sock.settimeout(1.0)
    print(f"[UDP-Sensing] Listening on port {UDP_SENSING_PORT}")

    while True:
        try:
            data, _ = sock.recvfrom(2048)
            now     = time.time()
            rssi    = None

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
                    state.last_rssi     = rssi
                    state.last_udp_time = now

        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP-Sensing] Error: {e}")

def udp_privacy_listener():
    """Listen to ESP32 #2 privacy toggle on port 5007."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PRIVACY_PORT))
    sock.settimeout(1.0)
    print(f"[UDP-Privacy] Listening on port {UDP_PRIVACY_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(256)
            try:
                j       = json.loads(data.decode('utf-8'))
                privacy = bool(j.get('privacy', False))
                with state.lock:
                    changed = (state.privacy_mode != privacy)
                    state.privacy_mode    = privacy
                    state.privacy_changed = changed
                if changed:
                    status = "🔒 ON" if privacy else "🔓 OFF"
                    print(f"[Privacy] Mode changed → {status}")
            except:
                pass
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP-Privacy] Error: {e}")

# ── Vital Signs Processor ──────────────────────────────────────────────────
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

        br_bpm = float(np.clip(br_bpm, 6.0, 30.0))   if br_conf > 0.1 else 0.0
        hr_bpm = float(np.clip(hr_bpm, 40.0, 120.0))  if hr_conf > 0.1 else 0.0

        with state.lock:
            state.breathing_bpm  = br_bpm
            state.heart_rate_bpm = hr_bpm
            state.breathing_conf = br_conf
            state.heart_conf     = hr_conf
            state.motion_score   = motion
            state.presence       = presence
            state.signal_quality = sig_q

# ── MediaPipe camera ───────────────────────────────────────────────────────
MP_TO_COCO = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]

def camera_thread():
    mp_pose = mp.solutions.pose
    cap     = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[Camera] Failed to open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[Camera] Webcam opened")

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
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
                landmarks = results.pose_landmarks.landmark
                confs     = []
                for ci, mi in enumerate(MP_TO_COCO):
                    lm = landmarks[mi]
                    keypoints.append({
                        "name":       KEYPOINT_NAMES[ci],
                        "x":          float(lm.x * 640),
                        "y":          float(lm.y * 480),
                        "z":          float(lm.z * 100),
                        "confidence": float(lm.visibility),
                    })
                    confs.append(lm.visibility)
                pose_conf = float(np.mean(confs))
            with state.lock:
                state.keypoints      = keypoints
                state.pose_confidence = pose_conf
                state.camera_active  = True
    cap.release()

# ── Firebase Thread ────────────────────────────────────────────────────────
def firebase_thread():
    print("[Firebase] Starting sync thread")
    firebase_put("auratrack/status", {
        "online":     True,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version":    "2.0.0",
        "features":   ["privacy_mode", "fake_persons"],
    })
    print(f"[Firebase] Live → {FIREBASE_DB_URL}/auratrack/live.json")

    last_live = last_hist = 0.0

    while True:
        time.sleep(0.5)
        now = time.time()

        with state.lock:
            privacy  = state.privacy_mode
            br_bpm   = state.breathing_bpm
            hr_bpm   = state.heart_rate_bpm
            motion   = state.motion_score
            presence = state.presence
            rssi     = state.last_rssi

        if now - last_live >= FIREBASE_LIVE_INTERVAL:
            last_live = now
            if privacy:
                # Push fake data to Firebase
                fake_persons = generate_fake_persons(now)
                live_data = {
                    "privacy_mode":    True,
                    "presence":        True,
                    "persons_count":   len(fake_persons),
                    "breathing_bpm":   round(random.uniform(*FAKE_BR_RANGE), 1),
                    "heart_rate_bpm":  round(random.uniform(*FAKE_HR_RANGE), 1),
                    "motion_level":    "active",
                    "rssi_dbm":        round(random.uniform(*FAKE_RSSI_RANGE), 1),
                    "confidence":      round(random.uniform(0.75, 0.98), 2),
                    "timestamp":       now,
                    "timestamp_str":   time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note":            "🔒 Privacy Shield Active",
                }
                print(f"[Firebase] 🔒 Privacy → {len(fake_persons)} fake persons")
            else:
                live_data = {
                    "privacy_mode":   False,
                    "presence":       presence,
                    "persons_count":  1 if presence else 0,
                    "breathing_bpm":  round(br_bpm, 1),
                    "heart_rate_bpm": round(hr_bpm, 1),
                    "motion_level":   "active" if motion > 0.6 else "present_still" if presence else "absent",
                    "rssi_dbm":       round(rssi, 1),
                    "timestamp":      now,
                    "timestamp_str":  time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                print(f"[Firebase] 🔓 Real → presence={presence} BR={br_bpm:.1f} HR={hr_bpm:.1f}")
            firebase_put("auratrack/live", live_data)

        if now - last_hist >= FIREBASE_HISTORY_INTERVAL:
            last_hist = now
            firebase_post("auratrack/history", {
                "privacy_mode": privacy,
                "timestamp":    now,
                "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                **({"note": "privacy_shield"} if privacy else {
                    "breathing_bpm":  round(br_bpm, 1),
                    "heart_rate_bpm": round(hr_bpm, 1),
                    "presence":       presence,
                })
            })

# ── WebSocket Sender ───────────────────────────────────────────────────────
async def websocket_sender():
    print(f"[WS] Connecting to {SERVER_WS_URL}")
    while True:
        try:
            async with websockets.connect(SERVER_WS_URL, ping_interval=20) as ws:
                print("[WS] Connected!")
                while True:
                    await asyncio.sleep(0.1)
                    now = time.time()

                    with state.lock:
                        privacy   = state.privacy_mode
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

                    if privacy:
                        # ── PRIVACY MODE: send fake data ──────────────
                        fake_persons = generate_fake_persons(now)
                        n_fake       = len(fake_persons)
                        fake_rssi    = random.uniform(*FAKE_RSSI_RANGE)
                        fake_motion  = random.uniform(0.3, 0.9)

                        msg = {
                            "type":      "sensing_update",
                            "timestamp": now,
                            "source":    "auratrack_privacy",
                            "tick":      tick,
                            "privacy_mode": True,
                            "nodes": [{
                                "node_id":          1,
                                "rssi_dbm":         fake_rssi,
                                "position":         [0.0, 0.0, 0.0],
                                "amplitude":        [random.uniform(0.1, 1.0) for _ in range(56)],
                                "subcarrier_count": 56,
                            }],
                            "features": {
                                "mean_rssi":           fake_rssi,
                                "variance":            random.uniform(1.0, 8.0),
                                "motion_band_power":   random.uniform(5.0, 20.0),
                                "breathing_band_power": random.uniform(2.0, 8.0),
                                "dominant_freq_hz":    random.uniform(0.2, 0.4),
                                "change_points":       random.randint(5, 30),
                                "spectral_power":      random.uniform(50, 150),
                            },
                            "classification": {
                                "motion_level": random.choice(["active", "present_moving", "active"]),
                                "presence":     True,
                                "confidence":   round(random.uniform(0.75, 0.98), 2),
                            },
                            "vital_signs": {
                                "breathing_rate_bpm":   round(random.uniform(*FAKE_BR_RANGE), 1),
                                "heart_rate_bpm":       round(random.uniform(*FAKE_HR_RANGE), 1),
                                "breathing_confidence": round(random.uniform(0.6, 0.95), 2),
                                "heartbeat_confidence": round(random.uniform(0.6, 0.95), 2),
                                "signal_quality":       round(random.uniform(0.6, 0.95), 2),
                            },
                            "signal_field": {
                                "grid_size": [20, 1, 20],
                                "values":    [random.uniform(0, 1) for _ in range(400)],
                            },
                            "persons":           fake_persons,
                            "estimated_persons": n_fake,
                            "pose_source":       "privacy_shield",
                        }

                    else:
                        # ── REAL MODE: send actual sensor data ────────
                        motion_level = (
                            "active"          if motion > 0.6 else
                            "present_moving"  if motion > 0.2 else
                            "present_still"   if presence    else
                            "absent"
                        )
                        confidence = float(np.clip(
                            0.3 + sig_qual*0.3 + pose_conf*0.4, 0.0, 1.0
                        ))
                        persons = []
                        if keypoints and pose_conf > 0.3:
                            persons = [{
                                "id":           1,
                                "confidence":   float(pose_conf),
                                "keypoints":    keypoints,
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
                            "timestamp": now,
                            "source":    "auratrack_bridge",
                            "tick":      tick,
                            "privacy_mode": False,
                            "nodes": [{
                                "node_id":          1,
                                "rssi_dbm":         rssi,
                                "position":         [0.0, 0.0, 0.0],
                                "amplitude":        [abs(rssi)] * 56,
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

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AuraTrack Bridge v2.0 — Privacy Shield Edition")
    print("=" * 60)
    print(f"  Sensing UDP : port {UDP_SENSING_PORT}  (ESP32 #1)")
    print(f"  Privacy UDP : port {UDP_PRIVACY_PORT}  (ESP32 #2)")
    print(f"  WebSocket   : {SERVER_WS_URL}")
    print(f"  Firebase    : {FIREBASE_DB_URL}")
    print("=" * 60)
    print()
    print("  Press button on ESP32 #2 to toggle privacy mode")
    print("  🔒 ON  → website shows 3-5 FAKE persons + random vitals")
    print("  🔓 OFF → website shows REAL sensor data")
    print()

    threading.Thread(target=udp_sensing_listener, daemon=True).start()
    threading.Thread(target=udp_privacy_listener, daemon=True).start()
    threading.Thread(target=compute_vitals,        daemon=True).start()
    threading.Thread(target=camera_thread,         daemon=True).start()
    threading.Thread(target=firebase_thread,       daemon=True).start()

    print("[Main] All threads started!")
    asyncio.run(websocket_sender())

if __name__ == "__main__":
    main()
