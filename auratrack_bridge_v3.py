"""
AuraTrack Bridge v3.0 — Full Body Motion Animation
====================================================
Real person: animated skeleton with walking cycle, arm swing,
             weight shift driven by actual RSSI motion score.
Privacy mode: 3-5 fake persons all independently animated.

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

FAKE_PERSON_COUNT_MIN = 3
FAKE_PERSON_COUNT_MAX = 5
FAKE_BR_RANGE   = (10, 22)
FAKE_HR_RANGE   = (55, 105)
FAKE_RSSI_RANGE = (-45, -80)

KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
MP_TO_COCO = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]

# ── Full Body Animator ─────────────────────────────────────────────────────
class BodyAnimator:
    """
    Biomechanical walking cycle animator.

    Joints animated:
      - Hips:      lateral sway + forward lean
      - Shoulders: counter-rotate against hips (natural gait)
      - Arms:      swing opposite to legs (elbow/wrist follow-through)
      - Legs:      alternating knee lift + ankle plantarflex
      - Head/Neck: subtle bob + slight turn into stride
      - Torso:     breathing expansion (chest rise)

    Motion is driven by:
      - walk_speed : 0.0 = still, 1.0 = fast walk
      - breath_bpm : breathing frequency for chest expansion
      - phase      : global time phase (seconds)
    """

    def __init__(self, person_id=0, cx=320, cy=120):
        self.person_id  = person_id
        self.cx         = cx          # canvas center X
        self.cy         = cy          # top of figure Y
        # Each person gets a unique phase offset so they don't move in sync
        self.phase_offset = person_id * 1.37 + random.uniform(0, 2 * math.pi)
        # Walking direction: slowly drift left/right
        self.walk_dir   = random.choice([-1, 1])
        self.walk_x     = float(cx)
        self.last_t     = time.time()

    def _sin(self, freq, t, offset=0.0):
        return math.sin(2 * math.pi * freq * t + self.phase_offset + offset)

    def _cos(self, freq, t, offset=0.0):
        return math.cos(2 * math.pi * freq * t + self.phase_offset + offset)

    def get_keypoints(self, t, walk_speed=0.5, breath_bpm=15.0, motion_score=0.5):
        """
        Returns 17 COCO keypoints for a walking human figure.

        t           : current time in seconds
        walk_speed  : 0.0 (still) → 1.0 (fast walk), driven by motion_score
        breath_bpm  : breathing rate, drives chest expansion
        motion_score: overall motion, scales all movement amplitude
        """
        dt = t - self.last_t
        self.last_t = t

        # ── Gait frequency: ~1 full stride per second at walk_speed=1 ──
        gait_hz   = 0.8 + walk_speed * 0.7        # 0.8–1.5 Hz
        breath_hz = breath_bpm / 60.0

        # ── Base amplitudes scaled by motion ────────────────────────────
        amp       = 0.3 + motion_score * 0.7       # 0.3 (idle) → 1.0 (active)

        # ── Walk drift: figure slowly moves left/right, bounces at edges ─
        self.walk_x += self.walk_dir * walk_speed * amp * 0.8 * dt * 30
        if self.walk_x > self.cx + 150:
            self.walk_dir = -1
        elif self.walk_x < self.cx - 150:
            self.walk_dir = 1
        wx = self.walk_x

        # ── Body proportions (pixels, 640×480 canvas) ────────────────────
        HEAD_R   = 18
        NECK     = 28
        TORSO    = 90
        UPPER_ARM= 45
        LOWER_ARM= 40
        UPPER_LEG= 70
        LOWER_LEG= 65

        # ── Gait oscillators ─────────────────────────────────────────────
        # Vertical body bob (happens twice per stride)
        bob      = amp * 4.0  * abs(self._sin(gait_hz * 2, t))

        # Hip lateral sway
        hip_sway = amp * 8.0  * self._sin(gait_hz, t)

        # Hip forward/back tilt (pelvis rotation)
        hip_tilt = amp * 5.0  * self._sin(gait_hz, t)

        # Shoulder counter-rotation (opposite phase to hips)
        sh_rot   = amp * 6.0  * self._sin(gait_hz, t, math.pi)

        # Breathing chest expansion
        breath   = amp * 3.0  * self._sin(breath_hz, t)

        # Head bob & slight turn
        head_bob = amp * 2.5  * self._sin(gait_hz * 2, t)
        head_turn= amp * 4.0  * self._sin(gait_hz, t, math.pi * 0.5)

        # ── Compute joint positions ───────────────────────────────────────
        # Root: mid-hip
        root_x = wx + hip_sway
        root_y = self.cy + NECK + TORSO - bob

        # Spine/shoulder midpoint
        sp_x   = wx + sh_rot * 0.3
        sp_y   = root_y - TORSO - breath * 0.5

        # Nose (head center)
        nose_x = sp_x + head_turn
        nose_y = sp_y - NECK - HEAD_R + head_bob

        # Eyes & ears
        leye_x = nose_x - 6;  leye_y = nose_y - 4
        reye_x = nose_x + 6;  reye_y = nose_y - 4
        lear_x = nose_x - 12; lear_y = nose_y
        rear_x = nose_x + 12; rear_y = nose_y

        # Shoulders
        lsh_x  = sp_x - 30 + sh_rot;   lsh_y = sp_y + 10
        rsh_x  = sp_x + 30 - sh_rot;   rsh_y = sp_y + 10

        # Arms — swing opposite legs
        # Left arm swings forward when right leg swings forward
        l_arm_swing = amp * 28.0 * self._sin(gait_hz, t, math.pi)   # opposite phase
        r_arm_swing = amp * 28.0 * self._sin(gait_hz, t)

        # Elbow (mid-arm)
        lelbow_x = lsh_x - 8  + l_arm_swing * 0.6
        lelbow_y = lsh_y + UPPER_ARM
        relbow_x = rsh_x + 8  - r_arm_swing * 0.6
        relbow_y = rsh_y + UPPER_ARM

        # Wrist (follow-through, slightly lag behind elbow)
        lwrist_x = lelbow_x - 5 + l_arm_swing * 0.9
        lwrist_y = lelbow_y + LOWER_ARM
        rwrist_x = relbow_x + 5 - r_arm_swing * 0.9
        rwrist_y = relbow_y + LOWER_ARM

        # Hips
        lhip_x  = root_x - 18 - hip_sway * 0.3
        lhip_y  = root_y + hip_tilt * 0.3
        rhip_x  = root_x + 18 + hip_sway * 0.3
        rhip_y  = root_y - hip_tilt * 0.3

        # Legs — alternating stride
        l_leg_swing = amp * 32.0 * self._sin(gait_hz, t)         # left leg
        r_leg_swing = amp * 32.0 * self._sin(gait_hz, t, math.pi) # right leg (opposite)

        # Knee lift
        lknee_x = lhip_x + l_leg_swing * 0.5
        lknee_y = lhip_y + UPPER_LEG - abs(l_leg_swing) * 0.4
        rknee_x = rhip_x + r_leg_swing * 0.5
        rknee_y = rhip_y + UPPER_LEG - abs(r_leg_swing) * 0.4

        # Ankle / foot (plantarflex at toe-off)
        lankle_x = lknee_x + l_leg_swing * 0.4
        lankle_y = lknee_y + LOWER_LEG + abs(l_leg_swing) * 0.2
        rankle_x = rknee_x + r_leg_swing * 0.4
        rankle_y = rknee_y + LOWER_LEG + abs(r_leg_swing) * 0.2

        # ── Assemble keypoints ────────────────────────────────────────────
        coords = [
            (nose_x,    nose_y),    # 0  nose
            (leye_x,    leye_y),    # 1  left_eye
            (reye_x,    reye_y),    # 2  right_eye
            (lear_x,    lear_y),    # 3  left_ear
            (rear_x,    rear_y),    # 4  right_ear
            (lsh_x,     lsh_y),     # 5  left_shoulder
            (rsh_x,     rsh_y),     # 6  right_shoulder
            (lelbow_x,  lelbow_y),  # 7  left_elbow
            (relbow_x,  relbow_y),  # 8  right_elbow
            (lwrist_x,  lwrist_y),  # 9  left_wrist
            (rwrist_x,  rwrist_y),  # 10 right_wrist
            (lhip_x,    lhip_y),    # 11 left_hip
            (rhip_x,    rhip_y),    # 12 right_hip
            (lknee_x,   lknee_y),   # 13 left_knee
            (rknee_x,   rknee_y),   # 14 right_knee
            (lankle_x,  lankle_y),  # 15 left_ankle
            (rankle_x,  rankle_y),  # 16 right_ankle
        ]

        keypoints = []
        for i, (x, y) in enumerate(coords):
            keypoints.append({
                "name":       KEYPOINT_NAMES[i],
                "x":          float(x),
                "y":          float(y),
                "z":          float(self._sin(gait_hz * 0.5, t, i * 0.3) * 10),
                "confidence": float(random.uniform(0.82, 0.99)),
            })
        return keypoints, float(self.walk_x)

# ── Global animators ───────────────────────────────────────────────────────
# One animator for the real person + 5 for fake privacy persons
REAL_ANIMATOR = BodyAnimator(person_id=0, cx=320, cy=80)
FAKE_ANIMATORS = [
    BodyAnimator(person_id=i+1,
                 cx=random.choice([110, 220, 330, 440, 550]),
                 cy=random.randint(60, 100))
    for i in range(5)
]

# ── Shared state ───────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock             = threading.Lock()
        self.rssi_buffer      = deque(maxlen=BUFFER_SIZE)
        self.rssi_timestamps  = deque(maxlen=BUFFER_SIZE)
        self.last_rssi        = -60.0
        self.last_udp_time    = 0.0
        self.breathing_bpm    = 15.0
        self.heart_rate_bpm   = 72.0
        self.breathing_conf   = 0.5
        self.heart_conf       = 0.5
        self.motion_score     = 0.4
        self.presence         = True
        self.signal_quality   = 0.6
        self.keypoints        = []
        self.pose_confidence  = 0.0
        self.camera_active    = False
        self.tick             = 0
        self.privacy_mode     = False

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
    fft   = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    mask  = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0, 0.0
    bp    = fft[mask]
    bf    = freqs[mask]
    idx   = np.argmax(bp)
    conf  = min(bp[idx] / (np.mean(fft) * 3.0 + 1e-9), 1.0)
    return float(bf[idx] * 60.0), float(conf)

# ── Firebase helpers ───────────────────────────────────────────────────────
def firebase_put(path, data):
    try:
        r = requests.put(f"{FIREBASE_DB_URL}/{path}.json", json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"[Firebase] PUT error: {e}")
        return False

def firebase_post(path, data):
    try:
        r = requests.post(f"{FIREBASE_DB_URL}/{path}.json", json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"[Firebase] POST error: {e}")
        return False

# ── UDP Listeners ──────────────────────────────────────────────────────────
def udp_sensing_listener():
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
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PRIVACY_PORT))
    sock.settimeout(1.0)
    print(f"[UDP-Privacy] Listening on port {UDP_PRIVACY_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(256)
            try:
                j       = json.loads(data.decode('utf-8'))
                privacy = bool(j.get('privacy', False))
                with state.lock:
                    changed = (state.privacy_mode != privacy)
                    state.privacy_mode = privacy
                if changed:
                    print(f"[Privacy] {'🔒 ON' if privacy else '🔓 OFF'}")
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
        presence = bool(np.mean(rssi_array) > -80 and variance > 0.05)
        sig_q    = float(np.clip((np.mean(rssi_array) + 100) / 60.0, 0.0, 1.0))

        br_sig  = bandpass_filter(rssi_d, 0.1, 0.5, actual_fs)
        hr_sig  = bandpass_filter(rssi_d, 0.8, 2.0, actual_fs)
        br_bpm, br_conf = estimate_bpm(br_sig, actual_fs, 0.1, 0.5)
        hr_bpm, hr_conf = estimate_bpm(hr_sig, actual_fs, 0.8, 2.0)

        br_bpm = float(np.clip(br_bpm, 6.0, 30.0))    if br_conf > 0.05 else 15.0
        hr_bpm = float(np.clip(hr_bpm, 40.0, 120.0))  if hr_conf > 0.05 else 72.0

        with state.lock:
            state.breathing_bpm  = br_bpm
            state.heart_rate_bpm = hr_bpm
            state.breathing_conf = br_conf
            state.heart_conf     = hr_conf
            state.motion_score   = max(motion, 0.3)  # floor at 0.3 so figure always moves a little
            state.presence       = presence
            state.signal_quality = sig_q

# ── Camera Thread ──────────────────────────────────────────────────────────
def camera_thread():
    mp_pose = mp.solutions.pose
    cap     = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[Camera] No webcam — using animated skeleton only")
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
                state.keypoints       = keypoints
                state.pose_confidence = pose_conf
                state.camera_active   = True
    cap.release()

# ── Firebase Thread ────────────────────────────────────────────────────────
def firebase_thread():
    print("[Firebase] Starting sync")
    firebase_put("auratrack/status", {
        "online": True, "version": "3.0.0",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
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
                data = {
                    "privacy_mode": True,
                    "persons_count": random.randint(FAKE_PERSON_COUNT_MIN, FAKE_PERSON_COUNT_MAX),
                    "breathing_bpm": round(random.uniform(*FAKE_BR_RANGE), 1),
                    "heart_rate_bpm": round(random.uniform(*FAKE_HR_RANGE), 1),
                    "note": "🔒 Privacy Shield Active",
                    "timestamp": now,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            else:
                data = {
                    "privacy_mode": False, "presence": presence,
                    "breathing_bpm": round(br_bpm, 1),
                    "heart_rate_bpm": round(hr_bpm, 1),
                    "motion_score": round(motion, 3),
                    "rssi_dbm": round(rssi, 1),
                    "timestamp": now,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            firebase_put("auratrack/live", data)
            print(f"[Firebase] {'🔒' if privacy else '🔓'} pushed")

        if now - last_hist >= FIREBASE_HISTORY_INTERVAL:
            last_hist = now
            firebase_post("auratrack/history", {
                "privacy_mode": privacy, "timestamp": now,
                "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                **({"note": "privacy_shield"} if privacy else {
                    "breathing_bpm": round(br_bpm, 1),
                    "heart_rate_bpm": round(hr_bpm, 1),
                }),
            })

# ── Build person dict ──────────────────────────────────────────────────────
def build_person(animator, t, walk_speed, breath_bpm, motion_score,
                 person_id, br_bpm, hr_bpm, br_conf, hr_conf):
    keypoints, wx = animator.get_keypoints(
        t, walk_speed=walk_speed,
        breath_bpm=breath_bpm,
        motion_score=motion_score
    )
    xs = [k["x"] for k in keypoints]
    ys = [k["y"] for k in keypoints]
    return {
        "id":           person_id,
        "confidence":   round(random.uniform(0.80, 0.97), 2),
        "keypoints":    keypoints,
        "bbox": {
            "x":      float(min(xs) - 15),
            "y":      float(min(ys) - 15),
            "width":  float(max(xs) - min(xs) + 30),
            "height": float(max(ys) - min(ys) + 30),
        },
        "zone":         f"zone_{person_id}",
        "position":     [float((wx - 320) / 100), 0.0, 0.0],
        "motion_score": int(motion_score * 100),
        "pose":         "walking" if walk_speed > 0.3 else "standing",
        "vital_signs": {
            "breathing_rate_bpm":   round(br_bpm, 1),
            "heart_rate_bpm":       round(hr_bpm, 1),
            "breathing_confidence": round(br_conf, 2),
            "heartbeat_confidence": round(hr_conf, 2),
        },
    }

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
                        state.tick += 1
                        tick = state.tick

                    # Walk speed derived from motion score
                    walk_speed = float(np.clip(motion * 1.2, 0.1, 1.0))

                    if privacy:
                        # ── PRIVACY: 3-5 animated fake persons ────────
                        n = random.randint(FAKE_PERSON_COUNT_MIN, FAKE_PERSON_COUNT_MAX)
                        persons = []
                        for i in range(n):
                            anim = FAKE_ANIMATORS[i]
                            p = build_person(
                                anim, now,
                                walk_speed   = random.uniform(0.3, 0.9),
                                breath_bpm   = random.uniform(*FAKE_BR_RANGE),
                                motion_score = random.uniform(0.4, 0.9),
                                person_id    = i + 1,
                                br_bpm       = random.uniform(*FAKE_BR_RANGE),
                                hr_bpm       = random.uniform(*FAKE_HR_RANGE),
                                br_conf      = random.uniform(0.6, 0.95),
                                hr_conf      = random.uniform(0.6, 0.95),
                            )
                            persons.append(p)

                        msg = {
                            "type": "sensing_update", "timestamp": now,
                            "source": "auratrack_privacy", "tick": tick,
                            "privacy_mode": True,
                            "nodes": [{"node_id": 1, "rssi_dbm": random.uniform(*FAKE_RSSI_RANGE),
                                       "position": [0,0,0],
                                       "amplitude": [random.uniform(0.1,1.0) for _ in range(56)],
                                       "subcarrier_count": 56}],
                            "features": {
                                "mean_rssi": random.uniform(*FAKE_RSSI_RANGE),
                                "variance": random.uniform(1.0, 8.0),
                                "motion_band_power": random.uniform(5, 20),
                                "breathing_band_power": random.uniform(2, 8),
                                "dominant_freq_hz": random.uniform(0.2, 0.4),
                                "change_points": random.randint(5, 30),
                                "spectral_power": random.uniform(50, 150),
                            },
                            "classification": {
                                "motion_level": "active",
                                "presence": True,
                                "confidence": round(random.uniform(0.75, 0.98), 2),
                            },
                            "vital_signs": {
                                "breathing_rate_bpm": round(random.uniform(*FAKE_BR_RANGE), 1),
                                "heart_rate_bpm": round(random.uniform(*FAKE_HR_RANGE), 1),
                                "breathing_confidence": round(random.uniform(0.6, 0.95), 2),
                                "heartbeat_confidence": round(random.uniform(0.6, 0.95), 2),
                                "signal_quality": round(random.uniform(0.6, 0.95), 2),
                            },
                            "signal_field": {
                                "grid_size": [20, 1, 20],
                                "values": [random.uniform(0, 1) for _ in range(400)],
                            },
                            "persons": persons,
                            "estimated_persons": n,
                            "pose_source": "privacy_shield",
                        }

                    else:
                        # ── REAL: 1 animated person driven by RSSI ────
                        person = build_person(
                            REAL_ANIMATOR, now,
                            walk_speed   = walk_speed,
                            breath_bpm   = br_bpm if br_bpm > 0 else 15.0,
                            motion_score = motion,
                            person_id    = 1,
                            br_bpm       = br_bpm,
                            hr_bpm       = hr_bpm,
                            br_conf      = br_conf,
                            hr_conf      = hr_conf,
                        )

                        motion_level = (
                            "active"         if motion > 0.6 else
                            "present_moving" if motion > 0.2 else
                            "present_still"  if presence    else
                            "absent"
                        )

                        msg = {
                            "type": "sensing_update", "timestamp": now,
                            "source": "auratrack_bridge", "tick": tick,
                            "privacy_mode": False,
                            "nodes": [{"node_id": 1, "rssi_dbm": rssi,
                                       "position": [0,0,0],
                                       "amplitude": [abs(rssi)] * 56,
                                       "subcarrier_count": 56}],
                            "features": {
                                "mean_rssi": rssi,
                                "variance": float(motion * 10),
                                "motion_band_power": float(motion * 15),
                                "breathing_band_power": float(br_conf * 5),
                                "dominant_freq_hz": float(br_bpm / 60.0),
                                "change_points": int(motion * 20),
                                "spectral_power": float(abs(rssi) * 2),
                            },
                            "classification": {
                                "motion_level": motion_level,
                                "presence": presence,
                                "confidence": float(np.clip(0.4 + sig_qual*0.3 + motion*0.3, 0, 1)),
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
                                        motion * math.exp(-((i//20-10)**2+(i%20-10)**2)/30.0),
                                        0, 1)) for i in range(400)
                                ],
                            },
                            "persons":           [person],
                            "estimated_persons": 1,
                            "pose_source":       "auratrack_animated",
                        }

                    await ws.send(json.dumps(msg))

        except Exception as e:
            print(f"[WS] Disconnected: {e}, retrying in 3s...")
            await asyncio.sleep(3)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AuraTrack Bridge v3.0 — Full Body Motion")
    print("=" * 60)
    print(f"  Sensing UDP : port {UDP_SENSING_PORT}")
    print(f"  Privacy UDP : port {UDP_PRIVACY_PORT}")
    print(f"  WebSocket   : {SERVER_WS_URL}")
    print(f"  Firebase    : {FIREBASE_DB_URL}")
    print("=" * 60)
    print()
    print("  Skeleton animation: walking cycle + arm swing")
    print("  Motion speed driven by live RSSI variance")
    print("  🔒 Privacy: 3-5 independently animated fake persons")
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
