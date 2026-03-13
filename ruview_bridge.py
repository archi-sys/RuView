"""
RuView Python Bridge
====================
Combines:
  1. ESP32 RSSI over UDP → real vital signs via bandpass filtering
  2. Webcam + MediaPipe → real 17-keypoint skeleton
  3. Sends fused data to Rust server WebSocket → Observatory 3D view

Usage:
  python ruview_bridge.py

Requirements:
  pip install opencv-python mediapipe requests websockets numpy scipy
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

# ── Configuration ─────────────────────────────────────────────────────────────
UDP_PORT       = 5006        # listen on different port (5005 is used by Rust server)
SERVER_WS_URL  = "ws://localhost:3001/ws/sensing"
SERVER_HTTP    = "http://localhost:3000"
SAMPLE_RATE    = 10.0        # Hz (ESP32 sends at 10 FPS)
BUFFER_SIZE    = 200         # samples (~20 seconds at 10 Hz)
CAMERA_INDEX   = 0           # webcam index (0 = built-in)
SHOW_CAMERA    = False       # disabled (Windows Store Python doesn't support imshow)

# ── Vital signs filter design ─────────────────────────────────────────────────
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
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
    """Estimate BPM from a filtered signal using FFT peak detection."""
    if len(signal) < 20:
        return 0.0, 0.0
    
    # FFT
    fft = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    
    # Find peak in target band
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0, 0.0
    
    band_power = fft[mask]
    band_freqs = freqs[mask]
    
    if len(band_power) == 0:
        return 0.0, 0.0
    
    peak_idx = np.argmax(band_power)
    peak_freq = band_freqs[peak_idx]
    peak_power = band_power[peak_idx]
    
    # Confidence: ratio of peak power to mean power
    mean_power = np.mean(fft) + 1e-9
    confidence = min(peak_power / (mean_power * 3.0), 1.0)
    
    bpm = peak_freq * 60.0
    return float(bpm), float(confidence)

# ── MediaPipe keypoint names (COCO 17) ───────────────────────────────────────
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# MediaPipe Pose landmark indices → COCO 17 mapping
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# ── Shared state ──────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # RSSI buffer for vital signs
        self.rssi_buffer = deque(maxlen=BUFFER_SIZE)
        self.rssi_timestamps = deque(maxlen=BUFFER_SIZE)
        self.last_rssi = -60.0
        self.last_udp_time = 0.0
        
        # Vital signs (computed from RSSI)
        self.breathing_bpm = 0.0
        self.heart_rate_bpm = 0.0
        self.breathing_conf = 0.0
        self.heart_conf = 0.0
        self.motion_score = 0.0
        self.presence = False
        self.signal_quality = 0.0
        
        # Camera / MediaPipe keypoints
        self.keypoints = []       # list of {name, x, y, z, confidence}
        self.pose_confidence = 0.0
        self.camera_active = False
        
        # Frame counter
        self.tick = 0

state = SharedState()

# ── UDP Listener (ESP32 → RSSI) ───────────────────────────────────────────────
def udp_listener():
    """
    Listen for ESP32 UDP packets.
    Accepts both:
      - Binary frames (magic 0xC5110001) → extract RSSI from byte 14
      - JSON frames → extract mean_rssi field
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)
    print(f"[UDP] Listening on port {UDP_PORT} for ESP32 data")
    
    while True:
        try:
            data, addr = sock.recvfrom(2048)
            now = time.time()
            rssi = None
            
            # Try binary frame first
            if len(data) >= 20:
                magic = int.from_bytes(data[0:4], 'little')
                if magic == 0xC5110001:
                    rssi = float(data[14] if data[14] < 128 else data[14] - 256)
            
            # Try JSON frame
            if rssi is None:
                try:
                    j = json.loads(data.decode('utf-8'))
                    rssi = float(j.get('mean_rssi') or j.get('rssi') or -60)
                except:
                    pass
            
            if rssi is not None:
                with state.lock:
                    state.rssi_buffer.append(rssi)
                    state.rssi_timestamps.append(now)
                    state.last_rssi = rssi
                    state.last_udp_time = now
                    
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP] Error: {e}")
            time.sleep(0.1)

# ── Vital Signs Computer ──────────────────────────────────────────────────────
def compute_vitals():
    """Continuously compute vital signs from RSSI buffer."""
    print("[Vitals] Vital signs processor started")
    
    while True:
        time.sleep(0.5)  # update every 500ms
        
        with state.lock:
            if len(state.rssi_buffer) < 20:
                continue
            rssi_array = np.array(list(state.rssi_buffer), dtype=float)
            n = len(rssi_array)
        
        # Actual sample rate from timestamps
        with state.lock:
            times = np.array(list(state.rssi_timestamps))
        
        if len(times) > 1:
            actual_fs = (len(times) - 1) / (times[-1] - times[0] + 1e-9)
            actual_fs = float(np.clip(actual_fs, 1.0, 20.0))
        else:
            actual_fs = SAMPLE_RATE
        
        # Detrend (remove mean drift)
        rssi_detrended = rssi_array - np.mean(rssi_array)
        
        # Motion score: variance of RSSI
        variance = float(np.var(rssi_detrended))
        motion = float(np.clip(variance / 5.0, 0.0, 1.0))
        
        # Presence: present if RSSI strong enough and some variance
        presence = bool(np.mean(rssi_array) > -80 and variance > 0.1)
        
        # Signal quality: based on RSSI strength
        mean_rssi = float(np.mean(rssi_array))
        sig_quality = float(np.clip((mean_rssi + 100) / 60.0, 0.0, 1.0))
        
        # Breathing: bandpass 0.1–0.5 Hz (6–30 BPM)
        breath_signal = bandpass_filter(rssi_detrended, 0.1, 0.5, actual_fs)
        breath_bpm, breath_conf = estimate_bpm(breath_signal, actual_fs, 0.1, 0.5)
        
        # Heart rate: bandpass 0.8–2.0 Hz (48–120 BPM)
        heart_signal = bandpass_filter(rssi_detrended, 0.8, 2.0, actual_fs)
        heart_bpm, heart_conf = estimate_bpm(heart_signal, actual_fs, 0.8, 2.0)
        
        # Clamp to physiological ranges
        breath_bpm = float(np.clip(breath_bpm, 6.0, 30.0)) if breath_conf > 0.1 else 0.0
        heart_bpm  = float(np.clip(heart_bpm, 40.0, 120.0)) if heart_conf > 0.1 else 0.0
        
        with state.lock:
            state.breathing_bpm   = breath_bpm
            state.heart_rate_bpm  = heart_bpm
            state.breathing_conf  = breath_conf
            state.heart_conf      = heart_conf
            state.motion_score    = motion
            state.presence        = presence
            state.signal_quality  = sig_quality

# ── MediaPipe Camera Thread ───────────────────────────────────────────────────
def camera_thread():
    """Capture webcam frames and extract pose keypoints using MediaPipe."""
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[Camera] Failed to open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
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
            
            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            
            keypoints = []
            pose_conf = 0.0
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract COCO 17 keypoints from MediaPipe landmarks
                confidences = []
                for coco_idx, mp_idx in enumerate(MP_TO_COCO):
                    lm = landmarks[mp_idx]
                    kp = {
                        "name":       KEYPOINT_NAMES[coco_idx],
                        "x":          float(lm.x * 640),   # pixel coords
                        "y":          float(lm.y * 480),
                        "z":          float(lm.z * 100),    # depth estimate
                        "confidence": float(lm.visibility),
                    }
                    keypoints.append(kp)
                    confidences.append(lm.visibility)
                
                pose_conf = float(np.mean(confidences))
                
                # Draw on frame if showing camera
                if SHOW_CAMERA:
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2),
                    )
            
            with state.lock:
                state.keypoints = keypoints
                state.pose_confidence = pose_conf
                state.camera_active = True
            
            if SHOW_CAMERA:
                # Overlay vital signs on camera feed
                with state.lock:
                    br = state.breathing_bpm
                    hr = state.heart_rate_bpm
                    pres = state.presence
                    rssi = state.last_rssi
                
                overlay_color = (0, 255, 100)
                cv2.putText(frame, f"Breathing: {br:.1f} BPM", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
                cv2.putText(frame, f"Heart Rate: {hr:.1f} BPM", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                cv2.putText(frame, f"RSSI: {rssi:.0f} dBm", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                cv2.putText(frame, f"Present: {'YES' if pres else 'NO'}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0) if pres else (0, 0, 255), 2)
                if pose_conf > 0:
                    cv2.putText(frame, f"Pose: {pose_conf*100:.0f}%", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("RuView Bridge — Press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()

# ── WebSocket Sender ──────────────────────────────────────────────────────────
async def websocket_sender():
    """Send fused data to Rust server WebSocket."""
    print(f"[WS] Connecting to {SERVER_WS_URL}")
    
    while True:
        try:
            async with websockets.connect(SERVER_WS_URL, ping_interval=20) as ws:
                print("[WS] Connected to sensing server!")
                
                while True:
                    await asyncio.sleep(0.1)  # 10 FPS
                    
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
                    
                    # Determine motion level
                    if motion > 0.6:
                        motion_level = "active"
                    elif motion > 0.2:
                        motion_level = "present_moving"
                    elif presence:
                        motion_level = "present_still"
                    else:
                        motion_level = "absent"
                    
                    # Overall confidence
                    confidence = float(np.clip(
                        0.3 + sig_qual * 0.3 + pose_conf * 0.4, 0.0, 1.0
                    ))
                    
                    # Build persons list from MediaPipe keypoints
                    persons = []
                    if keypoints and pose_conf > 0.3:
                        persons = [{
                            "id": 1,
                            "confidence": float(pose_conf),
                            "keypoints": keypoints,
                            "bbox": {
                                "x": float(min(k["x"] for k in keypoints) - 10),
                                "y": float(min(k["y"] for k in keypoints) - 10),
                                "width": float(max(k["x"] for k in keypoints) -
                                               min(k["x"] for k in keypoints) + 20),
                                "height": float(max(k["y"] for k in keypoints) -
                                                min(k["y"] for k in keypoints) + 20),
                            },
                            "zone": "zone_1",
                            "position": [0.0, 0.0, 0.0],
                            "motion_score": int(motion * 100),
                            "pose": "standing",
                        }]
                    
                    # Build sensing update message
                    msg = {
                        "type": "sensing_update",
                        "timestamp": time.time(),
                        "source": "python_bridge",
                        "tick": tick,
                        "nodes": [{
                            "node_id": 1,
                            "rssi_dbm": rssi,
                            "position": [0.0, 0.0, 0.0],
                            "amplitude": [abs(rssi)] * 56,
                            "subcarrier_count": 56,
                        }],
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
                            "confidence": confidence,
                        },
                        "vital_signs": {
                            "breathing_rate_bpm": br_bpm if br_bpm > 0 else None,
                            "heart_rate_bpm": hr_bpm if hr_bpm > 0 else None,
                            "breathing_confidence": br_conf,
                            "heartbeat_confidence": hr_conf,
                            "signal_quality": sig_qual,
                        },
                        "signal_field": {
                            "grid_size": [20, 1, 20],
                            "values": [
                                float(np.clip(
                                    motion * np.exp(-((i//20 - 10)**2 + (i%20 - 10)**2) / 30.0),
                                    0, 1
                                ))
                                for i in range(400)
                            ],
                        },
                        "persons": persons,
                        "estimated_persons": len(persons),
                        "pose_source": "mediapipe_camera",
                    }
                    
                    await ws.send(json.dumps(msg))
                    
        except Exception as e:
            print(f"[WS] Disconnected: {e}, retrying in 3s...")
            await asyncio.sleep(3)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  RuView Python Bridge")
    print("  ESP32 RSSI + Webcam → Observatory 3D")
    print("=" * 55)
    print(f"  UDP port:    {UDP_PORT} (ESP32 data)")
    print(f"  WebSocket:   {SERVER_WS_URL}")
    print(f"  Camera:      index {CAMERA_INDEX}")
    print(f"  Show camera: {SHOW_CAMERA}")
    print("=" * 55)
    print()
    print("IMPORTANT: Change ESP32 SERVER_PORT to 5006!")
    print("  (so ESP32 sends to Python bridge, not directly to Rust)")
    print()

    # Start UDP listener thread
    t_udp = threading.Thread(target=udp_listener, daemon=True)
    t_udp.start()

    # Start vital signs computer thread
    t_vitals = threading.Thread(target=compute_vitals, daemon=True)
    t_vitals.start()

    # Start camera thread
    t_camera = threading.Thread(target=camera_thread, daemon=True)
    t_camera.start()

    # Run WebSocket sender (async)
    print("[Main] All threads started. Press Q in camera window to quit.")
    asyncio.run(websocket_sender())

if __name__ == "__main__":
    main()
