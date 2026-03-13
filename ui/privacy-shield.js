/**
 * AuraTrack Privacy Shield
 * =========================
 * Drop this file into:
 *   C:\Users\KIIT\Documents\GitHub\RuView\ui\privacy-shield.js
 *
 * Add ONE line to index.html just before </body>:
 *   <script src="privacy-shield.js"></script>
 *
 * What it does:
 *   - Polls Firebase every 500ms for privacy_mode flag
 *   - When ON  → overlays shield UI + injects fake persons/vitals
 *   - When OFF → removes overlay, real data resumes normally
 *   - Fake data: 3-5 animated walking persons, random vitals
 */

(function () {
  'use strict';

  const FIREBASE_URL =
    'https://auratrack-b29d0-default-rtdb.firebaseio.com/auratrack/privacy_mode.json';

  const POLL_MS      = 500;
  const FAKE_MIN     = 3;
  const FAKE_MAX     = 5;

  // ── State ──────────────────────────────────────────────────────────────
  let privacyActive  = false;
  let overlayEl      = null;
  let fakeInterval   = null;
  let animTime       = 0;

  // Each fake person has its own walking phase & position
  const fakePersons = Array.from({ length: FAKE_MAX }, (_, i) => ({
    id:          i + 1,
    cx:          100 + i * 110 + Math.random() * 30,
    phase:       i * 1.37 + Math.random() * Math.PI * 2,
    walkDir:     Math.random() > 0.5 ? 1 : -1,
    walkX:       100 + i * 110,
    speed:       0.4 + Math.random() * 0.5,
    breathBpm:   12 + Math.random() * 8,
    heartBpm:    60 + Math.random() * 40,
  }));

  // ── Biomechanical walking keypoints ───────────────────────────────────
  function getWalkingKeypoints(person, t) {
    const { cx, phase, speed } = person;
    const amp    = 0.6 + speed * 0.4;
    const gHz    = 0.8 + speed * 0.5;
    const bHz    = person.breathBpm / 60;
    const sin    = (f, o = 0) => Math.sin(2 * Math.PI * f * t + phase + o);

    // Drift walking position
    person.walkX += person.walkDir * speed * 0.5;
    if (person.walkX > cx + 120) person.walkDir = -1;
    if (person.walkX < cx - 120) person.walkDir =  1;
    const wx = person.walkX;
    const cy = 80;

    const bob      = amp * 4  * Math.abs(sin(gHz * 2));
    const hipSway  = amp * 8  * sin(gHz);
    const shRot    = amp * 6  * sin(gHz, Math.PI);
    const breath   = amp * 3  * sin(bHz);
    const headBob  = amp * 2.5 * sin(gHz * 2);

    const rootY  = cy + 118 - bob;
    const spY    = rootY - 90 - breath * 0.5;
    const noseY  = spY - 28 - 18 + headBob;
    const noseX  = wx + shRot * 0.3;

    const lArmSw = amp * 28 * sin(gHz, Math.PI);
    const rArmSw = amp * 28 * sin(gHz);
    const lLegSw = amp * 32 * sin(gHz);
    const rLegSw = amp * 32 * sin(gHz, Math.PI);

    return [
      // nose, eyes, ears
      { x: noseX,              y: noseY },
      { x: noseX - 6,          y: noseY - 4 },
      { x: noseX + 6,          y: noseY - 4 },
      { x: noseX - 12,         y: noseY },
      { x: noseX + 12,         y: noseY },
      // shoulders
      { x: wx - 30 + shRot,    y: spY + 10 },
      { x: wx + 30 - shRot,    y: spY + 10 },
      // elbows
      { x: wx - 38 + lArmSw * 0.6, y: spY + 55 },
      { x: wx + 38 - rArmSw * 0.6, y: spY + 55 },
      // wrists
      { x: wx - 42 + lArmSw * 0.9, y: spY + 95 },
      { x: wx + 42 - rArmSw * 0.9, y: spY + 95 },
      // hips
      { x: wx - 18 - hipSway * 0.3, y: rootY + 5 },
      { x: wx + 18 + hipSway * 0.3, y: rootY - 5 },
      // knees
      { x: wx - 20 + lLegSw * 0.5,  y: rootY + 70 - Math.abs(lLegSw) * 0.4 },
      { x: wx + 20 + rLegSw * 0.5,  y: rootY + 70 - Math.abs(rLegSw) * 0.4 },
      // ankles
      { x: wx - 20 + lLegSw * 0.4,  y: rootY + 135 + Math.abs(lLegSw) * 0.2 },
      { x: wx + 20 + rLegSw * 0.4,  y: rootY + 135 + Math.abs(rLegSw) * 0.2 },
    ].map((pt, i) => ({
      name: ['nose','left_eye','right_eye','left_ear','right_ear',
             'left_shoulder','right_shoulder','left_elbow','right_elbow',
             'left_wrist','right_wrist','left_hip','right_hip',
             'left_knee','right_knee','left_ankle','right_ankle'][i],
      x: pt.x, y: pt.y, z: 0,
      confidence: 0.85 + Math.random() * 0.14,
    }));
  }

  // ── Build fake WS frame ────────────────────────────────────────────────
  function buildFakeFrame() {
    animTime += 0.1;
    const n = FAKE_MIN + Math.floor(Math.random() * (FAKE_MAX - FAKE_MIN + 1));
    const persons = fakePersons.slice(0, n).map(p => {
      const kps = getWalkingKeypoints(p, animTime);
      const xs  = kps.map(k => k.x);
      const ys  = kps.map(k => k.y);
      return {
        id:           p.id,
        confidence:   0.80 + Math.random() * 0.17,
        keypoints:    kps,
        bbox: {
          x:      Math.min(...xs) - 15,
          y:      Math.min(...ys) - 15,
          width:  Math.max(...xs) - Math.min(...xs) + 30,
          height: Math.max(...ys) - Math.min(...ys) + 30,
        },
        zone:         'zone_' + p.id,
        position:     [(p.walkX - 320) / 100, 0, 0],
        motion_score: Math.floor(50 + Math.random() * 50),
        pose:         'walking',
        vital_signs: {
          breathing_rate_bpm:   +(p.breathBpm + Math.random() * 2 - 1).toFixed(1),
          heart_rate_bpm:       +(p.heartBpm  + Math.random() * 4 - 2).toFixed(1),
          breathing_confidence: +(0.7  + Math.random() * 0.25).toFixed(2),
          heartbeat_confidence: +(0.65 + Math.random() * 0.30).toFixed(2),
        },
      };
    });

    return {
      type:              'sensing_update',
      timestamp:         Date.now() / 1000,
      source:            'privacy_shield',
      privacy_mode:      true,
      estimated_persons: n,
      persons,
      classification: {
        motion_level: 'active',
        presence:     true,
        confidence:   0.90 + Math.random() * 0.09,
      },
      vital_signs: {
        breathing_rate_bpm:   +(12 + Math.random() * 8).toFixed(1),
        heart_rate_bpm:       +(60 + Math.random() * 40).toFixed(1),
        breathing_confidence: 0.88,
        heartbeat_confidence: 0.82,
        signal_quality:       0.91,
      },
      nodes: [{
        node_id:          1,
        rssi_dbm:         -45 - Math.random() * 35,
        position:         [0, 0, 0],
        amplitude:        Array.from({ length: 56 }, () => Math.random()),
        subcarrier_count: 56,
      }],
      features: {
        mean_rssi:            -55 - Math.random() * 20,
        variance:             2 + Math.random() * 6,
        motion_band_power:    8 + Math.random() * 12,
        breathing_band_power: 3 + Math.random() * 5,
        dominant_freq_hz:     0.2 + Math.random() * 0.3,
        change_points:        Math.floor(8 + Math.random() * 22),
        spectral_power:       80 + Math.random() * 70,
      },
      signal_field: {
        grid_size: [20, 1, 20],
        values:    Array.from({ length: 400 }, () => Math.random()),
      },
    };
  }

  // ── Intercept WebSocket to inject fake data ────────────────────────────
  const _WebSocket = window.WebSocket;
  const hookedSockets = new Set();

  function hookSocket(ws) {
    if (hookedSockets.has(ws)) return;
    hookedSockets.add(ws);

    const origAddListener = ws.addEventListener.bind(ws);
    const messageHandlers = [];

    // Collect message handlers
    origAddListener('message', (e) => {
      // If privacy is active, drop real message and we'll send fake ones
      // (fake injection happens via fakeInterval below)
    });

    // Expose so we can fire fake events
    ws._privacyHooked = true;
    ws._messageHandlers = messageHandlers;
  }

  // ── Overlay UI ─────────────────────────────────────────────────────────
  function createOverlay() {
    if (overlayEl) return;
    overlayEl = document.createElement('div');
    overlayEl.id = 'privacy-shield-overlay';
    overlayEl.innerHTML = `
      <div style="
        position: fixed; top: 0; left: 0; right: 0; z-index: 99999;
        background: linear-gradient(90deg, #1a0a2e, #0d1b2a);
        border-bottom: 2px solid #9333ea;
        padding: 8px 20px;
        display: flex; align-items: center; gap: 12px;
        font-family: monospace; font-size: 13px;
        box-shadow: 0 2px 20px rgba(147,51,234,0.4);
      ">
        <span style="font-size:18px;">🔒</span>
        <span style="color:#c084fc; font-weight:bold; letter-spacing:1px;">
          PRIVACY SHIELD ACTIVE
        </span>
        <span style="color:#7c3aed; margin-left:8px;">
          — displaying decoy data —
        </span>
        <span id="ps-person-count" style="
          margin-left:auto; color:#a855f7;
          background: rgba(147,51,234,0.2);
          padding: 2px 10px; border-radius:8px;
          border: 1px solid rgba(147,51,234,0.4);
        ">● 0 persons</span>
      </div>
    `;
    document.body.appendChild(overlayEl);
    // Push page content down
    document.body.style.marginTop = '44px';
  }

  function removeOverlay() {
    if (overlayEl) {
      overlayEl.remove();
      overlayEl = null;
      document.body.style.marginTop = '';
    }
  }

  function updatePersonCount(n) {
    const el = document.getElementById('ps-person-count');
    if (el) el.textContent = `● ${n} persons`;
  }

  // ── Inject fake data into the page's WebSocket handlers ───────────────
  function startFakeInjection() {
    if (fakeInterval) return;
    console.log('[PrivacyShield] 🔒 Starting fake data injection');

    fakeInterval = setInterval(() => {
      const frame = buildFakeFrame();
      updatePersonCount(frame.estimated_persons);

      // Fire a synthetic MessageEvent on ALL active WebSocket instances
      // The pose.service.js handlePoseMessage will receive it normally
      if (window._privacyShieldTargets) {
        window._privacyShieldTargets.forEach(handler => {
          try { handler({ data: JSON.stringify(frame) }); } catch(e) {}
        });
      }

      // Also try dispatching on wsService if accessible
      try {
        if (window.wsService && window.wsService._connections) {
          window.wsService._connections.forEach(conn => {
            if (conn && conn.onmessage) {
              conn.onmessage({ data: JSON.stringify(frame) });
            }
          });
        }
      } catch(e) {}

      // Patch poseService directly if available
      try {
        if (window.poseService) {
          window.poseService.handlePoseMessage(frame);
        }
      } catch(e) {}

    }, 100); // 10 FPS
  }

  function stopFakeInjection() {
    if (fakeInterval) {
      clearInterval(fakeInterval);
      fakeInterval = null;
      console.log('[PrivacyShield] 🔓 Stopped fake data injection');
    }
  }

  // ── Privacy state change handler ───────────────────────────────────────
  function applyPrivacyState(active) {
    if (active === privacyActive) return;
    privacyActive = active;

    if (active) {
      console.log('%c🔒 PRIVACY SHIELD ON — injecting fake data', 
                  'color:#c084fc; font-weight:bold; font-size:14px');
      createOverlay();
      startFakeInjection();
    } else {
      console.log('%c🔓 PRIVACY SHIELD OFF — real data restored',
                  'color:#4ade80; font-weight:bold; font-size:14px');
      removeOverlay();
      stopFakeInjection();
    }
  }

  // ── Poll Firebase ──────────────────────────────────────────────────────
  async function pollFirebase() {
    try {
      const res = await fetch(FIREBASE_URL);
      if (res.ok) {
        const val = await res.json();
        applyPrivacyState(val === true);
      }
    } catch (e) {
      // Network error — keep current state
    }
  }

  // ── Expose poseService handler for injection ───────────────────────────
  // Wait for app to load then grab poseService reference
  function grabPoseService() {
    // Try common global names
    const candidates = ['poseService', 'app', 'liveDemo'];
    for (const name of candidates) {
      if (window[name]?.handlePoseMessage) {
        window.poseService = window[name];
        console.log(`[PrivacyShield] Hooked into window.${name}`);
        return true;
      }
      if (window[name]?.poseService?.handlePoseMessage) {
        window.poseService = window[name].poseService;
        console.log(`[PrivacyShield] Hooked into window.${name}.poseService`);
        return true;
      }
    }
    return false;
  }

  // ── Boot ───────────────────────────────────────────────────────────────
  function boot() {
    console.log('[PrivacyShield] Loaded — polling Firebase every 500ms');
    console.log('[PrivacyShield] Firebase:', FIREBASE_URL);

    // Try to grab poseService, retry if not ready yet
    if (!grabPoseService()) {
      const retryInterval = setInterval(() => {
        if (grabPoseService()) clearInterval(retryInterval);
      }, 500);
    }

    // Start polling
    pollFirebase();
    setInterval(pollFirebase, POLL_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

})();
