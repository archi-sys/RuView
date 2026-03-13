/**
 * AuraTrack Firebase Pusher
 * ==========================
 * Drop into: C:\Users\KIIT\Documents\GitHub\RuView\ui\firebase-pusher.js
 * Add to index.html before </body>:
 *   <script src="firebase-pusher.js"></script>
 *
 * Reads from Rust server API every 2s → pushes to Firebase REST API
 * No Python, no extra dependencies.
 */

(function () {
  'use strict';

  const RUST_API    = 'http://localhost:3000';
  const FIREBASE    = 'https://auratrack-b29d0-default-rtdb.firebaseio.com';
  const PUSH_EVERY  = 2000;   // ms — live data
  const HIST_EVERY  = 10000;  // ms — history entry

  let lastHistPush  = 0;

  // ── Firebase REST PUT ────────────────────────────────────────────────────
  async function fbPut(path, data) {
    try {
      await fetch(`${FIREBASE}/${path}.json`, {
        method:  'PUT',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(data),
      });
    } catch (e) {
      console.warn('[Firebase] PUT failed:', e.message);
    }
  }

  // ── Firebase REST POST (push with unique key) ────────────────────────────
  async function fbPost(path, data) {
    try {
      await fetch(`${FIREBASE}/${path}.json`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(data),
      });
    } catch (e) {
      console.warn('[Firebase] POST failed:', e.message);
    }
  }

  // ── Fetch from Rust server ───────────────────────────────────────────────
  async function fetchRust(endpoint) {
    const res = await fetch(`${RUST_API}${endpoint}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  // ── Main push loop ───────────────────────────────────────────────────────
  async function pushToFirebase() {
    try {
      // Fetch latest sensing data + vitals in parallel
      const [latest, vitals, health] = await Promise.all([
        fetchRust('/api/v1/sensing/latest').catch(() => null),
        fetchRust('/api/v1/vital-signs').catch(() => null),
        fetchRust('/health').catch(() => null),
      ]);

      const now     = Date.now() / 1000;
      const nowStr  = new Date().toLocaleString();

      // ── Build live payload ──────────────────────────────────────────────
      const live = {
        timestamp:      now,
        timestamp_str:  nowStr,
        source:         'rust_server',
        // From health
        status:         health?.status        ?? 'unknown',
        esp32_connected: health?.source === 'esp32',
        // From latest sensing
        presence:       latest?.classification?.presence      ?? false,
        motion_level:   latest?.classification?.motion_level  ?? 'unknown',
        confidence:     latest?.classification?.confidence    ?? 0,
        persons_count:  latest?.estimated_persons             ?? 0,
        // From vital signs
        breathing_bpm:  vitals?.breathing_rate_bpm            ?? 0,
        heart_rate_bpm: vitals?.heart_rate_bpm                ?? 0,
        breathing_conf: vitals?.breathing_confidence          ?? 0,
        heart_conf:     vitals?.heartbeat_confidence          ?? 0,
        signal_quality: vitals?.signal_quality                ?? 0,
        // RSSI
        rssi_dbm:       latest?.nodes?.[0]?.rssi_dbm          ?? 0,
      };

      // Push live data (overwrites every 2s)
      await fbPut('auratrack/live', live);

      // Push status
      await fbPut('auratrack/status', {
        online:      true,
        last_seen:   nowStr,
        version:     '3.0',
        source:      'browser_pusher',
      });

      console.log(
        `[Firebase] ✓ pushed — presence=${live.presence}` +
        ` BR=${live.breathing_bpm.toFixed(1)}` +
        ` HR=${live.heart_rate_bpm.toFixed(1)}` +
        ` persons=${live.persons_count}`
      );

      // Push history every 10s
      if (now - lastHistPush >= HIST_EVERY / 1000) {
        lastHistPush = now;
        await fbPost('auratrack/history', {
          timestamp:      now,
          timestamp_str:  nowStr,
          presence:       live.presence,
          motion_level:   live.motion_level,
          breathing_bpm:  live.breathing_bpm,
          heart_rate_bpm: live.heart_rate_bpm,
          persons_count:  live.persons_count,
          rssi_dbm:       live.rssi_dbm,
        });
        console.log('[Firebase] ✓ history entry saved');
      }

    } catch (e) {
      console.warn('[Firebase] Push cycle error:', e.message);
    }
  }

  // ── Boot ─────────────────────────────────────────────────────────────────
  function boot() {
    console.log('[Firebase Pusher] Started — pushing every 2s');
    console.log('[Firebase Pusher] Live data:', `${FIREBASE}/auratrack/live.json`);

    // Push immediately then on interval
    pushToFirebase();
    setInterval(pushToFirebase, PUSH_EVERY);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

})();
