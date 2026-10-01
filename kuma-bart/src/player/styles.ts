// Static shadow-DOM markup and styles for <kuma-player> (see KumaPlayer.ts).

export const ICON_PLAY = `<svg xmlns="http://www.w3.org/2000/svg" width="10" height="12" viewBox="0 0 10 12" fill="currentColor" aria-hidden="true"><path d="M0 0v12l10-6z"/></svg>`;
export const ICON_PAUSE = `<svg xmlns="http://www.w3.org/2000/svg" width="11" height="12" viewBox="0 0 11 12" fill="currentColor" aria-hidden="true"><rect width="4" height="12" rx="1"/><rect x="7" width="4" height="12" rx="1"/></svg>`;
const ICON_GEAR = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"/><path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.469l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z"/></svg>`;

// ── Styles ─────────────────────────────────────────────────────────────────────

export const STYLES = `
:host {
  display: block;
  --kp-accent: #ffffff;
  --kp-radius: 12px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; }

.kp {
  background: #111113;
  border-radius: var(--kp-radius);
  overflow: hidden;
}

/* ── Viewport ─────────────────────────────────────────────────────────────── */
.kp-viewport {
  position: relative;
  background: repeating-conic-gradient(#1d1f24 0% 25%, #16181c 0% 50%) 50% / 20px 20px;
  min-height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
}

canvas {
  display: block;
  max-width: 100%;
  height: auto;
}

.kp-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  color: rgba(255,255,255,0.2);
  letter-spacing: 0.05em;
  pointer-events: none;
}

.kp-status {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  padding: 20px 12px 7px;
  font-size: 10.5px;
  font-family: ui-monospace, 'Cascadia Code', 'SF Mono', monospace;
  color: rgba(255,255,255,0.35);
  background: linear-gradient(to top, rgba(0,0,0,0.5), transparent);
  pointer-events: none;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Control bar ──────────────────────────────────────────────────────────── */
.kp-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 10px;
  height: 46px;
  background: #0d0d10;
  border-top: 1px solid rgba(255,255,255,0.045);
}

.kp-play {
  flex-shrink: 0;
  width: 30px;
  height: 30px;
  border: none;
  background: rgba(255,255,255,0.08);
  border-radius: 50%;
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 120ms ease;
  padding: 0;
}
.kp-play:hover:not(:disabled) { background: rgba(255,255,255,0.15); }
.kp-play:active:not(:disabled) { background: rgba(255,255,255,0.22); }
.kp-play:disabled { opacity: 0.25; cursor: default; }

/* ── Progress bar ─────────────────────────────────────────────────────────── */
.kp-progress {
  flex: 1;
  position: relative;
  height: 20px;
  display: flex;
  align-items: center;
  cursor: pointer;
}

.kp-progress::before {
  content: '';
  position: absolute;
  left: 0; right: 0;
  height: 3px;
  background: rgba(255,255,255,0.1);
  border-radius: 99px;
  transition: height 150ms ease;
  pointer-events: none;
}

.kp-progress:hover::before { height: 5px; }

.kp-progress-fill {
  position: absolute;
  top: 50%;
  left: 0;
  transform: translateY(-50%);
  height: 3px;
  background: var(--kp-accent);
  border-radius: 99px;
  width: 0%;
  pointer-events: none;
  transition: height 150ms ease;
}

.kp-progress:hover .kp-progress-fill { height: 5px; }

.kp-scrubber {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  opacity: 0;
  cursor: pointer;
  z-index: 1;
}

.kp-time {
  flex-shrink: 0;
  min-width: 3.2ch;
  text-align: right;
  font-family: ui-monospace, 'Cascadia Code', 'SF Mono', monospace;
  font-size: 11.5px;
  font-variant-numeric: tabular-nums;
  color: rgba(255,255,255,0.38);
  white-space: nowrap;
}

.kp-debug-btn {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border: none;
  background: transparent;
  color: rgba(255,255,255,0.28);
  cursor: pointer;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 120ms ease, background 120ms ease;
  padding: 0;
}
.kp-debug-btn:hover {
  color: rgba(255,255,255,0.6);
  background: rgba(255,255,255,0.07);
}
.kp-debug-btn.active {
  color: rgba(255,255,255,0.85);
  background: rgba(255,255,255,0.1);
}

/* ── Debug panel ──────────────────────────────────────────────────────────── */
.kp-debug {
  border-top: 1px solid rgba(255,255,255,0.055);
  padding: 12px 14px;
  background: #090909;
}
.kp-debug.hidden { display: none; }

:host([vanilla]) .kp-status,
:host([vanilla]) .kp-debug-btn,
:host([vanilla]) .kp-debug {
  display: none;
}

.kp-debug-row {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.kp-debug-row button {
  font: inherit;
  font-size: 12px;
  padding: 5px 12px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.04);
  color: rgba(255,255,255,0.6);
  border-radius: 6px;
  cursor: pointer;
  transition: background 120ms, color 120ms;
}
.kp-debug-row button:hover:not(:disabled) {
  background: rgba(255,255,255,0.09);
  color: rgba(255,255,255,0.9);
}
.kp-debug-row button:disabled { opacity: 0.3; cursor: default; }

details {
  border-top: 1px solid rgba(255,255,255,0.055);
  padding-top: 8px;
}
details summary {
  cursor: pointer;
  font-size: 11.5px;
  color: rgba(255,255,255,0.32);
  padding: 2px 0;
  list-style: none;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 5px;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
  content: '▸';
  display: inline-block;
  font-size: 9px;
  transition: transform 150ms;
}
details[open] summary::before { transform: rotate(90deg); }
details summary:hover { color: rgba(255,255,255,0.6); }

.kp-log {
  margin-top: 8px;
  padding: 10px 12px;
  background: #050507;
  color: rgba(72, 230, 120, 0.8);
  font-family: ui-monospace, 'Cascadia Code', 'SF Mono', monospace;
  font-size: 11px;
  line-height: 1.65;
  border-radius: 6px;
  max-height: 260px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
  scrollbar-width: thin;
  scrollbar-color: rgba(255,255,255,0.1) transparent;
}
`;

// ── Template ───────────────────────────────────────────────────────────────────

export const TEMPLATE = `
<div class="kp">
  <div class="kp-viewport">
    <canvas class="kp-canvas" style="display:none"></canvas>
    <div class="kp-placeholder">No model loaded</div>
    <div class="kp-status"></div>
  </div>
  <div class="kp-controls">
    <button class="kp-play" disabled aria-label="Play">${ICON_PLAY}</button>
    <div class="kp-progress" role="presentation">
      <div class="kp-progress-fill"></div>
      <input class="kp-scrubber" type="range" min="0" max="1" step="0.001" value="0"
             disabled aria-label="Scrub position"/>
    </div>
    <span class="kp-time">0.000</span>
    <button class="kp-debug-btn" aria-label="Toggle debug panel" title="Debug">${ICON_GEAR}</button>
  </div>
  <div class="kp-debug hidden">
    <div class="kp-debug-row">
      <button class="kp-verify" disabled>Verify</button>
      <button class="kp-profile" disabled>Profile</button>
    </div>
    <details>
      <summary>Log</summary>
      <pre class="kp-log">(idle)</pre>
    </details>
  </div>
</div>
`;
