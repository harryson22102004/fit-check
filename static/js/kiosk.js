const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("cam-status");
const feed = document.getElementById("feed");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnRetry = document.getElementById("btn-retry");

let scanning = false;
let timer = null;
let lastLandmarks = null;
let motionHits = 0;
let blinkHits = 0;
let lastEAR = null;

function log(html, cls) {
  const empty = feed.querySelector(".muted");
  if (empty) empty.remove();
  const li = document.createElement("li");
  if (cls) li.className = cls;
  li.innerHTML = html;
  feed.prepend(li);
}

function livenessFrom(det) {
  const pts = det.landmarks.positions;
  const motion = Camera.landmarkMotion(lastLandmarks, pts);
  lastLandmarks = pts.map((p) => ({ x: p.x, y: p.y }));
  if (motion > 0.35 && motion < 18) motionHits += 1;
  const ear = (Camera.eyeAspect(det.landmarks, true) + Camera.eyeAspect(det.landmarks, false)) / 2;
  if (lastEAR != null && lastEAR > 0.22 && ear < 0.18) blinkHits += 1;
  lastEAR = ear;
  // Printed photos are almost still. Require a little live motion, or a blink.
  return motionHits >= 2 || blinkHits >= 1 || motion > 0.8;
}

async function loadHistory() {
  try {
    const res = await fetch("/api/events?limit=80");
    const data = await res.json();
    const events = data.events || [];
    if (!events.length) return;
    feed.innerHTML = "";
    events.forEach((e) => {
      const li = document.createElement("li");
      if (e.kind === "already") li.className = "muted";
      li.innerHTML = `<strong>${e.name || e.student_id || "Scan"}</strong> · ${e.message}<div class="muted">${e.created_at || ""}</div>`;
      feed.appendChild(li);
    });
  } catch (err) {
    console.error(err);
  }
}

async function boot() {
  await loadHistory();
  try {
    statusEl.textContent = "Loading face models…";
    await Camera.start(video, statusEl);
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Camera blocked";
    log(
      "Allow the camera in the browser, then Retry. This page needs a real webcam on HTTPS or localhost.",
      "muted"
    );
  }
}

async function tick() {
  if (!scanning || video.readyState < 2) return;
  const dets = await Camera.detect(video);
  if (!dets.length) {
    Camera.drawDetections(overlay, video, dets, []);
    statusEl.textContent = "Waiting for faces…";
    return;
  }

  const live = dets.some((d) => livenessFrom(d));
  const descriptors = dets.map((d) => Array.from(d.descriptor));
  let matches = [];
  try {
    const res = await fetch("/api/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ descriptors, liveness_ok: live }),
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = live ? "Recognize error" : "Hold still, then move slightly (liveness)";
      Camera.drawDetections(
        overlay,
        video,
        dets,
        dets.map(() => ({ ok: false, text: "liveness" }))
      );
      return;
    }
    matches = data.matches || [];
    (data.logged || []).forEach((row) => {
      log(
        `<strong>${row.name}</strong> · ${row.student_id} marked <em>${row.status}</em> at ${String(row.time_in).slice(11, 19)}`
      );
    });
    if ((data.logged || []).length && window.Persist) Persist.syncFromServer();
    matches
      .filter((m) => m.matched && m.already_logged)
      .forEach((m) => {
        log(`${m.name} already logged today — skipped extra pass.`, "muted");
      });
  } catch (err) {
    console.error(err);
  }

  Camera.drawDetections(
    overlay,
    video,
    dets,
    dets.map((d, i) => {
      const m = matches.find((x) => x.face_index === i);
      if (m?.matched) return { ok: true, text: m.name };
      return { ok: false, text: "unknown" };
    })
  );
  const named = matches.filter((m) => m.matched).map((m) => m.name);
  statusEl.textContent = named.length ? named.join(", ") : `${dets.length} face(s) — no roster match`;
}

btnStart.addEventListener("click", () => {
  scanning = true;
  btnStart.disabled = true;
  btnStop.disabled = false;
  motionHits = 0;
  blinkHits = 0;
  lastLandmarks = null;
  statusEl.textContent = "Scanning every few seconds";
  tick();
  timer = setInterval(tick, 2200);
});

btnStop.addEventListener("click", () => {
  scanning = false;
  btnStart.disabled = false;
  btnStop.disabled = true;
  if (timer) clearInterval(timer);
  statusEl.textContent = "Paused";
});

btnRetry.addEventListener("click", () => {
  scanning = false;
  if (timer) clearInterval(timer);
  btnStart.disabled = false;
  btnStop.disabled = true;
  Camera.stop();
  boot();
});

boot();
