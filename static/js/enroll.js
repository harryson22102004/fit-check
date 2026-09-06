const TARGET = 8;
const captures = [];

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("cam-status");
const hint = document.getElementById("hint");
const thumbs = document.getElementById("thumbs");
const count = document.getElementById("count");
const btnCapture = document.getElementById("btn-capture");
const btnRetry = document.getElementById("btn-retry");
const btnSave = document.getElementById("btn-save");
const form = document.getElementById("enroll-form");
const formMsg = document.getElementById("form-msg");

let loop = null;

function setMsg(text, ok) {
  formMsg.textContent = text;
  formMsg.className = "form-msg " + (ok ? "ok" : "err");
}

function renderThumbs() {
  thumbs.innerHTML = "";
  captures.forEach((c) => {
    const img = document.createElement("img");
    img.src = c.photo;
    thumbs.appendChild(img);
  });
  count.textContent = captures.length + " / " + TARGET + " photos";
  btnSave.disabled = captures.length < 5;
}

async function boot() {
  try {
    statusEl.textContent = "Loading face models…";
    await Camera.start(video, statusEl);
    btnCapture.disabled = false;
    hint.textContent = "Face the camera. Capture at least 5 angles (8 is better).";
    loop = setInterval(async () => {
      if (video.readyState < 2) return;
      const dets = await Camera.detect(video);
      Camera.drawDetections(
        overlay,
        video,
        dets,
        dets.map((d) => ({
          ok: d.detection.box.width > 90,
          text: dets.length === 1 ? "ready" : "one face only",
        }))
      );
      if (dets.length === 1) statusEl.textContent = "Face locked — capture when ready";
      else if (dets.length === 0) statusEl.textContent = "No face in frame";
      else statusEl.textContent = "Only one student per enrollment shot";
    }, 280);
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Camera blocked";
    hint.textContent =
      "Allow camera access in the browser address bar, then hit Retry camera. Use Chrome or Edge on HTTPS or localhost.";
    btnCapture.disabled = true;
  }
}

btnRetry.addEventListener("click", () => {
  if (loop) clearInterval(loop);
  Camera.stop();
  boot();
});

document.getElementById("file-photos").addEventListener("change", async (ev) => {
  const files = Array.from(ev.target.files || []);
  if (!files.length) return;
  await Camera.loadModels();
  for (const file of files) {
    if (captures.length >= 10) break;
    const url = URL.createObjectURL(file);
    const img = await new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = reject;
      el.src = url;
    });
    const det = await Camera.detectImage(img);
    URL.revokeObjectURL(url);
    if (!det) {
      setMsg("No face found in " + file.name, false);
      continue;
    }
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d").drawImage(img, 0, 0);
    captures.push({
      descriptor: Array.from(det.descriptor),
      photo: canvas.toDataURL("image/jpeg", 0.86),
    });
  }
  renderThumbs();
  setMsg("Added file photos. Total " + captures.length + ".", true);
  ev.target.value = "";
});

btnCapture.addEventListener("click", async () => {
  if (captures.length >= 10) {
    setMsg("Maximum 10 photos.", false);
    return;
  }
  const dets = await Camera.detect(video);
  if (dets.length !== 1) {
    setMsg("Need exactly one clear face in the frame.", false);
    return;
  }
  const box = dets[0].detection.box;
  if (box.width < 80 || box.height < 80) {
    setMsg("Move closer so the face fills more of the frame.", false);
    return;
  }
  const descriptor = Array.from(dets[0].descriptor);
  const photo = Camera.snapshotJpeg(video);
  captures.push({ descriptor, photo });
  renderThumbs();
  setMsg("Captured angle " + captures.length + ".", true);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (captures.length < 5) {
    setMsg("Capture at least 5 photos first.", false);
    return;
  }
  btnSave.disabled = true;
  setMsg("Saving embeddings…", true);
  const body = {
    student_id: form.student_id.value,
    name: form.name.value,
    class_section: form.class_section.value,
    guardian_whatsapp_number: form.guardian.value || null,
    embeddings: captures.map((c) => ({
      descriptor: c.descriptor,
      photo_jpeg_base64: c.photo,
    })),
  };
  try {
    const res = await fetch("/api/enroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Save failed");
    setMsg("Enrolled " + data.student_id + ". You can enroll another student or open the kiosk.", true);
    captures.length = 0;
    renderThumbs();
    form.reset();
  } catch (err) {
    setMsg(String(err.message || err), false);
  } finally {
    btnSave.disabled = captures.length < 5;
  }
});

boot();
