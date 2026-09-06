/* Shared camera helpers for enrollment and the classroom kiosk. */

const Camera = (() => {
  let stream = null;
  let modelsReady = false;

  async function loadModels() {
    if (modelsReady) return;
    const url = "/static/models";
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(url),
      faceapi.nets.faceLandmark68Net.loadFromUri(url),
      faceapi.nets.faceRecognitionNet.loadFromUri(url),
    ]);
    modelsReady = true;
  }

  function detectorOptions() {
    return new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.45 });
  }

  async function start(video, statusEl) {
    await loadModels();
    stop();
    const constraints = {
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    };
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      // Some desktops reject facingMode: user; try a generic camera.
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    video.srcObject = stream;
    video.muted = true;
    video.setAttribute("playsinline", "true");
    await video.play();
    if (statusEl) statusEl.textContent = "Camera live";
    return stream;
  }

  function stop() {
    if (!stream) return;
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  function sizeOverlay(video, canvas) {
    const w = video.videoWidth || video.clientWidth;
    const h = video.videoHeight || video.clientHeight;
    if (!w || !h) return { w: 0, h: 0 };
    canvas.width = w;
    canvas.height = h;
    return { w, h };
  }

  async function detect(video) {
    return faceapi
      .detectAllFaces(video, detectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();
  }

  function drawDetections(canvas, video, detections, labels) {
    const ctx = canvas.getContext("2d");
    const { w, h } = sizeOverlay(video, canvas);
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    const resized = faceapi.resizeResults(detections, { width: w, height: h });
    resized.forEach((det, i) => {
      const box = det.detection.box;
      ctx.strokeStyle = labels?.[i]?.ok ? "#34d399" : "#fb923c";
      ctx.lineWidth = 3;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      const tag = labels?.[i]?.text || "face";
      ctx.font = "16px IBM Plex Sans, sans-serif";
      const tw = ctx.measureText(tag).width + 14;
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillRect(box.x, Math.max(0, box.y - 26), tw, 24);
      ctx.fillStyle = "#111";
      ctx.fillText(tag, box.x + 7, Math.max(16, box.y - 8));
    });
  }

  function snapshotJpeg(video, quality = 0.86) {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", quality);
  }

  function landmarkMotion(prev, next) {
    if (!prev || !next || prev.length !== next.length) return 1;
    let s = 0;
    for (let i = 0; i < prev.length; i++) {
      const dx = prev[i].x - next[i].x;
      const dy = prev[i].y - next[i].y;
      s += Math.sqrt(dx * dx + dy * dy);
    }
    return s / prev.length;
  }

  function eyeAspect(landmarks, left) {
    const pts = left ? landmarks.getLeftEye() : landmarks.getRightEye();
    if (pts.length < 6) return 0.3;
    const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const v1 = dist(pts[1], pts[5]);
    const v2 = dist(pts[2], pts[4]);
    const h = dist(pts[0], pts[3]) || 1;
    return (v1 + v2) / (2 * h);
  }

  return { start, stop, detect, drawDetections, snapshotJpeg, landmarkMotion, eyeAspect, loadModels };
})();
