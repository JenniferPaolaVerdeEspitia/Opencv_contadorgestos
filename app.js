import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// (Opcional) silenciar logs ruidosos
(function () {
  const mute = [
    "FaceBlendshapesGraph acceleration",
    "GL version",
    "OpenGL error checking is disabled",
    "Graph successfully started running",
    "Created TensorFlow Lite XNNPACK delegate for CPU",
    "Feedback manager requires a model with a single signature",
    "Disabling support for feedback tensors",
  ];
  ["log", "warn", "error"].forEach((m) => {
    const orig = console[m].bind(console);
    console[m] = (...args) => {
      if (typeof args[0] === "string" && mute.some((t) => args[0].includes(t))) return;
      orig(...args);
    };
  });
})();

const $ = (s) => document.querySelector(s);
const video = $("#webcam");
const canvas = $("#overlay");
const ctx = canvas.getContext("2d");

const btnStart = $("#btnStart");
const btnStop = $("#btnStop");
const btnReset = $("#btnReset");

const dot = $("#dot-status");
const txt = $("#txt-status");
const hud = $("#hud");
const chkMesh = $("#chkMesh");

const thrBlinkEl = $("#thr-blink");
const thrMouthEl = $("#thr-mouth");
const thrBrowEl  = $("#thr-brow");
const valBlinkEl = $("#val-blink");
const valMouthEl = $("#val-mouth");
const valBrowEl  = $("#val-brow");

const countBlinkEl = $("#count-blink");
const countMouthEl = $("#count-mouth");
const countBrowEl  = $("#count-brow");

for (const [inp, out] of [
  [thrBlinkEl, valBlinkEl],
  [thrMouthEl, valMouthEl],
  [thrBrowEl,  valBrowEl],
]) {
  inp.addEventListener("input", () => (out.textContent = (+inp.value).toFixed(2)));
  out.textContent = (+inp.value).toFixed(2);
}

let stream = null;
let raf = 0;
let running = false;

let faceLandmarker = null;

let countBlink = 0;
let countMouth = 0;
let countBrow  = 0;

let prevBlink = false;
let prevMouth = false;

// Cejas: baseline + histéresis
let browBaseline = null;
let browState = "idle";
let browArmed = true;
let browReleaseStart = 0;
let calibratingUntil = 0;
let pauseUntil = 0;

function toast(msg) {
  const el = document.getElementById("liveToast");
  document.getElementById("toast-body").textContent = msg;
  const t = bootstrap.Toast.getOrCreateInstance(el);
  t.show();
}

function setStatus(on) {
  dot.classList.toggle("bg-success", on);
  dot.classList.toggle("bg-danger", !on);
  txt.textContent = on ? "Cámara encendida" : "Cámara detenida";
}

function getBlend(blendCats, keys) {
  if (!blendCats) return 0;
  for (const k of keys) {
    const exact = blendCats.find(
      (c) => (c.categoryName && c.categoryName === k) || (c.displayName && c.displayName === k)
    );
    if (exact) return exact.score ?? 0;
  }
  const part = blendCats.find((c) =>
    keys.some((k) => (c.categoryName || c.displayName || "").toLowerCase().includes(k.toLowerCase()))
  );
  return part ? part.score : 0;
}

function drawLandmarks(lms) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!chkMesh.checked || !lms) return;
  ctx.beginPath();
  for (const p of lms) {
    const x = p.x * canvas.width;
    const y = p.y * canvas.height;
    ctx.moveTo(x + 1, y);
    ctx.arc(x, y, 1, 0, Math.PI * 2);
  }
  ctx.fillStyle = "rgba(0,255,255,0.9)";
  ctx.fill();
}

async function loadFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
}

async function ensureDevice() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  if (!devices.some((d) => d.kind === "videoinput")) {
    throw new Error("No se detectó ninguna cámara en el sistema.");
  }
}

async function startCamera() {
  if (running) return;
  try {
    await ensureDevice();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await new Promise((res) => (video.readyState >= 1 ? res() : (video.onloadedmetadata = res)));
    await video.play();

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    if (!faceLandmarker) await loadFaceLandmarker();

    browBaseline = null;
    calibratingUntil = performance.now() + 800;
    browState = "idle";
    browArmed = true;
    pauseUntil = 0;

    running = true;
    setStatus(true);
    loop();
  } catch (err) {
    console.error(err);
    toast("No se pudo iniciar la cámara/modelo. Revisa permisos o si otra app usa la cámara.");
    setStatus(false);
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(raf);
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setStatus(false);
}

function loop() {
  if (!running) return;
  const now = performance.now();

  const res = faceLandmarker?.detectForVideo(video, now);
  const lm = res?.faceLandmarks?.[0] || null;
  drawLandmarks(lm);

  const blends = res?.faceBlendshapes?.[0]?.categories || [];
  const blink = (getBlend(blends, ["eyeBlinkLeft"]) + getBlend(blends, ["eyeBlinkRight"])) / 2;
  const mouth = Math.max(getBlend(blends, ["mouthOpen"]), getBlend(blends, ["jawOpen"]));

  const browInner = getBlend(blends, ["browInnerUp"]);
  const browOuter = (getBlend(blends, ["browOuterUpLeft"]) + getBlend(blends, ["browOuterUpRight"])) / 2;
  const brow = Math.max(browInner, browOuter);

  if (browBaseline === null) browBaseline = brow;

  const delta = brow - browBaseline;
  const thrBrowHi = +thrBrowEl.value;                 // delta alto
  const thrBrowLo = Math.max(0.05, thrBrowHi * 0.4);  // delta bajo (liberación)

  if (now < calibratingUntil) {
    const a = 0.35;
    browBaseline = browBaseline * (1 - a) + brow * a;
  } else if (browState === "idle" && Math.abs(delta) < thrBrowLo * 0.8) {
    const a = 0.03;
    browBaseline = browBaseline * (1 - a) + brow * a;
  }

  hud.textContent =
    `blink: ${blink.toFixed(2)} | mouth: ${mouth.toFixed(2)} | ` +
    `brow: ${brow.toFixed(2)} base: ${browBaseline.toFixed(2)} Δ:${delta.toFixed(2)}`;

  if (now >= pauseUntil) {
    const thrBlink = +thrBlinkEl.value;
    const thrMouth = +thrMouthEl.value;
    const isBlink = blink >= thrBlink;
    const isMouth = mouth >= thrMouth;

    if (isBlink && !prevBlink) countBlink++;
    if (isMouth && !prevMouth) countMouth++;
    prevBlink = isBlink;
    prevMouth = isMouth;

    if (browArmed && browState === "idle" && delta >= thrBrowHi) {
      countBrow++;
      browState = "raised";
      browArmed = false;
      browReleaseStart = 0;
    } else if (!browArmed) {
      if (delta <= thrBrowLo) {
        if (!browReleaseStart) browReleaseStart = now;
        else if (now - browReleaseStart >= 150) {
          browState = "idle";
          browArmed = true;
          browReleaseStart = 0;
        }
      } else {
        browReleaseStart = 0;
      }
    }
  }

  countBlinkEl.textContent = countBlink;
  countMouthEl.textContent = countMouth;
  countBrowEl.textContent  = countBrow;

  raf = requestAnimationFrame(loop);
}

btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", stopCamera);

btnReset.addEventListener("click", () => {
  countBlink = countMouth = countBrow = 0;
  countBlinkEl.textContent = "0";
  countMouthEl.textContent = "0";
  countBrowEl.textContent  = "0";

  prevBlink = prevMouth = false;
  browState = "idle";
  browArmed = true;
  browReleaseStart = 0;
  calibratingUntil = performance.now() + 800;
  pauseUntil = performance.now() + 400;
});

setStatus(false);




