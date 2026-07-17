"use strict";

/* X-Ray Vision — in-browser demo.
 *
 * Runs the same DenseNet-121 (original 2021 trained weights, converted to
 * ONNX) entirely client-side with onnxruntime-web. Prediction is identical
 * to the server app; the explanation heatmap here is occlusion-based
 * (slide a gray patch over the image, measure how much the abnormality
 * probability drops) because in-browser runtimes provide no gradients.
 */

const MODEL_URL = "model/densenet121_mura.onnx";
const INPUT_NAME = "inputs:0";
const PROB_NAME = "densenet121/probability:0";
const SIZE = 224;
const RESIZE = 256;
const GRID = 8;          // occlusion grid (GRID x GRID probes)
const PATCH = 48;        // occlusion patch size in pixels
const BATCH = 8;
const MAX_BYTES = 16 * 1024 * 1024;

// Same viridis anchors as the server app.
const VIRIDIS = [
  [68, 1, 84], [71, 24, 106], [72, 40, 120], [71, 57, 132],
  [66, 73, 140], [60, 88, 145], [54, 103, 148], [47, 117, 149],
  [42, 130, 150], [37, 144, 150], [33, 158, 148], [32, 171, 144],
  [41, 184, 135], [61, 195, 122], [91, 206, 101], [127, 214, 75],
  [167, 219, 51], [208, 222, 39], [253, 231, 37],
];

function viridis(v) {
  const x = Math.min(Math.max(v, 0), 1) * (VIRIDIS.length - 1);
  const lo = Math.floor(x), hi = Math.min(lo + 1, VIRIDIS.length - 1), f = x - lo;
  return [0, 1, 2].map((c) => VIRIDIS[lo][c] * (1 - f) + VIRIDIS[hi][c] * f);
}

const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone"), fileInput = $("file-input");

let session = null;

async function loadModel() {
  const bar = $("model-bar"), text = $("model-text");
  try {
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`model fetch failed (${res.status})`);
    const total = +res.headers.get("Content-Length") || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      if (total) bar.style.width = `${Math.round((got / total) * 100)}%`;
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }

    text.textContent = "Initializing model…";
    if (!crossOriginIsolated) ort.env.wasm.numThreads = 1;
    for (const providers of [["webgpu"], ["wasm"]]) {
      try {
        session = await ort.InferenceSession.create(buf, { executionProviders: providers });
        console.log("onnxruntime-web using:", providers[0]);
        break;
      } catch (e) {
        console.warn(`${providers[0]} init failed:`, e);
      }
    }
    if (!session) throw new Error("could not initialize onnxruntime-web");

    $("model-status").hidden = true;
    dropzone.hidden = false;
    $("sample-row").hidden = false;
  } catch (err) {
    text.textContent = "";
    showError(`Failed to load the model: ${err.message}`);
  }
}

/* ---------- preprocessing (mirrors app/preprocessing.py) ---------- */

function preprocess(img) {
  const c256 = document.createElement("canvas");
  c256.width = c256.height = RESIZE;
  const g = c256.getContext("2d");
  g.fillStyle = "#000";
  g.fillRect(0, 0, RESIZE, RESIZE);
  const ratio = RESIZE / Math.max(img.width, img.height);
  const w = Math.round(img.width * ratio), h = Math.round(img.height * ratio);
  g.imageSmoothingQuality = "high";
  g.drawImage(img, (RESIZE - w) / 2, (RESIZE - h) / 2, w, h);

  const off = (RESIZE - SIZE) / 2;
  const pix = g.getImageData(off, off, SIZE, SIZE).data;
  const data = new Float32Array(SIZE * SIZE * 3);
  for (let i = 0; i < SIZE * SIZE; i++) {
    data[i * 3] = pix[i * 4] / 255;
    data[i * 3 + 1] = pix[i * 4 + 1] / 255;
    data[i * 3 + 2] = pix[i * 4 + 2] / 255;
  }
  return data; // NHWC, single image
}

async function predict(batchData, n) {
  const tensor = new ort.Tensor("float32", batchData, [n, SIZE, SIZE, 3]);
  const out = await session.run({ [INPUT_NAME]: tensor });
  const prob = out[PROB_NAME].data; // [n * 2], [normal, abnormal] per row
  const abn = new Array(n);
  for (let i = 0; i < n; i++) abn[i] = prob[i * 2 + 1];
  return abn;
}

/* ---------- occlusion sensitivity map ---------- */

async function occlusionMap(base, p0, onProgress) {
  const positions = [];
  for (let gy = 0; gy < GRID; gy++)
    for (let gx = 0; gx < GRID; gx++)
      positions.push([
        Math.round((gx * (SIZE - PATCH)) / (GRID - 1)),
        Math.round((gy * (SIZE - PATCH)) / (GRID - 1)),
      ]);

  const importance = new Float32Array(GRID * GRID);
  for (let start = 0; start < positions.length; start += BATCH) {
    const slice = positions.slice(start, start + BATCH);
    const batch = new Float32Array(slice.length * SIZE * SIZE * 3);
    slice.forEach(([px, py], b) => {
      batch.set(base, b * SIZE * SIZE * 3);
      for (let y = py; y < py + PATCH; y++)
        for (let x = px; x < px + PATCH; x++) {
          const o = b * SIZE * SIZE * 3 + (y * SIZE + x) * 3;
          batch[o] = batch[o + 1] = batch[o + 2] = 0.5;
        }
    });
    const abn = await predict(batch, slice.length);
    abn.forEach((p, b) => { importance[start + b] = Math.max(0, p0 - p); });
    onProgress(Math.min(start + BATCH, positions.length), positions.length);
    await new Promise((r) => setTimeout(r)); // let the UI breathe
  }

  const max = Math.max(...importance);
  if (max > 0) for (let i = 0; i < importance.length; i++) importance[i] /= max;
  return importance;
}

/* ---------- rendering ---------- */

function drawInput(base) {
  const ctx = $("canvas-input").getContext("2d");
  const img = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    img.data[i * 4] = base[i * 3] * 255;
    img.data[i * 4 + 1] = base[i * 3 + 1] * 255;
    img.data[i * 4 + 2] = base[i * 3 + 2] * 255;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function drawHeatmap(base, importance) {
  // Upscale the GRID x GRID importance map smoothly to SIZE x SIZE.
  const small = document.createElement("canvas");
  small.width = small.height = GRID;
  const sg = small.getContext("2d");
  const simg = sg.createImageData(GRID, GRID);
  for (let i = 0; i < GRID * GRID; i++) {
    simg.data[i * 4] = importance[i] * 255;
    simg.data[i * 4 + 3] = 255;
  }
  sg.putImageData(simg, 0, 0);

  const up = document.createElement("canvas");
  up.width = up.height = SIZE;
  const ug = up.getContext("2d");
  ug.imageSmoothingEnabled = true;
  ug.imageSmoothingQuality = "high";
  ug.drawImage(small, 0, 0, SIZE, SIZE);
  const heat = ug.getImageData(0, 0, SIZE, SIZE).data;

  // Blend viridis(heat) over the input at 35% alpha, like the server app.
  const ctx = $("canvas-heatmap").getContext("2d");
  const out = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const [r, g, b] = viridis(heat[i * 4] / 255);
    out.data[i * 4] = base[i * 3] * 255 * 0.65 + r * 0.35;
    out.data[i * 4 + 1] = base[i * 3 + 1] * 255 * 0.65 + g * 0.35;
    out.data[i * 4 + 2] = base[i * 3 + 2] * 255 * 0.65 + b * 0.35;
    out.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

/* ---------- UI flow ---------- */

async function analyze(source) {
  $("error").hidden = true;
  $("results").hidden = true;
  dropzone.hidden = true;
  $("sample-row").hidden = true;
  $("progress").hidden = false;

  try {
    const img = await loadImage(source);
    const base = preprocess(img);

    $("progress-text").textContent = "Classifying…";
    const [p0] = await predict(base, 1);

    const importance = await occlusionMap(base, p0, (done, total) => {
      $("progress-text").textContent = `Computing sensitivity map… ${done}/${total}`;
    });

    drawInput(base);
    drawHeatmap(base, importance);

    const pct = Math.round(p0 * 1000) / 10;
    const verdict = $("verdict");
    verdict.textContent = `${pct}% chance of abnormality`;
    verdict.style.color = pct >= 50 ? "var(--danger)" : "var(--ok)";
    $("results").hidden = false;
    requestAnimationFrame(() => { $("gauge-fill").style.width = `${pct}%`; });
    $("results").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (err) {
    console.error(err);
    showError(err.message || "Analysis failed. Please try another image.");
    dropzone.hidden = false;
    $("sample-row").hidden = false;
  } finally {
    $("progress").hidden = true;
  }
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not read that image."));
    if (typeof source === "string") {
      img.src = source;
    } else {
      if (!["image/png", "image/jpeg"].includes(source.type))
        return reject(new Error("Please upload a PNG or JPEG X-ray image."));
      if (source.size > MAX_BYTES)
        return reject(new Error("That image is larger than 16 MB."));
      img.src = URL.createObjectURL(source);
    }
  });
}

function showError(message) {
  const e = $("error");
  e.textContent = message;
  e.hidden = false;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) analyze(fileInput.files[0]);
});
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]);
});
$("try-sample").addEventListener("click", (e) => {
  e.preventDefault();
  analyze("docs/example_input.png");
});
$("reset").addEventListener("click", () => {
  $("results").hidden = true;
  $("error").hidden = true;
  dropzone.hidden = false;
  $("sample-row").hidden = false;
  fileInput.value = "";
  dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
});

loadModel();
