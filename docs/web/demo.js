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
const LOGITS_NAME = "densenet121/densenet121/TargetSpatialSqueeze:0";
const SIZE = 224;
const RESIZE = 256;
const BATCH = 8;
const MAX_BYTES = 16 * 1024 * 1024;

// Occlusion resolution: finer on WebGPU (fast), coarser on WASM (slow).
const OCCLUSION = {
  webgpu: { patch: 32, stride: 16 }, // 13x13 = 169 probes
  wasm: { patch: 48, stride: 22 },   //  9x9  =  81 probes
};
let provider = "wasm";

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
        provider = providers[0];
        console.log("onnxruntime-web using:", provider);
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
  const prob = out[PROB_NAME].data;     // [n * 2], [normal, abnormal] per row
  const logits = out[LOGITS_NAME].data; // [n * 2]
  const abn = new Array(n), logit = new Array(n);
  for (let i = 0; i < n; i++) {
    abn[i] = prob[i * 2 + 1];
    logit[i] = logits[i * 2 + 1];
  }
  return { abn, logit };
}

/* ---------- test-time augmentation (multi-view) ----------
 *
 * Averages the abnormal probability over a few label-preserving views
 * (h-flip, small rotations, mild zoom) — the "Test Time Multi-View" stage
 * from the original project. View 0 stays the original, so its logit still
 * anchors the occlusion map.
 */

function transformView(base, { flip = false, rotate = 0, zoom = 1 }) {
  const src = document.createElement("canvas");
  src.width = src.height = SIZE;
  const sg = src.getContext("2d");
  const img = sg.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    img.data[i * 4] = base[i * 3] * 255;
    img.data[i * 4 + 1] = base[i * 3 + 1] * 255;
    img.data[i * 4 + 2] = base[i * 3 + 2] * 255;
    img.data[i * 4 + 3] = 255;
  }
  sg.putImageData(img, 0, 0);

  const dst = document.createElement("canvas");
  dst.width = dst.height = SIZE;
  const g = dst.getContext("2d");
  g.fillStyle = "#000";
  g.fillRect(0, 0, SIZE, SIZE);
  g.translate(SIZE / 2, SIZE / 2);
  if (flip) g.scale(-1, 1);
  if (zoom !== 1) g.scale(zoom, zoom);
  if (rotate) g.rotate((rotate * Math.PI) / 180);
  g.drawImage(src, -SIZE / 2, -SIZE / 2);

  const px = g.getImageData(0, 0, SIZE, SIZE).data;
  const out = new Float32Array(SIZE * SIZE * 3);
  for (let i = 0; i < SIZE * SIZE; i++) {
    out[i * 3] = px[i * 4] / 255;
    out[i * 3 + 1] = px[i * 4 + 1] / 255;
    out[i * 3 + 2] = px[i * 4 + 2] / 255;
  }
  return out;
}

async function predictTTA(base) {
  const views = [
    base,
    transformView(base, { flip: true }),
    transformView(base, { rotate: -8 }),
    transformView(base, { rotate: 8 }),
    transformView(base, { zoom: 1.11 }),
  ];
  const n = views.length;
  const batch = new Float32Array(n * SIZE * SIZE * 3);
  views.forEach((v, i) => batch.set(v, i * SIZE * SIZE * 3));
  const { abn, logit } = await predict(batch, n);
  const pAbn = abn.reduce((a, b) => a + b, 0) / n;
  return { pAbn, logit0: logit[0] }; // view 0 = original, for the occlusion map
}

/* ---------- occlusion sensitivity map ----------
 *
 * Importance of a patch = how much the *abnormal-class logit* drops when
 * that patch is grayed out. Logits are used instead of probabilities
 * because softmax saturates near 100% confidence and flattens the signal.
 * Overlapping patches are accumulated per pixel, then the map is floored
 * at a high percentile (like the original SmoothGrad post-processing,
 * which kept only top-percentile activity) so only the peak region shows.
 */

async function occlusionMap(base, logit0, onProgress) {
  const { patch, stride } = OCCLUSION[provider] || OCCLUSION.wasm;

  // Fill occluded patches with the image's own mean color: a fixed gray on
  // an X-ray's black background is itself an anomaly and pollutes the map.
  const mean = [0, 0, 0];
  for (let i = 0; i < SIZE * SIZE; i++)
    for (let c = 0; c < 3; c++) mean[c] += base[i * 3 + c];
  for (let c = 0; c < 3; c++) mean[c] /= SIZE * SIZE;

  // Skip near-uniform patches (plain background) — occluding nothing tells
  // us nothing, and it makes the map noisy at the image borders.
  const positions = [];
  for (let py = 0; py + patch <= SIZE; py += stride)
    for (let px = 0; px + patch <= SIZE; px += stride) {
      let sum = 0, sq = 0;
      const n = patch * patch;
      for (let y = py; y < py + patch; y++)
        for (let x = px; x < px + patch; x++) {
          const v = base[(y * SIZE + x) * 3];
          sum += v; sq += v * v;
        }
      const std = Math.sqrt(Math.max(0, sq / n - (sum / n) ** 2));
      if (std > 0.02) positions.push([px, py]);
    }

  const heat = new Float32Array(SIZE * SIZE);
  const cover = new Float32Array(SIZE * SIZE);

  for (let start = 0; start < positions.length; start += BATCH) {
    const slice = positions.slice(start, start + BATCH);
    const batch = new Float32Array(slice.length * SIZE * SIZE * 3);
    slice.forEach(([px, py], b) => {
      batch.set(base, b * SIZE * SIZE * 3);
      for (let y = py; y < py + patch; y++)
        for (let x = px; x < px + patch; x++) {
          const o = b * SIZE * SIZE * 3 + (y * SIZE + x) * 3;
          batch[o] = mean[0];
          batch[o + 1] = mean[1];
          batch[o + 2] = mean[2];
        }
    });
    const { logit } = await predict(batch, slice.length);
    slice.forEach(([px, py], b) => {
      const imp = Math.max(0, logit0 - logit[b]);
      for (let y = py; y < py + patch; y++)
        for (let x = px; x < px + patch; x++) {
          heat[y * SIZE + x] += imp;
          cover[y * SIZE + x] += 1;
        }
    });
    onProgress(Math.min(start + BATCH, positions.length), positions.length);
    await new Promise((r) => setTimeout(r)); // let the UI breathe
  }

  for (let i = 0; i < heat.length; i++) heat[i] /= Math.max(cover[i], 1);

  // Floor at the 80th percentile, normalize, sharpen.
  const sorted = Float32Array.from(heat).sort();
  const floor = sorted[Math.floor(sorted.length * 0.8)];
  let max = 0;
  for (let i = 0; i < heat.length; i++) {
    heat[i] = Math.max(0, heat[i] - floor);
    if (heat[i] > max) max = heat[i];
  }
  if (max > 0)
    for (let i = 0; i < heat.length; i++)
      heat[i] = Math.pow(heat[i] / max, 1.5);

  return boxBlur(heat, SIZE, 4);
}

function boxBlur(src, size, radius) {
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  const norm = 2 * radius + 1;
  for (let y = 0; y < size; y++) {        // horizontal pass
    let acc = 0;
    for (let x = -radius; x <= radius; x++) acc += src[y * size + Math.min(Math.max(x, 0), size - 1)];
    for (let x = 0; x < size; x++) {
      tmp[y * size + x] = acc / norm;
      acc += src[y * size + Math.min(x + radius + 1, size - 1)];
      acc -= src[y * size + Math.max(x - radius, 0)];
    }
  }
  for (let x = 0; x < size; x++) {        // vertical pass
    let acc = 0;
    for (let y = -radius; y <= radius; y++) acc += tmp[Math.min(Math.max(y, 0), size - 1) * size + x];
    for (let y = 0; y < size; y++) {
      out[y * size + x] = acc / norm;
      acc += tmp[Math.min(y + radius + 1, size - 1) * size + x];
      acc -= tmp[Math.max(y - radius, 0) * size + x];
    }
  }
  return out;
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

function drawHeatmap(base, heat) {
  // Blend viridis(heat) over the input at 35% alpha, like the server app.
  const ctx = $("canvas-heatmap").getContext("2d");
  const out = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const [r, g, b] = viridis(heat[i]);
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

    $("progress-text").textContent = "Classifying (multi-view)…";
    const { pAbn: p0, logit0 } = await predictTTA(base);

    const heat = await occlusionMap(base, logit0, (done, total) => {
      $("progress-text").textContent = `Computing sensitivity map… ${done}/${total}`;
    });

    drawInput(base);
    drawHeatmap(base, heat);

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
