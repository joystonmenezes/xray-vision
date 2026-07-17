"use strict";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const progress = document.getElementById("progress");
const progressText = document.getElementById("progress-text");
const errorBox = document.getElementById("error");
const results = document.getElementById("results");
const gaugeFill = document.getElementById("gauge-fill");
const verdict = document.getElementById("verdict");
const resultInput = document.getElementById("result-input");
const resultHeatmap = document.getElementById("result-heatmap");
const resetBtn = document.getElementById("reset");

const MAX_BYTES = 16 * 1024 * 1024;

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) analyze(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) analyze(file);
});

document.getElementById("try-sample").addEventListener("click", async (e) => {
  e.preventDefault();
  const res = await fetch("/static/example_input.png");
  const blob = await res.blob();
  analyze(new File([blob], "example_input.png", { type: "image/png" }));
});

resetBtn.addEventListener("click", () => {
  results.hidden = true;
  errorBox.hidden = true;
  dropzone.hidden = false;
  fileInput.value = "";
  dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
});

async function analyze(file) {
  errorBox.hidden = true;
  results.hidden = true;

  if (!["image/png", "image/jpeg"].includes(file.type)) {
    return showError("Please upload a PNG or JPEG X-ray image.");
  }
  if (file.size > MAX_BYTES) {
    return showError("That image is larger than 16 MB.");
  }

  dropzone.hidden = true;
  progress.hidden = false;
  progressText.textContent = "Analyzing X-ray… (computing sensitivity map)";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Server error (${res.status})`);
    }
    const data = await res.json();
    render(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
    dropzone.hidden = false;
  } finally {
    progress.hidden = true;
  }
}

function render(data) {
  const p = data.abnormal_probability;
  resultInput.src = data.input_image;
  resultHeatmap.src = data.heatmap;

  verdict.textContent = `${p}% chance of abnormality`;
  verdict.style.color =
    p >= 50 ? "var(--danger)" : "var(--ok)";

  results.hidden = false;
  requestAnimationFrame(() => {
    gaugeFill.style.width = `${p}%`;
  });
  results.scrollIntoView({ behavior: "smooth", block: "center" });
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
