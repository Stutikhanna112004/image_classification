const GAUGE_CIRCUMFERENCE = 251;

const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const previewStrip = document.getElementById("preview-strip");
const previewImg = document.getElementById("preview-img");
const rescanBtn = document.getElementById("rescan-btn");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const shutterBtn = document.getElementById("shutter-btn");
const tagCard = document.getElementById("tag-card");
const tagId = document.getElementById("tag-id");
const fieldSpecies = document.getElementById("field-species");
const fieldConfidence = document.getElementById("field-confidence");
const gaugeFill = document.getElementById("gauge-fill");
const gaugeReadout = document.getElementById("gauge-readout");
const stampPlaceholder = document.getElementById("stamp-placeholder");
const stampMark = document.getElementById("stamp-mark");
const tagFootnote = document.getElementById("tag-footnote");

let mediaStream = null;

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    tabContents.forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");

    if (btn.dataset.tab === "camera") startCamera();
    else stopCamera();
  });
});

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

rescanBtn.addEventListener("click", () => {
  previewStrip.hidden = true;
  fileInput.value = "";
  resetTag();
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewStrip.hidden = false;
  };
  reader.readAsDataURL(file);
  sendToServer(file);
}

async function startCamera() {
  if (mediaStream) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    video.srcObject = mediaStream;
  } catch (err) {
    tagFootnote.textContent = "Camera unavailable — check browser permissions";
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}

shutterBtn.addEventListener("click", () => {
  if (!mediaStream) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    previewImg.src = canvas.toDataURL("image/jpeg");
    previewStrip.hidden = false;
    sendToServer(blob);
  }, "image/jpeg", 0.92);
});

function resetTag() {
  tagCard.classList.remove("verdict-dog", "verdict-cat");
  fieldSpecies.textContent = "— — — —";
  fieldSpecies.classList.remove("is-dog", "is-cat");
  fieldConfidence.textContent = "— — %";
  gaugeFill.style.stroke = "var(--dog)";
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  gaugeReadout.textContent = "0%";
  stampPlaceholder.style.opacity = "0.4";
  stampPlaceholder.textContent = "AWAITING SCAN";
  stampMark.classList.remove("stamp-in", "is-dog", "is-cat");
  stampMark.textContent = "";
  tagFootnote.textContent = "Insert a photo to begin analysis";
  tagId.textContent = "NO. ---";
}

function randomTagId() {
  return "NO. " + Math.floor(1000 + Math.random() * 8999);
}

async function sendToServer(fileOrBlob) {
  stampPlaceholder.textContent = "SCANNING…";
  stampPlaceholder.style.opacity = "0.7";
  tagFootnote.textContent = "Reading the frame…";
  tagId.textContent = randomTagId();

  const formData = new FormData();
  formData.append("image", fileOrBlob, "capture.jpg");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Prediction failed");
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    tagFootnote.textContent = "Scan failed — is the server running?";
    stampPlaceholder.textContent = "ERROR";
  }
}

function renderResult(data) {
  const isDog = data.label === "dog";
  const confidencePct = Math.round(data.confidence * 100);

  tagCard.classList.remove("verdict-dog", "verdict-cat");
  tagCard.classList.add(isDog ? "verdict-dog" : "verdict-cat");

  fieldSpecies.textContent = isDog ? "DOG" : "CAT";
  fieldSpecies.classList.remove("is-dog", "is-cat");
  fieldSpecies.classList.add(isDog ? "is-dog" : "is-cat");

  fieldConfidence.textContent = `${confidencePct}%`;

  const offset = GAUGE_CIRCUMFERENCE * (1 - data.confidence);
  gaugeFill.style.stroke = isDog ? "var(--dog)" : "var(--cat)";
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = offset;
  });

  animateReadout(confidencePct);

  stampPlaceholder.style.opacity = "0";
  stampMark.textContent = isDog ? "Dog" : "Cat";
  stampMark.classList.remove("is-dog", "is-cat", "stamp-in");
  stampMark.classList.add(isDog ? "is-dog" : "is-cat");
  void stampMark.offsetWidth;
  stampMark.classList.add("stamp-in");

  tagFootnote.textContent =
    confidencePct >= 90 ? "High confidence match" :
    confidencePct >= 70 ? "Likely match" :
    "Low confidence — try a clearer photo";
}

function animateReadout(target) {
  let current = 0;
  const step = Math.max(1, Math.round(target / 20));
  const interval = setInterval(() => {
    current += step;
    if (current >= target) {
      current = target;
      clearInterval(interval);
    }
    gaugeReadout.textContent = `${current}%`;
  }, 25);
}

resetTag();