// ===== ELEMENT REFERENCES =====
let isLiveRecording = false;
const liveNotice = document.getElementById("liveAudioNotice");
const fileInput = document.getElementById("fileInput");
const filenameLabel = document.getElementById("filename");
const audioPlayer = document.getElementById("audioPlayer");
const seekBar = document.getElementById("seekBar");
const currentTimeLabel = document.getElementById("currentTime");
const totalTimeLabel = document.getElementById("totalTime");
const trackTitle = document.getElementById("trackTitle");
const durationLabel = document.getElementById("durationLabel");
const sourceLabel = document.getElementById("sourceLabel");

const startRecordBtn = document.getElementById("startRecordBtn");
const stopRecordBtn = document.getElementById("stopRecordBtn");
const playRecordedBtn = document.getElementById("playRecordedBtn");
const recordDot = document.getElementById("recordDot");
const recordStatus = document.getElementById("recordStatus");
const recordTimer = document.getElementById("recordTimer");

const spectrogramToggle = document.getElementById("spectrogramToggle");
const spectrogramPanel = document.getElementById("spectrogramPanel");
const spectrogramCanvas = document.getElementById("spectrogramCanvas");
const spectrogramToggleText = document.getElementById("spectrogramToggleText");

const gaugeCircle = document.getElementById("gaugeCircle");
const gaugeValue = document.getElementById("gaugeValue");
const predictionClass = document.getElementById("predictionClass");
const predictionConfidence = document.getElementById("predictionConfidence");

const classifyBtn = document.getElementById("classifyBtn");
const toast = document.getElementById("toast");
const toastMessage = document.getElementById("toastMessage");

// ===== HELPERS =====
function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function showToast(message) {
  toastMessage.textContent = message;
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 2600);
}

// ===== AUDIO CONTEXT + WAV CONVERSION =====
let audioCtx = null;
function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioCtx;
}

async function blobToAudioBuffer(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const ctx = getAudioContext();
  return await ctx.decodeAudioData(arrayBuffer);
}

function audioBufferToWav(abuffer) {
  const numOfChan = abuffer.numberOfChannels;
  const sampleRate = abuffer.sampleRate;
  const length = abuffer.length * numOfChan * 2 + 44;
  const buffer = new ArrayBuffer(length);
  const view = new DataView(buffer);

  let offset = 0;

  function writeString(str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset++, str.charCodeAt(i));
    }
  }

  // RIFF identifier
  writeString("RIFF");
  // RIFF chunk length
  view.setUint32(offset, length - 8, true); offset += 4;
  // RIFF type
  writeString("WAVE");
  // format chunk identifier
  writeString("fmt ");
  // format chunk length
  view.setUint32(offset, 16, true); offset += 4;
  // sample format (raw)
  view.setUint16(offset, 1, true); offset += 2;
  // channel count
  view.setUint16(offset, numOfChan, true); offset += 2;
  // sample rate
  view.setUint32(offset, sampleRate, true); offset += 4;
  // byte rate (sample rate * block align)
  const byteRate = sampleRate * numOfChan * 2;
  view.setUint32(offset, byteRate, true); offset += 4;
  // block align (channel count * bytes per sample)
  const blockAlign = numOfChan * 2;
  view.setUint16(offset, blockAlign, true); offset += 2;
  // bits per sample
  view.setUint16(offset, 16, true); offset += 2;
  // data chunk identifier
  writeString("data");
  // data chunk length
  const dataSize = abuffer.length * numOfChan * 2;
  view.setUint32(offset, dataSize, true); offset += 4;

  // write interleaved data
  const channels = [];
  for (let i = 0; i < numOfChan; i++) {
    channels.push(abuffer.getChannelData(i));
  }

  let sampleIndex = 0;
  while (sampleIndex < abuffer.length) {
    for (let ch = 0; ch < numOfChan; ch++) {
      let sample = channels[ch][sampleIndex];
      sample = Math.max(-1, Math.min(1, sample));
      sample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, sample, true);
      offset += 2;
    }
    sampleIndex++;
  }

  return buffer;
}

async function convertRecordingToWavBlob(blob) {
  const audioBuffer = await blobToAudioBuffer(blob);
  const wavArrayBuffer = audioBufferToWav(audioBuffer);
  return new Blob([wavArrayBuffer], { type: "audio/wav" });
}

// ===== STATE =====
let recordedChunks = [];
let mediaRecorder = null;
let recordedBlob = null;
let recordStartTime = null;
let recordTimerId = null;

// spectrogram state
let analyser = null;
let mediaElementSource = null;
let spectrogramCtx = spectrogramCanvas.getContext("2d");
let spectrogramAnimationId = null;
let spectrogramActive = false;

// ===== FILE INPUT HANDLING (UPLOAD .WAV) =====
fileInput.addEventListener("change", () => {
  isLiveRecording = false;
if (liveNotice) liveNotice.style.display = "none";

  const file = fileInput.files[0];
  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".wav")) {
    showToast("Only .wav files are supported.");
    fileInput.value = "";
    filenameLabel.textContent = "No file chosen";
    return;
  }

  filenameLabel.textContent = file.name;
  trackTitle.textContent = `File: ${file.name}`;
  sourceLabel.textContent = "Source: uploaded file";

  const url = URL.createObjectURL(file);
  audioPlayer.src = url;
  audioPlayer.load();

  audioPlayer.onloadedmetadata = () => {
    totalTimeLabel.textContent = formatTime(audioPlayer.duration);
    durationLabel.textContent = `Duration: ${audioPlayer.duration.toFixed(1)}s`;
  };
});

// ===== RECORDING CONTROLS =====
function startTimer() {
  recordStartTime = Date.now();
  recordTimerId = setInterval(() => {
    const elapsed = (Date.now() - recordStartTime) / 1000;
    recordTimer.textContent = formatTime(elapsed);
  }, 500);
}

function stopTimer() {
  if (recordTimerId) {
    clearInterval(recordTimerId);
    recordTimerId = null;
  }
}

startRecordBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    recordedChunks = [];
    recordedBlob = null;
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stopTimer();
      recordDot.classList.remove("active");
      recordStatus.textContent = "Processing…";

      const originalBlob = new Blob(recordedChunks, { type: "audio/webm" });

      try {
        // Convert recorded audio → WAV
        recordedBlob = await convertRecordingToWavBlob(originalBlob);
        isLiveRecording = true;

        // Load into audio player
        const url = URL.createObjectURL(recordedBlob);
        audioPlayer.src = url;
        audioPlayer.load();
        audioPlayer.onloadedmetadata = () => {
          totalTimeLabel.textContent = formatTime(audioPlayer.duration);
          durationLabel.textContent = `Duration: ${audioPlayer.duration.toFixed(1)}s`;
          trackTitle.textContent = "File: recorded clip (WAV)";
          sourceLabel.textContent = "Source: recording (converted to WAV)";
        };

        // Automatically classify the recorded WAV
        showToast("Recording ready. Classifying...");
        await uploadAndPredict(recordedBlob, "recorded.wav");
      } catch (err) {
        console.error(err);
        showToast("Could not convert recording to WAV.");
      }

      // stop tracks
      stream.getTracks().forEach((t) => t.stop());
      recordStatus.textContent = "Idle";
    };

    // Start recording
    mediaRecorder.start();
    recordDot.classList.add("active");
    recordStatus.textContent = "Recording…";
    recordTimer.textContent = "0:00";
    startTimer();
  } catch (err) {
    console.error(err);
    showToast("Could not start recording. Use HTTPS/localhost and allow microphone.");
  }
});

stopRecordBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  } else {
    showToast("Not recording currently.");
  }
});

// Play button now plays *whatever* is loaded in audioPlayer
playRecordedBtn.addEventListener("click", async () => {
  if (!audioPlayer.src) {
    showToast("No audio loaded. Upload or record first.");
    return;
  }

  try {
    const ctx = getAudioContext();
    await ctx.resume(); // needed on some browsers
  } catch (e) {
    console.warn("AudioContext resume failed or not required.", e);
  }

  audioPlayer.play().catch((err) => {
    console.error(err);
    showToast("Could not play audio.");
  });
});

// ===== AUDIO PLAYER PROGRESS BAR =====
audioPlayer.addEventListener("timeupdate", () => {
  if (!audioPlayer.duration) return;
  currentTimeLabel.textContent = formatTime(audioPlayer.currentTime);
  seekBar.value = (audioPlayer.currentTime / audioPlayer.duration) * 100;
});

audioPlayer.addEventListener("loadedmetadata", () => {
  totalTimeLabel.textContent = formatTime(audioPlayer.duration);
});

seekBar.addEventListener("input", () => {
  if (!audioPlayer.duration) return;
  const fraction = seekBar.value / 100;
  audioPlayer.currentTime = audioPlayer.duration * fraction;
});

// ===== SPECTROGRAM (REAL VISUALIZER) =====
function setupSpectrogramCanvas() {
  // match canvas internal size to CSS size
  const rect = spectrogramCanvas.getBoundingClientRect();
  spectrogramCanvas.width = rect.width;
  spectrogramCanvas.height = rect.height;
}

function startSpectrogram() {
  if (spectrogramActive) return;
  if (!audioPlayer.src) {
    showToast("Load or record audio first, then play it.");
  }

  const ctx = getAudioContext();
  ctx.resume().catch(() => {});

  if (!mediaElementSource) {
    mediaElementSource = ctx.createMediaElementSource(audioPlayer);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    mediaElementSource.connect(analyser);
    analyser.connect(ctx.destination);
  }

  setupSpectrogramCanvas();
  spectrogramActive = true;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    if (!spectrogramActive) return;

    spectrogramAnimationId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    const width = spectrogramCanvas.width;
    const height = spectrogramCanvas.height;

    // Shift existing image left
    const imageData = spectrogramCtx.getImageData(1, 0, width - 1, height);
    spectrogramCtx.putImageData(imageData, 0, 0);

    // Draw new column on the right
    for (let y = 0; y < height; y++) {
      const dataIndex = Math.floor((y / height) * bufferLength);
      const value = dataArray[dataIndex]; // 0-255
      const brightness = value / 255;

      const r = Math.floor(20 + 235 * brightness);
      const g = Math.floor(60 + 120 * brightness);
      const b = Math.floor(180 + 60 * (1 - brightness));

      spectrogramCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      spectrogramCtx.fillRect(width - 1, height - 1 - y, 1, 1);
    }
  }

  spectrogramCtx.clearRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
  draw();
}

function stopSpectrogram() {
  spectrogramActive = false;
  if (spectrogramAnimationId) {
    cancelAnimationFrame(spectrogramAnimationId);
    spectrogramAnimationId = null;
  }
}

spectrogramToggle.addEventListener("click", () => {
  const isVisible = spectrogramPanel.style.display === "block";
  if (isVisible) {
    spectrogramPanel.style.display = "none";
    spectrogramToggleText.textContent = "view spectrogram";
    stopSpectrogram();
  } else {
    spectrogramPanel.style.display = "block";
    spectrogramToggleText.textContent = "hide spectrogram";
    startSpectrogram();
  }
});

window.addEventListener("resize", () => {
  if (spectrogramActive) {
    setupSpectrogramCanvas();
  }
});

// ===== CLASSIFICATION – SEND .WAV TO FLASK BACKEND =====
async function uploadAndPredict(fileBlob, filename = "clip.wav") {
  const fd = new FormData();
  fd.append("file", fileBlob, filename);

  classifyBtn.classList.add("loading");
  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: fd,
    });

    let data;
    try {
      data = await res.json();
    } catch {
      data = {};
    }

    classifyBtn.classList.remove("loading");

    if (!res.ok) {
      showToast("Prediction failed: " + (data.error || res.statusText));
      return;
    }

    // Expected response from backend:
    // { prediction: "dog_bark", top_prob: 0.93 }
    predictionClass.textContent = data.prediction || "—";
        if (liveNotice) {
      liveNotice.style.display = isLiveRecording ? "block" : "none";
    }
    const prob = Number(data.top_prob) || 0;
    const probPercent = (prob * 100).toFixed(1);

    predictionConfidence.textContent = `Confidence: ${probPercent}%`;
    gaugeCircle.style.setProperty("--confidence", probPercent);
    gaugeValue.textContent = `${Math.round(prob * 100)}%`;
  } catch (err) {
    console.error(err);
    classifyBtn.classList.remove("loading");
    showToast("Request failed: " + err.message);
  }
}

// Manual classify (for uploaded .wav or already-converted recording)
classifyBtn.addEventListener("click", async () => {
  // If currently recording, don't allow classify
  if (mediaRecorder && mediaRecorder.state === "recording") {
    showToast("Stop recording first, then classify.");
    return;
  }

  // Prefer recordedBlob if available
  if (recordedBlob) {
    await uploadAndPredict(recordedBlob, "recorded.wav");
    return;
  }

  // Otherwise use uploaded file
  const file = fileInput.files[0];
  if (!file) {
    showToast("Upload or record audio before classifying.");
    return;
  }

  if (!file.name.toLowerCase().endsWith(".wav")) {
    showToast("Only .wav files are supported.");
    return;
  }

  await uploadAndPredict(file, file.name);
});
