const SAMPLE_RATE = 24000;

const els = {
  refreshStatusBtn: document.getElementById("refreshStatusBtn"),
  textInput: document.getElementById("textInput"),
  outputName: document.getElementById("outputName"),
  synthesizeBtn: document.getElementById("synthesizeBtn"),
  streamBtn: document.getElementById("streamBtn"),
  stopStreamBtn: document.getElementById("stopStreamBtn"),
  eventLog: document.getElementById("eventLog"),
  statusApi: document.getElementById("statusApi"),
  statusVox: document.getElementById("statusVox"),
  statusDevice: document.getElementById("statusDevice"),
  statusLlm: document.getElementById("statusLlm"),
  statusLanguage: document.getElementById("statusLanguage"),
  statusEndpoint: document.getElementById("statusEndpoint"),
  metricFirstChunk: document.getElementById("metricFirstChunk"),
  metricTotal: document.getElementById("metricTotal"),
  metricDuration: document.getElementById("metricDuration"),
  metricOutput: document.getElementById("metricOutput"),
  slotEars: document.getElementById("slotEars"),
  slotBrain: document.getElementById("slotBrain"),
  slotMemory: document.getElementById("slotMemory"),
  slotTools: document.getElementById("slotTools"),
};

let ws = null;
let audioCtx = null;
let nextPlaybackAt = 0;

function logLine(message, kind = "info") {
  const ts = new Date().toLocaleTimeString("pl-PL", { hour12: false });
  const className = kind === "error" ? "err" : kind === "ok" ? "ok" : "";
  els.eventLog.innerHTML += `<span class="${className}">[${ts}] ${message}</span>\n`;
  els.eventLog.scrollTop = els.eventLog.scrollHeight;
}

function getWsUrl(path) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${path}`;
}

function parseChunkPacket(buffer) {
  const view = new DataView(buffer);
  const payloadLength = view.getUint32(0, false);
  return {
    payloadLength,
    payload: buffer.slice(4, 4 + payloadLength),
  };
}

function playFloat32Chunk(payload) {
  if (!audioCtx) {
    audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  }

  const floatArray = new Float32Array(payload);
  const buffer = audioCtx.createBuffer(1, floatArray.length, SAMPLE_RATE);
  buffer.getChannelData(0).set(floatArray);

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  const startAt = Math.max(nextPlaybackAt, now + 0.01);
  source.start(startAt);
  nextPlaybackAt = startAt + buffer.duration;
}

async function refreshStatus() {
  try {
    const [statusResp, configResp] = await Promise.all([
      fetch("/status"),
      fetch("/config/public"),
    ]);

    if (!statusResp.ok || !configResp.ok) {
      throw new Error("Nie udalo sie pobrac statusu lub konfiguracji");
    }

    const status = await statusResp.json();
    const config = await configResp.json();

    els.statusApi.textContent = status.status;
    els.statusVox.textContent = String(status.vox_loaded);
    els.statusDevice.textContent = status.device;
    els.statusLlm.textContent = status.llm_model;
    els.statusLanguage.textContent = status.language;
    els.statusEndpoint.textContent = `${status.api_host}:${status.api_port}`;

    els.slotEars.textContent = `Wake model: ${config.wake_word_model} | STT: ${config.whisper_model}`;
    els.slotBrain.textContent = `LLM: ${config.llm_model} | Ollama: ${config.ollama_url}`;
    els.slotMemory.textContent = "ChromaDB + SQLite: oczekuje na modul memory";
    els.slotTools.textContent = "Web tools: oczekuje na modul function-calling";
  } catch (error) {
    logLine(`Status error: ${error.message}`, "error");
  }
}

async function synthesizeViaRest() {
  const text = els.textInput.value.trim();
  const output = els.outputName.value.trim() || "gui_out.wav";

  if (!text) {
    logLine("Podaj tekst do syntezy", "error");
    return;
  }

  els.synthesizeBtn.disabled = true;
  try {
    const response = await fetch("/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, output }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    els.metricFirstChunk.textContent = `${data.latency_first_chunk}s`;
    els.metricTotal.textContent = `${data.total_time}s`;
    els.metricDuration.textContent = `${data.audio_duration}s`;
    els.metricOutput.textContent = data.output;

    logLine(`Synthesize OK: ${data.output}`, "ok");
  } catch (error) {
    logLine(`Synthesize error: ${error.message}`, "error");
  } finally {
    els.synthesizeBtn.disabled = false;
  }
}

function stopStream() {
  if (ws) {
    ws.close();
    ws = null;
  }
  nextPlaybackAt = 0;
  logLine("Streaming zatrzymany", "ok");
}

function streamViaWs() {
  const text = els.textInput.value.trim();
  if (!text) {
    logLine("Podaj tekst do streamingu", "error");
    return;
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }

  ws = new WebSocket(getWsUrl("/ws/synthesize"));
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    logLine("WebSocket polaczony", "ok");
    nextPlaybackAt = 0;
    ws.send(text);
  };

  ws.onmessage = (event) => {
    const { payloadLength, payload } = parseChunkPacket(event.data);

    if (payloadLength === 0) {
      logLine("Streaming zakonczony", "ok");
      return;
    }

    playFloat32Chunk(payload);
  };

  ws.onerror = () => {
    logLine("Blad WebSocket", "error");
  };

  ws.onclose = () => {
    logLine("WebSocket rozlaczony");
  };
}

function bindEvents() {
  els.refreshStatusBtn.addEventListener("click", refreshStatus);
  els.synthesizeBtn.addEventListener("click", synthesizeViaRest);
  els.streamBtn.addEventListener("click", streamViaWs);
  els.stopStreamBtn.addEventListener("click", stopStream);
}

function start() {
  bindEvents();
  refreshStatus();
  setInterval(refreshStatus, 10000);
  logLine("GUI gotowe");
}

start();
