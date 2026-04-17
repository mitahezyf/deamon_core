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
  statusStt: document.getElementById("statusStt"),
  statusEndpoint: document.getElementById("statusEndpoint"),
  metricFirstChunk: document.getElementById("metricFirstChunk"),
  metricTotal: document.getElementById("metricTotal"),
  metricDuration: document.getElementById("metricDuration"),
  metricOutput: document.getElementById("metricOutput"),
  slotEars: document.getElementById("slotEars"),
  slotBrain: document.getElementById("slotBrain"),
  slotMemory: document.getElementById("slotMemory"),
  slotTools: document.getElementById("slotTools"),
  connectMicBtn: document.getElementById("connectMicBtn"),
  startCaptureBtn: document.getElementById("startCaptureBtn"),
  stopCaptureBtn: document.getElementById("stopCaptureBtn"),
  disconnectMicBtn: document.getElementById("disconnectMicBtn"),
  metricMicRate: document.getElementById("metricMicRate"),
  metricEarsEvent: document.getElementById("metricEarsEvent"),
  sttTranscript: document.getElementById("sttTranscript"),
};

let ws = null;
let earsWs = null;
let audioCtx = null;
let nextPlaybackAt = 0;
let micStream = null;
let micCtx = null;
let micProcessor = null;
let micMutedGain = null;
let micCapturing = false;

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
    els.statusStt.textContent = String(status.stt_loaded);
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

function setMicButtons() {
  const isConnected = earsWs && earsWs.readyState === WebSocket.OPEN;
  els.connectMicBtn.disabled = isConnected;
  els.disconnectMicBtn.disabled = !isConnected;
  els.startCaptureBtn.disabled = !isConnected || micCapturing;
  els.stopCaptureBtn.disabled = !isConnected || !micCapturing;
}

function floatToPcm16(float32Array) {
  const pcm = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));
    pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
  }
  return pcm;
}

async function setupMicrophonePipeline() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  micCtx = new AudioContext();
  const source = micCtx.createMediaStreamSource(micStream);
  micProcessor = micCtx.createScriptProcessor(4096, 1, 1);
  micMutedGain = micCtx.createGain();
  micMutedGain.gain.value = 0;

  els.metricMicRate.textContent = `${micCtx.sampleRate} Hz`;

  if (earsWs && earsWs.readyState === WebSocket.OPEN) {
    earsWs.send(`set_sample_rate:${micCtx.sampleRate}`);
  }

  micProcessor.onaudioprocess = (event) => {
    if (!earsWs || earsWs.readyState !== WebSocket.OPEN) {
      return;
    }

    const pcm = floatToPcm16(event.inputBuffer.getChannelData(0));
    earsWs.send(pcm.buffer);
  };

  source.connect(micProcessor);
  micProcessor.connect(micMutedGain);
  micMutedGain.connect(micCtx.destination);
}

async function teardownMicrophonePipeline() {
  if (micProcessor) {
    micProcessor.disconnect();
    micProcessor.onaudioprocess = null;
  }
  if (micMutedGain) {
    micMutedGain.disconnect();
  }
  if (micCtx) {
    await micCtx.close();
  }
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
  }

  micCtx = null;
  micProcessor = null;
  micMutedGain = null;
  micStream = null;
  micCapturing = false;
  setMicButtons();
}

function handleEarsEvent(data) {
  const eventName = data.event || "unknown";
  els.metricEarsEvent.textContent = eventName;

  if (eventName === "transcript") {
    els.sttTranscript.value = data.text || "";
    logLine(`STT: ${data.text || "(pusto)"}`, data.text ? "ok" : "info");
    micCapturing = false;
    setMicButtons();
    return;
  }

  if (eventName === "capture_started") {
    micCapturing = true;
    setMicButtons();
    logLine(`Capture aktywny (${data.source || "unknown"})`, "ok");
    return;
  }

  if (eventName === "stt_error") {
    micCapturing = false;
    setMicButtons();
    logLine(`STT error: ${data.message || "unknown"}`, "error");
    return;
  }

  if (eventName === "wake_word_detected") {
    logLine(`Wake word: ${data.label} score=${data.score}`, "ok");
    return;
  }

  if (eventName === "sample_rate_set") {
    els.metricMicRate.textContent = `${data.sample_rate} Hz`;
    return;
  }
}

async function connectMic() {
  if (earsWs && earsWs.readyState === WebSocket.OPEN) {
    return;
  }

  earsWs = new WebSocket(getWsUrl("/ws/ears/listen"));
  earsWs.binaryType = "arraybuffer";

  earsWs.onopen = async () => {
    logLine("Mikrofon WS polaczony", "ok");
    try {
      await setupMicrophonePipeline();
      setMicButtons();
    } catch (error) {
      logLine(`Nie udalo sie uruchomic mikrofonu: ${error.message}`, "error");
      disconnectMic();
    }
  };

  earsWs.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleEarsEvent(data);
    } catch (_error) {
      logLine("Niepoprawny event JSON z ears", "error");
    }
  };

  earsWs.onerror = () => {
    logLine("Blad WS ears", "error");
  };

  earsWs.onclose = async () => {
    logLine("Mikrofon WS rozlaczony");
    await teardownMicrophonePipeline();
    earsWs = null;
    setMicButtons();
  };
}

function startCapture() {
  if (!earsWs || earsWs.readyState !== WebSocket.OPEN) {
    logLine("Najpierw polacz mikrofon", "error");
    return;
  }

  micCapturing = true;
  setMicButtons();
  earsWs.send("start_capture");
}

function stopCapture() {
  if (!earsWs || earsWs.readyState !== WebSocket.OPEN) {
    return;
  }
  earsWs.send("stop_capture");
}

function disconnectMic() {
  if (earsWs) {
    earsWs.close();
    earsWs = null;
  }
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
  els.connectMicBtn.addEventListener("click", connectMic);
  els.startCaptureBtn.addEventListener("click", startCapture);
  els.stopCaptureBtn.addEventListener("click", stopCapture);
  els.disconnectMicBtn.addEventListener("click", disconnectMic);
}

function start() {
  bindEvents();
  setMicButtons();
  refreshStatus();
  setInterval(refreshStatus, 10000);
  logLine("GUI gotowe");
}

start();
