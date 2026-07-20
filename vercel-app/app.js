const form = document.getElementById("form");
const svgFile = document.getElementById("svgFile");
const fileName = document.getElementById("fileName");
const runButton = document.getElementById("runButton");
const statusTitle = document.getElementById("statusTitle");
const statusText = document.getElementById("statusText");
const progressBar = document.getElementById("progressBar");
const downloadLink = document.getElementById("downloadLink");
const logEl = document.getElementById("log");
const canvases = {
  mask: document.getElementById("maskCanvas"),
  groove: document.getElementById("grooveCanvas"),
  line: document.getElementById("lineCanvas"),
};

let activeObjectUrl = null;

svgFile.addEventListener("change", () => {
  fileName.textContent = svgFile.files[0]?.name || "未选择文件";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = svgFile.files[0];
  if (!file) return;

  resetOutput();
  runButton.disabled = true;
  statusTitle.textContent = "读取 SVG";
  statusText.textContent = file.name;
  appendLog(`file: ${file.name}`);

  try {
    const options = readOptions();
    const svgText = await file.text();
    const raster = await renderSvg(svgText, options.rasterWidth);
    drawSourcePreview(raster);

    statusTitle.textContent = "提取线稿和底板";
    const processed = await runPipeline(raster, options);
    drawMaskPreview(canvases.mask, processed.mask, processed.width, processed.height, [236, 247, 245]);
    drawMaskPreview(canvases.groove, processed.groove, processed.width, processed.height, [255, 110, 104]);

    statusTitle.textContent = "多线程生成 STL";
    const mesh = await buildMesh(processed, options);
    const stl = assembleStl(mesh.chunks, mesh.triangles);
    const blob = new Blob([stl], { type: "model/stl" });
    activeObjectUrl = URL.createObjectURL(blob);
    downloadLink.href = activeObjectUrl;
    downloadLink.download = `${stripExt(file.name)}_relief_card.stl`;
    downloadLink.classList.remove("disabled");
    statusTitle.textContent = "完成";
    statusText.textContent = `${formatBytes(blob.size)} · ${mesh.triangles.toLocaleString()} triangles · ${processed.physicalWidth.toFixed(1)} x ${processed.physicalHeight.toFixed(1)} mm`;
    setProgress(100);
    appendLog(statusText.textContent);
  } catch (error) {
    statusTitle.textContent = "失败";
    statusText.textContent = error.message || String(error);
    appendLog(`error: ${statusText.textContent}`);
  } finally {
    runButton.disabled = false;
  }
});

function readOptions() {
  const data = new FormData(form);
  const number = (name) => Number(data.get(name));
  const maxWorkers = navigator.hardwareConcurrency || 4;
  return {
    longMm: number("longMm"),
    rasterWidth: number("rasterWidth"),
    baseThickness: number("baseThickness"),
    featureHeight: number("featureHeight"),
    lineThreshold: number("lineThreshold"),
    grooveWidthPx: number("grooveWidthPx"),
    maskKeepComponents: number("maskKeepComponents"),
    bridgeWidthPx: number("bridgeWidthPx"),
    workerCount: Math.max(1, Math.min(number("workerCount"), maxWorkers, 12)),
  };
}

async function renderSvg(svgText, rasterWidth) {
  const blob = new Blob([svgText], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    await img.decode();
    const naturalWidth = img.naturalWidth || 1000;
    const naturalHeight = img.naturalHeight || naturalWidth;
    const width = Math.round(rasterWidth);
    const height = Math.max(1, Math.round(width * naturalHeight / naturalWidth));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    return { width, height, data: imageData.data.buffer, source: imageData };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function runPipeline(raster, options) {
  return new Promise((resolve, reject) => {
    const worker = new Worker("/worker.js");
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "progress") {
        setProgress(msg.value);
        statusText.textContent = msg.label;
      } else if (msg.type === "complete") {
        worker.terminate();
        appendLog(`raster: ${msg.width} x ${msg.height}`);
        appendLog(`size: ${msg.physicalWidth.toFixed(2)} x ${msg.physicalHeight.toFixed(2)} mm`);
        resolve(msg);
      } else if (msg.type === "error") {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message));
    };
    worker.postMessage({
      type: "processImage",
      width: raster.width,
      height: raster.height,
      data: raster.data,
      options,
    }, [raster.data]);
  });
}

async function buildMesh(processed, options) {
  const shared = supportsSharedBuffers();
  const maskSource = new Uint8Array(processed.mask);
  const grooveSource = new Uint8Array(processed.groove);
  const maskBuffer = shared ? toShared(maskSource) : processed.mask;
  const grooveBuffer = shared ? toShared(grooveSource) : processed.groove;
  const workers = Math.max(1, options.workerCount);
  const rowStep = Math.ceil(processed.height / workers);
  const tasks = [];

  for (let i = 0; i < workers; i += 1) {
    const startRow = i * rowStep;
    const endRow = Math.min(processed.height, startRow + rowStep);
    if (startRow >= endRow) continue;
    tasks.push(runMeshChunk({
      width: processed.width,
      height: processed.height,
      startRow,
      endRow,
      pxMm: processed.pxMm,
      baseThickness: options.baseThickness,
      featureHeight: options.featureHeight,
      maskBuffer: shared ? maskBuffer : maskSource.slice().buffer,
      grooveBuffer: shared ? grooveBuffer : grooveSource.slice().buffer,
      workerIndex: i,
      transfer: !shared,
    }));
  }

  const chunks = await Promise.all(tasks);
  chunks.sort((a, b) => a.startRow - b.startRow);
  const triangles = chunks.reduce((sum, chunk) => sum + chunk.triangles, 0);
  return { chunks, triangles };
}

function runMeshChunk(payload) {
  return new Promise((resolve, reject) => {
    const worker = new Worker("/worker.js");
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "meshProgress") {
        const base = 62 + msg.workerIndex * (30 / Math.max(1, readOptions().workerCount));
        setProgress(Math.min(94, base + msg.value));
      } else if (msg.type === "meshChunk") {
        worker.terminate();
        resolve(msg);
      } else if (msg.type === "error") {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message));
    };
    const transfers = payload.transfer ? [payload.maskBuffer, payload.grooveBuffer] : [];
    worker.postMessage({ type: "meshChunk", payload }, transfers);
  });
}

function assembleStl(chunks, triangleCount) {
  const totalBytes = 84 + triangleCount * 50;
  const out = new ArrayBuffer(totalBytes);
  const view = new DataView(out);
  const header = new TextEncoder().encode("SVG relief card generated in browser");
  new Uint8Array(out, 0, header.length).set(header);
  view.setUint32(80, triangleCount, true);
  let offset = 84;
  for (const chunk of chunks) {
    new Uint8Array(out, offset, chunk.buffer.byteLength).set(new Uint8Array(chunk.buffer));
    offset += chunk.buffer.byteLength;
  }
  return out;
}

function drawSourcePreview(raster) {
  const canvas = canvases.line;
  canvas.width = raster.source.width;
  canvas.height = raster.source.height;
  canvas.getContext("2d").putImageData(raster.source, 0, 0);
}

function drawMaskPreview(canvas, buffer, width, height, color) {
  canvas.width = width;
  canvas.height = height;
  const source = new Uint8Array(buffer);
  const image = new ImageData(width, height);
  for (let i = 0; i < source.length; i += 1) {
    const p = i * 4;
    if (source[i]) {
      image.data[p] = color[0];
      image.data[p + 1] = color[1];
      image.data[p + 2] = color[2];
      image.data[p + 3] = 255;
    } else {
      image.data[p] = 7;
      image.data[p + 1] = 16;
      image.data[p + 2] = 19;
      image.data[p + 3] = 255;
    }
  }
  canvas.getContext("2d").putImageData(image, 0, 0);
}

function resetOutput() {
  if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
  activeObjectUrl = null;
  downloadLink.classList.add("disabled");
  downloadLink.href = "#";
  logEl.textContent = "";
  setProgress(0);
}

function setProgress(value) {
  progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`;
}

function appendLog(line) {
  logEl.textContent += `${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function supportsSharedBuffers() {
  return typeof SharedArrayBuffer !== "undefined" && crossOriginIsolated;
}

function toShared(array) {
  const shared = new SharedArrayBuffer(array.byteLength);
  new Uint8Array(shared).set(array);
  return shared;
}

function stripExt(name) {
  return name.replace(/\.[^.]+$/, "");
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
