const form = document.getElementById("form");
const svgFile = document.getElementById("svgFile");
const fileName = document.getElementById("fileName");
const runButton = document.getElementById("runButton");
const statusTitle = document.getElementById("statusTitle");
const statusText = document.getElementById("statusText");
const progressBar = document.getElementById("progressBar");
const downloadLink = document.getElementById("downloadLink");
const logEl = document.getElementById("log");
const sourceCanvas = document.getElementById("sourceCanvas");
const lineCanvas = document.getElementById("lineCanvas");

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
  setProgress(5);
  statusTitle.textContent = "读取 SVG";
  statusText.textContent = file.name;
  appendLog(`file: ${file.name}`);

  try {
    const options = readOptions();
    const svgText = await file.text();
    const raster = await renderSvg(svgText, options.rasterWidth);
    drawSourcePreview(raster);
    drawLinePreview(raster, options.lineThreshold);
    appendLog(`raster: ${raster.width} x ${raster.height}`);

    setProgress(32);
    statusTitle.textContent = "上传到 Worker";
    statusText.textContent = "正在生成底板和凸起线条";
    const stl = await requestStl(raster, options, file.name);

    setProgress(95);
    if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
    activeObjectUrl = URL.createObjectURL(stl.blob);
    downloadLink.href = activeObjectUrl;
    downloadLink.download = stl.filename || `${stripExt(file.name)}_relief_card.stl`;
    downloadLink.classList.remove("disabled");

    statusTitle.textContent = "完成";
    statusText.textContent = `${formatBytes(stl.blob.size)} · ${stl.triangles || "?"} triangles · ${stl.widthMm || "?"} x ${stl.heightMm || "?"} mm`;
    appendLog(statusText.textContent);
    setProgress(100);
  } catch (error) {
    statusTitle.textContent = "失败";
    statusText.textContent = error.message || String(error);
    appendLog(`error: ${statusText.textContent}`);
    setProgress(0);
  } finally {
    runButton.disabled = false;
  }
});

checkHealth();

function readOptions() {
  const data = new FormData(form);
  const number = (name) => Number(data.get(name));
  return {
    longMm: number("longMm"),
    rasterWidth: number("rasterWidth"),
    baseThickness: number("baseThickness"),
    featureHeight: number("featureHeight"),
    lineThreshold: number("lineThreshold"),
    grooveWidthPx: number("grooveWidthPx"),
    maskKeepComponents: number("maskKeepComponents"),
    bridgeWidthPx: number("bridgeWidthPx"),
    basePaddingPx: number("basePaddingPx"),
  };
}

async function renderSvg(svgText, rasterLongPx) {
  const blob = new Blob([svgText], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    await img.decode();
    const size = parseSvgSize(svgText, img);
    const scale = rasterLongPx / Math.max(size.width, size.height);
    const width = Math.max(1, Math.round(size.width * scale));
    const height = Math.max(1, Math.round(size.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    return { width, height, imageData };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function parseSvgSize(svgText, img) {
  const fallbackWidth = img.naturalWidth || 1000;
  const fallbackHeight = img.naturalHeight || fallbackWidth;
  try {
    const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
    const svg = doc.documentElement;
    const viewBox = svg.getAttribute("viewBox");
    if (viewBox) {
      const parts = viewBox.replace(/,/g, " ").split(/\s+/).map(Number);
      if (parts.length >= 4 && parts[2] > 0 && parts[3] > 0) return { width: parts[2], height: parts[3] };
    }
    const width = parseDimension(svg.getAttribute("width"), fallbackWidth);
    const height = parseDimension(svg.getAttribute("height"), fallbackHeight);
    return { width, height };
  } catch {
    return { width: fallbackWidth, height: fallbackHeight };
  }
}

function parseDimension(value, fallback) {
  if (!value) return fallback;
  const match = value.match(/[0-9.]+/);
  const parsed = match ? Number(match[0]) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

async function requestStl(raster, options, filename) {
  const params = new URLSearchParams({
    width: String(raster.width),
    height: String(raster.height),
    filename,
    longMm: String(options.longMm),
    baseThickness: String(options.baseThickness),
    featureHeight: String(options.featureHeight),
    lineThreshold: String(options.lineThreshold),
    grooveWidthPx: String(options.grooveWidthPx),
    maskKeepComponents: String(options.maskKeepComponents),
    bridgeWidthPx: String(options.bridgeWidthPx),
    basePaddingPx: String(options.basePaddingPx),
  });

  const response = await fetch(`/api/generate?${params.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: raster.imageData.data.buffer,
  });
  if (!response.ok) {
    let message = `Worker failed with HTTP ${response.status}`;
    try {
      const payload = await response.json();
      if (payload.error) message = payload.error;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }

  const blob = await response.blob();
  return {
    blob,
    filename: parseContentDisposition(response.headers.get("Content-Disposition")),
    triangles: response.headers.get("X-Card-Triangles"),
    widthMm: response.headers.get("X-Card-Width-Mm"),
    heightMm: response.headers.get("X-Card-Height-Mm"),
  };
}

function drawSourcePreview(raster) {
  sourceCanvas.width = raster.width;
  sourceCanvas.height = raster.height;
  sourceCanvas.getContext("2d").putImageData(raster.imageData, 0, 0);
}

function drawLinePreview(raster, threshold) {
  lineCanvas.width = raster.width;
  lineCanvas.height = raster.height;
  const src = raster.imageData.data;
  const out = new ImageData(raster.width, raster.height);
  for (let i = 0; i < raster.width * raster.height; i += 1) {
    const p = i * 4;
    const gray = src[p] * 0.299 + src[p + 1] * 0.587 + src[p + 2] * 0.114;
    const on = gray < threshold && src[p + 3] > 24;
    out.data[p] = on ? 255 : 18;
    out.data[p + 1] = on ? 255 : 18;
    out.data[p + 2] = on ? 255 : 18;
    out.data[p + 3] = 255;
  }
  lineCanvas.getContext("2d").putImageData(out, 0, 0);
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const payload = await response.json();
    appendLog(`health: ${payload.ok ? "ok" : "failed"}`);
  } catch (error) {
    appendLog(`health: ${error.message || String(error)}`);
  }
}

function parseContentDisposition(header) {
  if (!header) return null;
  const match = header.match(/filename="?([^";]+)"?/i);
  return match ? match[1] : null;
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

function stripExt(name) {
  return name.replace(/\.[^.]+$/, "");
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
