const EPS = 1e-9;
const MAX_DIMENSION = 900;
const MAX_PIXELS = 900_000;
const MAX_BODY_BYTES = MAX_PIXELS * 4;
const MAX_TRIANGLES = 1_800_000;
const API_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Expose-Headers": "Content-Disposition,Content-Length,X-Card-Width-Mm,X-Card-Height-Mm,X-Card-Triangles",
};

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    try {
      if (request.method === "OPTIONS" && url.pathname.startsWith("/api/")) {
        return new Response(null, { status: 204, headers: API_HEADERS });
      }
      if (url.pathname === "/api/health") return json({ ok: true, runtime: "cloudflare-worker" });
      if (url.pathname === "/api/generate") return await generateModel(request, url);
      if (env.ASSETS) return await env.ASSETS.fetch(request);
      return new Response("Not found", { status: 404 });
    } catch (error) {
      console.error(JSON.stringify({ level: "error", message: error.message || String(error) }));
      return json({ ok: false, error: error.message || String(error) }, 400);
    }
  },
};

async function generateModel(request, url) {
  if (request.method !== "POST") return json({ ok: false, error: "Use POST" }, 405);

  const width = readRequiredInt(url, "width", 1, MAX_DIMENSION);
  const height = readRequiredInt(url, "height", 1, MAX_DIMENSION);
  const pixels = width * height;
  if (pixels > MAX_PIXELS) throw new Error(`Raster too large: ${pixels} px. Lower raster width.`);

  const contentLength = Number(request.headers.get("Content-Length") || "0");
  if (contentLength > MAX_BODY_BYTES) throw new Error("RGBA upload is too large");

  const body = await request.arrayBuffer();
  if (body.byteLength !== pixels * 4) {
    throw new Error(`Expected RGBA body of ${pixels * 4} bytes, got ${body.byteLength}`);
  }

  const options = readOptions(url);
  const rgba = new Uint8ClampedArray(body);
  const processed = processRaster(rgba, width, height, options);
  const stl = buildStl(processed, options);

  const safeName = sanitizeFilename(url.searchParams.get("filename") || "relief-card.svg").replace(/\.svg$/i, "");
  const headers = new Headers(API_HEADERS);
  headers.set("Content-Type", "model/stl");
  headers.set("Content-Length", String(stl.byteLength));
  headers.set("Content-Disposition", `attachment; filename=\"${safeName}_relief_card.stl\"`);
  headers.set("Cache-Control", "no-store");
  headers.set("X-Card-Width-Mm", processed.physicalWidth.toFixed(2));
  headers.set("X-Card-Height-Mm", processed.physicalHeight.toFixed(2));
  headers.set("X-Card-Triangles", String(processed.triangles));

  console.log(JSON.stringify({
    level: "info",
    event: "model_generated",
    width: processed.width,
    height: processed.height,
    triangles: processed.triangles,
    bytes: stl.byteLength,
  }));

  return new Response(stl, { headers });
}

function readOptions(url) {
  return {
    longMm: readNumber(url, "longMm", 120, 20, 240),
    baseThickness: readNumber(url, "baseThickness", 1.5, 0.4, 6),
    featureHeight: readNumber(url, "featureHeight", 2, 0.2, 6),
    lineThreshold: readNumber(url, "lineThreshold", 225, 20, 250),
    grooveWidthPx: readNumber(url, "grooveWidthPx", 2.2, 0.5, 8),
    maskKeepComponents: readInt(url, "maskKeepComponents", 8, 1, 20),
    bridgeWidthPx: readInt(url, "bridgeWidthPx", 34, 0, 100),
    basePaddingPx: readInt(url, "basePaddingPx", 6, 0, 40),
  };
}

function processRaster(rgba, width, height, options) {
  let lineRaw = thresholdLines(rgba, width, height, options.lineThreshold);

  let barrier = dilate(lineRaw, width, height, 2);
  barrier = close(barrier, width, height, 2);
  const outside = floodOutside(barrier, width, height);
  let mask = invert(outside);
  mask = close(mask, width, height, 2);
  mask = fillHoles(mask, width, height);

  let strokePlate = dilate(lineRaw, width, height, Math.max(2, options.basePaddingPx));
  strokePlate = close(strokePlate, width, height, Math.max(2, Math.ceil(options.basePaddingPx / 2)));
  strokePlate = fillHoles(strokePlate, width, height);
  orInPlace(mask, strokePlate);

  mask = selectAndMergeMask(mask, width, height, {
    keep: options.maskKeepComponents,
    minArea: Math.max(500, Math.floor((width * height) / 1400)),
    bottomGap: 16,
    bridgeWidth: options.bridgeWidthPx,
  });
  mask = dilate(mask, width, height, options.basePaddingPx);
  mask = close(mask, width, height, 2);
  mask = fillHoles(mask, width, height);

  const crop = boundingBox(mask, width, height, 4);
  mask = cropArray(mask, width, crop);
  lineRaw = cropArray(lineRaw, width, crop);
  const croppedWidth = crop.x1 - crop.x0;
  const croppedHeight = crop.y1 - crop.y0;
  andInPlace(lineRaw, mask);

  let line = dilate(lineRaw, croppedWidth, croppedHeight, 1);
  line = close(line, croppedWidth, croppedHeight, 1);
  line = removeSmallComponents(line, croppedWidth, croppedHeight, 28);

  let groove = dilate(line, croppedWidth, croppedHeight, Math.ceil(options.grooveWidthPx));
  andInPlace(groove, mask);
  groove = close(groove, croppedWidth, croppedHeight, 1);
  groove = removeSmallComponents(groove, croppedWidth, croppedHeight, 36);
  groove = close(groove, croppedWidth, croppedHeight, 1);

  const pxMm = options.longMm / Math.max(croppedWidth, croppedHeight);
  return {
    width: croppedWidth,
    height: croppedHeight,
    pxMm,
    physicalWidth: croppedWidth * pxMm,
    physicalHeight: croppedHeight * pxMm,
    mask,
    groove,
    triangles: 0,
  };
}

function thresholdLines(rgba, width, height, threshold) {
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i += 1) {
    const p = i * 4;
    const gray = rgba[p] * 0.299 + rgba[p + 1] * 0.587 + rgba[p + 2] * 0.114;
    out[i] = gray < threshold && rgba[p + 3] > 24 ? 1 : 0;
  }
  return out;
}

function dilate(src, width, height, iterations) {
  let current = src;
  for (let n = 0; n < iterations; n += 1) {
    const next = new Uint8Array(current.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let on = 0;
        for (let dy = -1; dy <= 1 && !on; dy += 1) {
          const yy = y + dy;
          if (yy < 0 || yy >= height) continue;
          for (let dx = -1; dx <= 1; dx += 1) {
            const xx = x + dx;
            if (xx >= 0 && xx < width && current[yy * width + xx]) {
              on = 1;
              break;
            }
          }
        }
        next[y * width + x] = on;
      }
    }
    current = next;
  }
  return current;
}

function erode(src, width, height, iterations) {
  let current = src;
  for (let n = 0; n < iterations; n += 1) {
    const next = new Uint8Array(current.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let on = 1;
        for (let dy = -1; dy <= 1 && on; dy += 1) {
          const yy = y + dy;
          if (yy < 0 || yy >= height) {
            on = 0;
            break;
          }
          for (let dx = -1; dx <= 1; dx += 1) {
            const xx = x + dx;
            if (xx < 0 || xx >= width || !current[yy * width + xx]) {
              on = 0;
              break;
            }
          }
        }
        next[y * width + x] = on;
      }
    }
    current = next;
  }
  return current;
}

function close(src, width, height, iterations) {
  return erode(dilate(src, width, height, iterations), width, height, iterations);
}

function floodOutside(barrier, width, height) {
  const outside = new Uint8Array(width * height);
  const queue = new Int32Array(width * height);
  let head = 0;
  let tail = 0;
  const push = (idx) => {
    if (!barrier[idx] && !outside[idx]) {
      outside[idx] = 1;
      queue[tail] = idx;
      tail += 1;
    }
  };

  for (let x = 0; x < width; x += 1) {
    push(x);
    push((height - 1) * width + x);
  }
  for (let y = 0; y < height; y += 1) {
    push(y * width);
    push(y * width + width - 1);
  }

  while (head < tail) {
    const idx = queue[head];
    head += 1;
    const x = idx % width;
    const y = Math.floor(idx / width);
    if (x > 0) push(idx - 1);
    if (x < width - 1) push(idx + 1);
    if (y > 0) push(idx - width);
    if (y < height - 1) push(idx + width);
  }
  return outside;
}

function invert(src) {
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i += 1) out[i] = src[i] ? 0 : 1;
  return out;
}

function fillHoles(src, width, height) {
  return invert(floodOutside(src, width, height));
}

function labelComponents(src, width, height) {
  const labels = new Int32Array(src.length);
  const queue = new Int32Array(src.length);
  const areas = [0];
  const boxes = [null];
  let current = 0;
  for (let i = 0; i < src.length; i += 1) {
    if (!src[i] || labels[i]) continue;
    current += 1;
    let head = 0;
    let tail = 0;
    queue[tail++] = i;
    labels[i] = current;
    let area = 0;
    let x0 = width;
    let y0 = height;
    let x1 = 0;
    let y1 = 0;
    while (head < tail) {
      const idx = queue[head++];
      const x = idx % width;
      const y = Math.floor(idx / width);
      area += 1;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      const neighbors = [idx - 1, idx + 1, idx - width, idx + width];
      for (const next of neighbors) {
        if (next < 0 || next >= src.length) continue;
        if ((next === idx - 1 && x === 0) || (next === idx + 1 && x === width - 1)) continue;
        if (src[next] && !labels[next]) {
          labels[next] = current;
          queue[tail++] = next;
        }
      }
    }
    areas[current] = area;
    boxes[current] = { x0, y0, x1, y1 };
  }
  return { labels, areas, boxes, count: current };
}

function selectAndMergeMask(mask, width, height, settings) {
  const { labels, areas, boxes } = labelComponents(mask, width, height);
  const ids = [];
  for (let id = 1; id < areas.length; id += 1) ids.push(id);
  ids.sort((a, b) => areas[b] - areas[a]);
  if (!ids.length) return mask;

  const main = ids[0];
  const mainBox = boxes[main];
  const keep = [main];
  for (const id of ids.slice(1)) {
    if (areas[id] < settings.minArea) continue;
    const box = boxes[id];
    const mostlyBelow = box.y0 > mainBox.y1 + settings.bottomGap;
    const substantial = areas[id] > areas[main] * 0.015;
    if (mostlyBelow || substantial) keep.push(id);
    if (keep.length >= settings.keep) break;
  }

  const kept = new Set(keep);
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < labels.length; i += 1) {
    if (kept.has(labels[i])) out[i] = 1;
  }

  if (settings.bridgeWidth > 0 && keep.length > 1) {
    const mainProj = projection(labels, width, height, main);
    for (const id of keep.slice(1)) {
      const compProj = projection(labels, width, height, id);
      let x0 = width;
      let x1 = 0;
      let hasOverlap = false;
      for (let x = 0; x < width; x += 1) {
        if (mainProj[x] && compProj[x]) {
          hasOverlap = true;
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
        }
      }
      const bw = Math.round(settings.bridgeWidth);
      if (!hasOverlap) {
        x0 = Math.max(0, boxes[id].x0 - bw);
        x1 = Math.min(width - 1, boxes[id].x1 + bw);
      } else {
        x0 = Math.max(0, x0 - bw);
        x1 = Math.min(width - 1, x1 + bw);
      }
      const y0 = Math.max(0, mainBox.y1 - bw);
      const y1 = Math.min(height - 1, boxes[id].y0 + bw);
      for (let y = y0; y <= y1; y += 1) {
        for (let x = x0; x <= x1; x += 1) out[y * width + x] = 1;
      }
    }
  }

  return fillHoles(close(out, width, height, Math.max(1, Math.floor(settings.bridgeWidth / 12))), width, height);
}

function projection(labels, width, height, id) {
  const out = new Uint8Array(width);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (labels[y * width + x] === id) out[x] = 1;
    }
  }
  return out;
}

function removeSmallComponents(src, width, height, minArea) {
  const { labels, areas } = labelComponents(src, width, height);
  const out = new Uint8Array(src.length);
  for (let i = 0; i < labels.length; i += 1) {
    if (areas[labels[i]] >= minArea) out[i] = 1;
  }
  return out;
}

function boundingBox(src, width, height, pad) {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!src[y * width + x]) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  if (x1 < x0 || y1 < y0) throw new Error("No printable area detected");
  return {
    x0: Math.max(0, x0 - pad),
    y0: Math.max(0, y0 - pad),
    x1: Math.min(width, x1 + 1 + pad),
    y1: Math.min(height, y1 + 1 + pad),
  };
}

function cropArray(src, sourceWidth, crop) {
  const width = crop.x1 - crop.x0;
  const height = crop.y1 - crop.y0;
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const start = (crop.y0 + y) * sourceWidth + crop.x0;
    out.set(src.subarray(start, start + width), y * width);
  }
  return out;
}

function andInPlace(a, b) {
  for (let i = 0; i < a.length; i += 1) a[i] = a[i] && b[i] ? 1 : 0;
}

function orInPlace(a, b) {
  for (let i = 0; i < a.length; i += 1) a[i] = a[i] || b[i] ? 1 : 0;
}

function buildStl(processed, options) {
  const { mask, groove, width, height, pxMm } = processed;
  const payload = {
    width,
    height,
    pxMm,
    baseThickness: options.baseThickness,
    featureHeight: options.featureHeight,
  };
  const triangles = countTriangles(mask, groove, payload);
  if (triangles > MAX_TRIANGLES) {
    throw new Error(`Model too complex: ${triangles} triangles. Lower raster width or simplify the SVG.`);
  }
  processed.triangles = triangles;
  const out = new ArrayBuffer(84 + triangles * 50);
  const view = new DataView(out);
  const header = new TextEncoder().encode("SVG relief card generated by Cloudflare Worker");
  new Uint8Array(out, 0, header.length).set(header);
  view.setUint32(80, triangles, true);

  let offset = 84;
  offset = writePlanarRuns(view, offset, mask, groove, payload);
  offset = writeVerticalWalls(view, offset, mask, groove, payload);
  if (offset !== out.byteLength) throw new Error("Internal STL size mismatch");
  return out;
}

function countTriangles(mask, groove, payload) {
  let total = 0;
  forEachRun(mask, groove, payload.width, payload.height, () => {
    total += 4;
  });

  for (let y = 0; y < payload.height; y += 1) {
    for (let x = 0; x < payload.width; x += 1) {
      const idx = y * payload.width + x;
      const z = heightAt(mask, groove, idx, payload);
      if (z <= 0) continue;
      const left = x > 0 ? heightAt(mask, groove, idx - 1, payload) : 0;
      const right = x < payload.width - 1 ? heightAt(mask, groove, idx + 1, payload) : 0;
      const up = y > 0 ? heightAt(mask, groove, idx - payload.width, payload) : 0;
      const down = y < payload.height - 1 ? heightAt(mask, groove, idx + payload.width, payload) : 0;
      if (z > left + EPS) total += 2;
      if (z > right + EPS) total += 2;
      if (z > up + EPS) total += 2;
      if (z > down + EPS) total += 2;
    }
  }
  return total;
}

function writePlanarRuns(view, offset, mask, groove, payload) {
  forEachRun(mask, groove, payload.width, payload.height, (xStart, xEnd, y, isGroove) => {
    const z = payload.baseThickness + payload.featureHeight * isGroove;
    const x0 = xStart * payload.pxMm;
    const x1 = xEnd * payload.pxMm;
    const y0 = (payload.height - y - 1) * payload.pxMm;
    const y1 = (payload.height - y) * payload.pxMm;
    offset = tri(view, offset, x0, y0, z, x1, y0, z, x1, y1, z);
    offset = tri(view, offset, x0, y0, z, x1, y1, z, x0, y1, z);
    offset = tri(view, offset, x0, y0, 0, x1, y1, 0, x1, y0, 0);
    offset = tri(view, offset, x0, y0, 0, x0, y1, 0, x1, y1, 0);
  });
  return offset;
}

function writeVerticalWalls(view, offset, mask, groove, payload) {
  for (let y = 0; y < payload.height; y += 1) {
    const y0 = (payload.height - y - 1) * payload.pxMm;
    const y1 = (payload.height - y) * payload.pxMm;
    for (let x = 0; x < payload.width; x += 1) {
      const idx = y * payload.width + x;
      const z = heightAt(mask, groove, idx, payload);
      if (z <= 0) continue;
      const x0 = x * payload.pxMm;
      const x1 = (x + 1) * payload.pxMm;
      const left = x > 0 ? heightAt(mask, groove, idx - 1, payload) : 0;
      const right = x < payload.width - 1 ? heightAt(mask, groove, idx + 1, payload) : 0;
      const up = y > 0 ? heightAt(mask, groove, idx - payload.width, payload) : 0;
      const down = y < payload.height - 1 ? heightAt(mask, groove, idx + payload.width, payload) : 0;
      if (z > left + EPS) {
        offset = tri(view, offset, x0, y0, left, x0, y1, z, x0, y1, left);
        offset = tri(view, offset, x0, y0, left, x0, y0, z, x0, y1, z);
      }
      if (z > right + EPS) {
        offset = tri(view, offset, x1, y0, right, x1, y1, right, x1, y1, z);
        offset = tri(view, offset, x1, y0, right, x1, y1, z, x1, y0, z);
      }
      if (z > down + EPS) {
        offset = tri(view, offset, x0, y0, down, x1, y0, down, x1, y0, z);
        offset = tri(view, offset, x0, y0, down, x1, y0, z, x0, y0, z);
      }
      if (z > up + EPS) {
        offset = tri(view, offset, x0, y1, up, x1, y1, z, x1, y1, up);
        offset = tri(view, offset, x0, y1, up, x0, y1, z, x1, y1, z);
      }
    }
  }
  return offset;
}

function forEachRun(mask, groove, width, height, visit) {
  for (let y = 0; y < height; y += 1) {
    let x = 0;
    while (x < width) {
      const idx = y * width + x;
      if (!mask[idx]) {
        x += 1;
        continue;
      }
      const isGroove = groove[idx];
      const xStart = x;
      x += 1;
      while (x < width) {
        const next = y * width + x;
        if (!mask[next] || groove[next] !== isGroove) break;
        x += 1;
      }
      visit(xStart, x, y, isGroove);
    }
  }
}

function heightAt(mask, groove, idx, payload) {
  if (!mask[idx]) return 0;
  return payload.baseThickness + payload.featureHeight * groove[idx];
}

function tri(view, offset, ax, ay, az, bx, by, bz, cx, cy, cz) {
  const normal = faceNormal(ax, ay, az, bx, by, bz, cx, cy, cz);
  const values = [normal[0], normal[1], normal[2], ax, ay, az, bx, by, bz, cx, cy, cz];
  for (const value of values) {
    view.setFloat32(offset, value, true);
    offset += 4;
  }
  view.setUint16(offset, 0, true);
  return offset + 2;
}

function faceNormal(ax, ay, az, bx, by, bz, cx, cy, cz) {
  const ux = bx - ax;
  const uy = by - ay;
  const uz = bz - az;
  const vx = cx - ax;
  const vy = cy - ay;
  const vz = cz - az;
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const mag = Math.hypot(nx, ny, nz);
  return mag ? [nx / mag, ny / mag, nz / mag] : [0, 0, 0];
}

function readNumber(url, name, fallback, min, max) {
  const raw = url.searchParams.get(name);
  const value = raw === null ? fallback : Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${name} must be a number`);
  return Math.min(max, Math.max(min, value));
}

function readInt(url, name, fallback, min, max) {
  return Math.round(readNumber(url, name, fallback, min, max));
}

function readRequiredInt(url, name, min, max) {
  const raw = url.searchParams.get(name);
  if (raw === null) throw new Error(`${name} query param is required`);
  const value = Number(raw);
  if (!Number.isInteger(value)) throw new Error(`${name} must be an integer`);
  if (value < min || value > max) throw new Error(`${name} must be between ${min} and ${max}`);
  return value;
}

function sanitizeFilename(name) {
  return name.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 80) || "relief-card";
}

function json(payload, status = 200) {
  const headers = new Headers(API_HEADERS);
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("Cache-Control", "no-store");
  return new Response(JSON.stringify(payload), { status, headers });
}
