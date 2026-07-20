const EPS = 1e-9;

self.onmessage = (event) => {
  const { type } = event.data;
  try {
    if (type === "processImage") processImage(event.data);
    if (type === "meshChunk") buildMeshChunk(event.data.payload);
  } catch (error) {
    self.postMessage({ type: "error", message: error.message || String(error) });
  }
};

function processImage({ width, height, data, options }) {
  const rgba = new Uint8ClampedArray(data);
  progress(8, "读取像素");
  let lineRaw = thresholdLines(rgba, width, height, options.lineThreshold);

  progress(16, "生成底板边界");
  let barrier = dilate(lineRaw, width, height, 2);
  barrier = close(barrier, width, height, 2);
  const outside = floodOutside(barrier, width, height);
  let mask = invert(outside);
  mask = close(mask, width, height, 2);
  mask = fillHoles(mask, width, height);

  progress(30, "筛选主体和底部文字");
  mask = selectAndMergeMask(mask, width, height, {
    keep: options.maskKeepComponents,
    minArea: 700,
    bottomGap: 12,
    bridgeWidth: options.bridgeWidthPx,
  });

  const crop = boundingBox(mask, width, height, 3);
  mask = cropArray(mask, width, crop);
  lineRaw = cropArray(lineRaw, width, crop);
  const croppedWidth = crop.x1 - crop.x0;
  const croppedHeight = crop.y1 - crop.y0;
  andInPlace(lineRaw, mask);

  progress(44, "修补线条");
  let line = dilate(lineRaw, croppedWidth, croppedHeight, 1);
  line = close(line, croppedWidth, croppedHeight, 1);
  line = removeSmallComponents(line, croppedWidth, croppedHeight, 28);

  progress(56, "生成凸起区域");
  let groove = dilate(line, croppedWidth, croppedHeight, Math.ceil(options.grooveWidthPx));
  andInPlace(groove, mask);
  groove = close(groove, croppedWidth, croppedHeight, 1);
  groove = removeSmallComponents(groove, croppedWidth, croppedHeight, 36);
  groove = close(groove, croppedWidth, croppedHeight, 1);

  const pxMm = options.longMm / Math.max(croppedWidth, croppedHeight);
  self.postMessage({
    type: "complete",
    width: croppedWidth,
    height: croppedHeight,
    pxMm,
    physicalWidth: croppedWidth * pxMm,
    physicalHeight: croppedHeight * pxMm,
    mask: mask.buffer,
    groove: groove.buffer,
  }, [mask.buffer, groove.buffer]);
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
    if (box.y0 > mainBox.y1 + settings.bottomGap) keep.push(id);
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
  let x1 = 0;
  let y1 = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!src[y * width + x]) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
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

function buildMeshChunk(payload) {
  const mask = new Uint8Array(payload.maskBuffer);
  const groove = new Uint8Array(payload.grooveBuffer);
  const triangles = countChunkTriangles(mask, groove, payload);
  const buffer = new ArrayBuffer(triangles * 50);
  const view = new DataView(buffer);
  let offset = 0;
  const heightAt = (idx) => payload.baseThickness + payload.featureHeight * groove[idx];

  for (let y = payload.startRow; y < payload.endRow; y += 1) {
    if ((y - payload.startRow) % 24 === 0) {
      self.postMessage({
        type: "meshProgress",
        workerIndex: payload.workerIndex,
        value: ((y - payload.startRow) / Math.max(1, payload.endRow - payload.startRow)) * 28,
      });
    }
    const y0 = (payload.height - y - 1) * payload.pxMm;
    const y1 = (payload.height - y) * payload.pxMm;
    for (let x = 0; x < payload.width; x += 1) {
      const idx = y * payload.width + x;
      if (!mask[idx]) continue;
      const x0 = x * payload.pxMm;
      const x1 = (x + 1) * payload.pxMm;
      const z = heightAt(idx);
      offset = tri(view, offset, x0, y0, z, x1, y0, z, x1, y1, z);
      offset = tri(view, offset, x0, y0, z, x1, y1, z, x0, y1, z);
      offset = tri(view, offset, x0, y0, 0, x1, y1, 0, x1, y0, 0);
      offset = tri(view, offset, x0, y0, 0, x0, y1, 0, x1, y1, 0);

      const left = x > 0 && mask[idx - 1] ? heightAt(idx - 1) : 0;
      const right = x < payload.width - 1 && mask[idx + 1] ? heightAt(idx + 1) : 0;
      const up = y > 0 && mask[idx - payload.width] ? heightAt(idx - payload.width) : 0;
      const down = y < payload.height - 1 && mask[idx + payload.width] ? heightAt(idx + payload.width) : 0;
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
  self.postMessage({ type: "meshChunk", startRow: payload.startRow, triangles, buffer }, [buffer]);
}

function countChunkTriangles(mask, groove, payload) {
  let total = 0;
  const heightAt = (idx) => payload.baseThickness + payload.featureHeight * groove[idx];
  for (let y = payload.startRow; y < payload.endRow; y += 1) {
    for (let x = 0; x < payload.width; x += 1) {
      const idx = y * payload.width + x;
      if (!mask[idx]) continue;
      total += 4;
      const z = heightAt(idx);
      const left = x > 0 && mask[idx - 1] ? heightAt(idx - 1) : 0;
      const right = x < payload.width - 1 && mask[idx + 1] ? heightAt(idx + 1) : 0;
      const up = y > 0 && mask[idx - payload.width] ? heightAt(idx - payload.width) : 0;
      const down = y < payload.height - 1 && mask[idx + payload.width] ? heightAt(idx + payload.width) : 0;
      if (z > left + EPS) total += 2;
      if (z > right + EPS) total += 2;
      if (z > up + EPS) total += 2;
      if (z > down + EPS) total += 2;
    }
  }
  return total;
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

function progress(value, label) {
  self.postMessage({ type: "progress", value, label });
}
