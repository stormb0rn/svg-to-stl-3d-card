"""
SVG → STL 3D Card Web Converter
Upload SVG files, get 3D printable STL cards back.
Supports concurrent multi-file processing via thread pool.
"""

import asyncio
import os
import uuid
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
import svgpathtools
import xml.etree.ElementTree as ET

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# === Config ===
UPLOAD_DIR = Path(__file__).parent / "uploads"
OUTPUT_DIR = Path(__file__).parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_WIDTH_MM = 85.0
TOTAL_HEIGHT_MM = 3.0
BASE_THICKNESS_MM = 1.0
LINE_HEIGHT_MM = 2.0
CARD_MARGIN_MM = 1.5
SAMPLE_POINTS = 300
MIN_AREA_SVG = 5.0
BRIDGE_RADIUS = 80

# Thread pool for CPU-bound SVG→STL conversion
executor = ThreadPoolExecutor(max_workers=4)

# Track job status: {job_id: {status, filename, progress, error, output_file}}
jobs: dict[str, dict] = {}

app = FastAPI(title="SVG → STL 3D Card Converter")


# ──────────────────────────────────────────────
#  Core conversion logic (runs in thread pool)
# ──────────────────────────────────────────────

def svg_path_to_polygons(path):
    subpaths, current = [], []
    for seg in path:
        if current and abs(seg.start - current[-1].end) > 0.5:
            subpaths.append(current); current = [seg]
        else:
            current.append(seg)
    if current:
        subpaths.append(current)

    shells, holes = [], []
    for sub in subpaths:
        n_segs = len(sub)
        if n_segs == 0:
            continue
        pts_per_seg = max(4, SAMPLE_POINTS // n_segs)
        points = []
        for seg in sub:
            for t in np.linspace(0, 1, pts_per_seg, endpoint=False):
                pt = seg.point(t)
                points.append((pt.real, pt.imag))
        if len(points) < 3:
            continue
        points.append(points[0])
        n = len(points)
        area = sum(
            points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
            for i in range(n)
        ) / 2
        if abs(area) < MIN_AREA_SVG:
            continue
        (shells if area > 0 else holes).append(points)

    result = []
    for shell_pts in shells:
        shell_poly = Polygon(shell_pts)
        if not shell_poly.is_valid:
            shell_poly = make_valid(shell_poly)
        if shell_poly.is_empty:
            continue
        my_holes = []
        for hole_pts in holes:
            try:
                if shell_poly.contains(Polygon(hole_pts).centroid):
                    my_holes.append(hole_pts)
            except Exception:
                continue
        try:
            poly = Polygon(shell_pts, my_holes) if my_holes else shell_poly
            if not poly.is_valid:
                poly = make_valid(poly)
            if not poly.is_empty:
                result.append(poly)
        except Exception:
            if not shell_poly.is_empty:
                result.append(shell_poly)
    return result


def collect_polygons(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polys = []
        for g in geom.geoms:
            polys.extend(collect_polygons(g))
        return polys
    return []


def extrude_geometry(geom, z_bottom, z_top):
    polys = collect_polygons(geom)
    meshes = []
    height = z_top - z_bottom
    if height <= 0:
        return None
    for poly in polys:
        if poly.is_empty or poly.area < 0.01:
            continue
        try:
            ext = trimesh.creation.extrude_polygon(poly, height)
            ext.apply_translation([0, 0, z_bottom])
            meshes.append(ext)
        except Exception:
            try:
                s = poly.simplify(0.05)
                if s.is_empty or s.area < 0.01:
                    continue
                ext = trimesh.creation.extrude_polygon(s, height)
                ext.apply_translation([0, 0, z_bottom])
                meshes.append(ext)
            except Exception:
                continue
    return trimesh.util.concatenate(meshes) if meshes else None


def convert_svg_to_stl(job_id: str, svg_path: str, stl_path: str):
    """Main conversion — runs in a worker thread."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10

        # Parse SVG
        paths, attrs = svgpathtools.svg2paths(svg_path)
        tree = ET.parse(svg_path)
        root = tree.getroot()
        svg_w = float(root.get('width', 800))
        svg_h = float(root.get('height', 800))

        jobs[job_id]["progress"] = 20

        # Process paths in SVG paint order
        visible_lines = Polygon()
        all_content = Polygon()
        bg_done = False
        total = len(paths)

        for i, (path, attr) in enumerate(zip(paths, attrs)):
            fill = attr.get('fill', None)
            if fill == 'white' and not bg_done:
                polys = svg_path_to_polygons(path)
                if polys and polys[0].area > svg_w * svg_h * 0.8:
                    bg_done = True
                    continue

            polygons = svg_path_to_polygons(path)
            if not polygons:
                continue
            pu = unary_union(polygons)
            if pu.is_empty:
                continue
            if not pu.is_valid:
                pu = make_valid(pu)
            try:
                all_content = all_content.union(pu)
            except Exception:
                pass
            if fill is None:
                try:
                    visible_lines = visible_lines.union(pu)
                except Exception:
                    pass
            elif fill == 'white':
                try:
                    visible_lines = visible_lines.difference(pu)
                except Exception:
                    pass

            jobs[job_id]["progress"] = 20 + int(40 * (i + 1) / total)

        if not visible_lines.is_valid:
            visible_lines = make_valid(visible_lines)
        if not all_content.is_valid:
            all_content = make_valid(all_content)

        if all_content.is_empty:
            raise ValueError("SVG contains no parseable content")

        jobs[job_id]["progress"] = 65

        # Card outline
        cb = all_content.bounds
        cw = cb[2] - cb[0]
        ch = cb[3] - cb[1]
        scale = TARGET_WIDTH_MM / cw
        h_mm = ch * scale
        margin_svg = CARD_MARGIN_MM / scale

        card_outline = all_content.buffer(BRIDGE_RADIUS, resolution=64)
        card_outline = card_outline.buffer(-BRIDGE_RADIUS + margin_svg, resolution=64)
        # Smooth pass: small buffer out+in rounds off sharp corners
        smooth_r = margin_svg * 0.5
        card_outline = card_outline.buffer(smooth_r, resolution=64).buffer(-smooth_r, resolution=64)
        card_outline = card_outline.simplify(0.3)

        outline_polys = collect_polygons(card_outline)
        if len(outline_polys) > 1:
            card_outline = max(outline_polys, key=lambda p: p.area)

        jobs[job_id]["progress"] = 75

        # Scale to mm
        from shapely.affinity import translate, scale as sh_scale

        def scale_geom(geom):
            geom = translate(geom, -cb[0], -cb[1])
            geom = sh_scale(geom, xfact=scale, yfact=scale, origin=(0, 0))
            geom = sh_scale(geom, xfact=1, yfact=-1, origin=(0, h_mm / 2))
            return geom

        card_mm = scale_geom(card_outline)
        lines_mm = scale_geom(visible_lines)

        jobs[job_id]["progress"] = 80

        # Build mesh
        all_meshes = []
        base = extrude_geometry(card_mm, 0, BASE_THICKNESS_MM)
        if base:
            all_meshes.append(base)
        lines = extrude_geometry(lines_mm, BASE_THICKNESS_MM, TOTAL_HEIGHT_MM)
        if lines:
            all_meshes.append(lines)

        if not all_meshes:
            raise ValueError("Failed to generate 3D mesh")

        jobs[job_id]["progress"] = 90

        final = trimesh.util.concatenate(all_meshes)
        centroid = final.bounds.mean(axis=0)
        final.apply_translation([-centroid[0], -centroid[1], -final.bounds[0][2]])
        final.export(stl_path)

        dims = final.bounds[1] - final.bounds[0]
        sz = os.path.getsize(stl_path) / (1024 * 1024)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["output_file"] = stl_path
        jobs[job_id]["dimensions"] = f"{dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f} mm"
        jobs[job_id]["file_size"] = f"{sz:.1f} MB"
        jobs[job_id]["faces"] = len(final.faces)

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 0


# ──────────────────────────────────────────────
#  API Endpoints
# ──────────────────────────────────────────────

@app.post("/upload")
async def upload_svg(files: list[UploadFile] = File(...)):
    """Upload one or more SVG files. Returns job IDs for tracking."""
    results = []
    for f in files:
        if not f.filename.lower().endswith('.svg'):
            results.append({"filename": f.filename, "error": "Not an SVG file"})
            continue

        job_id = str(uuid.uuid4())[:8]
        basename = Path(f.filename).stem
        svg_path = str(UPLOAD_DIR / f"{job_id}_{f.filename}")
        stl_path = str(OUTPUT_DIR / f"{basename}.stl")

        content = await f.read()
        with open(svg_path, "wb") as fp:
            fp.write(content)

        jobs[job_id] = {
            "status": "queued",
            "filename": f.filename,
            "progress": 0,
            "error": None,
            "output_file": None,
            "started": time.time(),
        }

        # Submit to thread pool
        executor.submit(convert_svg_to_stl, job_id, svg_path, stl_path)
        results.append({"filename": f.filename, "job_id": job_id})

    return JSONResponse(results)


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {
        "job_id": job_id,
        "filename": j["filename"],
        "status": j["status"],
        "progress": j["progress"],
        "error": j.get("error"),
        "dimensions": j.get("dimensions"),
        "file_size": j.get("file_size"),
        "faces": j.get("faces"),
    }


@app.get("/download/{job_id}")
async def download_stl(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "done" or not j.get("output_file"):
        raise HTTPException(400, "File not ready")
    basename = Path(j["filename"]).stem + ".stl"
    return FileResponse(j["output_file"], filename=basename, media_type="application/octet-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# ──────────────────────────────────────────────
#  Frontend
# ──────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SVG → STL 3D Card Converter</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }
  .container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
  h1 { font-size: 28px; font-weight: 600; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; margin-bottom: 32px; font-size: 14px; }

  .drop-zone {
    border: 2px dashed #444; border-radius: 16px; padding: 60px 20px;
    text-align: center; cursor: pointer; transition: all 0.2s;
    background: #1a1a1a; margin-bottom: 32px;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: #5b8def; background: #1a1a2e;
  }
  .drop-zone svg { width: 48px; height: 48px; margin-bottom: 16px; fill: #555; }
  .drop-zone p { font-size: 16px; color: #888; }
  .drop-zone .hint { font-size: 13px; color: #555; margin-top: 8px; }

  .job-list { display: flex; flex-direction: column; gap: 12px; }
  .job-card {
    background: #1a1a1a; border-radius: 12px; padding: 16px 20px;
    border: 1px solid #2a2a2a; transition: border-color 0.2s;
  }
  .job-card.done { border-color: #2d5a2d; }
  .job-card.error { border-color: #5a2d2d; }

  .job-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .job-name { font-weight: 500; font-size: 15px; color: #fff; }
  .job-status {
    font-size: 12px; padding: 3px 10px; border-radius: 20px;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .job-status.queued { background: #2a2a2a; color: #888; }
  .job-status.processing { background: #1a2a4a; color: #5b8def; }
  .job-status.done { background: #1a3a1a; color: #5bef5b; }
  .job-status.error { background: #3a1a1a; color: #ef5b5b; }

  .progress-bar {
    height: 4px; background: #2a2a2a; border-radius: 2px;
    overflow: hidden; margin-bottom: 10px;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, #5b8def, #8b5bef);
    border-radius: 2px; transition: width 0.3s ease;
  }

  .job-info { font-size: 13px; color: #666; display: flex; gap: 16px; flex-wrap: wrap; }
  .job-info span { white-space: nowrap; }
  .job-error { font-size: 13px; color: #ef5b5b; margin-top: 4px; }

  .download-btn {
    display: inline-block; margin-top: 10px; padding: 8px 20px;
    background: #5b8def; color: #fff; border: none; border-radius: 8px;
    font-size: 14px; font-weight: 500; cursor: pointer; text-decoration: none;
    transition: background 0.2s;
  }
  .download-btn:hover { background: #4a7cde; }

  .config { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
  .config label { font-size: 13px; color: #888; display: flex; align-items: center; gap: 6px; }
  .config input {
    width: 60px; padding: 4px 8px; background: #1a1a1a; border: 1px solid #333;
    border-radius: 6px; color: #fff; font-size: 13px; text-align: center;
  }
  .config span.unit { font-size: 12px; color: #555; }
</style>
</head>
<body>
<div class="container">
  <h1>SVG → STL 3D Card</h1>
  <p class="subtitle">Upload SVG line art, get 3D printable contour cards. Supports batch processing.</p>

  <div class="config">
    <label>Width <input type="number" id="cfg-width" value="85" step="5"> <span class="unit">mm</span></label>
    <label>Base <input type="number" id="cfg-base" value="1.0" step="0.5"> <span class="unit">mm</span></label>
    <label>Lines <input type="number" id="cfg-lines" value="2.0" step="0.5"> <span class="unit">mm</span></label>
  </div>

  <div class="drop-zone" id="dropZone">
    <svg viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
    <p>Drop SVG files here or click to upload</p>
    <p class="hint">Supports multiple files — processed in parallel</p>
    <input type="file" id="fileInput" accept=".svg" multiple hidden>
  </div>

  <div class="job-list" id="jobList"></div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const jobList = document.getElementById('jobList');
const activePolls = new Set();

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

async function handleFiles(files) {
  const svgFiles = [...files].filter(f => f.name.toLowerCase().endsWith('.svg'));
  if (!svgFiles.length) return;

  const formData = new FormData();
  svgFiles.forEach(f => formData.append('files', f));

  // Create placeholder cards
  svgFiles.forEach(f => {
    const card = createJobCard(null, f.name);
    jobList.prepend(card);
  });

  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const results = await resp.json();
    // Update placeholders with real job IDs
    const placeholders = jobList.querySelectorAll('.job-card[data-job="pending"]');
    results.forEach((r, i) => {
      if (placeholders[svgFiles.length - 1 - i]) {
        const card = placeholders[svgFiles.length - 1 - i];
        if (r.error) {
          card.dataset.job = 'error';
          card.querySelector('.job-status').textContent = 'Error';
          card.querySelector('.job-status').className = 'job-status error';
          card.querySelector('.job-error').textContent = r.error;
        } else {
          card.dataset.job = r.job_id;
          pollJob(r.job_id, card);
        }
      }
    });
  } catch (e) {
    console.error('Upload failed:', e);
  }
  fileInput.value = '';
}

function createJobCard(jobId, filename) {
  const card = document.createElement('div');
  card.className = 'job-card';
  card.dataset.job = jobId || 'pending';
  card.innerHTML = `
    <div class="job-header">
      <span class="job-name">${filename}</span>
      <span class="job-status queued">Queued</span>
    </div>
    <div class="progress-bar"><div class="progress-fill" style="width:0%"></div></div>
    <div class="job-info"></div>
    <div class="job-error" style="display:none"></div>
    <div class="job-actions"></div>
  `;
  return card;
}

async function pollJob(jobId, card) {
  if (activePolls.has(jobId)) return;
  activePolls.add(jobId);

  const poll = async () => {
    try {
      const resp = await fetch(`/status/${jobId}`);
      const data = await resp.json();

      const statusEl = card.querySelector('.job-status');
      statusEl.textContent = data.status === 'done' ? 'Done' :
                             data.status === 'error' ? 'Error' :
                             data.status === 'processing' ? 'Processing' : 'Queued';
      statusEl.className = `job-status ${data.status}`;
      card.className = `job-card ${data.status}`;

      card.querySelector('.progress-fill').style.width = data.progress + '%';

      if (data.status === 'done') {
        card.querySelector('.job-info').innerHTML = `
          <span>${data.dimensions}</span>
          <span>${data.file_size}</span>
          <span>${data.faces?.toLocaleString()} faces</span>
        `;
        card.querySelector('.job-actions').innerHTML = `
          <a class="download-btn" href="/download/${jobId}">Download STL</a>
        `;
        activePolls.delete(jobId);
        return;
      }
      if (data.status === 'error') {
        const errEl = card.querySelector('.job-error');
        errEl.style.display = 'block';
        errEl.textContent = data.error || 'Unknown error';
        activePolls.delete(jobId);
        return;
      }

      setTimeout(poll, 500);
    } catch (e) {
      setTimeout(poll, 1000);
    }
  };
  poll();
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    print(f"Starting SVG → STL converter on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
