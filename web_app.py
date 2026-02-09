"""
SVG → STL 3D Card Web Converter
Upload SVG files, get 3D printable STL cards back.
Supports concurrent multi-file processing via thread pool.
History stored in Supabase.
"""

import asyncio
import os
import uuid
import time
import shutil
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
import svgpathtools
import xml.etree.ElementTree as ET

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

# Supabase config
SUPABASE_URL = "https://fwytawawmtenhbnwhunc.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ3eXRhd2F3bXRlbmhibndodW5jIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIzODA3MjcsImV4cCI6MjA3Nzk1NjcyN30.oJSo5rG7U4HcA0L5lAPechmyKWLLcB0ce0nNmSxnqhA"
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY",
    "REDACTED")
STL_BUCKET = "stl-files"

# Thread pool for CPU-bound SVG→STL conversion
executor = ThreadPoolExecutor(max_workers=4)

# Track job status
jobs: dict[str, dict] = {}

app = FastAPI(title="SVG → STL 3D Card Converter")


# ──────────────────────────────────────────────
#  Supabase helpers
# ──────────────────────────────────────────────

def upload_to_supabase(stl_path: str, storage_name: str):
    """Upload STL file to Supabase Storage."""
    import urllib.request
    with open(stl_path, "rb") as f:
        data = f.read()
    url = f"{SUPABASE_URL}/storage/v1/object/{STL_BUCKET}/{quote(storage_name)}"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("apikey", SUPABASE_SERVICE_KEY)
    req.add_header("Authorization", f"Bearer {SUPABASE_SERVICE_KEY}")
    req.add_header("Content-Type", "application/octet-stream")
    try:
        urllib.request.urlopen(req)
        return f"{SUPABASE_URL}/storage/v1/object/public/{STL_BUCKET}/{quote(storage_name)}"
    except Exception as e:
        print(f"  Storage upload failed: {e}")
        return None


def save_history(filename, dimensions, file_size, faces, stl_url, config):
    """Save generation record to Supabase."""
    import urllib.request
    import json
    url = f"{SUPABASE_URL}/rest/v1/stl_generations"
    payload = json.dumps({
        "filename": filename,
        "dimensions": dimensions,
        "file_size": file_size,
        "faces": faces,
        "stl_storage_path": stl_url,
        "config": config,
    }).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("apikey", SUPABASE_SERVICE_KEY)
    req.add_header("Authorization", f"Bearer {SUPABASE_SERVICE_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Prefer", "return=minimal")
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"  History save failed: {e}")


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


def convert_svg_to_stl(job_id: str, svg_path: str, stl_path: str,
                       target_width=TARGET_WIDTH_MM, base_thick=BASE_THICKNESS_MM, total_h=TOTAL_HEIGHT_MM):
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
        scale = target_width / cw
        h_mm = ch * scale
        margin_svg = CARD_MARGIN_MM / scale

        card_outline = all_content.buffer(BRIDGE_RADIUS, resolution=64)
        card_outline = card_outline.buffer(-BRIDGE_RADIUS + margin_svg, resolution=64)
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
        base_mesh = extrude_geometry(card_mm, 0, base_thick)
        if base_mesh:
            all_meshes.append(base_mesh)
        lines = extrude_geometry(lines_mm, base_thick, total_h)
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

        dim_str = f"{dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm"
        size_str = f"{sz:.1f} MB"
        face_count = len(final.faces)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["output_file"] = stl_path
        jobs[job_id]["dimensions"] = dim_str
        jobs[job_id]["file_size"] = size_str
        jobs[job_id]["faces"] = face_count

        # Upload to Supabase in background
        try:
            storage_name = f"{job_id}_{Path(stl_path).name}"
            stl_url = upload_to_supabase(stl_path, storage_name)
            if stl_url:
                jobs[job_id]["stl_url"] = stl_url
                save_history(
                    filename=jobs[job_id]["filename"],
                    dimensions=dim_str,
                    file_size=size_str,
                    faces=face_count,
                    stl_url=stl_url,
                    config={"width": target_width, "base": base_thick, "total_height": total_h},
                )
        except Exception as e:
            print(f"  Supabase sync failed: {e}")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 0


# ──────────────────────────────────────────────
#  API Endpoints
# ──────────────────────────────────────────────

@app.post("/upload")
async def upload_svg(
    files: list[UploadFile] = File(...),
    width: float = Form(default=TARGET_WIDTH_MM),
    base: float = Form(default=BASE_THICKNESS_MM),
    lines: float = Form(default=LINE_HEIGHT_MM),
):
    """Upload one or more SVG files. Returns job IDs for tracking."""
    total_h = base + lines
    results = []
    for f in files:
        if not f.filename.lower().endswith('.svg'):
            results.append({"filename": f.filename, "error": "Not an SVG file"})
            continue

        job_id = str(uuid.uuid4())[:8]
        basename = Path(f.filename).stem
        svg_path = str(UPLOAD_DIR / f"{job_id}_{f.filename}")
        stl_path = str(OUTPUT_DIR / f"{job_id}_{basename}.stl")

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

        executor.submit(convert_svg_to_stl, job_id, svg_path, stl_path, width, base, total_h)
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
        "stl_url": j.get("stl_url"),
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
#  Frontend — Industrial / Maker Aesthetic
# ──────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FORGE — SVG to STL</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0a;
    --surface: #111111;
    --surface2: #1a1a1a;
    --border: #252525;
    --border-active: #d4861a;
    --text: #c8c8c8;
    --text-dim: #666;
    --text-bright: #f0f0f0;
    --amber: #d4861a;
    --amber-glow: #e8a030;
    --amber-dim: #8a5a10;
    --green: #3ddc84;
    --red: #ff6b6b;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Subtle grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(212, 134, 26, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(212, 134, 26, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .app { position: relative; z-index: 1; }

  /* ── Header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 20px 0;
    background: linear-gradient(180deg, rgba(212,134,26,0.04) 0%, transparent 100%);
  }
  .header-inner {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .logo-icon {
    width: 32px; height: 32px;
    background: var(--amber);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono);
    font-weight: 700;
    font-size: 16px;
    color: var(--bg);
    transform: rotate(-3deg);
  }
  .logo-text {
    font-family: var(--mono);
    font-weight: 700;
    font-size: 18px;
    color: var(--text-bright);
    letter-spacing: 3px;
  }
  .logo-sub {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 1px;
  }

  /* ── Tabs ── */
  .tabs {
    display: flex;
    gap: 2px;
  }
  .tab {
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: 1px;
    padding: 8px 16px;
    background: transparent;
    color: var(--text-dim);
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.15s;
    text-transform: uppercase;
  }
  .tab:hover { color: var(--text); }
  .tab.active {
    color: var(--amber);
    border-color: var(--border);
    border-bottom-color: var(--bg);
    background: var(--surface);
  }

  /* ── Main Layout ── */
  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px;
  }
  .panel { display: none; }
  .panel.active { display: block; }

  /* ── Convert Panel ── */
  .config-bar {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 16px 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    margin-bottom: 24px;
  }
  .config-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .config-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .config-input {
    width: 64px;
    padding: 6px 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--amber);
    font-family: var(--mono);
    font-size: 13px;
    text-align: center;
    outline: none;
    transition: border-color 0.15s;
  }
  .config-input:focus { border-color: var(--amber); }
  .config-unit {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
  }

  /* ── Drop Zone ── */
  .drop-zone {
    border: 1px dashed var(--border);
    padding: 64px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--surface);
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
  }
  .drop-zone::before {
    content: '';
    position: absolute;
    top: 8px; left: 8px; right: 8px; bottom: 8px;
    border: 1px solid rgba(212,134,26,0.0);
    transition: border-color 0.2s;
    pointer-events: none;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: var(--amber-dim);
    background: rgba(212,134,26,0.03);
  }
  .drop-zone:hover::before, .drop-zone.dragover::before {
    border-color: rgba(212,134,26,0.1);
  }
  .drop-icon {
    width: 48px; height: 48px;
    margin: 0 auto 16px;
    border: 2px solid var(--border);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.2s;
  }
  .drop-zone:hover .drop-icon { border-color: var(--amber-dim); }
  .drop-icon svg { width: 20px; height: 20px; fill: var(--text-dim); }
  .drop-text {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text-dim);
    letter-spacing: 0.5px;
  }
  .drop-hint {
    font-family: var(--mono);
    font-size: 11px;
    color: #444;
    margin-top: 8px;
  }

  /* ── Job Cards ── */
  .job-list { display: flex; flex-direction: column; gap: 2px; }

  .job-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    padding: 16px 20px;
    transition: all 0.2s;
  }
  .job-card.processing { border-left-color: var(--amber); }
  .job-card.done { border-left-color: var(--green); }
  .job-card.error { border-left-color: var(--red); }

  .job-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .job-name {
    font-family: var(--mono);
    font-weight: 500;
    font-size: 13px;
    color: var(--text-bright);
  }
  .job-badge {
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 8px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid;
  }
  .job-badge.queued { color: var(--text-dim); border-color: var(--border); }
  .job-badge.processing { color: var(--amber); border-color: var(--amber-dim); }
  .job-badge.done { color: var(--green); border-color: #1a5a3a; }
  .job-badge.error { color: var(--red); border-color: #5a1a1a; }

  .progress-track {
    height: 2px;
    background: var(--border);
    margin-bottom: 10px;
    overflow: hidden;
  }
  .progress-bar {
    height: 100%;
    background: var(--amber);
    transition: width 0.3s ease;
    box-shadow: 0 0 8px rgba(212,134,26,0.4);
  }

  .job-meta {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
  .job-error-msg {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--red);
    margin-top: 6px;
  }

  .job-actions {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 12px;
  }

  .btn {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 8px 16px;
    border: 1px solid;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .btn-primary {
    background: var(--amber);
    color: var(--bg);
    border-color: var(--amber);
  }
  .btn-primary:hover {
    background: var(--amber-glow);
    border-color: var(--amber-glow);
  }
  .btn-ghost {
    background: transparent;
    color: var(--text-dim);
    border-color: var(--border);
  }
  .btn-ghost:hover {
    color: var(--text);
    border-color: var(--text-dim);
  }

  /* ── 3D Preview ── */
  .preview-container {
    margin-top: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
    background: #0d0d0d;
    position: relative;
  }
  .preview-container canvas {
    display: block;
    width: 100%;
    height: 320px;
  }

  /* ── History Panel ── */
  .history-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 2px;
  }
  .history-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 16px 20px;
    transition: border-color 0.15s;
    cursor: default;
  }
  .history-card:hover {
    border-color: var(--amber-dim);
  }
  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
  }
  .history-name {
    font-family: var(--mono);
    font-weight: 500;
    font-size: 13px;
    color: var(--text-bright);
    word-break: break-all;
  }
  .history-date {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    white-space: nowrap;
    margin-left: 12px;
  }
  .history-meta {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }
  .history-actions {
    display: flex;
    gap: 8px;
  }
  .history-empty {
    text-align: center;
    padding: 80px 20px;
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text-dim);
  }
  .history-loading {
    text-align: center;
    padding: 60px 20px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-dim);
  }

  /* ── Animations ── */
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .job-card, .history-card {
    animation: fadeIn 0.3s ease forwards;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  .processing .job-badge { animation: pulse 1.5s ease-in-out infinite; }

  /* ── Footer ── */
  footer {
    max-width: 1100px;
    margin: 60px auto 0;
    padding: 20px 24px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 10px;
    color: #333;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
</style>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/"}}
</script>
</head>
<body>
<div class="app">

  <!-- Header -->
  <header>
    <div class="header-inner">
      <div class="logo">
        <div class="logo-icon">F</div>
        <div>
          <div class="logo-text">FORGE</div>
          <div class="logo-sub">SVG to STL converter</div>
        </div>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="convert">Convert</button>
        <button class="tab" data-tab="history">History</button>
      </div>
    </div>
  </header>

  <main>
    <!-- Convert Panel -->
    <div class="panel active" id="panel-convert">
      <div class="config-bar">
        <div class="config-item">
          <span class="config-label">Width</span>
          <input class="config-input" type="number" id="cfg-width" value="85" step="5">
          <span class="config-unit">mm</span>
        </div>
        <div class="config-item">
          <span class="config-label">Base</span>
          <input class="config-input" type="number" id="cfg-base" value="1.0" step="0.5">
          <span class="config-unit">mm</span>
        </div>
        <div class="config-item">
          <span class="config-label">Relief</span>
          <input class="config-input" type="number" id="cfg-lines" value="2.0" step="0.5">
          <span class="config-unit">mm</span>
        </div>
      </div>

      <div class="drop-zone" id="dropZone">
        <div class="drop-icon">
          <svg viewBox="0 0 24 24"><path d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/></svg>
        </div>
        <div class="drop-text">Drop SVG files here or click to upload</div>
        <div class="drop-hint">Supports batch processing with parallel conversion</div>
        <input type="file" id="fileInput" accept=".svg" multiple hidden>
      </div>

      <div class="job-list" id="jobList"></div>
    </div>

    <!-- History Panel -->
    <div class="panel" id="panel-history">
      <div class="history-loading" id="historyLoading">Loading history...</div>
      <div class="history-grid" id="historyGrid"></div>
      <div class="history-empty" id="historyEmpty" style="display:none">
        No generation history yet.<br>Convert some SVGs to see them here.
      </div>
    </div>
  </main>

  <footer>
    Forge &mdash; 3D printable contour cards from SVG line art
  </footer>
</div>

<script type="module">
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Supabase config ──
const SB_URL = '""" + SUPABASE_URL + """';
const SB_KEY = '""" + SUPABASE_ANON_KEY + """';

// ── Tab switching ──
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
    if (tab.dataset.tab === 'history') loadHistory();
  });
});

// ── Drop zone ──
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
fileInput.addEventListener('change', () => { handleFiles(fileInput.files); fileInput.value = ''; });

async function handleFiles(files) {
  const svgFiles = [...files].filter(f => f.name.toLowerCase().endsWith('.svg'));
  if (!svgFiles.length) return;

  const formData = new FormData();
  svgFiles.forEach(f => formData.append('files', f));
  formData.append('width', document.getElementById('cfg-width').value);
  formData.append('base', document.getElementById('cfg-base').value);
  formData.append('lines', document.getElementById('cfg-lines').value);

  svgFiles.forEach(f => {
    jobList.prepend(createJobCard(null, f.name));
  });

  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const results = await resp.json();
    const placeholders = jobList.querySelectorAll('.job-card[data-job="pending"]');
    results.forEach((r, i) => {
      const card = placeholders[svgFiles.length - 1 - i];
      if (!card) return;
      if (r.error) {
        card.dataset.job = 'error';
        updateCardStatus(card, 'error', r.error);
      } else {
        card.dataset.job = r.job_id;
        pollJob(r.job_id, card);
      }
    });
  } catch (e) {
    console.error('Upload failed:', e);
  }
}

function createJobCard(jobId, filename) {
  const card = document.createElement('div');
  card.className = 'job-card';
  card.dataset.job = jobId || 'pending';
  card.innerHTML = `
    <div class="job-header">
      <span class="job-name">${filename}</span>
      <span class="job-badge queued">Queued</span>
    </div>
    <div class="progress-track"><div class="progress-bar" style="width:0%"></div></div>
    <div class="job-meta"></div>
    <div class="job-error-msg" style="display:none"></div>
    <div class="job-actions"></div>
  `;
  return card;
}

function updateCardStatus(card, status, error) {
  const badge = card.querySelector('.job-badge');
  badge.className = 'job-badge ' + status;
  badge.textContent = status === 'done' ? 'Complete' :
                      status === 'error' ? 'Failed' :
                      status === 'processing' ? 'Processing' : 'Queued';
  card.className = 'job-card ' + status;
  if (error) {
    const errEl = card.querySelector('.job-error-msg');
    errEl.style.display = 'block';
    errEl.textContent = error;
  }
}

async function pollJob(jobId, card) {
  if (activePolls.has(jobId)) return;
  activePolls.add(jobId);

  const poll = async () => {
    try {
      const data = await (await fetch('/status/' + jobId)).json();
      updateCardStatus(card, data.status, data.error);
      card.querySelector('.progress-bar').style.width = data.progress + '%';

      if (data.status === 'done') {
        card.querySelector('.job-meta').innerHTML =
          `<span>${data.dimensions}</span><span>${data.file_size}</span><span>${data.faces?.toLocaleString()} faces</span>`;
        card.querySelector('.job-actions').innerHTML = `
          <a class="btn btn-primary" href="/download/${jobId}">Download STL</a>
          <div class="preview-box" style="width:100%"></div>
        `;
        initPreview('/download/' + jobId, card.querySelector('.preview-box'));
        activePolls.delete(jobId);
        return;
      }
      if (data.status === 'error') {
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

// ── 3D Preview ──
function initPreview(stlUrl, container) {
  const w = container.clientWidth || 700;
  const h = 320;
  container.classList.add('preview-container');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d0d);

  const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 2000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);

  // Warm industrial lighting
  scene.add(new THREE.AmbientLight(0x2a2018, 3));
  const key = new THREE.DirectionalLight(0xffeedd, 2);
  key.position.set(2, 3, 2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x8899bb, 0.8);
  fill.position.set(-2, -1, 1);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xd4861a, 0.6);
  rim.position.set(0, -2, -1);
  scene.add(rim);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.5;

  const loader = new STLLoader();
  loader.load(stlUrl, (geometry) => {
    const material = new THREE.MeshStandardMaterial({
      color: 0xccaa77,
      roughness: 0.35,
      metalness: 0.15,
    });
    const mesh = new THREE.Mesh(geometry, material);
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    mesh.position.sub(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.set(maxDim * 0.8, -maxDim * 1.0, maxDim * 0.9);
    controls.target.set(0, 0, 0);
    controls.update();
    scene.add(mesh);
  });

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();

  new ResizeObserver(() => {
    const nw = container.clientWidth;
    camera.aspect = nw / h;
    camera.updateProjectionMatrix();
    renderer.setSize(nw, h);
  }).observe(container);
}

// ── History ──
async function loadHistory() {
  const grid = document.getElementById('historyGrid');
  const loading = document.getElementById('historyLoading');
  const empty = document.getElementById('historyEmpty');

  loading.style.display = 'block';
  grid.innerHTML = '';
  empty.style.display = 'none';

  try {
    const resp = await fetch(
      SB_URL + '/rest/v1/stl_generations?order=created_at.desc&limit=50',
      { headers: { 'apikey': SB_KEY, 'Authorization': 'Bearer ' + SB_KEY } }
    );
    const data = await resp.json();
    loading.style.display = 'none';

    if (!data.length) {
      empty.style.display = 'block';
      return;
    }

    data.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = 'history-card';
      card.style.animationDelay = (idx * 0.05) + 's';
      const date = new Date(item.created_at);
      const dateStr = date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

      card.innerHTML = `
        <div class="history-header">
          <span class="history-name">${item.filename || 'Unknown'}</span>
          <span class="history-date">${dateStr}</span>
        </div>
        <div class="history-meta">
          <span>${item.dimensions || '-'}</span>
          <span>${item.file_size || '-'}</span>
          ${item.faces ? '<span>' + item.faces.toLocaleString() + ' faces</span>' : ''}
        </div>
        <div class="history-actions">
          ${item.stl_storage_path ? '<a class="btn btn-primary" href="' + item.stl_storage_path + '" target="_blank">Download</a>' : ''}
          ${item.stl_storage_path ? '<button class="btn btn-ghost preview-history-btn">3D Preview</button>' : ''}
        </div>
        <div class="history-preview-box" style="width:100%"></div>
      `;
      grid.appendChild(card);

      const previewBtn = card.querySelector('.preview-history-btn');
      if (previewBtn) {
        previewBtn.addEventListener('click', () => {
          previewBtn.style.display = 'none';
          initPreview(item.stl_storage_path, card.querySelector('.history-preview-box'));
        });
      }
    });
  } catch (e) {
    loading.style.display = 'none';
    empty.style.display = 'block';
    empty.textContent = 'Failed to load history.';
    console.error(e);
  }
}
</script>
</div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    print(f"Starting FORGE on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
