# SVG to STL 3D Card

This repo contains two deployment targets for generating 3D-printable STL relief cards from SVG artwork.

## Local / Render FastAPI app

The original app is `web_app.py`. It runs a Python backend, stores uploads and outputs locally, and can save history to Supabase.

```bash
pip install -r requirements.txt
python web_app.py
```

Render deployment is configured in `render.yaml`.

## Vercel static app

The Vercel-friendly version lives in `vercel-app/`.

It runs SVG processing and STL generation in the browser with Web Workers. This avoids Vercel serverless limits and avoids relying on server-side ImageMagick/OpenSCAD binaries.

Local preview:

```bash
python3 -m http.server 8080 -d vercel-app
```

Preview deploy:

```bash
vercel deploy vercel-app -y
```

For GitHub import on Vercel, set the project Root Directory to `vercel-app`.

