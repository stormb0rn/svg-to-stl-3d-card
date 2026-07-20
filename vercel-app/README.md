# SVG Relief Card Vercel App

This is a Vercel-friendly migration of the original local FastAPI tool.

The original server used local binaries such as ImageMagick and OpenSCAD. Those are not reliable in Vercel serverless functions, so this version runs the conversion in the browser with Web Workers and exports STL directly.

## Local preview

```bash
python3 -m http.server 8080 -d vercel-app
```

## Deploy

```bash
vercel deploy vercel-app -y
```

