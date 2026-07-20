# Cloudflare Worker SVG Relief Card

This deployment target hosts a static SVG upload page and a Cloudflare Worker API that generates binary STL files.

## Architecture

The Worker runtime does not provide DOM or Canvas APIs, so the browser renders the uploaded SVG into an RGBA raster. The browser POSTs that binary raster to `/api/generate`, and the Worker handles mask cleanup, bottom-board merging, raised-line generation, and STL writing.

Default card settings match the current print target:

- Base plate: `1.5 mm`
- Raised line height above base: `2.0 mm`
- Long edge: `120 mm`
- Raster long edge: `640 px`

## Local development

```bash
npm install
npm run dev
```

Open the Wrangler local URL, upload a clean black/white SVG, and download the STL.

Health check:

```bash
curl http://localhost:8787/api/health
```

## Deploy

```bash
npm run deploy
```

If this is the first deploy, Wrangler will prompt for Cloudflare auth. The Worker uses Workers Static Assets for `public/` and routes `/api/*` through the Worker script.

## Notes

- This target outputs STL only. 3MF/OpenSCAD output remains in the Python app because Workers cannot run the required native binaries.
- If the Worker returns `Model too complex`, lower raster width or simplify the SVG. The STL generator coalesces horizontal spans to reduce triangle count, but very dense masks can still exceed Worker memory limits.
