"""
Generate a contour-shaped (异形) 3D printable card directly from SVG vector paths.
No rasterization or potrace needed — SVG paths are parsed directly into 3D geometry.

Usage:
  python generate_svg_to_3d.py input.svg [output.stl]
"""

import sys
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
import svgpathtools
import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# === Configuration ===
TARGET_WIDTH_MM = 85.0
TOTAL_HEIGHT_MM = 3.0
BASE_THICKNESS_MM = 1.0
LINE_HEIGHT_MM = 2.0
CARD_MARGIN_MM = 1.5
SAMPLE_POINTS_PER_PATH = 300
MIN_AREA_SVG = 5.0
BRIDGE_RADIUS_SVG = 80


def svg_path_to_polygons(path, sample_pts=SAMPLE_POINTS_PER_PATH):
    """Convert one svgpathtools Path into list of Shapely polygons."""
    subpaths = []
    current = []
    for seg in path:
        if current and abs(seg.start - current[-1].end) > 0.5:
            subpaths.append(current)
            current = [seg]
        else:
            current.append(seg)
    if current:
        subpaths.append(current)

    shells = []
    holes = []

    for sub in subpaths:
        n_segs = len(sub)
        if n_segs == 0:
            continue
        pts_per_seg = max(4, sample_pts // n_segs)
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

        if area > 0:
            shells.append(points)
        else:
            holes.append(points)

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
                hp = Polygon(hole_pts)
                if shell_poly.contains(hp.centroid):
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
    """Extract all Polygon objects from any Shapely geometry."""
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
    """Extrude a Shapely geometry to a trimesh mesh."""
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
                simplified = poly.simplify(0.05)
                if simplified.is_empty or simplified.area < 0.01:
                    continue
                ext = trimesh.creation.extrude_polygon(simplified, height)
                ext.apply_translation([0, 0, z_bottom])
                meshes.append(ext)
            except Exception:
                continue

    return trimesh.util.concatenate(meshes) if meshes else None


def generate_preview(visible_lines, card_outline, svg_width, svg_height, debug_dir):
    """Generate a preview image of the card."""
    W = 800
    s = W / svg_width
    H = int(svg_height * s)
    img = Image.new('RGB', (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    def draw_poly(poly, fc):
        if isinstance(poly, (MultiPolygon, GeometryCollection)):
            for g in poly.geoms:
                draw_poly(g, fc)
            return
        if not isinstance(poly, Polygon) or poly.is_empty:
            return
        coords = [(x * s, y * s) for x, y in poly.exterior.coords]
        draw.polygon(coords, fill=fc)
        for hole in poly.interiors:
            hc = [(x * s, y * s) for x, y in hole.coords]
            draw.polygon(hc, fill=(30, 30, 30))

    draw_poly(card_outline, (80, 80, 80))
    draw_poly(visible_lines, (255, 255, 255))
    img.save(os.path.join(debug_dir, "preview.png"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_svg_to_3d.py input.svg [output.stl]")
        sys.exit(1)

    input_svg = sys.argv[1]
    basename = os.path.splitext(os.path.basename(input_svg))[0]
    output_stl = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{basename}.stl"
    )
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"debug_{basename}")
    os.makedirs(debug_dir, exist_ok=True)

    print(f"=== 3D Contour Card — Direct SVG Vector Pipeline ===")
    print(f"  Input: {input_svg}")

    # Step 1: Parse SVG
    print("\n[1/5] Parsing SVG paths...")
    paths, attrs = svgpathtools.svg2paths(input_svg)
    print(f"  Total paths: {len(paths)}")

    tree = ET.parse(input_svg)
    root = tree.getroot()
    svg_width = float(root.get('width', 800))
    svg_height = float(root.get('height', 800))
    print(f"  SVG canvas: {svg_width} x {svg_height}")

    # Step 2: Process paths in SVG paint order
    print(f"\n[2/5] Processing {len(paths)} paths in SVG paint order...")

    visible_lines = Polygon()
    all_content = Polygon()
    bg_done = False

    for i, (path, attr) in enumerate(zip(paths, attrs)):
        fill = attr.get('fill', None)

        # Auto-detect and skip background rectangle
        if fill == 'white' and not bg_done:
            polys = svg_path_to_polygons(path)
            if polys and polys[0].area > svg_width * svg_height * 0.8:
                bg_done = True
                print(f"  Skipped background rectangle (path {i})")
                continue

        polygons = svg_path_to_polygons(path)
        if not polygons:
            continue

        poly_union = unary_union(polygons)
        if poly_union.is_empty:
            continue
        if not poly_union.is_valid:
            poly_union = make_valid(poly_union)

        try:
            all_content = all_content.union(poly_union)
        except Exception:
            pass

        if fill is None:
            # No fill attr = default black in SVG = line work
            try:
                visible_lines = visible_lines.union(poly_union)
            except Exception:
                pass
        elif fill == 'white':
            # White fills cover black underneath
            try:
                visible_lines = visible_lines.difference(poly_union)
            except Exception:
                pass

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(paths)} paths...")

    if not visible_lines.is_valid:
        visible_lines = make_valid(visible_lines)
    if not all_content.is_valid:
        all_content = make_valid(all_content)

    # Step 3: Compute card outline and scale
    print("\n[3/5] Computing card outline...")

    content_bounds = all_content.bounds
    content_width_svg = content_bounds[2] - content_bounds[0]
    content_height_svg = content_bounds[3] - content_bounds[1]
    scale = TARGET_WIDTH_MM / content_width_svg
    content_height_mm = content_height_svg * scale

    margin_svg = CARD_MARGIN_MM / scale
    # Morphological close to bridge disconnected parts, then add margin
    card_outline = all_content.buffer(BRIDGE_RADIUS_SVG, resolution=64)
    card_outline = card_outline.buffer(-BRIDGE_RADIUS_SVG + margin_svg, resolution=64)
    # Light smoothing pass: small buffer out+in rounds off sharp corners
    smooth_r = margin_svg * 0.5
    card_outline = card_outline.buffer(smooth_r, resolution=64).buffer(-smooth_r, resolution=64)
    card_outline = card_outline.simplify(0.3)  # Very gentle simplify in SVG units (~0.03mm)

    # Check connectivity
    outline_polys = collect_polygons(card_outline)
    if len(outline_polys) > 1:
        print(f"  WARNING: outline has {len(outline_polys)} pieces, keeping largest")
        card_outline = max(outline_polys, key=lambda p: p.area)

    visible_count = len(collect_polygons(visible_lines))
    print(f"  Visible line polygons: {visible_count}")
    print(f"  Scale: {scale:.4f} mm/svg_unit")
    print(f"  Card size: {TARGET_WIDTH_MM:.0f} x {content_height_mm:.0f} mm")

    # Save preview
    generate_preview(visible_lines, card_outline, svg_width, svg_height, debug_dir)
    print(f"  Preview saved to {debug_dir}/preview.png")

    # Step 4: Scale to mm and build 3D mesh
    print("\n[4/5] Building 3D mesh...")

    def scale_geom(geom):
        from shapely.affinity import translate, scale as sh_scale
        geom = translate(geom, -content_bounds[0], -content_bounds[1])
        geom = sh_scale(geom, xfact=scale, yfact=scale, origin=(0, 0))
        geom = sh_scale(geom, xfact=1, yfact=-1, origin=(0, content_height_mm / 2))
        return geom

    card_outline_mm = scale_geom(card_outline)
    visible_lines_mm = scale_geom(visible_lines)

    all_meshes = []

    base_mesh = extrude_geometry(card_outline_mm, 0, BASE_THICKNESS_MM)
    if base_mesh:
        all_meshes.append(base_mesh)
        print(f"  Base plate: {len(base_mesh.faces)} faces")

    line_mesh = extrude_geometry(visible_lines_mm, BASE_THICKNESS_MM, TOTAL_HEIGHT_MM)
    if line_mesh:
        all_meshes.append(line_mesh)
        print(f"  Line ridges: {len(line_mesh.faces)} faces")

    if not all_meshes:
        print("\nError: No meshes generated!")
        return

    # Step 5: Combine & export
    print("\n[5/5] Combining & exporting...")
    final = trimesh.util.concatenate(all_meshes)

    centroid = final.bounds.mean(axis=0)
    final.apply_translation([-centroid[0], -centroid[1], -final.bounds[0][2]])

    final.export(output_stl)
    sz = os.path.getsize(output_stl) / (1024 * 1024)
    dims = final.bounds[1] - final.bounds[0]

    print(f"\n=== Done! ===")
    print(f"  File: {output_stl}")
    print(f"  Size: {sz:.1f} MB")
    print(f"  Dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
    print(f"  Height: {TOTAL_HEIGHT_MM}mm (base {BASE_THICKNESS_MM}mm + lines {LINE_HEIGHT_MM}mm)")
    print(f"  Card shape: CONTOUR (异形)")


if __name__ == "__main__":
    main()
