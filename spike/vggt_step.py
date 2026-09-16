"""VGGT -> COLMAP sparse reconstruction — SPIKE.md §3.2.

Runs the PATCHED demo_colmap.py (see spike/patches/) and then refuses to hand
anything downstream until the poses have been sanity-checked.

    "Inspect the resulting point cloud before going further -- if the poses are
     wrong, nothing downstream can recover."  -- SPIKE.md §3.2

gsplat reads the flat `<scene>/sparse/` that VGGT writes: its parser tries
`sparse/0/` and falls back to `sparse/` (examples/datasets/colmap.py:71-73). No
rename shim needed -- the two documented interfaces really do meet.

Runs inside image A.
"""

from __future__ import annotations

import os
import subprocess

DEMO_COLMAP = "/opt/vggt/demo_colmap.py"
FALLBACK_CONF = 1.0


def count_points(scene_dir: str) -> int:
    import pycolmap
    return len(pycolmap.Reconstruction(os.path.join(scene_dir, "sparse")).points3D)


def run_vggt(scene_dir: str, conf_thres: float = 5.0) -> dict:
    """Invoke the patched demo_colmap.py on <scene_dir>/images/.

    --use_ba is deliberately absent and must stay absent. It is not a cost
    decision: demo_colmap.py:162 hardcodes keypoint_extractor="aliked+sp", and
    the "sp" is SuperPoint, whose weights carry Magic Leap's NON-COMMERCIAL
    licence. See LICENSES.md. AGENTS.md §1 makes that a hard stop.
    """
    images = os.path.join(scene_dir, "images")
    n_images = len([f for f in os.listdir(images) if not f.startswith(".")])
    if n_images < 2:
        raise RuntimeError(f"Need at least 2 images, found {n_images} in {images}")

    ckpt = os.environ["VGGT_COMMERCIAL_CKPT"]
    assert os.path.exists(ckpt), f"Commercial checkpoint missing: {ckpt}"

    def invoke(conf: float) -> None:
        cmd = ["python", DEMO_COLMAP, f"--scene_dir={scene_dir}",
               f"--conf_thres_value={conf}"]
        print(f"$ {' '.join(cmd)}   ({n_images} images, --use_ba OFF by licence)")
        proc = subprocess.run(cmd, cwd="/opt/vggt", capture_output=True, text=True)
        print(proc.stdout[-3000:])          # always, not only on failure
        if proc.returncode != 0:
            print(proc.stderr[-4000:])
            raise RuntimeError(f"demo_colmap.py failed ({proc.returncode})")

        sparse = os.path.join(scene_dir, "sparse")
        written = sorted(os.listdir(sparse)) if os.path.isdir(sparse) else []
        for required in ("cameras.bin", "images.bin", "points3D.bin"):
            if required not in written:
                raise RuntimeError(f"VGGT did not write {required}; got {written}")

    invoke(conf_thres)
    n_points = count_points(scene_dir)
    used = conf_thres

    # demo_colmap.py filters points with `depth_conf >= conf_thres_value`
    # (default 5.0) and will happily write a VALID, EMPTY reconstruction if
    # nothing clears the bar. Retry once, loudly, and record which threshold
    # produced the result -- a silently relaxed confidence threshold is exactly
    # how a plausible-looking but meaningless reconstruction gets through.
    if n_points == 0 and conf_thres > FALLBACK_CONF:
        print(f"\n*** 0 points at conf_thres={conf_thres}. "
              f"Retrying at {FALLBACK_CONF} — treat the result with suspicion. ***\n")
        invoke(FALLBACK_CONF)
        n_points = count_points(scene_dir)
        used = FALLBACK_CONF

    return {
        "n_images": n_images,
        "sparse_dir": os.path.join(scene_dir, "sparse"),
        "n_points3D": n_points,
        "conf_thres_used": used,
        "conf_thres_relaxed": used != conf_thres,
    }


def read_reconstruction(scene_dir: str) -> dict:
    """Pose and point statistics, straight from the written COLMAP model."""
    import numpy as np
    import pycolmap

    rec = pycolmap.Reconstruction(os.path.join(scene_dir, "sparse"))

    centres = np.array([img.projection_center() for img in rec.images.values()])
    xyz = np.array([p.xyz for p in rec.points3D.values()])

    extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))) if len(xyz) else 0.0
    baseline = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0))) if len(centres) else 0.0

    return {
        "n_registered": len(rec.images),
        "n_points3D": len(xyz),
        "scene_extent": round(extent, 4),
        "camera_baseline": round(baseline, 4),
        "baseline_ratio": round(baseline / extent, 4) if extent else 0.0,
        "centres": centres,
        "xyz": xyz,
        "rgb": np.array([p.color for p in rec.points3D.values()]) if len(xyz) else np.zeros((0, 3)),
        "has_nan": bool(np.isnan(centres).any() or np.isnan(xyz).any()),
    }


def gate(stats: dict, n_input: int) -> tuple[bool, list[str]]:
    """Cheap pre-flight checks. AGENTS.md §4: fail before spending GPU.

    These catch *catastrophic* pose failure only. They are not the quality gate —
    that gets built next phase from SPIKE-LOG.md. Passing here means "worth
    training on", never "this is good".
    """
    reasons: list[str] = []

    if stats["has_nan"]:
        reasons.append("NaN in camera centres or points — reconstruction is broken")

    if stats["n_registered"] < n_input:
        reasons.append(
            f"only {stats['n_registered']}/{n_input} images registered — "
            "VGGT dropped frames"
        )

    if stats["n_points3D"] < 1000:
        reasons.append(
            f"only {stats['n_points3D']} 3D points — too sparse to train a splat"
        )

    # Every camera in the same place means no parallax, so no recoverable geometry.
    if stats["baseline_ratio"] < 0.01:
        reasons.append(
            f"camera baseline is {stats['baseline_ratio']:.4f} of scene extent — "
            "cameras are effectively coincident, no parallax"
        )

    return (not reasons), reasons


def pose_preview(stats: dict, out_png: str, size: int = 420) -> str:
    """Three orthographic views of the sparse cloud with camera centres in red.

    Deliberately written with numpy + PIL only. Adding matplotlib to image A for
    a debug render would mean rebuilding the layer that bakes the 4 GB checkpoint.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    xyz, rgb, centres = stats["xyz"], stats["rgb"], stats["centres"]
    if not len(xyz):
        raise RuntimeError("no 3D points to preview")

    allpts = np.vstack([xyz, centres])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    scale = float((hi - lo).max()) or 1.0

    def norm(p):
        return (p - lo) / scale

    nxyz, ncentres = norm(xyz), norm(centres)
    pad = 0.06
    views = [("front (XY)", 0, 1), ("top (XZ)", 0, 2), ("side (ZY)", 2, 1)]
    panels = []

    for label, a, b in views:
        panel = Image.new("RGB", (size, size), (16, 16, 20))

        def to_px(pts):
            u = pad * size + pts[:, a] * size * (1 - 2 * pad)
            v = size - (pad * size + pts[:, b] * size * (1 - 2 * pad))
            return np.stack([u, v], axis=-1).astype(int)

        # Painter's order so nearer points land on top. The axis indices are a
        # permutation of {0,1,2}, so the one not being plotted is always 3-a-b.
        depth_axis = 3 - a - b
        order = np.argsort(-xyz[:, depth_axis])
        px = to_px(nxyz[order])
        cols = rgb[order].astype(int)
        buf = np.array(panel)
        ok = (px[:, 0] >= 0) & (px[:, 0] < size) & (px[:, 1] >= 0) & (px[:, 1] < size)
        buf[px[ok, 1], px[ok, 0]] = cols[ok]
        panel = Image.fromarray(buf)
        draw = ImageDraw.Draw(panel)

        for u, v in to_px(ncentres):
            draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 70, 70))

        draw.text((8, 8), label, fill=(210, 210, 210))
        panels.append(panel)

    sheet = Image.new("RGB", (size * len(panels), size + 26), (16, 16, 20))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * size, 0))
    ImageDraw.Draw(sheet).text(
        (8, size + 6),
        f"{stats['n_registered']} cameras (red) · {stats['n_points3D']} points · "
        f"baseline/extent {stats['baseline_ratio']:.3f}",
        fill=(210, 210, 210),
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    sheet.save(out_png, quality=92)
    return out_png
