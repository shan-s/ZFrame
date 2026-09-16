"""Keyframe extraction — SPIKE.md §3.1.

Target 60-150 keyframes selected on sharpness and pose spread, not a fixed interval.

A NOTE ON "POSE SPREAD"
----------------------
§3.1 asks for selection on "sharpness and pose spread". Taken literally that is
circular: there are no poses until VGGT has run, and VGGT is what consumes these
frames. The proxy used here is **cumulative optical-flow displacement** — how far
the image content has travelled since the last kept frame. It is monotonic in
camera motion for a rigid scene, which is what "spread" actually needs to mean at
this stage.

Consequence worth knowing: it cannot distinguish a camera orbiting an object from
a camera panning across a static scene. That is fine for the spike (every capture
is a deliberate orbit) and is recorded in SPIKE-LOG.md as a known limit.

Runs on CPU inside image A. ffmpeg is invoked as a subprocess (AGENTS.md §1) and
is the LGPL build compiled in modal_app.py.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass

FFMPEG = "/usr/local/bin/ffmpeg"
FFPROBE = "/usr/local/bin/ffprobe"

TARGET_MIN, TARGET_MAX = 60, 150
CANDIDATE_OVERSAMPLE = 3.0  # decode ~3x the target so selection has real choice
LONG_EDGE = 1024           # VGGT works at 518, but gsplat trains on these pixels
BLUR_FLOOR_RATIO = 0.40    # drop frames below this fraction of median sharpness


@dataclass
class FrameScore:
    path: str
    index: int
    sharpness: float        # variance of Laplacian
    displacement: float     # median optical-flow magnitude vs previous candidate
    cumulative: float       # cumulative displacement from frame 0


def probe(video: str) -> dict:
    """Container metadata. Rotation matters: phones record it as a side-data tag."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", "-select_streams", "v:0", video],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]

    num, den = (stream.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0

    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": round(fps, 3),
        "duration_s": round(float(data["format"].get("duration", 0.0)), 2),
        "codec": stream.get("codec_name"),
        "rotation": _rotation(stream),
    }


def _rotation(stream: dict) -> int:
    """ffmpeg auto-applies this on decode; we read it only to log it.

    AGENTS.md §6 exists because geometry and texture silently disagree when an
    orientation tag is honoured in one path and ignored in another. Recording the
    value makes that visible if it ever goes wrong.
    """
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            return int(side["rotation"])
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) if tag else 0


def candidate_fps(meta: dict, target_frames: int) -> float:
    """Decode rate that yields ~CANDIDATE_OVERSAMPLE x target candidates.

    A fixed rate is wrong at both ends: on a 4-second clip it yields fewer
    candidates than the 60-frame floor, leaving selection nothing to choose
    between; on a 3-minute clip it decodes thousands of frames to throw nearly
    all of them away. Never exceeds the source rate — decoding faster than the
    video was shot just duplicates frames.
    """
    duration = meta.get("duration_s") or 0.0
    source_fps = meta.get("fps") or 30.0
    if duration <= 0:
        return min(8.0, source_fps)
    wanted = (target_frames * CANDIDATE_OVERSAMPLE) / duration
    return float(max(1.0, min(wanted, source_fps)))


def extract_candidates(video: str, workdir: str, fps: float) -> list[str]:
    """Decode to JPEG stills. Decode only — never re-encode video.

    H.264/HEVC *decoding* is native to FFmpeg, which is why the LGPL build with
    --disable-gpl is sufficient (see LICENSES.md).
    """
    os.makedirs(workdir, exist_ok=True)
    pattern = os.path.join(workdir, "cand_%05d.jpg")

    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", video,
         "-vf", f"fps={fps},scale='if(gt(iw,ih),{LONG_EDGE},-2)':"
                f"'if(gt(iw,ih),-2,{LONG_EDGE})':flags=lanczos",
         "-qscale:v", "2",
         "-pix_fmt", "yuvj420p",
         pattern],
        check=True,
    )

    frames = sorted(
        os.path.join(workdir, f) for f in os.listdir(workdir) if f.startswith("cand_")
    )
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames from {video}")
    return frames


def score_frames(paths: list[str]) -> list[FrameScore]:
    """Sharpness per frame, plus displacement against the previous frame."""
    import cv2
    import numpy as np

    scores: list[FrameScore] = []
    prev_gray = None
    prev_pts = None
    cumulative = 0.0

    for i, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        displacement = 0.0
        if prev_gray is not None and prev_pts is not None and len(prev_pts):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            if nxt is not None and status is not None:
                ok = status.ravel() == 1
                if ok.sum() >= 8:
                    delta = np.linalg.norm(nxt[ok] - prev_pts[ok], axis=-1)
                    displacement = float(np.median(delta))

        cumulative += displacement
        scores.append(FrameScore(path, i, sharpness, displacement, cumulative))

        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=400, qualityLevel=0.01, minDistance=12
        )

    return scores


def select_keyframes(scores: list[FrameScore], target: int) -> tuple[list[FrameScore], dict]:
    """Even coverage in *camera travel*, sharpest frame within each bin.

    Binning on cumulative displacement rather than on time is the whole point: a
    handheld capture that lingers on one side of the object would otherwise spend
    most of its frame budget there and leave the far side unreconstructed.
    """
    import numpy as np

    if not scores:
        raise RuntimeError("no scored frames")

    target = max(TARGET_MIN, min(TARGET_MAX, target))
    sharp = np.array([s.sharpness for s in scores])
    median_sharp = float(np.median(sharp))
    floor = median_sharp * BLUR_FLOOR_RATIO

    usable = [s for s in scores if s.sharpness >= floor]
    rejected_blur = len(scores) - len(usable)

    # A near-static capture has no travel to bin on; fall back to even time spacing.
    total_travel = usable[-1].cumulative - usable[0].cumulative if usable else 0.0
    if total_travel <= 1e-6:
        step = max(1, len(usable) // target)
        chosen = usable[::step][:target]
        return chosen, {
            "strategy": "uniform-time (no camera travel detected)",
            "candidates": len(scores),
            "rejected_blur": rejected_blur,
            "selected": len(chosen),
            "median_sharpness": round(median_sharp, 1),
            "total_travel_px": 0.0,
        }

    edges = np.linspace(usable[0].cumulative, usable[-1].cumulative, target + 1)
    chosen: list[FrameScore] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [s for s in usable if lo <= s.cumulative < hi]
        if bucket:
            chosen.append(max(bucket, key=lambda s: s.sharpness))
    if usable[-1] not in chosen:
        chosen.append(usable[-1])

    chosen.sort(key=lambda s: s.index)
    return chosen, {
        "strategy": "travel-binned, sharpest-per-bin",
        "candidates": len(scores),
        "rejected_blur": rejected_blur,
        "selected": len(chosen),
        "median_sharpness": round(median_sharp, 1),
        "total_travel_px": round(total_travel, 1),
    }


def write_scene(chosen: list[FrameScore], scene_dir: str, fps: float) -> str:
    """Write selected frames to <scene>/images/ — the layout demo_colmap.py expects.

    AGENTS.md §6: apply EXIF orientation and strip the tag, so no downstream
    consumer can apply it a second time.

    Also writes frame_map.json. Without it the link back to a source timestamp is
    lost, and with it any chance of checking estimated poses against a known
    ground truth (see modal_app.validate_synthetic).
    """
    import json

    from PIL import Image, ImageOps

    images_dir = os.path.join(scene_dir, "images")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    for n, score in enumerate(chosen):
        with Image.open(score.path) as im:
            im = ImageOps.exif_transpose(im)      # honour orientation exactly once
            im = im.convert("RGB")
            data = list(im.getdata())
            clean = Image.new("RGB", im.size)     # rebuild without any EXIF block
            clean.putdata(data)
            clean.save(os.path.join(images_dir, f"frame_{n:04d}.jpg"), quality=95)

    with open(os.path.join(scene_dir, "frame_map.json"), "w") as fh:
        json.dump({
            "candidate_fps": fps,
            "frames": [
                {
                    "output": f"frame_{n:04d}.jpg",
                    "candidate_index": s.index,
                    "source_time_s": round(s.index / fps, 5),
                    "sharpness": round(s.sharpness, 2),
                }
                for n, s in enumerate(chosen)
            ],
        }, fh, indent=2)

    return images_dir


def contact_sheet(chosen: list[FrameScore], out_path: str, cols: int = 10) -> str:
    """A single image of every selected frame.

    SPIKE.md §3.5's principle applied one stage earlier: frame selection is the
    cheapest thing to get wrong and the cheapest to check by eye.
    """
    from PIL import Image

    thumb_w = 160
    rows = (len(chosen) + cols - 1) // cols
    thumbs = []
    for score in chosen:
        with Image.open(score.path) as im:
            im = im.convert("RGB")
            h = int(im.height * thumb_w / im.width)
            thumbs.append(im.resize((thumb_w, h)))

    thumb_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (18, 18, 18))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * thumb_w, (i // cols) * thumb_h))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, quality=90)
    return out_path
