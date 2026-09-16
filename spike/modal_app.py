"""ZFrame reconstruction spike — Modal app.

Answers one question (SPIKE.md): does VGGT + gsplat beat Polycam, KIRI and Luma
on captures those tools fail on? Disposable by design.

    video.mp4 -> ffmpeg keyframes -> VGGT -> sparse/{cameras,images,points3D}.bin
              -> gsplat -> depth render -> Open3D TSDF -> .glb

WHY TWO IMAGES (this is forced by upstream, not a preference)
-------------------------------------------------------------
VGGT's demo_colmap.py needs the OFFICIAL pycolmap (C++ bindings, `Reconstruction`).
gsplat v1.5.3's COLMAP parser needs rmbrualla's pycolmap (MIT, pure-Python
`SceneManager`). Both install under the module name `pycolmap`, so they cannot
coexist in one environment.

gsplat main would dodge that (it uses official pycolmap) but requires numpy>=2
against VGGT's numpy<2, and its examples pin torch==2.9.1 described upstream as
"internal builds" -- not reproducible. So: v1.5.3, two images.

The two stages already communicate through COLMAP files on disk, which is the
documented bridge, so the split costs nothing. It also satisfies AGENTS.md §4 for
free: the gsplat/export image has no VGGT in it at all.
"""

import os

import modal

# --- Upstream pins (mirrored in pyproject.toml [tool.zframe]) -----------------
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
GSPLAT_TAG = "v1.5.3"
FFMPEG_VERSION = "6.1.1"

# AGENTS.md §1: the COMMERCIAL checkpoint. Never facebook/VGGT-1B.
VGGT_HF_REPO = "facebook/VGGT-1B-Commercial"
VGGT_HF_FILE = "vggt_1B_commercial.pt"
CKPT_DIR = "/opt/vggt_weights"
CKPT_PATH = f"{CKPT_DIR}/{VGGT_HF_FILE}"

# A10G is sm_86, L4 is sm_89. Build AOT for both so no GPU-time JIT compile.
CUDA_ARCH_LIST = "8.6;8.9"

CUDA_BASE = "nvidia/cuda:12.1.1-devel-ubuntu22.04"

app = modal.App("zframe-spike")
volume = modal.Volume.from_name("zframe-spike", create_if_missing=True)
VOL = "/vol"
hf_secret = modal.Secret.from_name("huggingface")


# =============================================================================
# FFmpeg — built from source, LGPL only
# =============================================================================
# SPIKE.md §2 says `apt: ffmpeg (LGPL build)`. On ubuntu22.04 that is false:
# libavcodec58 depends on libx264-163 and libx265-199, i.e. a GPL build linked
# against x264/x265, forbidden by name in AGENTS.md §1.
#
# Those are ENCODERS. This pipeline only decodes video into stills, and H.264 /
# HEVC decoding is native to FFmpeg, so --disable-gpl costs us nothing.
_FFMPEG_BUILD = [
    f"wget -q https://ffmpeg.org/releases/ffmpeg-{FFMPEG_VERSION}.tar.xz -O /tmp/ff.tar.xz",
    "mkdir -p /tmp/ff && tar xf /tmp/ff.tar.xz -C /tmp/ff --strip-components=1",
    (
        "cd /tmp/ff && ./configure --prefix=/usr/local "
        "--disable-gpl --disable-nonfree --disable-version3 "
        "--disable-doc --disable-debug --disable-ffplay --enable-pic"
    ),
    "cd /tmp/ff && make -j$(nproc) && make install && ldconfig",
    "rm -rf /tmp/ff /tmp/ff.tar.xz",
]

# The assertion lives in spike/checks/ffmpeg_licence.sh rather than inline: Modal
# renders run_commands into Dockerfile RUN lines, which cannot hold a multi-line
# script, and a reviewable file is better than a wall of escaped shell anyway.


def _download_vggt_weights() -> None:
    """Bake the commercial checkpoint into the image (SPIKE.md §2)."""
    import os

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert token, (
        "No HF token in the 'huggingface' Modal secret. "
        f"{VGGT_HF_REPO} is a manually-gated repo; approval is required."
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    path = hf_hub_download(
        repo_id=VGGT_HF_REPO,
        filename=VGGT_HF_FILE,
        local_dir=CKPT_DIR,
        token=token,
    )

    size = os.path.getsize(path)
    assert size > 1_000_000_000, f"Checkpoint implausibly small ({size} bytes): {path}"
    # Name-check as a second guard against ever baking facebook/VGGT-1B's model.pt.
    assert "commercial" in os.path.basename(path).lower(), (
        f"Refusing a checkpoint that is not the commercial one: {path}"
    )
    print(f"Baked {VGGT_HF_REPO}/{VGGT_HF_FILE} ({size / 1e9:.2f} GB) -> {path}")


# =============================================================================
# Image A — frames + VGGT
# =============================================================================
image_vggt = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.11")
    .apt_install(
        "git", "wget", "xz-utils",
        # ffmpeg source build
        "build-essential", "nasm", "yasm", "pkg-config",
        # OpenCV runtime (SPIKE.md §2)
        "libglib2.0-0", "libsm6", "libxrender-dev", "libxext6", "libgl1",
    )
    .run_commands(*_FFMPEG_BUILD)
    .add_local_file("spike/checks/ffmpeg_licence.sh", "/opt/ffmpeg_licence.sh", copy=True)
    .run_commands("bash /opt/ffmpeg_licence.sh")
    # VGGT pins torch==2.3.1 / torchvision==0.18.1 in its own requirements.txt.
    .pip_install(
        "torch==2.3.1", "torchvision==0.18.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4",          # VGGT requires numpy<2 (SPIKE.md §2 agrees)
        "Pillow", "einops", "safetensors", "huggingface_hub",
        "opencv-python", "scipy", "trimesh", "tqdm", "requests",
        "hydra-core", "omegaconf",
        # Unavoidable: demo_colmap.py imports pycolmap at module level and writes
        # the reconstruction through it, with or without --use_ba. BSD-3-Clause.
        "pycolmap==3.10.0",
        # Also unavoidable: demo_colmap.py:31 -> track_predict -> vggsfm_utils:15
        # does `from lightglue import ...` at import time. Apache-2.0.
        # See the SuperPoint warning on the generate() function below.
        "lightglue @ git+https://github.com/jytime/LightGlue.git",
    )
    .run_commands(
        f"git clone https://github.com/facebookresearch/vggt.git /opt/vggt",
        f"cd /opt/vggt && git checkout {VGGT_COMMIT}",
        "cd /opt/vggt && pip install -e . --no-deps",
    )
    .add_local_file(
        "spike/patches/patch_demo_colmap.py",
        "/opt/patch_demo_colmap.py",
        copy=True,
    )
    .run_commands("python /opt/patch_demo_colmap.py")
    .env({"VGGT_COMMERCIAL_CKPT": CKPT_PATH, "PYTHONUNBUFFERED": "1"})
    .run_function(_download_vggt_weights, secrets=[hf_secret])
)


# =============================================================================
# Image B — gsplat training, mesh, export
# =============================================================================
# No VGGT here, by construction. AGENTS.md §4: /export imports no ML stack.
image_gsplat = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.11")
    .apt_install(
        "git", "wget", "build-essential", "ninja-build",
        "libglib2.0-0", "libsm6", "libxrender-dev", "libxext6", "libgl1",
    )
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # setuptools/wheel must be present BEFORE any --no-build-isolation install.
    .pip_install("numpy<2.0", "ninja", "setuptools", "wheel")
    .env({"TORCH_CUDA_ARCH_LIST": CUDA_ARCH_LIST, "MAX_JOBS": "4"})
    # Modal's add_python build reports clang as its compiler in sysconfig, so
    # setuptools links extensions with clang++ even though nvcc/gcc compiled the
    # objects. Installed HERE rather than in apt_install above so the ~2.5 GB torch
    # layer stays cached across this fix.
    .run_commands("apt-get update && apt-get install -y clang && rm -rf /var/lib/apt/lists/*")
    .run_commands(
        # --no-build-isolation is REQUIRED: gsplat's setup.py imports torch inside
        # get_extensions(), and pip's isolated build env cannot see the torch we
        # installed above. Without it the build dies with ModuleNotFoundError.
        #
        # setup.py AOT-compiles via CUDAExtension (needs nvcc, NOT a GPU), so the
        # kernels are baked in and _backend.py never falls back to runtime JIT.
        "pip install --no-build-isolation "
        f"git+https://github.com/nerfstudio-project/gsplat.git@{GSPLAT_TAG}",
    )
    .run_commands(
        f"git clone --depth 1 --branch {GSPLAT_TAG} "
        "https://github.com/nerfstudio-project/gsplat.git /opt/gsplat",
        # Drop the [ffmpeg] extra. imageio-ffmpeg ships a bundled ffmpeg BINARY
        # whose build licence its own docs never state -- AGENTS.md §1: "if you
        # cannot determine a licence, do not add the dependency". It is only used
        # for optional training-progress .mp4s (simple_trainer.py:1043); the
        # module-level `import imageio` and imwrite() paths need no binary.
        r"sed -i 's/^imageio\[ffmpeg\]/imageio/' /opt/gsplat/examples/requirements.txt",
        # fused-ssim and fused-bilagrid are CUDA extensions with the same torch
        # build-isolation problem, so they are split out and installed separately.
        "grep -v 'fused-' /opt/gsplat/examples/requirements.txt > /tmp/reqs.txt",
        "pip install -r /tmp/reqs.txt",
        "pip install --no-build-isolation "
        "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5 "
        "git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe",
        # Prove the bundled ffmpeg binary did not sneak back in transitively.
        'python -c "'
        "import importlib.util,sys; "
        "sys.exit(1) if importlib.util.find_spec('imageio_ffmpeg') else "
        "print('no bundled ffmpeg binary in image B - OK')"
        '"',
    )
    .add_local_file("spike/patches/patch_simple_trainer.py",
                    "/opt/patch_simple_trainer.py", copy=True)
    .run_commands("python /opt/patch_simple_trainer.py")
    # Mesh + export stage (AGENTS.md §1: Open3D TSDF is the only licence-clean path)
    .pip_install("open3d==0.19.0", "trimesh", "Pillow")
    .env({"PYTHONUNBUFFERED": "1", "ZFRAME_SKIP_TRAJ_VIDEO": "1"})
)


# =============================================================================
# Verification — cheap, runs in seconds, catches breakage before GPU time
# =============================================================================
@app.function(image=image_vggt, volumes={VOL: volume}, timeout=600)
def verify_vggt() -> dict:
    import os
    import subprocess

    import numpy as np
    import pycolmap
    import torch

    cfg = subprocess.run(
        ["/usr/local/bin/ffmpeg", "-version"], capture_output=True, text=True
    ).stdout
    config_line = next(l for l in cfg.splitlines() if l.startswith("configuration:"))

    forbidden = [f for f in ("--enable-gpl", "libx264", "libx265") if f in config_line]
    assert not forbidden, f"LICENCE VIOLATION at runtime: {forbidden}"

    ckpt = os.environ["VGGT_COMMERCIAL_CKPT"]
    assert os.path.exists(ckpt), f"Commercial checkpoint missing: {ckpt}"

    with open("/opt/vggt/demo_colmap.py") as fh:
        patched = fh.read()
    assert "facebook/VGGT-1B/resolve" not in patched, "Forbidden checkpoint URL is back!"
    assert "PATCHED (ZFrame" in patched, "demo_colmap.py patch is missing"

    import lightglue  # noqa: F401  -- import-time dependency of demo_colmap

    info = {
        # str() matters: torch.__version__ is a torch-defined str subclass, and
        # returning it raw makes the local driver need torch just to unpickle.
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": str(np.__version__),
        "pycolmap": str(getattr(pycolmap, "__version__", "unknown")),
        "checkpoint_gb": round(os.path.getsize(ckpt) / 1e9, 2),
        "ffmpeg_gpl_free": True,
    }
    print("VGGT image OK:", info)
    return info


@app.function(image=image_gsplat, gpu="L4", volumes={VOL: volume}, timeout=900)
def verify_gsplat() -> dict:
    import os

    import numpy as np
    import open3d as o3d
    import torch

    import gsplat

    # Touch the CUDA kernels so a JIT fallback shows up here, not mid-training.
    means = torch.rand(100, 3, device="cuda")
    quats = torch.nn.functional.normalize(torch.rand(100, 4, device="cuda"), dim=-1)
    scales = torch.rand(100, 3, device="cuda") * 0.01
    opacities = torch.rand(100, device="cuda")
    colors = torch.rand(100, 3, device="cuda")
    viewmat = torch.eye(4, device="cuda")[None]
    K = torch.tensor([[[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]]], device="cuda")

    rendered, alpha, _ = gsplat.rasterization(
        means, quats, scales, opacities, colors, viewmat, K, 128, 128,
        render_mode="RGB+ED",   # the depth channel the mesh stage depends on
    )

    assert os.path.exists("/opt/gsplat/examples/simple_trainer.py"), \
        "gsplat examples/simple_trainer.py missing -- clone step failed"
    # Prove the CUDA kernels were compiled ahead of time rather than JIT-ed just now.
    from gsplat.cuda._backend import _C  # noqa: F401

    info = {
        "torch": str(torch.__version__),   # see note in verify_vggt
        "gsplat": str(gsplat.__version__),
        "numpy": str(np.__version__),
        "open3d": str(o3d.__version__),
        "render_shape": tuple(int(d) for d in rendered.shape),
        "depth_channel": bool(rendered.shape[-1] == 4),
    }
    print("gsplat image OK:", info)
    return info


@app.local_entrypoint()
def verify() -> None:
    """modal run spike/modal_app.py::verify"""
    print("=== Image A (frames + VGGT) ===")
    print(verify_vggt.remote())
    print("\n=== Image B (gsplat + mesh + export) ===")
    print(verify_gsplat.remote())
    print("\nBoth images verified.")


# =============================================================================
# Pipeline stages
# =============================================================================
# AGENTS.md §4 wants /generate and /export kept separate, never merged. SPIKE.md
# forbids building a web surface. Resolved as Modal functions driven by
# `modal run`: the boundary is real (image B contains no VGGT at all), but there
# are no HTTP endpoints yet. Promoting these to @modal.fastapi_endpoint later is
# a decorator change.
#
# The reconstruct/train split inside "generate" is forced by the two-image split
# described at the top of this file, not by choice.

SCENES = f"{VOL}/scenes"

image_vggt = image_vggt.add_local_dir("spike", "/root/spike")
image_gsplat = image_gsplat.add_local_dir("spike", "/root/spike")


@app.function(image=image_vggt, gpu="L4", volumes={VOL: volume}, timeout=1800)
def stage_reconstruct(scene: str, target_frames: int = 120,
                      conf_thres: float = 5.0) -> dict:
    """Frames -> VGGT -> COLMAP sparse. Image A. (AGENTS.md §4: /generate)"""
    import json
    import sys

    sys.path.insert(0, "/root/spike")
    import frames as F
    import vggt_step as V

    scene_dir = f"{SCENES}/{scene}"
    video = f"{scene_dir}/input.mp4"
    assert os.path.exists(video), f"No video at {video}"

    meta = F.probe(video)
    print(f"input: {meta}")

    fps = F.candidate_fps(meta, target_frames)
    candidates = F.extract_candidates(video, f"{scene_dir}/candidates", fps)
    print(f"decoded {len(candidates)} candidates at {fps:.2f} fps")
    scores = F.score_frames(candidates)
    chosen, selection = F.select_keyframes(scores, target_frames)
    selection["candidate_fps"] = round(fps, 3)
    print(f"frame selection: {selection}")

    F.write_scene(chosen, scene_dir, fps)
    F.contact_sheet(chosen, f"{scene_dir}/preview/contact_sheet.jpg")

    vggt_result = V.run_vggt(scene_dir, conf_thres)
    stats = V.read_reconstruction(scene_dir)
    passed, reasons = V.gate(stats, vggt_result["n_images"])

    # Best-effort: a debug render must never destroy the diagnostic manifest that
    # explains WHY there was nothing to render.
    preview_error = None
    try:
        V.pose_preview(stats, f"{scene_dir}/preview/poses.jpg")
    except Exception as exc:  # noqa: BLE001
        preview_error = str(exc)
        print(f"pose preview unavailable: {exc}")

    manifest = {
        "scene": scene,
        "video": meta,
        "selection": selection,
        "vggt": vggt_result,
        "reconstruction": {k: v for k, v in stats.items()
                           if k not in ("centres", "xyz", "rgb")},
        "gate_passed": passed,
        "gate_reasons": reasons,
        "preview_error": preview_error,
    }
    with open(f"{scene_dir}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    volume.commit()

    # SPIKE.md §3.2 -- refuse to spend training GPU on broken poses.
    if not passed:
        print("\n*** POSE GATE FAILED ***")
        for r in reasons:
            print(f"  - {r}")
        print("Inspect preview/poses.jpg before going further.")

    return manifest


@app.function(image=image_gsplat, gpu="L4", volumes={VOL: volume}, timeout=3600)
def stage_train(scene: str, max_steps: int = 30_000) -> dict:
    """COLMAP sparse -> trained splat. Image B. SPIKE.md §3.3: no tuning yet."""
    import glob
    import subprocess

    scene_dir = f"{SCENES}/{scene}"
    result_dir = f"{scene_dir}/gsplat"

    cmd = [
        "python", "simple_trainer.py", "default",
        "--data_dir", scene_dir,
        "--result_dir", result_dir,
        "--data_factor", "1",       # our frames are full-res, not COLMAP images_N
        "--max_steps", str(max_steps),
        "--save_ply",
        "--disable_viewer",
    ]
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd="/opt/gsplat/examples", capture_output=True, text=True)
    print(proc.stdout[-6000:])
    if proc.returncode != 0:
        print(proc.stderr[-6000:])
        raise RuntimeError(f"simple_trainer failed ({proc.returncode})")

    ckpts = sorted(glob.glob(f"{result_dir}/ckpts/*.pt"))
    plys = sorted(glob.glob(f"{result_dir}/ply/*.ply"))
    if not ckpts:
        raise RuntimeError(f"no checkpoint written under {result_dir}/ckpts")

    volume.commit()
    return {"checkpoint": ckpts[-1], "ply": plys[-1] if plys else None,
            "result_dir": result_dir}


@app.function(image=image_gsplat, gpu="L4", volumes={VOL: volume}, timeout=1800)
def stage_export(scene: str, checkpoint: str) -> dict:
    """Splat -> depth render -> Open3D TSDF -> cleaned .glb. (AGENTS.md §4: /export)

    No VGGT in this image, by construction.
    """
    import json
    import sys

    sys.path.insert(0, "/root/spike")
    import mesh as M

    scene_dir = f"{SCENES}/{scene}"

    raw, extent, voxel, centre, support = M.fuse_tsdf(scene_dir, checkpoint)
    M.export_glb(raw, f"{scene_dir}/export/raw.glb")

    # Floor removal comes FIRST: while the floor is still attached, the subject
    # and the ground are one connected component, so nothing can be separated.
    cleaned, plane_info = M.align_to_support_plane(raw, extent, voxel, centre, support)
    print(f"support plane: {plane_info}")
    dropped = plane_info["components_removed"]
    cleaned, post_decimate_dropped = M.finalise(cleaned)
    out = M.export_glb(cleaned, f"{scene_dir}/export/{scene}.glb")

    report = M.contract_report(cleaned)
    report.update(plane_info)
    report["components_removed"] = dropped
    report["fragments_after_decimation"] = post_decimate_dropped
    report["glb"] = out
    print(f"export contract (delivered subset): {json.dumps(report, indent=2)}")

    with open(f"{scene_dir}/export/contract.json", "w") as fh:
        json.dump(report, fh, indent=2)

    volume.commit()
    return report


@app.local_entrypoint()
def run(
    video: str,
    scene: str = "",
    target_frames: int = 120,
    max_steps: int = 30_000,
    skip_train: bool = False,
) -> None:
    """End-to-end. `modal run spike/modal_app.py::run --video capture.mp4`

    AGENTS.md §7: this exiting zero does NOT mean it worked. Pull the artifacts
    and open the .glb before believing anything.
    """
    import json
    import pathlib

    src = pathlib.Path(video)
    assert src.exists(), f"No such video: {src}"
    scene = scene or src.stem.replace(" ", "_")

    print(f"Uploading {src.name} -> scene '{scene}'")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(src), f"scenes/{scene}/input.mp4")

    manifest = stage_reconstruct.remote(scene, target_frames)
    print(json.dumps(manifest, indent=2))

    if not manifest["gate_passed"]:
        print("\nStopping before GPU training: pose gate failed (SPIKE.md §3.2).")
        print(f"Download preview: modal volume get zframe-spike scenes/{scene}/preview .")
        return

    if skip_train:
        print("skip_train set — stopping after reconstruction.")
        return

    trained = stage_train.remote(scene, max_steps)
    print(json.dumps(trained, indent=2))

    report = stage_export.remote(scene, trained["checkpoint"])
    print(json.dumps(report, indent=2))

    print(f"\nPull artifacts:\n  modal volume get zframe-spike scenes/{scene} ./out")
    print("Then OPEN THE .glb IN BLENDER. AGENTS.md §7.")


@app.function(image=image_vggt, volumes={VOL: volume}, timeout=600)
def validate_synthetic(scene: str, ground_truth_json: str, render_fps: float = 24.0) -> dict:
    """Check VGGT's poses against Blender ground truth (plumbing validation only).

    VGGT's output is gauge-free: arbitrary global rotation, translation and
    SCALE. Comparing raw coordinates would be meaningless, so the estimated
    camera centres are first Umeyama-aligned (similarity) onto the true ones and
    the residual is reported relative to the true scene size.

    Only camera CENTRES are compared. Orientation would require converting
    Blender's OpenGL-style convention to COLMAP's, and a silent convention bug
    there would produce a confident, wrong number — worse than no number. Centre
    agreement under a similarity fit is already a strong signal, and it is the
    one this check can make honestly.

    This says the pipeline is wired correctly. It says NOTHING about SPIKE.md's
    thesis — the synthetic scene has none of the properties being tested.
    """
    import json

    import numpy as np
    import pycolmap

    scene_dir = f"{SCENES}/{scene}"

    with open(ground_truth_json) as fh:
        truth = json.load(fh)
    with open(f"{scene_dir}/frame_map.json") as fh:
        frame_map = {f["output"]: f["source_time_s"] for f in json.load(fh)["frames"]}

    true_centres = {
        (p["frame"] - 1) / render_fps: np.array(p["camera_to_world"])[:3, 3]
        for p in truth["poses"]
    }
    true_times = np.array(sorted(true_centres))

    rec = pycolmap.Reconstruction(f"{scene_dir}/sparse")

    estimated, actual, unmatched = [], [], 0
    for img in rec.images.values():
        t = frame_map.get(img.name)
        if t is None:
            unmatched += 1
            continue
        nearest = float(true_times[np.argmin(np.abs(true_times - t))])
        estimated.append(img.projection_center())
        actual.append(true_centres[nearest])

    if len(estimated) < 3:
        return {"error": f"only {len(estimated)} matched cameras", "unmatched": unmatched}

    X = np.array(estimated)   # VGGT, arbitrary gauge
    Y = np.array(actual)      # Blender, true

    # Umeyama: least-squares similarity transform X -> Y.
    mu_x, mu_y = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_x, Y - mu_y
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / ((Xc ** 2).sum() / len(X)))
    aligned = (scale * (R @ X.T)).T + (mu_y - scale * (R @ mu_x))

    residual = np.linalg.norm(aligned - Y, axis=1)
    true_extent = float(np.linalg.norm(Y.max(0) - Y.min(0)))

    result = {
        "matched_cameras": len(estimated),
        "unmatched": unmatched,
        "rmse_world_units": round(float(np.sqrt((residual ** 2).mean())), 4),
        "rmse_pct_of_scene": round(float(np.sqrt((residual ** 2).mean()) / true_extent * 100), 3),
        "max_error_pct_of_scene": round(float(residual.max() / true_extent * 100), 3),
        "recovered_scale_factor": round(scale, 5),
        "note": "centres only; VGGT scale is arbitrary so this is gauge-free",
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(image=image_gsplat, gpu="L4", volumes={VOL: volume}, timeout=900)
def debug_render(scene: str, checkpoint: str, n: int = 3) -> dict:
    """Dump RGB + depth from the trained splat at training poses.

    Diagnostic only. When the mesh comes out wrong the question is always which
    link broke — the splat, the depth channel, or the fusion — and this answers
    it directly instead of by inference.
    """
    import sys

    sys.path.insert(0, "/root/spike")
    import numpy as np
    import torch
    from PIL import Image

    import mesh as M
    from datasets.colmap import Parser

    scene_dir = f"{SCENES}/{scene}"
    out_dir = f"{scene_dir}/preview/debug"
    os.makedirs(out_dir, exist_ok=True)

    parser = Parser(data_dir=scene_dir, factor=1, normalize=True, test_every=1)
    splats = M.load_splats(checkpoint)
    centre, radius, cam_distance = M.subject_region(parser.camtoworlds)

    stats = []
    step = max(1, len(parser.camtoworlds) // n)
    for i in range(0, len(parser.camtoworlds), step):
        if len(stats) >= n:
            break
        camtoworld = parser.camtoworlds[i]
        cam_id = parser.camera_ids[i]
        K = parser.Ks_dict[cam_id]
        width, height = parser.imsize_dict[cam_id]

        viewmat = torch.from_numpy(np.linalg.inv(camtoworld)).float().cuda()
        rgb, depth = M.render_depth_and_colour(
            splats, viewmat, torch.from_numpy(K).float().cuda(), width, height
        )

        Image.fromarray(rgb).save(f"{out_dir}/rgb_{i:04d}.jpg", quality=90)

        valid = depth[depth > 0]
        if len(valid):
            lo, hi = float(valid.min()), float(valid.max())
            vis = np.zeros_like(depth)
            vis[depth > 0] = (depth[depth > 0] - lo) / max(hi - lo, 1e-6)
            Image.fromarray((vis * 255).astype(np.uint8)).save(f"{out_dir}/depth_{i:04d}.png")
        else:
            lo = hi = 0.0

        stats.append({
            "view": i,
            "depth_min": round(lo, 4),
            "depth_max": round(hi, 4),
            "valid_px_pct": round(100.0 * float((depth > 0).mean()), 2),
            "cam_to_subject": round(float(np.linalg.norm(camtoworld[:3, 3] - centre)), 4),
        })

    volume.commit()
    result = {
        "subject_centre": [round(float(v), 4) for v in centre],
        "subject_radius": round(radius, 4),
        "median_cam_distance": round(cam_distance, 4),
        "views": stats,
    }
    print(result)
    return result
