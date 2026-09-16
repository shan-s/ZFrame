# SPIKE.md — Validate the reconstruction claim

**Goal:** answer one question as cheaply as possible — *does VGGT + gsplat produce
visibly better results than Polycam, KIRI and Luma on captures those tools are
known to fail on?*

Everything else in `ARCHITECTURE-CAPTURE.md` depends on the answer. If it is no,
the positioning, pricing and go-to-market are all moot and should be rethought
before any product is built.

**Budget:** roughly a day of work and under £10 of GPU.
**Out of scope:** web app, auth, payments, accounts, UI, generative fill,
delighting, credits. Do not build them. Do not scaffold a frontend.

---

## 1. What you are building

One Modal script. Video in, `.glb` and splat out, run from your laptop.

```
video.mp4
  → ffmpeg keyframes (subprocess, LGPL build)
  → VGGT-1B-Commercial  → sparse/{cameras,images,points3D}.bin
  → gsplat simple_trainer → trained splat
  → depth render + Open3D TSDF → mesh → .glb
```

The VGGT→gsplat bridge is already solved and documented: VGGT ships
`demo_colmap.py`, which writes `cameras.bin`, `images.bin` and `points3D.bin`
into `<scene>/sparse/` in COLMAP format. gsplat's `simple_trainer.py default`
consumes exactly that layout. **You are wiring two documented interfaces
together, not inventing a format.**

---

## 2. Container

Build one Modal image for the whole spike — a single cold start covers the run.

- Base: `nvidia/cuda:12.1.1-devel-ubuntu22.04`, `add_python="3.11"`
- `apt`: `git`, `ninja-build`, `ffmpeg` *(LGPL build — see AGENTS.md §1)*, the
  usual OpenCV runtime libs (`libglib2.0-0`, `libsm6`, `libxrender-dev`,
  `libxext6`)
- `pip`: `torch`, `torchvision` matching the CUDA version, `numpy<2.0`,
  `opencv-python`, `trimesh`, `open3d`, `huggingface_hub`
- Clone and install `facebookresearch/vggt` and `nerfstudio-project/gsplat`
- Bake the **`facebook/VGGT-1B-Commercial`** weights into the image so they are
  cached, not downloaded per run

GPU: **A10G or L4** is enough for the spike. Set `timeout=1800` — the first
runs will be slow and you do not want the container reclaimed mid-training.

---

## 3. Steps

**3.1 Frame extraction.** `ffmpeg` as a subprocess. Target **60–150 keyframes**,
selected on sharpness and pose spread rather than fixed interval. Run
`ImageOps.exif_transpose()` on anything derived from stills.

**3.2 VGGT.** Run `demo_colmap.py --scene_dir=<scene>`. Inspect the resulting
point cloud before going further — if the poses are wrong, nothing downstream
can recover. Note that `--use_ba` enables bundle adjustment and may pull in
`pycolmap`; check its licence before using that flag (AGENTS.md §1).

**3.3 Splat.** `python simple_trainer.py default --data_dir <scene> --result_dir
<out>`. Start with default iterations; do not tune yet.

**3.4 Mesh.** Render depth from the trained splat at the training camera poses,
TSDF-fuse with Open3D, extract the mesh, export `.glb` via trimesh. **This is
the only licence-clean mesh path** — do not substitute 2DGS, GOF or SuGaR
(AGENTS.md §1).

**3.5 Look at it.** Open every `.glb` in Blender or a browser model-viewer. The
spike is not done because the script exited zero.

---

## 4. The three test captures

Shoot these yourself on a phone, handheld, no special technique — that is the
point. Then run the same footage through Polycam, KIRI and Luma free tiers and
put the results side by side.

| # | Subject | What it is testing | Expected incumbent failure |
| :--- | :--- | :--- | :--- |
| 1 | **Glossy black coffee maker against a plain white wall** | Low-texture background + specular dark subject | Polycam: photogrammetry has no features on the wall; LiDAR scatters on gloss and is absorbed by black. Expect a hole. |
| 2 | **Chair with thin wire or metal legs** | Thin intersecting geometry | KIRI: meshing culls thin structures as noise. Expect legs to vanish or blob. |
| 3 | **Any object, exported as mesh** | Export cleanliness | Luma: exports carry background floor, floaters, and a melted-wax subject. |

Add a fourth if you can: **a video you shot months ago for something else** — an
unplanned, casual clip. That one tests the actual positioning claim
(`ARCHITECTURE-CAPTURE.md` §1), not just reconstruction quality.

---

## 5. Pass / fail

**Pass** — on at least two of the three targets, your result is *visibly* better
to someone who is not you: no gaping hole where the incumbent has one, thin
structures intact, no background geometry in the export. Screenshot everything.
That gallery becomes the most persuasive marketing asset you have.

**Partial** — reconstruction is good but the mesh or export is poor. That is an
engineering problem, not a thesis problem. Continue, but the zero-cleanup claim
(AGENTS.md §5) is now the risk to chase.

**Fail** — results are comparable to or worse than the incumbents. Stop. The
capability claim is not real, and the positioning needs rethinking before
anything else is built. This is the cheapest possible version of that discovery
and finding it here is a success, not a wasted day.

---

## 6. Record as you go

Keep `LICENSES.md` updated from the first dependency. It is trivial now and
painful to reconstruct later — and this is the one area where a mistake is not
recoverable by refactoring.

Also keep a plain log of what failed and why: which captures broke, at which
stage, with what symptom. That log is what the quality gate gets built from in
the next phase, and you will not remember the details.
