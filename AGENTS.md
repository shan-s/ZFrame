# AGENTS.md — ZFrame

Rules for any coding agent working in this repo. Read fully before writing code.
`ARCHITECTURE-CAPTURE.md` holds the reasoning; **this file holds the rules.**
Where they appear to conflict, follow this file and flag the conflict.

---

## 1. Licence rules — these are absolute

This product is sold commercially from the **United Kingdom**. Several of the
best-known, best-documented libraries in 3D reconstruction are non-commercial or
territory-restricted. An agent reaching for "the standard approach" will pick one
of them and the code will look completely correct. Do not.

**Never add, import, vendor, copy from, or install any of these:**

| Forbidden | Reason |
| :--- | :--- |
| `graphdeco-inria/gaussian-splatting` and any fork or submodule of it | Non-commercial: *"THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE SOFTWARE FOR COMMERCIAL PURPOSES WITHOUT PRIOR AND EXPLICIT CONSENT"* |
| `diff-gaussian-rasterization` | Same Inria licence, usually vendored as a submodule |
| **2DGS** (`hbb1/2d-gaussian-splatting`) | Carries the Inria non-commercial licence — **verified** |
| **GOF** (`autonomousvision/gaussian-opacity-fields`) | Carries the Inria non-commercial licence — **verified** |
| **SuGaR** | Built on Inria 3DGS — treat as forbidden unless proven otherwise |
| **DUSt3R / MASt3R** weights (Naver) | Checkpoints inherit non-commercial terms from training data (3D_Street_View) |
| **Hunyuan3D** (any version) | Licence text: *"THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"* |
| **FLUX.1 Kontext [dev]** | Non-commercial (BFL licence) |
| FFmpeg linked against `libx264` / `libx265` | GPL — would oblige open-sourcing this backend |

**Use these instead:**

| Approved | Licence | Role |
| :--- | :--- | :--- |
| `facebook/VGGT-1B-Commercial` | vggt-aup (commercial OK, no military) | Camera poses, depth, point maps |
| `nerfstudio-project/gsplat` | Apache 2.0 | Splat training and rasterisation |
| Open3D | MIT | TSDF fusion / Poisson meshing |
| Qwen-Image-Edit | Apache 2.0 | Generative fill and touch-up |
| FFmpeg, LGPL build, called as a **subprocess** | LGPL | Frame extraction |

### Model checkpoint trap

The VGGT README and every tutorial use `facebook/VGGT-1B`. **That is the wrong
checkpoint.** Always load `facebook/VGGT-1B-Commercial`. If you copy example code,
change this line.

### Mesh extraction

The academic mesh-extraction repos are all Inria-derived and unusable. The
licence-clean path is: **render depth maps from the trained gsplat, then TSDF-fuse
or Poisson-reconstruct with Open3D.** Do not introduce a new mesh library to
avoid writing this.

### Standing rule

Every new dependency gets an entry in `LICENSES.md`: name, version, licence,
territory restriction, and a link to the licence text. **If you cannot determine
a licence, do not add the dependency** — stop and ask.

---

## 2. What this is

ZFrame turns ordinary video — including footage shot for some other purpose — into
clean, production-ready 3D assets, delivered on the open web. No mobile app.

The two claims the product lives or dies on:

1. **It works on input that fails elsewhere** — low-texture, glossy, thin
   structures, sparse coverage, motion blur.
2. **The output needs no cleanup** — see §5.

---

## 3. Stack — fixed, do not re-decide

**Spike (current phase) — Python only. No web stack. Do not scaffold a frontend.**

- GPU compute: **Modal**, pinned SDK version, `@modal.fastapi_endpoint`
- Object storage: **Cloudflare R2** (S3-compatible)
- Python 3.11, PyTorch, CUDA 12.x devel base image

Later phases, when we get there:

- Web: **Next.js** (App Router) on Vercel
- DB: **Postgres** + Drizzle
- Auth: **Clerk**
- Payments: **Stripe** (web checkout — there is no app store in this product)
- Viewer: Three.js

If a task seems to need something not listed here, ask rather than choosing.

---

## 4. Pipeline shape

```
video → frame extraction → quality gate → VGGT (poses + points)
      → gsplat (splat) → depth render → Open3D (mesh) → .glb
```

**Two separate endpoints, always** — do not merge them:

- `/generate` — photo/video in, reconstruction bundle out. Expensive, ML-heavy.
- `/export` — bundle in, rendered artifact out. **No ML, no VGGT import.**

**Fail before spending GPU.** The quality gate (blur, frame overlap, coverage)
runs first and rejects bad input with a specific, visual reason. Never run a full
reconstruction on input already known to be unusable.

---

## 5. The export contract

Every exported `.glb` must be:

- Centred at `[0,0,0]`, origin at the **base** of the object, not the bounding-box
  centre and not wherever the camera started
- Upright, sitting flat on the ground plane, derived from the dominant support
  plane and gravity — never left at the capture's arbitrary rotation
- Scaled in **real-world metres**
- Background removed, subject isolated, no floaters, no floor geometry
- Watertight, decimated, with LODs
- PBR maps separated (albedo / roughness / normal), not baked lighting

This contract is the product. A mesh that violates it is a failed build, not a
rough edge.

---

## 6. Correctness traps

- **EXIF orientation:** call `ImageOps.exif_transpose()` on ingest, before
  inference, and strip the orientation tag from derived images. Otherwise
  geometry and texture end up rotated relative to each other with no error.
- **Never present generated geometry as measurement.** Dimensions read off a
  generated region are suppressed, not estimated.
- **Delighting is maths first:** normals → spherical-harmonic lighting estimate →
  divide → *then* a generative touch-up. Never prompt a model to "remove
  shadows"; it changes materials.

---

## 7. Definition of done

A change is done when it runs end to end on a real video and the output is
inspected — not when it compiles. For pipeline work that means opening the
resulting `.glb` in Blender or a browser model-viewer and looking at it.

Do not mark work complete on the basis that no exception was raised.
