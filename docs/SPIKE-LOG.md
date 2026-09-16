# SPIKE-LOG.md — what failed, where, and with what symptom

Required by `SPIKE.md` §6. Kept as we go, not reconstructed afterwards: *"that log is
what the quality gate gets built from in the next phase, and you will not remember
the details."*

Format per entry: **date · capture · stage · symptom · resolution**.
Stages: `frames` · `vggt` · `gsplat` · `mesh` · `export` · `inspect`.

---

## Deliberate deferrals (debt, not failures)

Carried knowingly, agreed before implementation started.

| Item | Why deferred | What it would take |
| :--- | :--- | :--- |
| PBR map separation (albedo / roughness / normal) | Requires the delighting chain in AGENTS.md §6 (normals → SH lighting estimate → divide → generative touch-up). Out of a one-day budget. | A delighting pass; the maths first, never a "remove shadows" prompt. |
| LODs | Pure post-process on a mesh we do not yet trust. | Decimation ladder once the base mesh is good. |
| **True metric scale** (AGENTS.md §5 "real-world metres") | VGGT output is **scale-arbitrary**. Bare phone video carries no scale reference. This is an open problem, not an engineering task. | A known-size reference in frame, ARKit/ARCore metadata, or a metric-depth model. |
| Quality gate (AGENTS.md §4 "fail before spending GPU") | SPIKE.md §6 says the gate is built *from this log*, so the log must exist first. | Next phase, informed by the failures below. |

## Substitutions made against the spec

| Spec says | What was done | Why |
| :--- | :--- | :--- |
| SPIKE §3.1: select keyframes on "sharpness and **pose spread**" | Sharpness (variance-of-Laplacian) + **inter-frame feature displacement** as a pose-spread proxy | Circular as written — there are no poses until VGGT has run, and VGGT is what consumes the frames. |
| SPIKE §2: "`apt`: ffmpeg (LGPL build)" | FFmpeg **compiled from source** with `--disable-gpl --disable-nonfree` | Ubuntu's ffmpeg links `libx264`/`libx265` and is GPL. See `LICENSES.md`. |
| SPIKE §3.2: "`--use_ba` **may** pull in pycolmap" | pycolmap accepted as a **direct, unavoidable** dependency (BSD-3-Clause) | Required on both sides of the bridge regardless of `--use_ba`. See `LICENSES.md`. |
| SPIKE §3.2: check pycolmap's licence "before using that flag" | **`--use_ba` forbidden outright** | Right instinct, wrong target. pycolmap is fine; `--use_ba` hardcodes `keypoint_extractor="aliked+sp"` → **SuperPoint**, whose weights are Magic Leap **non-commercial**. See `LICENSES.md`. |
| SPIKE §1: "**One** Modal script" | **Two Modal images**, one app | Forced by upstream: VGGT needs official `pycolmap`, gsplat v1.5.3 needs rmbrualla's `pycolmap`. Same module name — cannot coexist. Handoff is the COLMAP `sparse/` dir, which is the designed interface anyway. |
| SPIKE §2: gsplat (unversioned) | Pinned **v1.5.3**, not `main` | `main` needs `numpy>=2` (VGGT needs `<2`) and pins `torch==2.9.1` described upstream as "internal builds". |
| gsplat examples' `imageio[ffmpeg]` | Base `imageio` only | The extra bundles an ffmpeg binary of undocumented build licence. Only needed for optional progress videos. |
| SPIKE §3.2: run `demo_colmap.py` as shipped | `demo_colmap.py` **patched** before use | Upstream hardcodes the non-commercial `facebook/VGGT-1B` checkpoint at line 114. |

---

## Run log

### 2026-09-15 · synthetic (Blender turntable) · `vggt`
**Symptom:** `demo_colmap.py` exited 0, wrote valid `cameras/images/points3D.bin`,
and the reconstruction contained **zero 3D points**. Nothing errored anywhere.
**Cause:** the non-BA path filters with `depth_conf >= conf_thres_value`
(`demo_colmap.py:~213`), default **5.0**. VGGT's depth confidence on a synthetic
EEVEE render sits below that everywhere, so every point was discarded. An empty
reconstruction is a perfectly valid COLMAP model, so nothing downstream objected.
**Resolution:** `conf_thres_value` is now an explicit pipeline parameter. On zero
points the run retries once at **1.0**, logs loudly, and records
`conf_thres_used` / `conf_thres_relaxed` in the manifest. At 1.0: **100,000
points** (the `max_points_for_colmap` cap), 82/82 images registered.
**Gate implication:** *"wrote the files"* is not *"produced a reconstruction"*.
The quality gate must assert a **minimum point count** and surface the confidence
threshold that produced it — a relaxed threshold silently buys plausible garbage.
Whether real phone captures also fall below 5.0 is still unknown and matters: if
they do, the default is wrong for everyone, not just for synthetic input.

### 2026-09-15 · synthetic · `vggt` — pose accuracy vs ground truth
**Result (not a failure):** VGGT poses checked against Blender's known camera
path via Umeyama similarity alignment: **82/82 matched, RMSE 0.46% of scene
extent, max error 1.10%**.
**Why it is recorded:** it separates "the pipeline ran" from "the poses are
right", and it is the only stage where ground truth exists. It says nothing about
SPIKE.md's thesis — the synthetic scene is well-textured, matte and evenly lit,
i.e. the easy case.

### 2026-09-15 · build environment · `container`
Three build failures worth remembering, none of them about reconstruction:
1. `pip install gsplat` → `ModuleNotFoundError: No module named 'torch'`.
   gsplat's `setup.py` imports torch inside `get_extensions()`; pip's build
   isolation hides it. Needs `--no-build-isolation` (same for `fused-ssim` and
   `fused-bilagrid`).
2. `error: command 'clang++' failed`. Modal's `add_python` reports clang in
   sysconfig, so setuptools links extensions with clang++ although nvcc/gcc
   compiled them. Installed clang late, so the ~2.5 GB torch layer stays cached.
3. `could not parse Dockerfile`. `run_commands` renders to Dockerfile `RUN`
   lines and cannot hold a multi-line script — hence
   `spike/checks/ffmpeg_licence.sh` as a file.

Also: a `set -o pipefail` + `grep -q` pipeline reported the h264 decoder as
**missing** when it was present — `grep -q` exits on first match, ffmpeg takes
SIGPIPE, and pipefail turns a successful match into a failed pipeline. The
licence check now captures output into variables first. Worth remembering
because the failure mode is inverted: it cries wolf rather than staying silent.

<!--
Template:

### YYYY-MM-DD · <capture name> · <stage>
**Symptom:** what was actually observed, visually — not the exception text alone.
**Cause:** if known.
**Resolution:** what changed, or "unresolved".
**Gate implication:** what a pre-flight check could have caught, for the next phase.
-->

### 2026-09-15 · synthetic · `mesh` — three traps that all produce a clean, wrong mesh

Every one of these passed the export contract while being completely wrong. They
are recorded in detail because the quality gate has to catch them, and because
none announces itself — each yields a single-component, base-at-origin,
centred mesh that looks like a pass in JSON.

**1. "Keep the largest connected component" keeps the FLOOR.**
An object on a surface reconstructs as a big sheet with a small island on top,
so the largest component is reliably the ground and the subject is what gets
discarded. The result was a tidy, single-component, contract-passing mesh of a
patch of floor.
*Fix:* the capture tells us where the subject is — every camera points at it.
Intersecting the optical axes gives that point, and the component is chosen by
proximity to it, never by size.

**2. Sizing the TSDF from the scene extent resolves the room, not the object.**
`voxel = scene_extent / 512` with a wide floor in view gave a voxel of 0.03 for
a subject 0.33 across — about 11 voxels. The subject essentially vanished.
*Fix:* size the volume from the subject region. Not a synthetic-scene artifact:
a phone capture of an object on a table has the same shape.

**3. A splat renders the empty background, and its depth there reads as NEAR.**
Expected depth (`render_mode="RGB+ED"`) over low-opacity background Gaussians
returns a near value rather than "nothing". Fused across an orbit those readings
agree, so TSDF builds a solid shell wrapped around the subject. It survives every
2D cleanup because in 3D it is a genuine, consistent surface.
*Fix:* unproject each depth pixel and drop anything outside the subject sphere,
before fusion. This also does half of AGENTS.md §5's "background removed" for free.

**And the measurement that made 1–3 fixable:** the subject's radius is measured
from rendered depth at the centre of frame (where the cameras are aimed), not
from the orbit radius. Orbit radius measures how far back the photographer stood,
which is a property of the room, not the object. Measured 0.2238 against a true
0.164 half-extent — the right order, which nothing derived from orbit radius was.

**Where it ended up (7k steps, a quarter of default):** recognisable subject,
upright, base at origin, centred, background removed, single component,
**recovered height 0.3354 vs a true 0.329** — under 2% error on a scale nothing
in the pipeline was told. Still **not watertight**, still holey, with one
debris shard attached above the head. That is SPIKE.md §5's "Partial": an
engineering problem, not a thesis problem.

**Gate implications for the next phase.** None of these is detectable from the
mesh alone, so the gate needs cross-checks, not self-consistency checks:
- subject volume as a fraction of the region — a floor patch is thin and wide
- whether the kept component contains the camera convergence point
- voxel size relative to the SUBJECT, never to the scene
- component count after decimation, which re-fragments a clean mesh (273 pieces
  here); the contract must be measured after every destructive step, not once

### 2026-09-15 · gsplat · `train` — trainer dies after training succeeds
**Symptom:** `ValueError: Could not find a backend to open .../traj_6999.mp4`.
**Cause:** `simple_trainer.py` calls `render_traj()` unconditionally and writes an
.mp4 through `imageio`, which needs `imageio-ffmpeg` — excluded on licence
grounds (bundled binary of undocumented licence, see `LICENSES.md`).
`render_traj_path` selects the trajectory type and has no "off" value.
**Resolution:** `spike/patches/patch_simple_trainer.py` guards both call sites
behind `ZFRAME_SKIP_TRAJ_VIDEO=1`. Note the checkpoint and .ply are written
*before* this point in the loop, so the earlier crashed run had usable output —
worth knowing before paying to retrain.
