# LICENSES.md — ZFrame dependency licence record

Maintained per **AGENTS.md §1 (standing rule)**: every dependency gets an entry with
name, version, licence, territory restriction, and a link to the licence text.
**If a licence cannot be determined, the dependency is not added** — stop and ask.

ZFrame is sold commercially from the **United Kingdom**. Territory restrictions are
therefore recorded explicitly, not assumed absent.

Last verified: **2026-09-15**

---

## Core reconstruction stack

| Dependency | Version / pin | Licence | Territory restriction | Licence text |
| :--- | :--- | :--- | :--- | :--- |
| `facebook/VGGT-1B-Commercial` **(weights)** | HF rev `ebb29a53` | `vggt-aup-license` (commercial permitted; AUP prohibits military use) | None found | [HF LICENSE](https://huggingface.co/facebook/VGGT-1B-Commercial/blob/main/LICENSE) |
| `facebookresearch/vggt` **(code)** | pin `a288dd0f14786c93483e45524328726ab7b1b4ce` | Meta "VGGT License" v1 (2025-07-29), Research Materials + AUP | None found | [LICENSE.txt](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) |
| `nerfstudio-project/gsplat` | tag **`v1.5.3`** | Apache-2.0 | None | [LICENSE](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE) |
| `pycolmap` (official, **image A**) | 3.10.0 (VGGT's own pin) | BSD-3-Clause ("new BSD") | None | [COLMAP LICENSE](https://github.com/colmap/colmap/blob/main/LICENSE) · [PyPI](https://pypi.org/project/pycolmap/) |
| `pycolmap` (rmbrualla fork, **image B**) | `cc7ea4b7` via gsplat v1.5.3 | MIT | None | [LICENSE](https://github.com/rmbrualla/pycolmap/blob/master/LICENSE) |
| `lightglue` (jytime fork) | git `main` | Apache-2.0 | None — **but see SuperPoint below** | [LICENSE](https://github.com/jytime/LightGlue/blob/main/LICENSE) |
| `open3d` | 0.19.0 | MIT | None | [LICENSE](https://github.com/isl-org/Open3D/blob/main/LICENSE) |
| FFmpeg (**self-built, `--disable-gpl`**) | 6.x source build | LGPL-2.1-or-later | None | [FFmpeg legal](https://www.ffmpeg.org/legal.html) · [LGPLv2.1](https://github.com/FFmpeg/FFmpeg/blob/master/COPYING.LGPLv2.1) |

### Two licences, not one — VGGT

The VGGT **code** and the VGGT **weights** are covered by *separate documents*. The
repository's `LICENSE.txt` is Meta's "VGGT License" (Research Materials + Acceptable
Use Policy); the commercial checkpoint carries `vggt-aup-license` from its own HF repo.
Both are recorded above. Neither contains a non-commercial restriction — verified by
reading the text, not by trusting the repo's GitHub licence label, which reads
`NOASSERTION`.

### Why FFmpeg is built from source — do not "simplify" this

`apt install ffmpeg` on `ubuntu22.04` pulls `libavcodec58`, which depends on
**`libx264-163` and `libx265-199`**. That is a **GPL build linked against x264/x265**,
forbidden by name in AGENTS.md §1 — it would oblige open-sourcing this backend.

`libx264`/`libx265` are **encoders**. This pipeline only decodes video into stills, and
H.264/HEVC *decoding* is native to FFmpeg. Building with `--disable-gpl --disable-nonfree`
therefore costs no functionality. The image build asserts on `ffmpeg -version` and
**fails** if `--enable-gpl`, `libx264` or `libx265` appear.

### pycolmap — unavoidable, and why that is fine

AGENTS.md's source docs treat pycolmap as an optional consequence of `--use_ba`. It is not.
`demo_colmap.py:23` imports it at module top level, the non-BA write path
(`reconstruction.write()`, line 246) is a pycolmap call, and gsplat's own COLMAP parser
(`examples/datasets/colmap.py:144`) constructs `pycolmap.Reconstruction`. It is required
on **both sides** of the VGGT→gsplat bridge regardless of flags.

BSD-3-Clause is commercially clean with no territory restriction, so this is recorded and
accepted rather than worked around. Note COLMAP's own licence text warns that *bundled
dependencies* (Ceres BSD-3, Eigen MPL-2.0, FLANN BSD) are separately licensed; all are
commercially permissive. Re-verify if we ever vendor or statically relink pycolmap.

---

## Supporting Python dependencies

| Dependency | Version | Licence | Territory restriction | Licence text |
| :--- | :--- | :--- | :--- | :--- |
| `modal` | 1.5.5 (pinned, AGENTS.md §3) | Apache-2.0 | None | [LICENSE](https://github.com/modal-labs/modal-client/blob/main/LICENSE) |
| `torch` / `torchvision` | CUDA 12.1 build | BSD-3-Clause | None | [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| `numpy` | `<2.0` | BSD-3-Clause | None | [LICENSE](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| `opencv-python` | latest | Apache-2.0 | None | [LICENSE](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| `trimesh` | latest | MIT | None | [LICENSE](https://github.com/mikedh/trimesh/blob/main/LICENSE.md) |
| `huggingface_hub` | latest | Apache-2.0 | None | [LICENSE](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| `Pillow` | latest | MIT-CMU | None | [LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |

---

## Rejected — do not reintroduce

Recorded so a future contributor (human or agent) does not "helpfully" add one back.
All are forbidden by AGENTS.md §1.

| Rejected | Reason |
| :--- | :--- |
| `graphdeco-inria/gaussian-splatting`, `diff-gaussian-rasterization` | Inria non-commercial |
| **2DGS** — including `gsplat/examples/simple_trainer_2dgs.py` | 2DGS is Inria-non-commercial. gsplat's reimplementation sits inside an Apache-2.0 repo and *looks* safe; AGENTS.md §1 and SPIKE.md §3.4 forbid it regardless. The mesh path is depth-render → Open3D TSDF. |
| **GOF**, **SuGaR** | Inria-derived |
| **DUSt3R / MASt3R** weights | Non-commercial inherited from training data |
| **Hunyuan3D** | Licence expressly excludes the **United Kingdom** |
| **FLUX.1 Kontext [dev]** | Non-commercial (BFL) |
| `facebook/VGGT-1B` (non-commercial checkpoint) | Wrong checkpoint. Hardcoded at `demo_colmap.py:114` upstream and patched out in this repo — see `spike/patches/`. |
| FFmpeg linked against `libx264`/`libx265` | GPL — see above |


---

## ⛔ `--use_ba` is FORBIDDEN — SuperPoint is non-commercial

**This is the most dangerous trap found in the spike, and SPIKE.md points straight at it.**

SPIKE.md §3.2 says: *"`--use_ba` enables bundle adjustment and may pull in `pycolmap`;
check its licence before using that flag."* The instinct was right, the reason was wrong.
pycolmap is BSD-3-Clause and is required anyway. The actual hazard is two levels deeper:

```
demo_colmap.py:142   if args.use_ba:
demo_colmap.py:155       predict_tracks(...)
demo_colmap.py:162           keypoint_extractor="aliked+sp"   <-- hardcoded, no CLI flag
                                                  ^^
                                            SuperPoint
```

LightGlue's own README states plainly:

> The pre-trained weights of LightGlue and the code provided in this repository are
> released under the Apache-2.0 license. DISK follows this license as well but
> **SuperPoint follows a different, restrictive license** (this includes its pre-trained
> weights and its inference file `lightglue/superpoint.py`). ALIKED was published under
> a BSD-3-Clause license.

SuperPoint's weights and inference file are **Magic Leap's non-commercial licence**.
ZFrame is sold commercially. Running `--use_ba` would therefore breach AGENTS.md §1 —
and because `"aliked+sp"` is hardcoded rather than exposed as an argument, there is no
command-line way to avoid it.

**Rule:** `--use_ba` is off. Not deferred on cost grounds — **forbidden on licence grounds.**

**If bundle adjustment is ever needed**, the route is to patch line 162 to
`keypoint_extractor="aliked"` (ALIKED alone is BSD-3-Clause), verify no SuperPoint
weights are fetched, and record it here first.

**Why LightGlue is still installed:** `demo_colmap.py:31` → `track_predict.py` →
`vggsfm_utils.py:15` does `from lightglue import ALIKED, SIFT, SuperPoint` at **import
time**, so the package must be present for the module to load at all. On the non-BA path
the SuperPoint class is imported but never instantiated, so its weights are never
downloaded or used. Importing a class is not exercising its licensed weights — but that
line is thin, and it is exactly why `--use_ba` stays off.

---

## Excluded despite being an upstream default

| Excluded | Reason |
| :--- | :--- |
| `imageio-ffmpeg` (the `imageio[ffmpeg]` extra in gsplat's `examples/requirements.txt`) | Ships **platform wheels containing a bundled ffmpeg binary**, and its documentation never states that binary's build licence. AGENTS.md §1: *"If you cannot determine a licence, do not add the dependency."* Only needed for optional training-progress `.mp4`s (`simple_trainer.py:1043`); base `imageio` covers the `imwrite` paths. The image build asserts `imageio_ffmpeg` is absent. |
| `gsplat` `main` branch | Requires `numpy>=2.0` against VGGT's `numpy<2`, and its examples pin `torch==2.9.1`, described upstream as *"internal builds"* — not publicly reproducible. Pinned to `v1.5.3` instead. |

---

## gsplat v1.5.3 example dependencies (image B)

Verified 2026-09-15. All permissive, no territory restrictions.

| Dependency | Licence | | Dependency | Licence |
| :--- | :--- | :--- | :--- | :--- |
| `viser` | MIT | | `tensorboard` | Apache-2.0 |
| `nerfview` | Apache-2.0 | | `tensorly` | BSD-3-Clause |
| `fused-ssim` | MIT | | `splines` | MIT |
| `fused-bilagrid` | MIT | | `scikit-learn` | BSD-3-Clause |
| `torchmetrics` | Apache-2.0 | | `matplotlib` | PSF-based (BSD-compatible) |
| `tyro` | MIT | | `imageio` (base only) | BSD-2-Clause |
| `tqdm` | MPL-2.0 AND MIT | | `pyyaml` | MIT |

`fused-ssim` was checked specifically for Inria provenance given its role in 3DGS
training — it is an independent MIT implementation, not derived from
`graphdeco-inria/gaussian-splatting`.
