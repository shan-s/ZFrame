# ZFrame spike — how to run it

Answers one question (`docs/SPIKE.md`): does VGGT + gsplat beat Polycam, KIRI and
Luma on captures those tools fail on? Disposable by design.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python modal==1.5.5
```

Needs a Modal `huggingface` secret with a token approved for the **manually
gated** `facebook/VGGT-1B-Commercial` repo.

## Verify the images

```bash
modal run spike/modal_app.py::verify
```

Builds both images and checks the things that fail silently: the ffmpeg build
carries no GPL components, the baked checkpoint is the commercial one, the
`demo_colmap.py` patch is applied, and gsplat's CUDA kernels are AOT-compiled
with a working depth channel.

## Run a capture

```bash
modal run spike/modal_app.py::run --video capture.mp4 --scene coffee_maker
```

Stops before GPU training if the pose gate fails. `--skip-train` stops after
reconstruction; `--max-steps` and `--target-frames` are the cost levers.

## Look at the result — this is the point

```bash
modal volume get zframe-spike scenes/coffee_maker ./out
blender --background --python tools/inspect_glb.py -- \
    --glb out/export/coffee_maker.glb --out out/shots
```

`inspect_glb.py` renders fixed azimuths under fixed lighting, framed by the
model's own bounding box, so ZFrame and the incumbents' exports can be compared
without flattering either. Run it on their `.glb`s too — that side-by-side is
the spike's actual deliverable (`SPIKE.md` §5).

**AGENTS.md §7: exiting zero is not done. Done is having looked at it.**

## Smoke test without footage

```bash
blender --background --python tools/make_synthetic_clip.py -- --out /tmp/synthetic
modal run spike/modal_app.py::run --video /tmp/synthetic/synthetic.mp4 --scene synthetic
modal volume put zframe-spike /tmp/synthetic/ground_truth.json \
    scenes/synthetic/ground_truth.json --force
modal run spike/modal_app.py::validate_synthetic --scene synthetic \
    --ground-truth-json /vol/scenes/synthetic/ground_truth.json
```

Validates plumbing against known camera poses. It proves nothing about the
thesis — the scene is well-textured, matte and evenly lit, i.e. the easy case.

`debug_render` dumps RGB and depth from the trained splat when a mesh comes out
wrong and you need to know which link broke.

## Files

| Path | Role |
| :--- | :--- |
| `modal_app.py` | Images, pipeline stages, entrypoints |
| `frames.py` | Keyframe extraction and selection |
| `vggt_step.py` | VGGT → COLMAP, pose gate, pose preview |
| `mesh.py` | Depth render → Open3D TSDF → cleaned `.glb` |
| `checks/ffmpeg_licence.sh` | Build-time GPL assertion |
| `patches/` | Upstream patches — read the docstrings before changing either |

## Two rules worth repeating

- **`--use_ba` is forbidden.** Not cost — licence. It hardcodes SuperPoint, whose
  weights are non-commercial. See `LICENSES.md`.
- **Never `simple_trainer_2dgs.py`.** It sits in the same Apache-2.0 directory and
  looks safe; 2DGS is Inria non-commercial. AGENTS.md §1.
