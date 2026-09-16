"""Patch gsplat's simple_trainer.py to skip trajectory-video rendering.

WHY
---
simple_trainer.py calls `render_traj()` unconditionally after evaluation
(line ~886), which writes an .mp4 via `imageio.get_writer`. That needs
`imageio-ffmpeg`, which ships a BUNDLED ffmpeg binary whose build licence its
documentation never states. AGENTS.md §1: "If you cannot determine a licence, do
not add the dependency." So the extra is excluded (see LICENSES.md) and the call
would otherwise kill the run AFTER training finishes:

    ValueError: Could not find a backend to open .../traj_6999.mp4 with iomode `w?`

`render_traj_path` selects the trajectory TYPE and has no "off" value -- an
unrecognised value raises -- so there is no configuration route to disable it.

Nothing is lost: the video exists for human eyeballing, and this spike does its
inspection in Blender from fixed viewpoints (tools/inspect_glb.py), which is both
reproducible and fair across the tools being compared.

Run at image build time. Idempotent, and fails loudly if upstream moves.
"""

import sys

TARGET = "/opt/gsplat/examples/simple_trainer.py"

EDITS = [
    (
        """            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)""",
        """            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                # PATCHED (ZFrame): see spike/patches/patch_simple_trainer.py
                if os.environ.get("ZFRAME_SKIP_TRAJ_VIDEO") != "1":
                    self.render_traj(step)""",
    ),
    (
        """        runner.eval(step=step)
        runner.render_traj(step=step)""",
        """        runner.eval(step=step)
        # PATCHED (ZFrame): see spike/patches/patch_simple_trainer.py
        if os.environ.get("ZFRAME_SKIP_TRAJ_VIDEO") != "1":
            runner.render_traj(step=step)""",
    ),
]

MARKER = "PATCHED (ZFrame)"


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as fh:
        source = fh.read()

    if MARKER in source:
        print("patch_simple_trainer: already applied")
        return 0

    for old, new in EDITS:
        if old not in source:
            print(
                "patch_simple_trainer: FAILED -- expected upstream text not found.\n"
                "gsplat's simple_trainer.py has changed. Verify by hand where\n"
                "render_traj() is called before updating this patch.\n"
                f"--- looked for ---\n{old}",
                file=sys.stderr,
            )
            return 1
        source = source.replace(old, new, 1)

    # `os` is imported at simple_trainer.py:3, so the guard has it in scope.
    if "\nimport os\n" not in source:
        print("patch_simple_trainer: FAILED -- `os` not imported upstream", file=sys.stderr)
        return 1

    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source)

    print("patch_simple_trainer: applied -- render_traj skipped when "
          "ZFRAME_SKIP_TRAJ_VIDEO=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
