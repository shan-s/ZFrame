"""Patch VGGT's demo_colmap.py to load the COMMERCIAL checkpoint.

AGENTS.md §1 ("model checkpoint trap"): upstream hardcodes a raw URL to
`facebook/VGGT-1B` -- the NON-COMMERCIAL checkpoint -- at demo_colmap.py:114.
Running the script as SPIKE.md §3.2 instructs would silently load
licence-incompatible weights and produce a perfectly plausible result.

This patcher is deliberately strict: it asserts the exact upstream text is
present and fails loudly if it is not. A silent no-op here would reintroduce
the trap on the next upstream bump, which is precisely the failure mode
AGENTS.md §1 calls unrecoverable.

Run at image build time. Idempotent.
"""

import sys

TARGET = "/opt/vggt/demo_colmap.py"

# Exact upstream text at commit a288dd0f (verified 2026-09-15).
UPSTREAM = '''    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))'''

REPLACEMENT = '''    # PATCHED (ZFrame, AGENTS.md §1): upstream hardcoded the non-commercial
    # facebook/VGGT-1B checkpoint here. Load the baked commercial weights instead.
    # No network fetch, no fallback -- an absent file must crash, never silently
    # download the wrong weights.
    import os as _os
    _ckpt = _os.environ["VGGT_COMMERCIAL_CKPT"]
    assert _os.path.exists(_ckpt), f"Commercial VGGT checkpoint missing: {_ckpt}"
    model.load_state_dict(torch.load(_ckpt, map_location="cpu"))'''

ALREADY = "PATCHED (ZFrame, AGENTS.md §1)"


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as fh:
        source = fh.read()

    if ALREADY in source:
        print("patch_demo_colmap: already applied, nothing to do")
        return 0

    if UPSTREAM not in source:
        print(
            "patch_demo_colmap: FAILED -- expected upstream checkpoint-loading text\n"
            "was not found in demo_colmap.py. Upstream has changed.\n"
            "DO NOT bypass this: verify by hand which checkpoint the new code loads\n"
            "and update UPSTREAM before rebuilding. See AGENTS.md §1.",
            file=sys.stderr,
        )
        return 1

    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source.replace(UPSTREAM, REPLACEMENT, 1))

    # Prove the forbidden checkpoint is gone from the file entirely.
    with open(TARGET, "r", encoding="utf-8") as fh:
        patched = fh.read()
    if "facebook/VGGT-1B/resolve" in patched:
        print("patch_demo_colmap: FAILED -- forbidden checkpoint URL still present", file=sys.stderr)
        return 1

    print("patch_demo_colmap: applied -- now loads $VGGT_COMMERCIAL_CKPT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
