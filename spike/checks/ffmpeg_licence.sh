#!/usr/bin/env bash
# Assert the built ffmpeg carries NO GPL components. Run at image build time.
#
# AGENTS.md §1 forbids "FFmpeg linked against libx264/libx265 -- GPL -- would
# oblige open-sourcing this backend". SPIKE.md §2 assumed `apt install ffmpeg`
# gives an LGPL build; on ubuntu22.04 it does not (libavcodec58 depends on
# libx264-163 and libx265-199), which is why modal_app.py compiles from source.
#
# x264/x265 are ENCODERS. This pipeline only decodes video into stills, and
# H.264/HEVC decoding is native to ffmpeg, so --disable-gpl costs no capability.
# This script proves both halves of that claim: no GPL in, decoders still there.
#
# Fails the BUILD. A licence breach found later in an audit is the one class of
# mistake AGENTS.md §1 says refactoring cannot recover.

set -euo pipefail

FF="${FFMPEG_BIN:-/usr/local/bin/ffmpeg}"
CONFIG="$("$FF" -version | grep configuration:)"
echo "$CONFIG"

fail() { echo "LICENCE VIOLATION: $1 (AGENTS.md §1)" >&2; exit 1; }

for TOKEN in "--enable-gpl" "--enable-nonfree" "libx264" "libx265"; do
  if echo "$CONFIG" | grep -q -- "$TOKEN"; then
    fail "ffmpeg built with $TOKEN"
  fi
done

# Capture once, then match against the variables. Piping ffmpeg into `grep -q`
# looks cleaner but is a trap here: grep exits on first match, ffmpeg takes
# SIGPIPE, and `set -o pipefail` turns that into a pipeline failure -- so a
# SUCCESSFUL match reports as a failure.
ENCODERS="$("$FF" -hide_banner -encoders 2>/dev/null)"
DECODERS="$("$FF" -hide_banner -decoders 2>/dev/null)"

if grep -qE ' (libx264|libx265) ' <<<"$ENCODERS"; then
  fail "x264/x265 encoder present in build"
fi

# Decoding is the whole reason ffmpeg is here; prove --disable-gpl kept it.
for DEC in h264 hevc vp9; do
  if ! grep -qE "^ *V[^ ]* +${DEC}( |$)" <<<"$DECODERS"; then
    echo "BUILD BROKEN: $DEC decoder missing" >&2
    echo "--- decoders matching '$DEC' ---" >&2
    grep -E "$DEC" <<<"$DECODERS" >&2 || echo "(none)" >&2
    exit 1
  fi
done

echo "ffmpeg licence assertion PASSED: LGPL, decoders intact, no GPL components"
