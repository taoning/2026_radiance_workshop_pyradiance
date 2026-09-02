"""
02_viz.py -- getting pictures out of Radiance and into matplotlib.

    python 02_viz.py

Radiance's HDR format stores real physical radiance, not display values. An
8-bit PNG cannot hold it. So every visualization is a decision about how to
squash four or five orders of magnitude into 256 levels, and the right answer
depends on what question you are asking.

We make the same image four ways:

    out/02_a_srgb.png         exposure + tone compression   "what it looks like"
    out/02_b_pcond.png        pcond -h + iso-luminance      "what the eye does"
    out/02_c_falsecolour.png  log falsecolour + colourbar   "what the numbers are"
    out/02_d_profile.png      a slice through the image     "where the glare is"

Only (a) is purely a picture. (b) is a picture with numbers drawn on top --
the greys come from pcond and are perceptual, the contour lines come from the
original luminance array and are physical.

Radiance ships a `falsecolor` script, but pyradiance does not bundle it -- run
00_verify.py and you will see it reported as absent. We do it in matplotlib
instead, which is more flexible anyway.
"""

from __future__ import annotations

import numpy as np
import pyradiance as pr

import wsviz
from wsvenv import MODEL, OUT, VIEWS, banner, checkpoint, render_view, step, timed

banner("02 -- visualization")

# ---------------------------------------------------------------------------
# Get an image. Reuse 01's preview if it is there.
# ---------------------------------------------------------------------------
hdr_path = MODEL / "preview.hdr"
if not hdr_path.exists():
    step("no preview from 01 found, rendering one")
    try:
        with timed("rpict"):
            hdr_path.write_bytes(render_view("office.oct", VIEWS / "hero.vf", 480, 330, quality='good'))
    except Exception as e:  # noqa: BLE001
        print(f"     could not render ({e}) -- falling back to the shipped image")
hdr_path = checkpoint(hdr_path, "preview.hdr")
hdr = hdr_path.read_bytes()


# ===========================================================================
#  THE ONE IDEA:  pvalue -> numpy
# ===========================================================================
# wsviz.hdr_to_array() wraps this, but do it by hand once so you know what is
# in the box. There is no magic and no file format parsing.
step("HDR -> numpy, the long way")

xres, yres = pr.get_image_dimensions(hdr)     # careful: returns (width, height)
print(f"     image is {xres} wide x {yres} high")

raw = pr.pvalue(
    hdr,
    header=False,     # -h   drop the text header
    resstr=False,     # -H   drop the resolution line
    outform="f",      # -df  raw 32-bit floats, not ascii; implies "data only"
)
print(f"     pvalue returned {len(raw)} bytes = {len(raw) // 4} floats")

arr = np.frombuffer(raw, dtype=np.single).reshape(yres, xres, 3)
print(f"     -> numpy array {arr.shape}, dtype {arr.dtype}")
# Radiance scans row by row, so ROWS (y) is the first axis. The pyradiance
# docs example reshapes (xres, yres, 3), which is only correct for a square
# image -- ours is not, so it would come out transposed.

# ---------------------------------------------------------------------------
# RGB -> luminance
# ---------------------------------------------------------------------------
# These three weights turn Radiance's RGB radiance into photometric luminance.
# You will see the same triple in rmtxop -c and in every Radiance workflow.
lum = arr @ np.array([47.4, 119.9, 11.6])
print(f"     luminance: min {lum.min():.2f}, median {np.median(lum):.1f}, "
      f"max {lum.max():.0f} cd/m2  (dynamic range {lum.max() / max(lum.min(), 1e-6):.0f}:1)")


# ===========================================================================
#  FOUR WAYS TO LOOK AT IT
# ===========================================================================
step("(a) exposure + Reinhard + sRGB -- the photograph")
wsviz.save_srgb(hdr, "02_a_srgb", "(a) auto-exposed sRGB -- colour preserved")

step("(b) pcond -h + iso-luminance contours -- the human visual system")
# pcond models adaptation, veiling glare and loss of colour vision in the dark.
# Compare with (a): the shadows go grey. That is deliberate, and physiological.
#
# But pcond has already destroyed the physical values, so the greys mean
# nothing quantitative. The fix is to overlay contours drawn from the ORIGINAL
# luminance array: the picture shows what the eye does, the lines show what the
# numbers are. Note we pass `lum`, NOT the tonemapped output.
wsviz.save_tonemap(hdr, "02_b_pcond",
                   "(b) pcond -h, contours = luminance [cd/m$^2$]",
                   contour=lum, levels=[100, 300, 1000, 3000, 10000])
# >>> TODO (you): drop sigma= to 0.5 and watch the contours turn to spaghetti.
# >>> That is rpict sampling noise, and it is why we blur before contouring.

step("(c) log falsecolour -- the measurement")
wsviz.save_falsecolor(lum, "02_c_falsecolour",
                      "(c) luminance, log scale",
                      vmin=1.0, vmax=6000)
# >>> TODO (you): switch to a linear scale (log=False) and see how much of the
# >>> room disappears into the bottom of the colour bar. That is why daylight
# >>> results are almost always plotted logarithmically.

step("(d) a horizontal slice")
wsviz.save_profile(lum, "02_d_profile",
                   "(d) luminance across the image, mid-height")


# ===========================================================================
#  WHY THE COLOURBAR MATTERS
# ===========================================================================
step("where is the glare?")
# A quick quantitative question we can now answer in two lines of numpy.
for thresh in (500, 2000, 10000):
    pct = (lum > thresh).mean() * 100
    print(f"     {pct:5.2f}% of pixels exceed {thresh:>6,} cd/m2")

hottest = np.unravel_index(np.argmax(lum), lum.shape)
print(f"     brightest pixel: row {hottest[0]}, col {hottest[1]} "
      f"= {lum[hottest]:,.0f} cd/m2")

# pextrem does the same thing inside Radiance, and returns the position too.
lo, hi = pr.pextrem(hdr, original=True)
print(f"     pextrem agrees: max at ({hi.x}, {hi.y}) rgb=({hi.R:.1f}, {hi.G:.1f}, {hi.B:.1f})")

banner("done")
print(f"  Four PNGs in {OUT.name}/. Open them side by side -- same data, four questions.")
print("  Next: 03_pointintime.py")
