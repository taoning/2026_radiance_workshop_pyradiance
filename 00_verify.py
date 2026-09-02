"""
00_verify.py -- prove your install works. Run this FIRST.

    python 00_verify.py

If this prints "ALL CHECKS PASSED" and writes out/00_verify.png, you are ready.
If it does not, flag a helper now -- do not wait until block 3.

What it checks:
  1. pyradiance imports and its bundled Radiance binaries are on disk
  2. RAYPATH is wired up so the .cal files can be found
  3. a Radiance binary actually executes (gensky)
  4. the full render -> pvalue -> numpy -> matplotlib -> PNG chain works
"""

import sys
from datetime import datetime

from wsvenv import NPROC, OUT, SCRATCH, banner, env_report, step

banner("pyradiance workshop -- environment check")

failures = []

# 1. import + binaries -------------------------------------------------------
try:
    import pyradiance as pr

    env_report()
except Exception as e:  # noqa: BLE001
    sys.exit(f"FAIL: cannot import pyradiance ({e})\n  Try: pip install --upgrade pyradiance")

step("checking the third-party python stack")
for mod, why in (("numpy", "arrays"),
                 ("matplotlib", "plotting"),
                 ("scipy", "contour smoothing in 02")):
    try:
        m = __import__(mod)
        print(f"     {mod:<11} {getattr(m, '__version__', '?'):<10} ({why})")
    except ImportError:
        failures.append(f"{mod} not installed -- pip install {mod}")

step("checking bundled binaries")
needed = ["oconv", "rtrace", "rpict", "gensky", "gendaylit", "gendaymtx",
          "rfluxmtx", "dctimestep", "rmtxop", "pvalue", "pcond", "genbox",
          "genblinds", "genrev", "xform", "getinfo", "genssky"]
have = {p.name for p in pr.BINPATH.iterdir()}
missing = [n for n in needed if n not in have and f"{n}.exe" not in have]
print(f"     {len(have)} binaries found; {len(needed) - len(missing)}/{len(needed)} required present")
if missing:
    failures.append(f"missing binaries: {missing}")

# Two tools the workshop deliberately does NOT use, because they are not shipped:
for absent in ("falsecolor", "genskyvec"):
    if absent not in have:
        print(f"     note: '{absent}' is not bundled (expected -- we don't use it)")

# 2. RAYPATH / .cal files ----------------------------------------------------
step("checking RAYPATH support files")
libdir = pr.BINPATH.parent / "lib"
cals = {p.name for p in libdir.iterdir()} if libdir.exists() else set()
for c in ("reinhartb.cal", "macbeth.cal", "cieresp.cal", "perezlum.cal"):
    if c not in cals:
        failures.append(f"missing support file {c}")
print(f"     {len(cals)} files in {libdir.name}/")

# 3. run a binary ------------------------------------------------------------
step("running gensky")
try:
    # NOTE: gensky's first argument is a datetime, not month/day/hour.
    sky = pr.gensky(datetime(2024, 6, 21, 12), latitude=40.7, longitude=74.0,
                    timezone=75).decode()
    print("     " + [ln for ln in sky.splitlines() if ln.strip()][0][:60])
except Exception as e:  # noqa: BLE001
    failures.append(f"gensky failed: {e}")

# 4. the whole chain ---------------------------------------------------------
step("rendering a tiny test scene and plotting it")
try:
    import numpy as np

    import wsviz

    # render() computes its ZONE with getbbox(). Writing the geometry to a file
    # and using add_surface(path) is still the clearest way to do it, and it is
    # what `rad` wants; bare Primitive objects now work too.
    geo = SCRATCH / "verify_geometry.rad"
    box = pr.genbox("grey", "block", 1.2, 1.2, 1.2)
    floor = pr.Primitive("grey", "polygon", "floor", [],
                         [-3, -3, 0, 3, -3, 0, 3, 3, 0, -3, 3, 0])
    geo.write_bytes(floor.bytes + b"\n" + box)

    # Scene id must be a bare name: internally it builds both "{sid}.oct" and
    # "m{sid}.oct" in the CWD, so a sid containing a "/" is rejected.
    scene = pr.Scene("verify")
    scene.add_material(pr.Primitive("void", "plastic", "grey", [], [0.6, 0.6, 0.6, 0, 0]))
    scene.add_material(pr.Primitive("void", "light", "bright", [], [40, 40, 40]))
    scene.add_surface(geo)
    scene.add_source(pr.Primitive("bright", "source", "sun", [], [0.3, -0.7, 1, 3]))
    scene.build()

    view = pr.create_default_view()
    view.vp = (2.5, -4, 2.2)
    view.vdir = (-0.45, 1, -0.4)
    hdr = pr.render(scene, view, resolution=(160, 160), ambbounce=1, nproc=NPROC)

    arr = wsviz.hdr_to_array(hdr)
    lum = wsviz.luminance(arr)
    print(f"     image {arr.shape}, mean luminance {lum.mean():.1f} cd/m2")
    wsviz.save_falsecolor(lum, "00_verify", "install check -- if you can see this, you're set",
                          vmin=0.1)
    if not np.isfinite(lum).all():
        failures.append("render produced non-finite values")
except Exception as e:  # noqa: BLE001
    failures.append(f"render/plot chain failed: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
banner("ALL CHECKS PASSED" if not failures else "PROBLEMS FOUND")
if failures:
    for f in failures:
        print(f"  FAIL: {f}")
    sys.exit(1)
print(f"  Open {OUT / '00_verify.png'} to confirm, then move on to 01_model.py")
