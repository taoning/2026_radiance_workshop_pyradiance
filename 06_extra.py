"""
06_extra.py -- complex fenestration: rolling a shade + the IGU into ONE BSDF.

    python 06_extra.py

Everything so far has modelled the shading device as GEOMETRY: genblinds gave
us 28 real slats, and every ray that entered the room had to intersect them.
That works, and it is what 01-05 do. It also has three problems:

    1. It is slow. Slats are small, numerous, and specular.
    2. It only works for devices you can draw. A woven screen, a fritted
       glass, a prismatic film -- you are not going to model the yarn.
    3. It cannot be measured. There is no goniophotometer reading you can
       feed to a pile of polygons.

The alternative is a BSDF: a Bidirectional Scattering Distribution Function.
Instead of geometry, you carry a MATRIX -- "light arriving from direction i
leaves in direction o with this much probability" -- and Radiance samples it.
For the Klems full basis that is a 145 x 145 matrix per component, four
components per side. It is what the whole three/five-phase method is built on.

`E Screen 1% Pearl-Grey.xml` in this directory is exactly that: a Mermet woven
solar screen, measured, Klems full basis, visible and solar, from the LBNL
Complex Glazing Database. 2 MB of matrix.

But here is the thing people get wrong. You have a shade XML, and you have an
IGU. You want the system. You CANNOT multiply the two XMLs together, because
the 50 mm cavity between the shade and the glass interreflects: light bounces
off the room-side of the glass, back onto the shade, off the shade again, and
some of it eventually gets through. A naive product misses all of that.

So you SIMULATE the assembly and re-measure it. That is what genBSDF does, and
it is what this script does in Python:

    device scene  ->  pr.generate_bsdf()  ->  pr.generate_xml()  ->  aBSDF

Outputs:
    out/06_a_klems.png     the system matrix, and what came out of it
    out/06_b_compare.png   geometric venetian blinds vs. the aBSDF system
    out/06_c_grid.png      workplane illuminance, both ways

Runtime is dominated by the sampling in block 2 (~45 s). If it fails, the
script falls back to the shipped checkpoint and carries on.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyradiance as pr
from pyradiance.genbsdf import SamplingBox

import wsviz
from wsvenv import (CKPT, DATA, GRID_PARAMS, HERE, MODEL, NPROC, OUT, SCRATCH,
                    VIEWS, banner, checkpoint, model_dir, render_view,
                    scene_files, step, timed)

# ===========================================================================
#  PARAMETERS
# ===========================================================================
SHADE_SRC = HERE / "E Screen 1% Pearl-Grey.xml"
SHADE_XML = MODEL / "shade_escreen.xml"     # space-free copy; see block 1
SYS_XML = MODEL / "sys_igu_escreen.xml"     # what we are about to build

GAP = 0.05                  # >>> TODO (you): shade-to-glass cavity, m. Try 0.15.
DEVICE_HALF = 3.0           # geometry half-extent -- deliberately huge, see below
SAMPLE_HALF = 0.5           # the 1 x 1 m patch we actually sample

NSAMP = 1000                # >>> TODO (you): 400 is ~15 s, 2000 is ~100 s
BSDF_PARAMS = ["-ab", "4", "-ad", "400", "-lw", "1e-5"]

WHEN = datetime(2024, 3, 21, 13, 0)
SITE = dict(latitude=40.7, longitude=74.0, timezone=75)
GRID = (15, 6)
XRES, YRES = 440, 320

banner("06 -- a system BSDF: woven shade + IGU, collapsed into one matrix")


# ===========================================================================
#  1. THE DEVICE SCENE
# ===========================================================================
# genBSDF's convention: the device lies BEHIND the z = 0 plane, and the room is
# on the +z side. So we build the assembly looking "up" out of the window:
#
#     z = 0        ---- the woven shade (a BSDF layer)      room side, +z
#                    |
#                    |  GAP = 50 mm cavity   <- this is the whole point
#                    |
#     z = -GAP     ---- the IGU (glaze_mat_igu from 01_model.py)
#                                                            exterior, -z
#
# The IGU is the two-pane assembly 01_model.py built with pr.genglaze_json():
# SageGlass SR2.0 outboard, Pilkington Optifloat inboard. genglaze already
# solved the interreflection BETWEEN those two lites and handed us one
# WGMDfunc material, so one surface here stands for both panes. What it could
# not know about is the shade, which is what we are adding.
step("device scene: shade + IGU with a 50 mm cavity")

if not SHADE_SRC.exists():
    raise SystemExit(f"Missing {SHADE_SRC.name} -- it ships with the workshop bundle.")

# ---------------------------------------------------------------------------
# GOTCHA 1: the filename.
# A Radiance BSDF primitive takes the XML path as a bare string argument, and
# the parser splits on whitespace. "E Screen 1% Pearl-Grey.xml" has two spaces
# in it and would be read as three separate arguments. Copy it somewhere sane.
# ---------------------------------------------------------------------------
if not SHADE_XML.exists() or SHADE_XML.stat().st_mtime < SHADE_SRC.stat().st_mtime:
    shutil.copyfile(SHADE_SRC, SHADE_XML)
print(f"     shade  {SHADE_SRC.name!r}  ->  {SHADE_XML.relative_to(HERE)}")

# ---------------------------------------------------------------------------
# GOTCHA 2: the up vector.
#
#     void BSDF shade_mat
#     6 thickness  file.xml  ux uy uz  funcfile
#
# ux uy uz orients the XML's own coordinate system on the surface -- it tells
# Radiance which way is "up" in the matrix. It must NOT be parallel to the
# surface normal, or the local frame is degenerate. Our shade faces +z, so
# `0 0 1` is exactly the wrong answer and `0 1 0` is right.
#
# The failure mode is nasty: Radiance does not stop. It quietly returns zero
# for most of the distribution, the sampling below finishes in a fraction of
# the time it should, and you get a plausible-looking XML that is wrong.
# If block 2 finishes suspiciously fast, this is the first thing to check.
#
# thickness = 0 because the XML already carries the fabric's own 0.52 mm; a
# non-zero thickness here would ask Radiance to treat the surface as a PROXY
# for geometry sitting that far behind it.
# ---------------------------------------------------------------------------
UP = "0 1 0"


def layer(mat: str, name: str, z: float, half: float) -> str:
    """A square, normal facing +z (toward the room), centred on the origin."""
    return (f"\n{mat} polygon {name}\n0\n0\n12\n"
            f"\t{-half}\t{-half}\t{z}\n"
            f"\t{ half}\t{-half}\t{z}\n"
            f"\t{ half}\t{ half}\t{z}\n"
            f"\t{-half}\t{ half}\t{z}\n")


# ---------------------------------------------------------------------------
# GOTCHA 3: make the geometry MUCH bigger than the box you sample.
#
# The sampler fires rays at the device from a plane covering the sampling box.
# At 60 degrees incidence, a ray entering the cavity is displaced sideways by
# GAP * tan(60) = 87 mm before it reaches the far layer. If the layers are the
# same size as the sampling box, every ray entering within 87 mm of the edge
# MISSES the second layer entirely and sails straight through.
#
# With 1 x 1 m layers and a 1 x 1 m box that leaks about a third of the area at
# grazing angles, and it inflated the system transmittance from 1.6% to 9.3%
# when we built this script. Six-metre layers, one-metre box: no leak.
# ---------------------------------------------------------------------------
device = "\n".join([
    f"!xform {model_dir() / 'materials.rad'}",
    "",
    f"void BSDF shade_mat\n6 0 {SHADE_XML} {UP} .\n0\n0",
    layer("shade_mat", "shade", 0.0, DEVICE_HALF),
    layer("glaze_mat_igu", "igu", -GAP, DEVICE_HALF),
])
device_rad = SCRATCH / "device.rad"
device_rad.write_text(device)
print(f"     wrote {device_rad.relative_to(HERE)} "
      f"({2 * DEVICE_HALF:.0f} m layers, sampled over "
      f"{2 * SAMPLE_HALF:.0f} x {2 * SAMPLE_HALF:.0f} m)")

# One more thing worth saying out loud: glaze_mat_igu is a WGMDfunc that refers
# to "igu_t.dat" and "igu_r.dat" by BARE FILENAME. Those live in the project
# root, and wsvenv chdir'd us there on import. Run this from anywhere else and
# Radiance will not find them.


# ===========================================================================
#  2. SAMPLE IT
# ===========================================================================
# pr.generate_bsdf() is genBSDF's algorithm, in Python. It sets up an rfluxmtx
# sender covering one face of the sampling box and two glow-source receivers
# (one hemisphere either side), fires rays, and collates four matrices per
# side: transmission and reflection, front and back.
#
#   basis="kf"   Klems full -- 145 patches per hemisphere, the industry default
#   outspec="y"  weight the result photopically. We only want visible here;
#                a solar-band XML would need a second pass with solar weights.
#   dim=         override the bounding box. You MUST pass this: our layers are
#                flat polygons, so the automatic bbox would be degenerate, and
#                besides we deliberately made the geometry oversized.
#
# "Front" in the LBNL XML convention means the EXTERIOR side, which here is
# -z -- the same convention SamplingBox uses. generate_xml() maps it through.
step(f"sampling the assembly, Klems full basis ({NSAMP} samples)")

DIM = SamplingBox(-SAMPLE_HALF, SAMPLE_HALF, -SAMPLE_HALF, SAMPLE_HALF, -GAP, 0.0)

try:
    t0 = time.time()
    with timed("generate_bsdf (145 x 145, both sides)"):
        res = pr.generate_bsdf(str(device_rad), basis="kf", dim=DIM,
                               nsamp=NSAMP, nproc=NPROC, outspec="y", nspec=3,
                               params=BSDF_PARAMS)
    elapsed = time.time() - t0
    if elapsed < 2.0:
        print("     !! that was suspiciously fast -- check the BSDF up vector")

    # wrapBSDF, wrapped. This is what turns eight matrices into a WINDOW XML.
    # t= is the overall device thickness. wrapBSDF warns if you leave it out,
    # and downstream tools use it to place proxy geometry.
    xml = pr.generate_xml(vis_results=res, basis="kf", unit="meter", t=GAP,
                          n="IGU + E Screen 1% Pearl-Grey",
                          m="pyradiance workshop")
    SYS_XML.write_bytes(xml)
    print(f"     wrote {SYS_XML.relative_to(HERE)}  ({len(xml) // 1024} kB)")
except Exception as e:  # noqa: BLE001
    print(f"     !! sampling failed ({e})")
    SYS_XML.write_bytes(b"")

sys_xml = checkpoint(SYS_XML, "sys_igu_escreen.xml")


# ===========================================================================
#  3. DOES IT MAKE SENSE?
# ===========================================================================
# Never ship a BSDF you have not checked. Radiance's `checkBSDF` prints the
# hemispherical values and, crucially, the RECIPROCITY ERROR: for any passive
# device, front-to-back and back-to-front transmission must agree. If they do
# not, either your device is not passive or your sampling is wrong.
step("checkBSDF -- sanity check the system XML")


def check(path) -> dict[str, float]:
    """Run checkBSDF and pull out the four hemispherical numbers.

    pyradiance does not wrap checkBSDF, but every Radiance binary is in
    pr.BINPATH -- so anything unwrapped is still one subprocess call away.
    That is worth remembering: the wheel ships 91 tools, the Python API
    covers most of them, and this is how you reach the rest.
    """
    out = subprocess.run([str(pr.BINPATH / "checkBSDF"), str(path)],
                         capture_output=True, text=True).stdout
    vals = {}
    for line in out.splitlines():
        for key in ("Interior Refl", "Exterior Refl",
                    "Int->Ext Trans", "Ext->Int Trans"):
            if line.startswith(key):
                vals[key] = float(line.split("%")[0].split()[-1])
    print("   " + "\n   ".join(out.splitlines()[7:]))
    return vals


print("\n   --- the shade on its own, as measured ---")
shade_v = check(SHADE_XML)
print("\n   --- the system we just built ---")
sys_v = check(sys_xml)

# The headline number, at normal incidence. Sampling the IGU on its own the
# same way gives 58.6% Tvis, which is what you would expect from a 60%-T
# electrochromic lite in its clear state behind clear float.
T_IGU = 58.6
t_shade = shade_v.get("Ext->Int Trans", 0.0)
t_sys = sys_v.get("Ext->Int Trans", 0.0)
naive = t_shade * T_IGU / 100
print(f"\n     shade alone (measured)   {t_shade:5.2f} %")
print(f"     IGU alone (genglaze)     {T_IGU:5.2f} %")
print(f"     naive product            {naive:5.2f} %")
print(f"     SIMULATED system         {t_sys:5.2f} %")
print(f"     cavity gains you         {100 * (t_sys / naive - 1):+5.1f} %")
print("     -- small here because the screen is dark and the glass is clear;")
print("        put a bright blind behind a low-e coating and it is worth 10%.")


# ===========================================================================
#  4. THE MATRIX, DRAWN
# ===========================================================================
# A WINDOW XML is not a binary blob -- it is 145 x 145 comma-separated floats
# inside a <ScatteringData> tag, one block per direction. So we can just read
# it. (pr.bsdf2klems() would re-project an arbitrary BSDF onto the Klems basis,
# but it emits another XML, not a matrix, so for a Klems file it is a no-op.)
#
# Column j is the outgoing distribution for light arriving in incident patch j;
# column 0 is normal incidence. The basis is patch 0 = the 0-5 degree cap, then
# rings of 8, 16, 20, 24, 24, 24, 16 and 12 out to the horizon: 145 in all.
step("read the Klems matrix back out of the XML and plot it")

KLEMS_RINGS = [(0, 1, 0.0), (10, 8, 5.0), (20, 16, 15.0), (30, 20, 25.0),
               (40, 24, 35.0), (50, 24, 45.0), (60, 24, 55.0),
               (70, 16, 65.0), (82.5, 12, 75.0)]


def klems_matrix(path, direction: str = "Transmission Front") -> np.ndarray:
    """Pull one 145 x 145 component out of a Klems WINDOW XML."""
    text = Path(path).read_text(errors="replace")
    blocks = re.findall(r"<WavelengthDataBlock>(.*?)</WavelengthDataBlock>",
                        text, re.S)
    for b in blocks:
        if f"<WavelengthDataDirection>{direction}<" not in b:
            continue
        data = re.search(r"<ScatteringData>(.*?)</ScatteringData>", b, re.S)
        # WINDOW writes commas, wrapBSDF writes whitespace. Accept either.
        vals = np.array([float(x) for x in re.split(r"[,\s]+", data.group(1))
                         if x.strip()])
        return vals.reshape(145, 145)
    raise LookupError(f"no {direction!r} block in {path}")


def klems_patches():
    """(theta_centre, phi_centre, theta_lower, theta_upper) for all 145."""
    out = []
    for i, (th, nphi, lo) in enumerate(KLEMS_RINGS):
        hi = KLEMS_RINGS[i + 1][2] if i + 1 < len(KLEMS_RINGS) else 90.0
        for k in range(nphi):
            out.append((th, 360.0 * k / nphi, lo, hi))
    return out


try:
    M = klems_matrix(sys_xml)
    patches = klems_patches()

    fig = plt.figure(figsize=(11, 4.6))

    # (a) the whole matrix
    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(np.log10(np.maximum(M, 1e-6)), cmap="magma",
                   interpolation="nearest")
    ax.set_xlabel("incident patch")
    ax.set_ylabel("outgoing patch")
    ax.set_title("front transmission, 145 x 145\n(log$_{10}$ BSDF)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # The bright diagonal is the COHERENT THROUGH-BEAM: light that goes
    # straight on. Everything off the diagonal is the woven fabric scattering
    # it. That ridge is exactly what aBSDF pulls out and traces as a real ray.
    ax.annotate("the through-beam\n(this is what aBSDF\nextracts)",
                xy=(28, 28), xytext=(58, 105), fontsize=7, color="w",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="w", lw=0.8))

    # (b) the normal-incidence column, on the hemisphere. Log scale, because
    # the peak is three orders of magnitude above the scattered halo.
    ax = fig.add_subplot(1, 2, 2, projection="polar")
    col = M[:, 0]
    lo_v, hi_v = max(col[col > 0].min(), col.max() * 1e-4), col.max()
    norm = plt.matplotlib.colors.LogNorm(vmin=lo_v, vmax=hi_v)
    ring_n = {th: n for th, n, _ in KLEMS_RINGS}
    for (th, phi, lo, hi), v in zip(patches, col):
        ax.bar(np.radians(phi), hi - lo, bottom=lo,
               width=2 * np.pi / ring_n[th],
               color=plt.cm.magma(norm(max(v, lo_v))), edgecolor="none")
    ax.set_ylim(0, 90)
    ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(["30$^\\circ$", "60$^\\circ$", ""], fontsize=7,
                       color="0.7")
    ax.set_xticklabels([])
    ax.set_title("outgoing distribution for light\narriving at normal incidence"
                 "\n(log scale: peak is 1000x the halo)", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax,
                 fraction=0.046, pad=0.08, label="BSDF [1/sr]")

    fig.suptitle("System BSDF: IGU + E Screen 1% Pearl-Grey", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "06_a_klems.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("     wrote out/06_a_klems.png")
except Exception as e:  # noqa: BLE001
    print(f"     !! could not plot the matrix "
          f"({type(e).__name__}: {str(e)[:80]}) -- carrying on")


# ===========================================================================
#  5. PUT IT IN THE SCENE: aBSDF
# ===========================================================================
# Radiance has two primitives for this:
#
#   BSDF    5 or 6 args:  thickness xmlfile ux uy uz funcfile
#           A general scattering surface, optionally standing in as a PROXY
#           for geometry `thickness` behind it.
#
#   aBSDF   5 args:       xmlfile ux uy uz funcfile        (NO thickness)
#           "Aperture BSDF". Same data, but Radiance pulls the COHERENT
#           through-beam out of the matrix and traces it as a real specular
#           ray. That is what you want for anything you can see through: with
#           plain BSDF the sun is smeared across a 5-degree Klems patch and
#           the view out of the window turns to soup; with aBSDF the sun stays
#           a sun and casts a sharp patch on the floor.
#
# Our system has glass in it, so aBSDF it is. Note the up vector rule applies
# here too -- the window faces -y, so `0 0 1` is fine THIS time.
step("aBSDF glazing + octrees")

MATERIALS, GEOMETRY = scene_files()
D = model_dir()

# Reuse the three panes 01_model.py already made -- same geometry, new
# modifier. parse_primitive() reads a .rad file into Primitive objects, so
# swapping a modifier is a one-liner rather than a regex.
panes = [pr.Primitive("sys_escreen", p.ptype, p.identifier, p.sargs, p.fargs)
         for p in pr.parse_primitive((D / "glazing.rad").read_text())]
absdf = (f"void aBSDF sys_escreen\n5 {sys_xml} 0 0 1 .\n0\n0\n\n"
         + "\n".join(p.bytes.decode() for p in panes))
(MODEL / "glazing_bsdf.rad").write_text(absdf + "\n")
print(f"     wrote model/glazing_bsdf.rad")

sky = SCRATCH / "sky_06.rad"
sky.write_bytes(
    pr.gensky(WHEN, **SITE, sunny_with_sun=True) + b"\n"
    + b"\n".join(p.bytes for p in [
        pr.Primitive("skyfunc", "glow", "skyglow", [], [1, 1, 1, 0]),
        pr.Primitive("skyglow", "source", "sky", [], [0, 0, 1, 180]),
        pr.Primitive("skyfunc", "glow", "groundglow", [], [1, 1, 1, 0]),
        pr.Primitive("groundglow", "source", "ground", [], [0, 0, -1, 180]),
    ]) + b"\n")

# Two scenes, differing only in how the shading is represented.
#   blinds : real slat geometry + plain IGU glazing   (what 01-05 use)
#   absdf  : no slats at all, the aBSDF panes carry both shade AND glass
geom_blinds = GEOMETRY
geom_absdf = [g for g in GEOMETRY
              if not g.endswith(("blinds.rad", "glazing.rad"))] \
             + [str(MODEL / "glazing_bsdf.rad")]

octrees = {}
for tag, geom in (("blinds", geom_blinds), ("absdf", geom_absdf)):
    p = SCRATCH / f"office_{tag}.oct"
    with timed(f"oconv [{tag}]"):
        p.write_bytes(pr.oconv(MATERIALS, *geom, str(sky), warning=False))
    octrees[tag] = p


# ===========================================================================
#  6. COMPARE
# ===========================================================================
step("rendering both, same view and same sky")
TITLES = {"blinds": "geometric venetian blinds\n(28 slats, plain IGU)",
          "absdf": "aBSDF system\n(one matrix: screen + IGU)"}

imgs, times = {}, {}
for tag, oct_path in octrees.items():
    t0 = time.time()
    with timed(f"rpict [{tag}]"):
        imgs[tag] = render_view(oct_path, VIEWS / "hero.vf", XRES, YRES,
                                quality="fast")
    times[tag] = time.time() - t0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, tag in zip(axes, ("blinds", "absdf")):
    ax.imshow(wsviz.to_srgb(wsviz.hdr_to_array(imgs[tag]), key=0.22))
    ax.set_axis_off()
    ax.set_title(f"{TITLES[tag]}\n{times[tag]:.1f} s", fontsize=9)
fig.suptitle(f"{WHEN:%d %B %H:%M}, CIE clear sky -- same view, same sky",
             fontsize=11, y=1.06)
fig.text(0.5, 0.99, "each panel is auto-exposed independently: read the lux "
                    "in 06_c, not the brightness here",
         ha="center", fontsize=8, color="0.35")
fig.tight_layout()
fig.savefig(OUT / "06_b_compare.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("     wrote out/06_b_compare.png")

# ---------------------------------------------------------------------------
# The numbers matter more than the picture. A 1%-openness screen is a much
# heavier shade than blinds at 15 degrees, so expect the aBSDF room to be
# considerably darker -- that is the device, not the method.
# ---------------------------------------------------------------------------
step("rtrace -I on the workplane, both ways")
points = (DATA / "points.txt").read_bytes()
lux = {}
for tag, oct_path in octrees.items():
    with timed(f"rtrace [{tag}]"):
        raw = pr.rtrace(points, str(oct_path), irradiance=True, outform="f",
                        outspec="v", nproc=NPROC, header=False,
                        params=GRID_PARAMS)
    lux[tag] = np.frombuffer(raw, dtype=np.single).reshape(-1, 3) @ wsviz.LUM
    v = lux[tag]
    print(f"       {tag:7s} mean {v.mean():6.0f} lux   "
          f"front row {v.reshape(GRID)[0].mean():6.0f}   "
          f"back row {v.reshape(GRID)[-1].mean():5.0f}")

fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1, 1.7]})
# A shared LINEAR scale would render the aBSDF panel solid black -- the two
# rooms are a factor of 25 apart. Share a LOG scale instead: still one colour
# bar, still directly comparable, and you can see both.
vmax = max(np.percentile(v, 99) for v in lux.values())
norm = plt.matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
for ax, tag in zip(axes, ("blinds", "absdf")):
    m = ax.imshow(np.flipud(lux[tag].reshape(GRID)), cmap="inferno", norm=norm)
    ax.set_title(f"{TITLES[tag]}\nmean {lux[tag].mean():.0f} lux", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].set_ylabel("distance from window  ->")
fig.colorbar(m, ax=axes[:2], location="bottom", fraction=0.05, pad=0.02,
             label="illuminance [lux], log scale")

ax = axes[2]
dist = np.linspace(0, 8, GRID[0])
for tag, style in (("blinds", "-o"), ("absdf", "-s")):
    ax.semilogy(dist, lux[tag].reshape(GRID).mean(axis=1), style,
                ms=3.5, lw=1.2, label=TITLES[tag].replace("\n", " "))
ax.axhline(300, color="k", ls=":", lw=0.9)
ax.text(0.1, 320, "300 lux", fontsize=7)
ax.set_xlabel("distance from window [m]")
ax.set_ylabel("row-mean illuminance [lux]")
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
ax.set_title("front-to-back falloff", fontsize=10)

fig.savefig(OUT / "06_c_grid.png", dpi=130)
plt.close(fig)
print("     wrote out/06_c_grid.png")


banner("06 done")
print("""
  What to take away:

    * A system BSDF is SIMULATED, not multiplied. The cavity between the
      shade and the glass is a real optical component.
    * generate_bsdf() needs three things right: a non-degenerate up vector,
      geometry larger than the sampling box, and an explicit `dim`.
    * checkBSDF's reciprocity error is your unit test. Ours is not zero --
      the source measurement is not perfectly reciprocal either -- but the
      hemispherical values agree to within the Monte Carlo noise.
    * Use aBSDF, not BSDF, for anything you can see through.
    * The XML you just made is the input to a three-phase annual run: drop it
      into the T matrix of 04_annual.py and you can swap shading devices
      without re-tracing a single ray.
""")
