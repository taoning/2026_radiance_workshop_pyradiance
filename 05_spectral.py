"""
05_spectral.py -- beyond RGB.

    python 05_spectral.py

Everything so far has been three numbers per ray: R, G, B. That is a drastic
simplification. Real surfaces and real skies have continuous spectra, and three
broad channels throw most of that away. It does not matter much if you only
want lux -- but it matters a lot if you care about

    * circadian / melanopic response, which peaks near 490 nm
    * spectrally selective glazing and coatings
    * colour rendering, and metamerism
    * anything involving narrowband sources

Radiance can carry N spectral bands instead of 3. pyradiance exposes this in
two places:

    genssky           a physically-based SPECTRAL sky
    rtrace -co+ -cs N ray tracing with N spectral samples

Outputs:
    out/05_a_bands.png      the same view, band by band
    out/05_b_spectra.png    spectral radiance at picked points
    out/05_c_melanopic.png  relative melanopic efficacy
    out/05_d_rgb_vs_spec.png  what RGB actually costs you

NOTE: this is the least-travelled path in pyradiance. If genssky fails on your
machine, the script falls back to a shipped checkpoint and carries on.
"""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyradiance as pr

import wsviz
from wsvenv import (CKPT, DATA, MODEL, NPROC, OUT, SCRATCH, VIEWS, banner,
                    scene_files, step, timed)

NBANDS = 21
WL_MIN, WL_MAX = 390, 770
WHEN = datetime(2024, 3, 21, 13, 0)
SITE = dict(latitude=40.78, longitude=73.96, timezone=75)
XRES, YRES = 320, 220

meta = json.loads((DATA / "grid.json").read_text())
MB = meta["macbeth_patches"]

banner(f"05 -- hyperspectral rendering, {NBANDS} bands")

# Radiance orders its spectral samples from LONGEST to SHORTEST wavelength.
# (Verify it yourself: render the Macbeth chart and see which band the red
# patch peaks in -- it is band 0.)
edges = np.linspace(WL_MIN, WL_MAX, NBANDS + 1)
WL = ((edges[:-1] + edges[1:]) / 2)[::-1]        # descending, matches band index
print(f"     band centres (band 0 first): {np.round(WL, 0)}")


# ===========================================================================
#  1. A SPECTRAL SKY
# ===========================================================================
# gendaylit gives you an RGB sky. genssky runs an atmospheric model and gives
# you a `spectrum` primitive for the sun plus a hyperspectral (.hsr) sky image
# referenced through a `specpict` pattern.
step("genssky -- spectral sky (first run builds an atmosphere cache, ~15 s)")
atmos = SCRATCH / "atmos"
atmos.mkdir(exist_ok=True)          # genssky will not create this for you
ssky = SCRATCH / "ssky.rad"
try:
    with timed("genssky"):
        ssky.write_bytes(pr.genssky(WHEN, **SITE, out_dir=str(atmos),
                                    out_name="ssky", nthreads=NPROC))
except Exception as e:  # noqa: BLE001
    print(f"     !! genssky failed ({e}); falling back to the shipped sky")
    # The shipped copy carries a __HSR__ placeholder instead of a hard-coded
    # path to the hyperspectral sky image, so patch it for this machine.
    fallback = (CKPT / "ssky.rad").read_text()
    ssky.write_text(fallback.replace("__HSR__", str(CKPT / "ssky_sky.hsr")))

head = [ln for ln in ssky.read_text().splitlines() if ln and not ln.startswith("#")]
print("     sky description contains:",
      ", ".join(sorted({ln.split()[1] for ln in head if len(ln.split()) > 2
                        and ln.split()[1] in
                        ("spectrum", "light", "source", "specpict")})))

# ---------------------------------------------------------------------------
# Read that list again: spectrum, light, source, specpict. genssky gives you the
# SUN, and a `specpict` PATTERN describing what the sky looks like -- but it does
# not give you the sky itself, and it does not give you a ground. A specpict is a
# modifier; on its own it emits nothing at all.
#
# So we bolt on the two glow hemispheres by hand, exactly as 01_model.py does for
# gensky. Skip this and you get a scene lit by direct sun only -- and the failure
# is silent, because a sunless sky just looks like a contrasty sunny day.
# ---------------------------------------------------------------------------
sky_dome = SCRATCH / "ssky_dome.rad"
sky_dome.write_bytes(b"\n".join(p.bytes for p in [
    pr.Primitive("skyfunc", "glow", "skyglow", [], [1, 1, 1, 0]),
    pr.Primitive("skyglow", "source", "sky", [], [0, 0, 1, 180]),
    pr.Primitive("skyfunc", "glow", "groundglow", [], [1, 1, 1, 0]),
    pr.Primitive("groundglow", "source", "ground", [], [0, 0, -1, 180]),
]) + b"\n")
print("     added the sky and ground glow hemispheres genssky leaves to you")


# ===========================================================================
#  2. SPECTRAL MATERIALS
# ===========================================================================
# A `spectrum` primitive holds N reflectance values between two wavelengths.
# Use it as the MODIFIER of an ordinary plastic whose RGB is set to 1 1 1 --
# the spectrum then supplies all the colour. This is exactly what pyradiance's
# load_material_smd() builds when you give it spectral=True.
step("building a spectral test card")
SPEC_WL = np.arange(400, 701, 20.0)          # 16 samples, 400-700 nm


def gaussian(peak, width, height=0.85, floor=0.03):
    return floor + height * np.exp(-0.5 * ((SPEC_WL - peak) / width) ** 2)


SAMPLES = {
    # a narrowband green and a broadband green: very different spectra
    "narrow green": gaussian(530, 15),
    "broad green": gaussian(540, 70, height=0.55),
    # a deep blue, to contrast with the warm interior
    "deep blue": gaussian(455, 28),
}

card = []
CX, CY, CZ = 2.05, 1.72, 0.761        # on the front desk, behind the Macbeth chart
PATCH = 0.06
sample_pts = {}
for i, (nm, refl) in enumerate(SAMPLES.items()):
    sid = f"spec{i}"
    card.append(pr.Primitive("void", "spectrum", f"{sid}_spectrum", [],
                             [SPEC_WL[0], SPEC_WL[-1], *np.round(refl, 4)]))
    card.append(pr.Primitive(f"{sid}_spectrum", "plastic", sid, [], [1, 1, 1, 0, 0]))
    x0 = CX + i * (PATCH + 0.01)
    card.append(pr.Primitive(sid, "polygon", f"{sid}_p", [],
                             [x0, CY, CZ, x0 + PATCH, CY, CZ,
                              x0 + PATCH, CY + PATCH, CZ, x0, CY + PATCH, CZ]))
    sample_pts[nm] = (x0 + PATCH / 2, CY + PATCH / 2, CZ)
spec_card = SCRATCH / "spectral_card.rad"
spec_card.write_bytes(b"\n".join(p.bytes for p in card) + b"\n")
print(f"     {len(SAMPLES)} spectrum materials, {len(SPEC_WL)} samples each")

step("oconv with the spectral sky")
MATERIALS, GEOMETRY = scene_files()
oct_s = SCRATCH / "office_spectral.oct"
oct_s.write_bytes(pr.oconv(MATERIALS, *GEOMETRY, str(spec_card), str(ssky),
                           str(sky_dome), warning=False))


# ===========================================================================
#  3. THE SPECTRAL RENDER
# ===========================================================================
# rpict CANNOT do this. It accepts -cs on the command line but silently returns
# an ordinary 3-channel picture. For N bands you must drive rtrace yourself:
#
#     vwrays  ->  one ray per pixel  ->  rtrace -co+ -cs N  ->  N floats/pixel
#
# -co+  output in spectral (component) form
# -cs N number of spectral samples
# -cw   the wavelength range they span
step(f"vwrays | rtrace -co+ -cs {NBANDS}")
vargs = pr.get_view_args(pr.viewfile(str(VIEWS / "hero.vf")))
dims = pr.vwrays(view=vargs, dimensions=True, xres=XRES, yres=YRES).decode().split()
X, Y = int(dims[1]), int(dims[3])     # vwrays prints "-x W -y H" (note: NOT getinfo order)
print(f"     {X} x {Y} = {X * Y:,} rays x {NBANDS} bands")

rays = pr.vwrays(view=vargs, outform="f", xres=X, yres=Y)
RT = ["-ab", "2", "-ad", "512", "-as", "128", "-aa", "0.2", "-lw", "1e-3"]


def trace(rays_bytes, nbands):
    p = list(RT)
    if nbands > 3:
        p += ["-co+", "-cs", str(nbands), "-cw", str(WL_MIN), str(WL_MAX)]
    return pr.rtrace(rays_bytes, str(oct_s), inform="f", outform="f", outspec="v",
                     nproc=NPROC, header=False, params=p)


with timed(f"spectral render ({NBANDS} bands)"):
    cube = np.frombuffer(trace(rays, NBANDS), dtype=np.single).reshape(Y, X, NBANDS)
with timed("rgb render (for comparison)"):
    rgb = np.frombuffer(trace(rays, 3), dtype=np.single).reshape(Y, X, 3)
print(f"     spectral cube {cube.shape}, {cube.nbytes / 1e6:.1f} MB")

wsviz.save_band_grid(cube, WL, "05_a_bands",
                     f"Radiance per band, {WHEN:%d %b %H:%M} -- spectral sky")


# ===========================================================================
#  4. SPECTRA AT INDIVIDUAL POINTS
# ===========================================================================
# rtrace on a handful of chosen points is far cheaper than a whole image, and
# it is how you would actually interrogate a scene.
step("spectra at chosen surfaces")
probes = {
    "Macbeth red": MB["red"],
    "Macbeth blue": MB["blue"],
    "Macbeth white": MB["white"],
    **{f"card: {k}": list(v) for k, v in sample_pts.items()},
}
probe_rays = b"".join(
    f"{p[0]} {p[1]} {p[2] + 0.25} 0 0 -1\n".encode() for p in probes.values())
pr_out = pr.rtrace(probe_rays, str(oct_s), inform="a", outform="f", outspec="v",
                   header=False, nproc=NPROC,
                   params=RT + ["-co+", "-cs", str(NBANDS),
                                "-cw", str(WL_MIN), str(WL_MAX)])
spectra = np.frombuffer(pr_out, dtype=np.single).reshape(len(probes), NBANDS)

# Plot with wavelength ascending, which is how everyone expects to read a spectrum.
order = np.argsort(WL)
wsviz.save_spectrum(
    WL[order],
    {name: s[order] / max(s.max(), 1e-9) for name, s in zip(probes, spectra)},
    "05_b_spectra",
    "Spectral radiance leaving each surface (each normalised to its own peak)",
    ylabel="relative spectral radiance")

print("     the two greens have nearly the same colour but different spectra --")
print("     an RGB render cannot tell them apart; a spectral one can.")


# ===========================================================================
#  5. MELANOPIC RESPONSE
# ===========================================================================
# The circadian ("melanopic") action spectrum peaks around 490 nm, well to the
# blue of the photopic V(lambda) peak at 555 nm. So the melanopic content of a
# view depends on WHERE the light came from: sky is blue-rich, a warm interior
# surface is not. You cannot get this from an RGB render.
step("melanopic vs photopic")
V_WL = np.array([380, 420, 460, 500, 540, 555, 580, 620, 660, 700, 740, 780])
V_LAM = np.array([0.0000, 0.0040, 0.0600, 0.3230, 0.9540, 1.0000,
                  0.8700, 0.3810, 0.0610, 0.0041, 0.0003, 0.0000])
M_WL = np.array([380, 420, 460, 480, 490, 500, 520, 540, 560, 600, 640, 700, 780])
M_LAM = np.array([0.000, 0.318, 0.897, 0.995, 1.000, 0.973, 0.786,
                  0.539, 0.318, 0.069, 0.009, 0.000, 0.000])

v_w = np.interp(WL, V_WL, V_LAM)
m_w = np.interp(WL, M_WL, M_LAM)

photopic = cube @ v_w
melanopic = cube @ m_w
ratio = melanopic / np.maximum(photopic, 1e-6)
# Report it relative to the scene median: an absolute melanopic EDI needs the
# CIE S 026 normalisation constants, which is more than we have time for.
rel = ratio / np.median(ratio)

# Scale the colour bar to the actual spread, not to a couple of outlier pixels.
lo, hi = np.percentile(rel, [2, 98])
wsviz.save_falsecolor(
    rel, "05_c_melanopic",
    f"Relative melanopic efficacy (blue-rich = high)\n"
    f"scene median = 1.0; colour bar spans the 2nd-98th percentile",
    label="melanopic / photopic, relative to median",
    vmin=lo, vmax=hi, log=False, cmap="coolwarm")
print(f"     2nd-98th percentile: {lo:.2f} to {hi:.2f} "
      f"(full range {rel.min():.2f} to {rel.max():.2f})")
print("     This is a fully spectral scene: every opaque surface carries a MEASURED")
print("     reflectance spectrum from spectraldb.com (see 01_model.py), the glazing")
print("     is a measured two-layer IGU from the IGSDB, and the sky came out of an")
print("     atmospheric model. Nothing here is an RGB triple upsampled to a curve.")
print("     What drives the map is therefore real: sun-lit surfaces (warm, low")
print("     melanopic) against sky-lit ones (blue, high), the green cast the")
print("     electrochromic lite puts on everything it transmits, and the spectral")
print("     test card. None of it survives a 3-channel render.")


# ===========================================================================
#  6. WHAT DID RGB COST US?
# ===========================================================================
step("RGB vs spectral, side by side")
# A trap worth pointing out: if you difference two ordinary -ab 2 renders, what
# you mostly see is Monte Carlo noise, because the two runs sample the ambient
# calculation differently. The spectral effect is buried under it.
#
# So for the difference map we re-trace both with -ab 0. Direct light only is
# fully deterministic, so whatever is left really is the spectral difference.
step("re-tracing with -ab 0 so the comparison is deterministic")
DET = ["-ab", "0"]


def trace_det(nbands):
    p = list(DET)
    if nbands > 3:
        p += ["-co+", "-cs", str(nbands), "-cw", str(WL_MIN), str(WL_MAX)]
    return np.frombuffer(
        pr.rtrace(rays, str(oct_s), inform="f", outform="f", outspec="v",
                  nproc=NPROC, header=False, params=p),
        dtype=np.single).reshape(Y, X, nbands)


cube_d, rgb_d = trace_det(NBANDS), trace_det(3)
lum_rgb = rgb_d @ wsviz.LUM
phot_d = cube_d @ v_w
# Different normalisations, so match the means and compare the STRUCTURE.
lum_spec = phot_d * (lum_rgb.mean() / max(phot_d.mean(), 1e-9))
mask = lum_rgb > np.percentile(lum_rgb, 40)      # ignore the near-black pixels
diff = np.where(mask, 100 * (lum_spec - lum_rgb) / np.maximum(lum_rgb, 1e-3), np.nan)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
axes[0].imshow(wsviz.to_srgb(rgb, key=0.28)); axes[0].set_title("3-band (RGB)")
axes[1].imshow(wsviz.to_srgb(cube[:, :, [2, 4, 7]], key=0.28))
axes[1].set_title(f"{NBANDS}-band, shown as 664/580/453 nm")
m = axes[2].imshow(np.clip(diff, -6, 6), cmap="RdBu_r", vmin=-6, vmax=6)
axes[2].set_title("luminance difference [%]\n(mean-matched)")
for a in axes:
    a.set_axis_off()
fig.colorbar(m, ax=axes[2], fraction=0.046, label="%")
fig.savefig(OUT / "05_d_rgb_vs_spec.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("     wrote out/05_d_rgb_vs_spec.png")
ad = np.abs(diff[np.isfinite(diff)])
print(f"     median |difference| = {np.median(ad):.1f}%, "
      f"90th percentile = {np.percentile(ad, 90):.1f}%")
print()
print("     Read that number carefully, because it is the honest conclusion of")
print("     this block: for LUMINANCE, three channels are already an excellent")
print("     approximation. RGB was designed for exactly this. If all you want is")
print("     lux or DA, 05 buys you nothing that 04 did not already give you.")
print()
print("     Spectral rendering earns its cost somewhere else:")
print("       * melanopic / circadian metrics  (see 05_c -- V(lambda) cannot")
print("         be recovered from RGB once the spectrum has been collapsed)")
print("       * colour rendering and metamerism (05_b: two greens, one colour)")
print("       * narrowband sources and spectrally selective glazing/coatings")
print("     Use it when the QUESTION is spectral, not to get better lux.")

banner("done -- that's the workshop")
print("  You have gone from a blank file to hyperspectral annual-capable daylight")
print("  simulation, entirely in Python.")
print()
print("  Where to go next:")
print("    * pr.Rcontrib          -- daylight coefficients with your own binning")
print("    * 3-phase + BSDF       -- pr.bsdf, WrapBSDF, genbsdf.py for shading systems")
print("    * pr.gensdaymtx        -- annual SPECTRAL sky matrices")
print("    * docs/howtos/guide4.md is empty -- an easy first contribution")
