"""
03_pointintime.py -- one moment in time, properly.

    python 03_pointintime.py

The canonical Radiance pipeline is three steps:

        sky description  +  geometry   -->  oconv  -->  octree
        octree  +  sensor rays         -->  rtrace -->  numbers
        octree  +  a view              -->  rpict  -->  a picture

Everything else in Radiance -- including the annual methods in 04 -- is a
variation on this. Here we do it once, carefully, and look at what the
parameters actually buy us.

Outputs:
    out/03_a_sky_compare.png    CIE clear sky vs real measured weather
    out/03_b_workplane.png      illuminance on the desk plane, in lux
    out/03_c_hero.png           the good render
    out/03_d_ambbounce.png      why -ab matters
    out/03_e_glare.png          fisheye + evalglare
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pyradiance as pr

import wsviz
from wsvenv import (DATA, GRID_PARAMS, MODEL, NPROC, OUT, SCRATCH, VIEWS,
                    banner, render_view, scene_files, step, timed)

meta = json.loads((DATA / "grid.json").read_text())
GRID = tuple(meta["shape"])
SITE = dict(latitude=40.78, longitude=73.96, timezone=75)   # from the EPW header

WHEN = datetime(2024, 3, 21, 13, 0)   # >>> TODO (you): try 12, 9, 0 (Dec 21) or 6, 21
MATERIALS, GEOMETRY = scene_files()

banner(f"03 -- point in time: {WHEN:%d %B, %H:%M}")


# ===========================================================================
#  1. TWO WAYS TO DESCRIBE A SKY
# ===========================================================================
# gensky  -- a CIE standard sky. An idealised, statistical sky. You give it a
#            date and a sky type; it gives you a smooth luminance distribution.
#            Good for code compliance and for daylight factor.
# gendaylit -- a Perez all-weather sky. You give it MEASURED irradiance, and it
#            reproduces that particular moment. This is what annual work uses.
step("building two skies for the same instant")


def epw_row(path, when):
    """Pull direct-normal and diffuse-horizontal irradiance out of an EPW.

    EPW data lines start at line 9. Fields (0-based): 1=month, 2=day, 3=hour
    (1..24, hour-ending), 14=direct normal W/m2, 15=diffuse horizontal W/m2.
    """
    for line in path.read_text().splitlines()[8:]:
        f = line.split(",")
        if int(f[1]) == when.month and int(f[2]) == when.day and int(f[3]) == when.hour + 1:
            return float(f[14]), float(f[15])
    raise LookupError(f"{when} not found in {path.name}")


dni, dhi = epw_row(DATA / "NewYork_TMY3.epw", WHEN)
print(f"     EPW says: direct normal {dni:.0f} W/m2, diffuse horizontal {dhi:.0f} W/m2")

HEMISPHERE = [
    pr.Primitive("skyfunc", "glow", "skyglow", [], [1, 1, 1, 0]),
    pr.Primitive("skyglow", "source", "sky", [], [0, 0, 1, 180]),
    pr.Primitive("skyfunc", "glow", "groundglow", [], [1, 1, 1, 0]),
    pr.Primitive("groundglow", "source", "ground", [], [0, 0, -1, 180]),
]


def sky_file(name: str, sky_bytes: bytes):
    """gensky/gendaylit give you the sun and a brightness function, but not the
    hemisphere that emits it. Bolt on the glow sources yourself."""
    p = SCRATCH / f"sky_{name}.rad"
    p.write_bytes(sky_bytes + b"\n" + b"\n".join(x.bytes for x in HEMISPHERE) + b"\n")
    return p


skies = {
    "CIE clear (gensky)":
        sky_file("cie", pr.gensky(WHEN, **SITE, sunny_with_sun=True)),
    "Perez measured (gendaylit)":
        sky_file("perez", pr.gendaylit(WHEN, **SITE, dirnorm=dni, diffhor=dhi)),
}

octrees = {}
for label, skyp in skies.items():
    tag = "cie" if "gensky" in label else "perez"
    oct_path = SCRATCH / f"office_{tag}.oct"
    with timed(f"oconv [{tag}]"):
        oct_path.write_bytes(pr.oconv(MATERIALS, *GEOMETRY, str(skyp), warning=False))
    octrees[label] = oct_path


# ===========================================================================
#  2. rtrace: NUMBERS ON THE WORKPLANE
# ===========================================================================
step("rtrace -I on the workplane grid")
points = (DATA / "points.txt").read_bytes()

# -I  (irradiance=True) is the important switch. Without it, rtrace returns the
#     RADIANCE seen looking along the ray. With it, you get the IRRADIANCE
#     arriving on a surface whose normal is the ray direction -- which, times
#     the photometric weights, is illuminance in lux.
RT_PARAMS = GRID_PARAMS

results = {}
for label, oct_path in octrees.items():
    with timed(f"rtrace [{label}]"):
        raw = pr.rtrace(points, str(oct_path), irradiance=True, outform="f",
                        outspec="v", nproc=NPROC, header=False, params=RT_PARAMS)
    irr = np.frombuffer(raw, dtype=np.single).reshape(-1, 3)
    results[label] = irr @ wsviz.LUM        # W/m2 -> lux
    lux = results[label]
    print(f"       {label:28s} mean {lux.mean():6.0f} lux, "
          f"front row {lux.reshape(GRID)[0].mean():6.0f}, "
          f"back row {lux.reshape(GRID)[-1].mean():5.0f}")

# The two skies disagree, and that is the point: a CIE clear sky is a idealised
# construct, while the Perez sky reproduces the weather that was actually
# recorded. Annual simulation always uses the latter.
fig_vmax = max(np.percentile(v, 97) for v in results.values())
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(8.5, 8))
for ax, (label, lux) in zip(axes, results.items()):
    m = ax.imshow(np.flipud(lux.reshape(GRID)), cmap="inferno", vmin=0, vmax=fig_vmax)
    ax.set_title(f"{label}\nmean {lux.mean():.0f} lux", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel("distance from window  ->")
fig.colorbar(m, ax=axes, fraction=0.046, pad=0.04, label="illuminance [lux]")
fig.savefig(OUT / "03_a_sky_compare.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"     wrote out/03_a_sky_compare.png")

# From here on, use the physically real sky.
OCT = octrees["Perez measured (gendaylit)"]
lux = results["Perez measured (gendaylit)"]

wsviz.save_grid(lux, GRID, "03_b_workplane",
                f"Workplane illuminance, {WHEN:%d %b %H:%M}\n(Perez sky from measured EPW data)",
                annotate=True, cmap='turbo')

# A couple of numbers people actually ask for.
print(f"     min {lux.min():.0f} / mean {lux.mean():.0f} / max {lux.max():.0f} lux")
print(f"     uniformity (min/mean) = {lux.min() / lux.mean():.2f}")
print(f"     {100 * (lux >= 300).mean():.0f}% of the workplane is above 300 lux")


# ===========================================================================
#  3. rpict: THE PICTURE
# ===========================================================================
step("rpict, good quality (this is the slow one)")
with timed("rpict hero"):
    hero = render_view(OCT, VIEWS / "hero.vf", 640, 440, quality="good")
(MODEL / "hero.hdr").write_bytes(hero)
wsviz.save_srgb(hero, "03_c_hero", f"{WHEN:%d %B, %H:%M} -- Perez sky")


# ===========================================================================
#  4. WHAT -ab BUYS YOU
# ===========================================================================
# -ab is the number of diffuse interreflections. With -ab 0 the only light in
# the room is what arrives directly from the sky and sun: the ceiling is black
# and the back of the room is unlit. Each bounce costs time and adds realism.
# Look at the terracotta accent wall -- with enough bounces it tints the whole
# back of the room. That is colour bleed, and it is real.
step("-ab sweep on the back view")
panels = {}
for ab in (0, 1, 3):
    with timed(f"rpict -ab {ab}"):
        panels[ab] = render_view(OCT, VIEWS / "back.vf", 380, 260, quality="fast",
                                 extra=["-ab", str(ab)])

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (ab, img) in zip(axes, panels.items()):
    ax.imshow(wsviz.to_srgb(wsviz.hdr_to_array(img), key=0.28))
    ax.set_title(f"-ab {ab}", fontsize=11)
    ax.set_axis_off()
fig.suptitle("Ambient bounces: the back wall is lit only by interreflection", fontsize=12)
fig.savefig(OUT / "03_d_ambbounce.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("     wrote out/03_d_ambbounce.png")


# ===========================================================================
#  5. GLARE FROM THE DESK
# ===========================================================================
# A 180-degree fisheye from the eye position is what glare metrics are defined
# on. evalglare reads that image and reports DGP (Daylight Glare Probability).
step("fisheye + evalglare")
with timed("rpict fisheye"):
    fish = render_view(OCT, VIEWS / "desk.vf", 420, 420, quality="fast")
fish_path = SCRATCH / "desk_fisheye.hdr"
fish_path.write_bytes(fish)

wsviz.save_srgb(fish, "03_e_glare", "180-degree fisheye from the seated eye position")

try:
    report = pr.evalglare(str(fish_path), detailed=False).decode().strip()
    print(f"     evalglare: {report.splitlines()[-1][:110]}")
    dgp = float(report.split()[1])
    verdict = ("imperceptible" if dgp < 0.35 else "perceptible" if dgp < 0.40
               else "disturbing" if dgp < 0.45 else "intolerable")
    print(f"     DGP = {dgp:.3f}  ->  {verdict} glare")
except Exception as e:  # noqa: BLE001
    print(f"     evalglare did not run cleanly ({e}); the fisheye PNG is still there")

banner("done")
print("  Compare 03_a (two skies), then 03_d (-ab 0/1/4).")
print("  >>> TODO: change WHEN at the top and re-run. Try December.")
print("  Next: 04_annual.py")
