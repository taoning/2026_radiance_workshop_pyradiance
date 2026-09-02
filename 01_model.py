"""
01_model.py -- build the workshop office, in Python.

    python 01_model.py

We are going to build a south-facing single-sided office, 3.6 m wide x 8 m deep
x 3.0 m high, entirely from pyradiance calls -- no hand-written .rad files.

Coordinate system (metres):

        z (up)
        |
        |          y = 8.0  ......  back wall (accent colour)
        |         /
        |        /
        +-------/------- x = 3.6
       (0,0,0)
        y = 0.0 .......  window wall, faces SOUTH

The room is deliberately deep. Daylight falls off roughly with the square of
the distance from the window, so an 8 m plan gives a strong gradient -- which
is what makes the hero render, the workplane map, and the annual heatmap in
the later scripts all tell the same story.

Three pyradiance ideas are on display here:

  1. Primitive  -- a Radiance primitive as a Python dataclass.
  2. gen*       -- genbox / genrev / genblinds return geometry as BYTES.
  3. Xform      -- a fluent wrapper over `xform` for placing that geometry.

Almost every pyradiance function accepts bytes-or-path and returns bytes.
Once you have internalised that, the whole library follows.

>>> TODO (you): after your first run, come back and change BLIND_ANGLE,
>>> a material id, or WINDOW_HEAD, then re-run. Everything downstream picks it up.

A note on materials
-------------------
Every opaque surface in this room is SPECTRAL. Instead of inventing three RGB
numbers per surface, we pull measured reflectance spectra from the Spectral
Materials Database (spectraldb.com, SUTD / U. Toronto) and let pyradiance's
load_material_smd() turn each one into a `spectrum` + `plastic` pair.

That does NOT make the rest of the workshop slower or more complicated: 02, 03
and 04 still render in ordinary RGB, and Radiance collapses the spectra to three
channels on the fly. It only starts to matter in 05_spectral.py, where we ask for
21 bands and the spectra are actually carried through.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pyradiance as pr

import wsviz
from wsvenv import DATA, MODEL, NPROC, SCRATCH, VIEWS, banner, render_view, step, timed

# ===========================================================================
#  PARAMETERS -- the knobs worth playing with
# ===========================================================================
ROOM_W, ROOM_D, ROOM_H = 3.6, 8.0, 3.0     # width (x), depth (y), height (z)

WINDOW_SILL = 0.4                          # >>> TODO (you): try 0.4 for a floor-to-ceiling window
WINDOW_HEAD = 2.45
WINDOW_INSET = 0.35                         # gap from the side walls
N_PANES = 3                                 # mullions divide the window into this many panes

WALL_T = 0.30                               # >>> TODO (you): south wall thickness -> depth of the reveal

BLIND_ANGLE = 15.0                          # >>> TODO (you): 0 = horizontal, 80 = nearly closed
BLIND_SLATS = 28
BLIND_DEPTH = 0.05

# --- the insulated glazing unit ------------------------------------------
# Two real measured products, downloaded as records from igsdb.lbl.gov and
# listed OUTBOARD -> INBOARD:
#
#   1. SageGlass SR2.0, 7.0 mm laminate  -- an ELECTROCHROMIC lite, measured
#      here in its fully-clear 60 %T state, coating on its back face so it
#      lands on surface 2. 624 measured wavelengths.
#   2. Pilkington Optifloat Clear, 9.4 mm monolithic float, uncoated.
#      369 measured wavelengths.
#
# >>> TODO (you): the same SageGlass product exists in several tint states.
# >>> Download a darker one, drop the filename in below, and re-run --
# >>> 03_pointintime.py and 05_spectral.py both pick it up.
GLAZING_JSONS = ["igsdb_product_7406.json", "igsdb_product_424.json"]
GLAZING_JSON_DIR = DATA                     # where those records live on disk
GLAZING_PREFIX = "igu"
GLAZING_MAT = f"glaze_mat_{GLAZING_PREFIX}"
GLAZING_WL = (380, 780, 5)                  # nm: start, end, interval

WORKPLANE_Z = 0.80                          # sensor height for every later script
GRID_NX, GRID_NY = 6, 15                    # 90 sensors

SITE = dict(latitude=40.7, longitude=74.0, timezone=75)  # New York City
WHEN = datetime(2024, 3, 21, 13)            # the preview sky: equinox, 13:00

banner("01 -- building the office model")


# ===========================================================================
#  Small helpers
# ===========================================================================
def poly(mod: str, name: str, *pts) -> pr.Primitive:
    """A polygon from (x, y, z) tuples. Vertex order sets the surface normal
    by the right-hand rule -- counter-clockwise seen from the front."""
    return pr.Primitive(mod, "polygon", name, [], [c for p in pts for c in p])


def plastic(name, rgb, spec=0.0, rough=0.0) -> pr.Primitive:
    """Radiance's workhorse opaque material: diffuse + optional specular lobe."""
    return pr.Primitive("void", "plastic", name, [], [*rgb, spec, rough])


def write(path, *chunks) -> "object":
    """Concatenate Primitives and raw bytes into one .rad file."""
    blob = b"\n".join(
        c.bytes if isinstance(c, pr.Primitive) else (c if isinstance(c, bytes) else str(c).encode())
        for c in chunks
    )
    path.write_bytes(blob + b"\n")
    print(f"     wrote {path.relative_to(MODEL.parent)}  ({len(blob) // 1024 + 1} kB)")
    return path


def box(mat, name, sx, sy, sz, at=(0, 0, 0), bevel=None) -> bytes:
    """genbox + xform: make a box of a given size and drop it at a position."""
    b = pr.genbox(mat, name, sx, sy, sz, beveled=bevel)
    return pr.Xform(b).translate(*at)()


# ===========================================================================
#  1. MATERIALS -- measured spectra from spectraldb.com
# ===========================================================================
# https://www.spectraldb.com is a free library of ~1300 measured surface
# reflectance spectra: real ceilings, floors, blinds, brick, asphalt, foliage.
# Every entry has a landing page like
#
#     https://www.spectraldb.com/measurements/00734/
#
# which shows you the spectrum, the L*a*b*, a ready-made Radiance definition to
# copy, and -- the part we care about -- a "Download Data" link to
#
#     https://www.spectraldb.com/measurements/00734/spectral.csv
#
# That CSV is three columns, `wl,sci,sce`: wavelength, specular-INCLUDED and
# specular-EXCLUDED reflectance in percent. And that is EXACTLY the format that
#
#     pr.load_material_smd(path, roughness=..., spectral=True)
#
# eats. It hands back two primitives:
#
#     void spectrum <stem>_spectrum        the measured SCE curve, 0..1
#     <stem>_spectrum plastic <stem>       1 1 1, so the spectrum is the colour
#
# and it derives the specular fraction for you as Y(SCI) - Y(SCE) -- the light
# the sphere caught only when the specular port was closed. So there is no need
# to copy anything by hand, and no invented RGB triples anywhere in this model.
#
# The material is named after the CSV's FILENAME STEM, which is why we cache
# each download as <slug>.csv: the slug becomes the Radiance modifier, and the
# geometry below refers to it through MAT[...].
#
# What the CSV does NOT carry is roughness, which the web page reports
# separately -- so we keep it in the table.
#
# >>> TODO (you): go browsing on spectraldb.com, find something you like, and
# >>> drop its 5-digit id into the table. Delete the cached CSV to re-fetch.
# ---------------------------------------------------------------------------
SDB_URL = "https://www.spectraldb.com/measurements/{id}/spectral.csv"
SDB_PAGE = "https://www.spectraldb.com/measurements/{id}/"
SDB_CACHE = DATA / "spectraldb"
SDB_CACHE.mkdir(exist_ok=True)

#     slot        id       slug (= Radiance modifier)          rough  fallback RGB
SDB = {
    "ceiling":  ("01168", "white_ceiling",                     0.30, (0.86, 0.86, 0.84)),
    "wall":     ("00734", "stone_ceramic_tile_wall",           0.20, (0.79, 0.78, 0.75)),
    "accent":   ("00733", "red_painted_wall",                  0.20, (0.44, 0.17, 0.12)),
    "floor":    ("00006", "dark_grey_floor_tiles_nonslip",     0.30, (0.24, 0.23, 0.22)),
    "desk":     ("00088", "wooden_textured_table_top",         0.15, (0.42, 0.33, 0.23)),
    "frame":    ("00097", "window_mullion",                    0.10, (0.20, 0.20, 0.20)),
    "blind":    ("00534", "opaque_roller_blind",               0.20, (0.81, 0.81, 0.81)),
    "chair":    ("01191", "fabric_chair",                      0.20, (0.05, 0.05, 0.05)),
    "screen":   ("01256", "black_monitor_plastic",             0.05, (0.05, 0.05, 0.05)),
    "ground":   ("00643", "road_asphalt",                      0.20, (0.10, 0.09, 0.09)),
    "context":  ("00083", "aluminium_grey_exterior_cladding",  0.15, (0.47, 0.47, 0.46)),
}
# Convenience: MAT["wall"] -> "stone_ceramic_tile_wall". Every piece of geometry
# below goes through this, so re-skinning the room is a one-line edit.
MAT = {slot: slug for slot, (_, slug, _, _) in SDB.items()}
MAT["chartbase"] = "m_chartbase"      # a prop, not a real measured surface


def fetch_spectral(mid: str, slug: str) -> Path | None:
    """Return the cached spectraldb CSV for `mid`, downloading it if need be.

    Returns None if we could not get it -- the caller then falls back to RGB, so
    a workshop on bad wifi still runs. The cache lives in data/spectraldb/ and is
    committed with the workshop, so in practice this never touches the network.
    """
    dst = SDB_CACHE / f"{slug}.csv"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = SDB_URL.format(id=mid)
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            body = r.read().decode()
    except (urllib.error.URLError, OSError, UnicodeDecodeError) as e:  # noqa: BLE001
        print(f"     !! could not download {mid} ({e}); falling back to RGB")
        print(f"        you can fetch it by hand from {SDB_PAGE.format(id=mid)}")
        print(f"        and save it as {dst.relative_to(MODEL.parent)}")
        return None
    if not body.lower().startswith("wl,sci,sce"):
        print(f"     !! {url} did not look like a spectraldb CSV; falling back to RGB")
        return None
    dst.write_text(body)
    print(f"     downloaded {mid} -> {dst.relative_to(MODEL.parent)}")
    return dst


step("spectral materials from spectraldb.com")
materials = []
for slot, (mid, slug, rough, rgb) in SDB.items():
    csv = fetch_spectral(mid, slug)
    if csv is None:
        materials.append(plastic(slug, rgb, rough=rough))
        continue
    prims = pr.load_material_smd(csv, roughness=rough, spectral=True)
    materials.extend(prims)
    # The plastic is always the last primitive; its 4th real arg is the
    # specular fraction that load_material_smd worked out from SCI - SCE.
    nwl = (len(prims[0].fargs) - 2) if len(prims) > 1 else 0
    print(f"     {slot:9s} {slug:34s} {nwl:2d} wavelengths, "
          f"spec {prims[-1].fargs[3]:.3f}, rough {rough:.2f}")

# The Macbeth chart's backing board is a prop, not a measured surface, so it
# stays an ordinary RGB plastic -- as do the 24 patches themselves (section 6).
materials.append(plastic("m_chartbase", (0.05, 0.05, 0.05)))

# ---------------------------------------------------------------------------
# The glazing gets its own treatment. Instead of `void glass` with three made-up
# RGB numbers, we run genglaze over two MEASURED products from the IGSDB and let
# it solve the multi-layer interreflection between them.
#
# genglaze_json() reads the IGSDB records straight off disk -- no unpacking on
# our side. Layer order is the order of the list: outboard first.
#
# genglaze emits three primitives plus two side-car data files:
#
#   specdata refl_spec_igu   -> igu_r.dat    reflectance
#   specdata trans_spec_igu  -> igu_t.dat    transmittance
#   WGMDfunc glaze_mat_igu                   ties them together
#
# Each .dat is 2-dimensional: transmittance as a function of WAVELENGTH and
# INCIDENCE ANGLE. That second axis is why the reveal darkens at grazing angles
# for free, and the first axis is what gives 05_spectral.py's `rtrace -co+ -cs N`
# something real to integrate instead of a flat RGB filter.
#
# Caveat worth knowing: the material refers to "igu_t.dat" by bare filename, so
# the .dat files must sit somewhere Radiance can find them -- CWD or RAYPATH.
# We write them to the project root, which is where every later script runs.
# ---------------------------------------------------------------------------
step("glazing from measured IGSDB data")
glaze = pr.genglaze_json(
    [str(GLAZING_JSON_DIR / j) for j in GLAZING_JSONS],
    prefix=GLAZING_PREFIX,
    wavelength_start=GLAZING_WL[0],
    wavelength_end=GLAZING_WL[1],
    wavelength_interval=GLAZING_WL[2],
)
print(f"     -> {GLAZING_MAT}, {GLAZING_PREFIX}_t.dat, {GLAZING_PREFIX}_r.dat")

# ---------------------------------------------------------------------------
#  Your own materials: model/materials_custom.rad
# ---------------------------------------------------------------------------
# 01_model.py OVERWRITES model/materials.rad every time it runs, which makes it
# a bad place to keep anything you typed yourself. So there is a second file
# that this script creates once and then never touches again, and whose contents
# are appended verbatim to materials.rad on every run.
#
# That is the home for the copy-paste route: every spectraldb page prints a
# ready-made "Radiance 6 Spectral Definition" block, and pasting it here is the
# fastest way to try a material out before committing it to the SDB table above.
# ---------------------------------------------------------------------------
CUSTOM = MODEL / "materials_custom.rad"
if not CUSTOM.exists():
    CUSTOM.write_text("""\
# materials_custom.rad -- YOUR materials. 01_model.py never overwrites this file;
# it just appends the contents to model/materials.rad on every run.
#
# The copy-paste route: open a material on https://www.spectraldb.com, e.g.
#
#     https://www.spectraldb.com/measurements/01271/
#
# and paste its "Radiance 6 Spectral Definition" block below. Those blocks all
# declare their spectrum as `void spectrum _diffuseSpectrum*`; the trailing '*'
# tells Radiance to make the name unique for you, so you can stack as many of
# them as you like without a "duplicate definition" error.
#
# The tidier route, once you know you want to keep a material, is to put its
# 5-digit id in the SDB table in 01_model.py instead -- then the spectrum is
# downloaded and versioned rather than pasted.
#
# Anything you define here can be used as a modifier straight away: point a
# MAT[...] entry at it, or reference it by name from your own geometry.
""")
    print(f"     created {CUSTOM.relative_to(MODEL.parent)} (paste your own materials there)")
custom_txt = CUSTOM.read_bytes()
ncustom = sum(1 for ln in custom_txt.decode().splitlines()
              if ln.strip() and not ln.startswith("#"))
if ncustom:
    print(f"     appending {CUSTOM.name} ({ncustom} non-comment lines)")

write(MODEL / "materials.rad", *materials, glaze,
      b"\n# --- appended from materials_custom.rad ---\n", custom_txt)


# ===========================================================================
#  2. ROOM SHELL
# ===========================================================================
step("room shell")
W, D, H = ROOM_W, ROOM_D, ROOM_H
wx0, wx1 = WINDOW_INSET, W - WINDOW_INSET
wz0, wz1 = WINDOW_SILL, WINDOW_HEAD

shell = [
    poly(MAT["floor"], "floor", (0, 0, 0), (W, 0, 0), (W, D, 0), (0, D, 0)),
    poly(MAT["ceiling"], "ceiling", (0, 0, H), (0, D, H), (W, D, H), (W, 0, H)),
    poly(MAT["accent"], "wall_back", (0, D, 0), (W, D, 0), (W, D, H), (0, D, H)),
    poly(MAT["wall"], "wall_left", (0, 0, 0), (0, D, 0), (0, D, H), (0, 0, H)),
    poly(MAT["wall"], "wall_right", (W, 0, 0), (W, 0, H), (W, D, H), (W, D, 0)),
]

# ---------------------------------------------------------------------------
#  The south wall: ONE POLYGON WITH A HOLE IN IT
# ---------------------------------------------------------------------------
# Everyone will tell you Radiance has no boolean subtraction, so you punch a
# window by drawing four polygons AROUND the opening. That is true, but it is
# not the whole story. A single Radiance polygon CAN have a hole -- you just
# have to draw it as a snake eating its own tail:
#
#   ^ v                                1>2>3>4   the OUTER contour
#   |     2---------------------3
#   |     |                     |      5>6       in along the SLIT
#   |     |     9---------8     |
#   |     |     |         |     |      6>7>8>9   the INNER contour, wound
#   |     |     6---------7     |                the OTHER way round
#   |     |     :                      10>11     back out along the SAME slit
#   |     1-----5---------------4
#   +---------------------------> u    (5 = 11 and 6 = 10: the slit is a
#                                       single segment, walked twice)
#
# The slit is traversed twice in opposite directions, so it encloses no area:
# the two edges cancel, and the crossing test correctly reports "outside" for
# anything in the opening. What makes the hole a hole is that the inner contour
# WINDS THE OPPOSITE WAY from the outer one. Get that backwards and the two
# windings reinforce instead of cancelling, and you get a solid wall.
#
# Better still, we do not have to build the thickness by hand. genprism takes a
# 2D contour and extrudes it, emitting the two end caps (.b and .t) plus one
# quad per contour edge -- so the reveal (jambs, head, sill soffit) and the
# caps around the exposed 300 mm edge all come out of a single call. It makes
# no convexity assumption; it writes your contour out verbatim, hole and all.
#
# genprism works in the u-v plane and extrudes along +w, but our wall face
# lies in world x-z and its thickness runs south. rotatex(90) maps
# (x, y, z) -> (x, -z, y), which sends contour-v to world-z and the extrusion
# to -y. Hence the vertex list below is (x, z) pairs, not (x, y).
#
# 13 polygons in total. Two of them are the slit edge extruded twice, giving a
# coincident pair sealed inside the solid at x = wx0, z = 0..wz0. No ray can
# reach them, but they are the same kind of untidiness that four stacked
# genboxes would have produced -- that route cost 24 polygons for the same
# wall, so this is still the better trade.
#
# (Verified with rtrace: a ray through the opening escapes, a ray aimed exactly
# at the slit line hits solid wall, the inner face normal is +y and the outer
# face normal is -y, and getbbox comes back x 0..W, y -WALL_T..0, z 0..H.)
#
# Reveals are worth the trouble: a 300 mm wall cuts the low-angle sky that the
# sensors at the back of the room can see, and the sunlit sill is the single
# brightest surface in the hero render.
# ---------------------------------------------------------------------------
KEYHOLE = [
    # outer contour
    0, 0,   0, H,   W, H,   W, 0,
    wx0, 0,                             # pause on the bottom edge: slit foot
    wx0, wz0,                           # in along the slit to the opening corner
    # inner contour, wound the opposite way
    wx1, wz0,   wx1, wz1,   wx0, wz1,   wx0, wz0,
    wx0, 0,                             # back out along the slit; closes to vertex 1
]
# no_connect / no_ends would drop the closing edge and the end caps -- we want
# both, so neither flag is set.
wall_s = pr.Xform(
    pr.genprism(MAT["wall"], "wall_s", KEYHOLE, lvect=(0, 0, WALL_T))
).rotatex(90)()
write(MODEL / "room.rad", *shell, wall_s)

# ---------------------------------------------------------------------------
# Glazing lives on its own so later scripts can swap or black it out.
#
# One gensurf patch per pane, sitting at mid-wall depth. gensurf takes three
# parametric expressions x(s,t), y(s,t), z(s,t) with s and t running 0..1, and
# tessellates them m x n. At 1 x 1 each pane is a single flat quad -- which is
# to say, exactly what poly() would have given us. What we gain is that the
# glass is written as a SURFACE rather than four corners: raise m and n and
# perturb y(s,t) (real insulated units bow a few mm under pressure) and you
# have curved glazing, with `smooth=True` to interpolate the normals.
# ---------------------------------------------------------------------------
GLASS_Y = -WALL_T / 2
pane_w = (wx1 - wx0) / N_PANES
panes = []
for k in range(N_PANES):
    panes.append(
        pr.gensurf(GLAZING_MAT, f"pane{k}",
               f"{wx0 + k * pane_w}+{pane_w}*s",     # x(s,t)
               str(GLASS_Y),                         # y(s,t) -- flat, mid-wall
               f"{wz0}+{wz1 - wz0}*t",               # z(s,t)
               1, 1))

write(MODEL / "glazing.rad", *panes)


# ===========================================================================
#  3. WINDOW FRAME + MULLIONS
# ===========================================================================
# Mullion shadows are the single cheapest thing you can add to make a daylight
# render look real, so it is worth the dozen lines.
step("window frame and mullions")
ww, wh = wx1 - wx0, wz1 - wz0
fw, fd = 0.06, 0.12            # frame face width, depth (front-to-back)
# The frame now sits INSIDE the wall, straddling the glass plane, instead of
# being stuck on the inner face -- that is what makes the mullions throw the
# long raking shadows onto the reveal that sell the thickness.
fy = GLASS_Y - fd / 2
frame = [
    box(MAT["frame"], "f_bot", ww + 2 * fw, fd, fw, (wx0 - fw, fy, wz0 - fw)),
    box(MAT["frame"], "f_top", ww + 2 * fw, fd, fw, (wx0 - fw, fy, wz1)),
    box(MAT["frame"], "f_lft", fw, fd, wh, (wx0 - fw, fy, wz0)),
    box(MAT["frame"], "f_rgt", fw, fd, wh, (wx1, fy, wz0)),
]
# Xform.array() repeats geometry -- the idiomatic way to make N of something.
if N_PANES > 1:
    pitch = ww / N_PANES
    mullion = pr.genbox(MAT["frame"], "mullion", 0.05, fd, wh)
    frame.append(
        pr.Xform(mullion)
        .translate(wx0 + pitch - 0.025, fy, wz0)
        .array(N_PANES - 1)
        .translate(pitch, 0, 0)()
    )
write(MODEL / "frame.rad", *frame)


# ===========================================================================
#  4. VENETIAN BLINDS
# ===========================================================================
# genblinds builds its slats in a fixed local frame: slat WIDTH runs along +Y,
# slat DEPTH along +X, and the slats stack up +Z. Our window is in the y=0
# plane, so we rotate -90 degrees about Z to swing the width onto +X, then
# translate into the opening. Transforms apply in the order you chain them.
# With a thick wall we can hang them in the reveal (y slightly negative), just
# inside the room-side face, where a real interior blind would live.
step(f"venetian blinds ({BLIND_SLATS} slats at {BLIND_ANGLE} deg)")
blinds = pr.genblinds(MAT["blind"], "blind", BLIND_DEPTH, ww, wh, BLIND_SLATS, BLIND_ANGLE)
write(MODEL / "blinds.rad",
      pr.Xform(blinds).rotatez(-90).translate(wx0, -0.02, wz0)())


# ===========================================================================
#  5. FURNITURE
# ===========================================================================
step("furniture")
furn = []


def workstation(tag, x0, y0):
    """A desk, a monitor and a task chair. The occupant sits on the +y side and
    faces the window, so the monitor goes at the window edge of the desk."""
    dw, dd, dt = 1.50, 0.75, 0.04            # desk width, depth, top thickness
    top_z = 0.72
    g = [
        box(MAT["desk"], f"{tag}_top", dw, dd, dt, (x0, y0, top_z)),
        box(MAT["desk"], f"{tag}_legL", 0.04, dd, top_z, (x0 + 0.05, y0, 0)),
        box(MAT["desk"], f"{tag}_legR", 0.04, dd, top_z, (x0 + dw - 0.09, y0, 0)),
        # A slim cylindrical monitor post rather than a box: it looks better,
        # and it is a much smaller obstacle for the sensor grid to collide with.
        pr.Primitive(MAT["chair"], "cylinder", f"{tag}_post", [],
                     [x0 + dw / 2, y0 + 0.165, top_z + dt,
                      x0 + dw / 2, y0 + 0.165, top_z + dt + 0.11, 0.030]),
        box(MAT["screen"], f"{tag}_scr", 0.58, 0.03, 0.34,
            (x0 + dw / 2 - 0.29, y0 + 0.15, top_z + dt + 0.11)),
    ]
    # Task chair: seat and back as bevelled boxes, then a proper pedestal.
    cx, cy = x0 + dw / 2 - 0.23, y0 + dd + 0.18
    g += [
        box(MAT["chair"], f"{tag}_seat", 0.46, 0.44, 0.06, (cx, cy, 0.44), bevel=0.02),
        box(MAT["chair"], f"{tag}_back", 0.46, 0.06, 0.48, (cx, cy + 0.38, 0.50), bevel=0.02),
    ]
    # Radiance has cylinder/ring primitives built in -- much better here than a
    # genrev cone, which reads as a traffic cone. A ring with an inner radius of
    # 0 is a disc.
    px, py = cx + 0.23, cy + 0.22
    g += [
        pr.Primitive(MAT["chair"], "cylinder", f"{tag}_pedpost", [],
                     [px, py, 0.03, px, py, 0.44, 0.035]),
        pr.Primitive(MAT["chair"], "ring", f"{tag}_base", [],
                     [px, py, 0.03, 0, 0, 1, 0, 0.30]),
        pr.Primitive(MAT["chair"], "ring", f"{tag}_baseu", [],
                     [px, py, 0.029, 0, 0, -1, 0, 0.30]),
    ]
    return g


for i, (x0, y0) in enumerate([(0.20, 1.30), (1.95, 1.30), (1.95, 4.40)]):
    furn += workstation(f"ws{i}", x0, y0)

# A storage run along the back wall: keeps the far end of the room from
# reading as an empty box, without cluttering the hero view.
furn.append(box(MAT["desk"], "credenza", 2.20, 0.42, 0.78, (0.35, 7.52, 0), bevel=0.01))
write(MODEL / "furniture.rad", *furn)


# ===========================================================================
#  6. MACBETH COLOUR CHECKER
# ===========================================================================
# This sits on the front desk. It is a nice prop in the point-in-time render,
# and in 05_spectral.py it becomes the thing we actually measure -- the 24
# patches give us a spread of reflectance spectra to compare RGB against.
step("Macbeth colour checker")
MACBETH_SRGB = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67),
    (133, 128, 177), (103, 189, 170), (214, 126, 44), (80, 91, 166),
    (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31),
    (187, 86, 149), (8, 133, 161), (243, 243, 242), (200, 200, 200),
    (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52),
]
MACBETH_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
    "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black",
]


def srgb_to_linear(v255):
    """Radiance works in LINEAR radiometric units. Never feed it sRGB directly."""
    c = v255 / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


PATCH, GAP = 0.050, 0.008
CHART_X, CHART_Y = 1.95 + 0.10, 1.30 + 0.16      # on the front-right desk
CHART_Z = 0.72 + 0.04 + 0.001                    # 1 mm above the desk top

chart = [pr.Primitive(MAT["chartbase"], "polygon", "chart_base", [], [
    CHART_X - 0.02, CHART_Y - 0.02, CHART_Z - 0.0005,
    CHART_X + 6 * (PATCH + GAP) + 0.012, CHART_Y - 0.02, CHART_Z - 0.0005,
    CHART_X + 6 * (PATCH + GAP) + 0.012, CHART_Y + 4 * (PATCH + GAP) + 0.012, CHART_Z - 0.0005,
    CHART_X - 0.02, CHART_Y + 4 * (PATCH + GAP) + 0.012, CHART_Z - 0.0005,
])]
patch_centres = {}
for idx, (srgb, nm) in enumerate(zip(MACBETH_SRGB, MACBETH_NAMES)):
    row, col = divmod(idx, 6)
    px = CHART_X + col * (PATCH + GAP)
    py = CHART_Y + (3 - row) * (PATCH + GAP)
    rgb = tuple(round(srgb_to_linear(v), 4) for v in srgb)
    chart.append(plastic(f"mb{idx:02d}", rgb))
    chart.append(poly(f"mb{idx:02d}", f"patch{idx:02d}",
                      (px, py, CHART_Z), (px + PATCH, py, CHART_Z),
                      (px + PATCH, py + PATCH, CHART_Z), (px, py + PATCH, CHART_Z)))
    patch_centres[nm] = [round(px + PATCH / 2, 4), round(py + PATCH / 2, 4), CHART_Z]
write(MODEL / "macbeth.rad", *chart)


# ===========================================================================
#  7. EXTERIOR CONTEXT
# ===========================================================================
# Without a ground plane the room floats in a void and the reflected component
# is wrong. The two context blocks sit across a street to the south, so they
# show up through the window and shade the low winter sun a little.
step("ground and context buildings")
ext = [
    poly(MAT["ground"], "ground", (-40, -50, -0.01), (40, -50, -0.01), (40, 40, -0.01), (-40, 40, -0.01)),
    box(MAT["context"], "ctxA", 12, 10, 8.0, (-19, -26, 0)),
    box(MAT["context"], "ctxB", 10, 13, 11.5, (7, -30, 0)),
]
write(MODEL / "exterior.rad", *ext)


# ===========================================================================
#  8. VIEWS
# ===========================================================================
step("views")


def view(vtype, vp, vd, vh, vv, vu=(0.0, 0.0, 1.0)) -> pr.View:
    """A pyradiance View object instead of a hand-typed -vp/-vd string.

    create_default_view() rather than View(): the bare constructor leaves -vu
    and the fore/aft clipping planes uninitialised, and you find out about it
    when a render comes back black. Every vector must be a TUPLE of floats --
    a list raises TypeError.
    """
    v = pr.create_default_view()
    v.type = vtype                  # 'v' perspective, 'a' angular fisheye, 'l' parallel
    v.vp, v.vdir, v.vu = vp, vd, vu
    v.horiz, v.vert = vh, vv
    return v


VIEWFILES = {
    # The hero shot. Standing near the back-left corner looking out through the
    # window, slightly off-axis so the side wall gives the image some depth.
    "hero": view("v", (1.80, 6.90, 1.62), (0.0, -1.0, -0.10), 62.0, 43.0),
    # Looking back into the room: this is the view that shows daylight falling off.
    "back": view("v", (1.80, 0.95, 1.62), (0.0, 1.0, -0.07), 62.0, 43.0),
    # 180-degree fisheye from a seated eye position -- what evalglare wants.
    "desk": view("a", (1.05, 2.10, 1.20), (0.0, -1.0, 0.0), 180.0, 180.0),
}
for nm, v in VIEWFILES.items():
    # str(View) gives " -vtv -vp ...", with no program name in front. Radiance's
    # viewfile() only recognises a line starting with "rvu" or "VIEW=", so that
    # prefix is not decoration -- drop it and pr.viewfile() raises.
    (VIEWS / f"{nm}.vf").write_text("rvu" + str(v) + "\n")
    print(f"     views/{nm}.vf")


# ===========================================================================
#  9. SENSOR GRID
# ===========================================================================
# One line per sensor: "x y z  dx dy dz".
step(f"workplane sensor grid at z = {WORKPLANE_Z} m")
xs = np.linspace(0.30, W - 0.30, GRID_NX)
ys = np.linspace(0.50, D - 0.50, GRID_NY)
pts = [(x, y, WORKPLANE_Z) for y in ys for x in xs]   # y outer => row-major in y
(DATA / "points.txt").write_text(
    "\n".join(f"{x:.3f} {y:.3f} {z:.3f} 0 0 1" for x, y, z in pts) + "\n"
)
meta = {
    "shape": [GRID_NY, GRID_NX],          # (rows = depth, cols = width)
    "npoints": len(pts),
    "workplane_z": WORKPLANE_Z,
    "room": [W, D, H],
    "site": SITE,
    "macbeth_patches": patch_centres,
    "window": {"x": [wx0, wx1], "z": [wz0, wz1]},
}
(DATA / "grid.json").write_text(json.dumps(meta, indent=2))
print(f"     {len(pts)} sensors, grid {GRID_NY} x {GRID_NX} -> data/points.txt")


# ===========================================================================
#  10. BUILD THE OCTREE AND TAKE A LOOK
# ===========================================================================
# Scene collects materials, surfaces and sources, then oconv's them.
#
# Two things that catch people out:
#   * add file PATHS, not Primitive objects, if you want render() to work --
#     render() sizes its ZONE with getbbox(), which ignores Primitives.
#   * the scene id must be a bare name: Scene also writes "m{sid}.oct".
step("building octree")
GEOMETRY = ["room.rad", "glazing.rad", "frame.rad", "blinds.rad",
            "furniture.rad", "macbeth.rad", "exterior.rad"]

# ---------------------------------------------------------------------------
#  A SPECTRAL SKY: genssky
# ---------------------------------------------------------------------------
# The materials are spectral, so the sky should be too. gensky would give us a
# CIE luminance distribution with no colour to speak of; genssky runs an actual
# atmospheric radiative-transfer model for the date, time and place and hands
# back a genuinely spectral description:
#
#     void spectrum sunrad       the solar spectrum, 390-770 nm in 20 bands
#     sunrad light solar         scaled to the extraterrestrial irradiance
#     solar source sun           the 0.533-degree solar disc
#     void specpict skyfunc      a HYPERSPECTRAL sky image (.hsr), angularly
#                                mapped, one spectrum per sky direction
#
# The .hsr and an atmosphere cache go in out_dir -- genssky will NOT create that
# directory for you. The path it writes into the specpict is absolute, so unlike
# genglaze's .dat files there is no CWD/RAYPATH trap here.
#
# THE TRAP: look at that list again. genssky gives you the sun, and a PATTERN
# describing what the sky looks like -- but no sky, and no ground. `specpict
# skyfunc` is a modifier, not a light source; on its own it emits nothing. If
# you oconv genssky's output as-is you get a scene lit by direct sun only, and
# the error is quiet -- the render just comes out contrasty and dark, exactly
# the way a sunny day is supposed to look. So we bolt on the same two glow
# hemispheres gensky needs. This is the single most common way to get genssky
# wrong.
#
# We still render this preview in RGB. rpict has no spectral mode -- it collapses
# `spectrum` and `specpict` to three channels for you. 05_spectral.py is where we
# actually keep the bands.
step("genssky -- spectral sky (first run builds an atmosphere cache, ~15 s)")
atmos = SCRATCH / "atmos"
atmos.mkdir(exist_ok=True)
try:
    with timed("genssky"):
        sky = pr.genssky(WHEN, **SITE, out_dir=str(atmos),
                         out_name="preview_sky", nthreads=NPROC)
except Exception as e:  # noqa: BLE001
    print(f"     !! genssky failed ({e}); falling back to a CIE clear sky")
    sky = pr.gensky(WHEN, **SITE, sunny_with_sun=True)

sky_rad = write(MODEL / "sky_preview.rad", sky,
                pr.Primitive("skyfunc", "glow", "skyglow", [], [1, 1, 1, 0]),
                pr.Primitive("skyglow", "source", "sky", [], [0, 0, 1, 180]),
                pr.Primitive("skyfunc", "glow", "groundglow", [], [1, 1, 1, 0]),
                pr.Primitive("groundglow", "source", "ground", [], [0, 0, -1, 180]))

scene = pr.Scene("office")
scene.add_material(MODEL / "materials.rad")
for g in GEOMETRY:
    scene.add_surface(MODEL / g)
scene.add_source(sky_rad)
scene.build()
print(f"     octree: office.oct")


# ---------------------------------------------------------------------------
#  Sanity check: is any sensor buried inside furniture?
# ---------------------------------------------------------------------------
# This is the single most common error in daylight modelling, and it is silent:
# a sensor inside a desk leg just reports 0 lux and drags your averages down.
# Cast a ray straight up from every sensor and complain about near hits.
# (`-o L` asks rtrace for the distance to the first intersection.)
step("checking no sensor is buried in furniture")
probe = b"".join(f"{x} {y} {z} 0 0 1\n".encode() for x, y, z in pts)
dist = pr.rtrace(probe, scene.octree, outform="f", outspec="L",
                 header=False, params=["-ab", "0"])
dist = np.frombuffer(dist, dtype=np.single)
buried = np.where(dist < 0.05)[0]
if len(buried):
    print(f"     !! {len(buried)} sensor(s) blocked within 50 mm: {buried.tolist()}")
    for i in buried:
        print(f"        sensor {i} at {pts[i]} -- clearance {dist[i] * 1000:.0f} mm")
    print("        move the grid or the furniture, then re-run.")
else:
    print(f"     all {len(pts)} sensors clear (nearest obstruction "
          f"{dist.min():.2f} m above the workplane)")

step("preview render (draft quality -- 03_pointintime.py does it properly)")
with timed("rpict"):
    hdr = render_view(scene.octree, VIEWS / "hero.vf", 480, 330, quality="fast",
                      extra=["-ps", "1"])  # -ps 1: the blind slats need every pixel
(MODEL / "preview.hdr").write_bytes(hdr)

wsviz.save_srgb(hdr, "01_preview",
                f"01 -- your office ({WHEN:%d %b, %H:%M}, spectral clear sky)")

banner("model built")
print("  Look at out/01_preview.png -- that is YOUR room.")
print()
print("  The green cast is not a bug and not a tone-mapping artefact. Measured on")
print("  its own, the SageGlass IGU transmits R/G = 0.88 and B/G = 0.72 -- real")
print("  electrochromic glass in its clear state is distinctly green. Every RGB")
print("  triple in this room now comes from a measurement rather than from")
print("  someone's taste, and this is what that costs you.")
print("  Now go back to the PARAMETERS block, change something, and re-run.")
print("  Then continue to 02_viz.py")
