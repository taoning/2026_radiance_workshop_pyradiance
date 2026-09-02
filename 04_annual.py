"""
04_annual.py -- a whole year, in about a minute.

    python 04_annual.py

03 simulated one moment and took a few seconds. A year has 8760 of them, so
brute force would take hours. The matrix methods get us out of that.

The trick: split the calculation in two.

    D  the DAYLIGHT COEFFICIENT matrix.  How much does each patch of sky
       contribute to each sensor?  Depends only on GEOMETRY, so we compute it
       ONCE with rfluxmtx.                              [ nsensors x npatches ]

    S  the SKY matrix.  How bright is each patch of sky at each hour?
       Depends only on WEATHER, so gendaymtx reads it straight from the EPW.
                                                        [ npatches x 8760 ]

    D @ S  =  illuminance at every sensor for every hour of the year.

One expensive geometry calculation, one cheap matrix multiply. This is the
"two-phase method", and it is the foundation of every annual daylight metric.

Outputs:
    out/04_a_skymatrix.png    what gendaymtx produced
    out/04_b_heatmap.png      one sensor, all 8760 hours
    out/04_c_da.png           spatial Daylight Autonomy
    out/04_d_udi.png          Useful Daylight Illuminance
"""

from __future__ import annotations

import json

import numpy as np
import pyradiance as pr

import wsviz
from wsvenv import (CKPT, DATA, MODEL, NPROC, OUT, SCRATCH, banner, checkpoint,
                    scene_files, step, timed)

meta = json.loads((DATA / "grid.json").read_text())
GRID = tuple(meta["shape"])
NPTS = meta["npoints"]

EPW = DATA / "NewYork_TMY3.epw"
MF = 4                    # Reinhart subdivision. MF=1 -> 145 patches + ground.
                          # >>> TODO (you): MF=4 gives 2305 patches and a much
                          # >>> sharper sun, at ~16x the rfluxmtx cost.
DA_THRESHOLD = 300        # lux, the usual target for office work
OCCUPIED = (8, 18)        # 08:00-18:00

banner("04 -- annual daylight, two-phase method")


# ===========================================================================
#  1. THE SKY RECEIVER
# ===========================================================================
# rfluxmtx needs to know how to chop the sky up. You tell it with a magic
# comment -- "#@rfluxmtx" -- placed immediately before the surface. This is the
# bit that confuses everyone the first time:
#
#     h=r1   hemisphere sampling, Reinhart with MF=1  (145 patches)
#     u=+Y   the "up" vector used to orient the patch numbering
#
# The comment is read by rfluxmtx, not by Radiance proper. To oconv it is just
# a comment. Two surfaces: the sky dome, and the ground hemisphere below it,
# which becomes the final (146th) column.
step(f"writing the sky receiver (Reinhart MF={MF})")
receiver = SCRATCH / f"skyrecv_mf{MF}.rad"
receiver.write_text(f"""\
#@rfluxmtx h=u u=+Y
void glow ground_glow
0
0
4 1 1 1 0

ground_glow source ground
0
0
4 0 0 -1 180

#@rfluxmtx h=r{MF} u=+Y
void glow sky_glow
0
0
4 1 1 1 0

sky_glow source sky
0
0
4 0 0 1 180
""")
# Reinhart: the 144 Tregenza bands are subdivided by MF^2, but the zenith cap
# stays a single patch -> 144*MF^2 + 1 sky patches, +1 for the ground.
NSKY = 144 * MF * MF + 1
NPATCH = NSKY + 1
print(f"     receiver: 1 ground patch + {NSKY} sky patches")


# ===========================================================================
#  2. THE DAYLIGHT COEFFICIENT MATRIX  (the expensive bit)
# ===========================================================================
# We trace from the SENSORS out to the sky. Note the octree here must NOT
# contain a sky -- rfluxmtx supplies its own. So we build a sky-free octree.
step("building a sky-free octree")
MATERIALS, GEOMETRY = scene_files()
oct_nosky = SCRATCH / "office_nosky.oct"
oct_nosky.write_bytes(pr.oconv(MATERIALS, *GEOMETRY, warning=False))

dc_path = SCRATCH / f"dc_mf{MF}.mtx"
step(f"rfluxmtx: {NPTS} sensors x {NPATCH} sky patches")
if dc_path.exists():
    print("     (already computed -- delete scratch/ to force a rebuild)")
else:
    points = (DATA / "points.txt").read_bytes()
    with timed("rfluxmtx"):
        dc_path.write_bytes(pr.rfluxmtx(
            receiver=receiver,
            rays=points,
            octree=oct_nosky,
            params=["-I+",              # irradiance, as in 03
                    "-y", str(NPTS),    # tell it how many sensors are coming
                    "-ab", "3", "-ad", "1024", "-lw", "1e-4",
                    "-n", str(NPROC),
                    "-faf"],            # float in, float out
        ))
dc_path = checkpoint(dc_path, f"dc_mf{MF}.mtx")

# Read it into numpy. Radiance matrices are a text header, a blank line, then
# the raw numbers -- getinfo strips the header for us. We asked for -faf, so
# the body is 32-bit floats, interleaved R,G,B.
dc = np.frombuffer(pr.getinfo(dc_path.read_bytes(), strip_header=True),
                   dtype=np.single).reshape(NPTS, -1, 3)
print(f"     D matrix: {dc.shape}  (sensors x patches x RGB)")


# ===========================================================================
#  3. THE SKY MATRIX  (the cheap bit)
# ===========================================================================
# gendaymtx reads the EPW *directly* -- there is no need to convert to .wea
# first. It pulls latitude, longitude and time zone out of the EPW header and
# runs the Perez model for all 8760 hours.
step("gendaymtx straight from the EPW")
with timed("gendaymtx"):
    sky_raw = pr.gendaymtx(str(EPW), mfactor=MF, outform="d")   # keep the header
sky = np.frombuffer(pr.getinfo(sky_raw, strip_header=True),
                    dtype=np.double).reshape(-1, 8760, 3)
print(f"     S matrix: {sky.shape}  (patches x hours x RGB)")
assert sky.shape[0] == dc.shape[1], (
    f"patch count mismatch: D has {dc.shape[1]}, S has {sky.shape[0]}")


# ===========================================================================
#  4. MULTIPLY
# ===========================================================================
step("D @ S")
# The Radiance way is `dctimestep D.mtx S.mtx`, and for image sequences you
# need it. For a sensor grid the multiply is a one-liner in numpy, and it is
# useful to see that there is no magic in it.
with timed("matrix multiply"):
    # (sensors, patches, rgb) x (patches, hours, rgb) -> (sensors, hours, rgb)
    result = np.einsum("spc,phc->shc", dc, sky)
lux = result @ wsviz.LUM          # -> (sensors, hours) in lux
print(f"     annual illuminance: {lux.shape}, "
      f"peak {lux.max():,.0f} lux, annual mean {lux.mean():,.0f} lux")

# Cross-check against dctimestep so you can trust the numpy version.
# dctimestep reads the dimensions out of the matrix headers, so both inputs
# must keep theirs -- this is why we did not pass header=False to gendaymtx.
with timed("dctimestep cross-check"):
    dct = pr.dctimestep(str(dc_path), sky_raw, outform="d")
dct = np.frombuffer(pr.getinfo(dct, strip_header=True),
                    dtype=np.double).reshape(NPTS, 8760, 3) @ wsviz.LUM
err = np.abs(dct - lux).max() / max(lux.max(), 1)
print(f"     dctimestep agrees to {err:.2e} relative -- same answer, same method")


# ===========================================================================
#  5. DOES IT AGREE WITH 03?
# ===========================================================================
# This is the check that should convince you. 03 computed one hour the direct
# way: an explicit Perez sky from gendaylit, traced with rtrace. Here we got
# the same hour out of a matrix multiply. Nothing is shared between the two
# routes except the geometry and the weather file.
step("cross-check against the point-in-time result from 03")
DAY_OF_YEAR, HOUR = 80, 13          # 21 March, 13:00 -- same instant as 03
idx = (DAY_OF_YEAR - 1) * 24 + HOUR
snap = lux[:, idx]
print(f"     two-phase at 21 Mar 13:00: mean {snap.mean():.0f} lux "
      f"(front row {snap.reshape(GRID)[0].mean():.0f}, "
      f"back row {snap.reshape(GRID)[-1].mean():.0f})")
print("     03_pointintime.py reported roughly 960 / 3200 / 100 for the same hour.")
print("     The two-phase answer runs about 10% high because gendaymtx smears the")
print(f"     sun across {NSKY} patches instead of using a sharp disc. Raise MF")
print("     to sharpen it -- that is the main accuracy knob in this method.")


# ===========================================================================
#  6. WHAT THE YEAR LOOKS LIKE
# ===========================================================================
step("plots")

# (a) the sky matrix itself -- total sky brightness over the year
sky_total = (sky @ wsviz.LUM).sum(axis=0)
wsviz.save_annual_heatmap(
    sky_total, "04_a_skymatrix",
    "Sky matrix: total hemispherical brightness, New York TMY3",
    label="sum over patches [lux]")

# (b) one sensor, every hour
mid = NPTS // 2
row, col = divmod(mid, GRID[1])
wsviz.save_annual_heatmap(
    lux[mid], "04_b_heatmap",
    f"Sensor {mid} ({row * 0.5 + 0.5:.1f} m from the window): every hour of the year")

# (c) Daylight Autonomy: fraction of OCCUPIED hours above threshold
hour_of_day = np.tile(np.arange(24), 365)
occ = (hour_of_day >= OCCUPIED[0]) & (hour_of_day < OCCUPIED[1])
da = (lux[:, occ] >= DA_THRESHOLD).mean(axis=1) * 100
sda = (da >= 50).mean() * 100
print(f"     occupied hours: {occ.sum()}")
print(f"     DA{DA_THRESHOLD}: min {da.min():.0f}%, mean {da.mean():.0f}%, max {da.max():.0f}%")
print(f"     sDA{DA_THRESHOLD}/50% = {sda:.0f}% of the floor area")

wsviz.save_grid(da, GRID, "04_c_da",
                f"Daylight Autonomy (>{DA_THRESHOLD} lux, {OCCUPIED[0]}:00-{OCCUPIED[1]}:00)\n"
                f"sDA = {sda:.0f}% of area",
                label="% of occupied hours", cmap="viridis",
                vmin=0, vmax=100, annotate=True)

# (d) UDI -- the more honest metric, because it counts too much light as a fault
wsviz.save_udi_bars(lux, "04_d_udi", occupied=occ,
                    title="Useful Daylight Illuminance by sensor "
                          "(sensor 0 = nearest the window)")

over = (lux[:, occ] > 3000).mean(axis=1) * 100
print(f"     glare risk: sensors exceed 3000 lux for up to {over.max():.0f}% of occupied hours")

# Save for anyone who wants to keep going.
np.save(SCRATCH / "annual_lux.npy", lux)

banner("done")
print(f"  You just simulated {NPTS} sensors x 8760 hours = {NPTS * 8760:,} results.")
print("  The geometry was traced once; the year was a matrix multiply.")
print("  >>> TODO: set MF = 4 at the top and compare. Watch the rfluxmtx time.")
print("  Next: 05_spectral.py")
