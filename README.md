# pyradiance in 2 hours

A hands-on introduction to [pyradiance](https://github.com/LBNL-ETA/pyradiance):
Python bindings for Radiance, with the Radiance binaries bundled in the wheel.

By the end you will have built a daylit office in Python, simulated one moment
and one whole year, and rendered it hyperspectrally.

---

## Before the workshop

You need **Python 3.10 or newer** (3.12 recommended) and about 10 minutes.

```bash
python -m venv pyrad-ws
source pyrad-ws/bin/activate          # Windows: pyrad-ws\Scripts\activate
pip install pyradiance numpy matplotlib scipy
```

That is the entire dependency list. **No Jupyter, no Radiance install.**
pyradiance ships all 91 Radiance binaries inside the wheel.

> **Install from PyPI wheels, not from source.** A source build needs git
> submodules and downloads libtiff at build time. If `pip install pyradiance`
> tries to compile anything, stop and ask for help.

Then, from this directory:

```bash
python 00_verify.py
```

If that prints `ALL CHECKS PASSED` and writes `out/00_verify.png`, you are ready.
If it does not, **say so at the start of the session**, not at block 3.

---

## Running the workshop

Seven scripts, in order. Each writes PNGs into `out/`.

```bash
python 00_verify.py         #  ~2 s   does my install work?
python 01_model.py          # ~10 s   build the office
python 02_viz.py            #  ~3 s   HDR -> numpy -> matplotlib
python 03_pointintime.py    # ~45 s   one moment: sky, sensors, render, glare
python 04_annual.py         # ~30 s   8760 hours by matrix multiplication
python 05_spectral.py       # ~40 s   9 spectral bands
python 06_extra.py          # ~90 s   a measured shade + the IGU, as one BSDF
```

Keep a file browser open on `out/` next to your editor. The PNGs refresh on
every run.

There is deliberately **no notebook**. Each script is standalone and re-runnable
from the top, so there is no hidden state and no out-of-order-cell confusion.
Each has `# %%` cell markers, so if you use VS Code with `ipykernel` installed
you get an interactive cell experience for free — but nothing requires it.

### If something breaks

Every script falls back to a pre-computed result in `checkpoints/` when its
input is missing. A failure in one block will not strand you in the next one.
You will see a line like:

```
  !! dc_mf1.mtx missing -- using shipped checkpoint dc_mf1.mtx
```

To force a clean rebuild, delete `scratch/`.

### Things to change

Each script has `>>> TODO (you):` markers on the lines worth playing with —
blind angle, accent colour, time of day, sky subdivision. Change one, re-run,
look at the PNG.

---

## What is in each block

| Script | Radiance tools | The idea |
|---|---|---|
| `01_model.py` | `genbox` `genrev` `genblinds` `xform` `oconv` `rpict` | A scene is just primitives; build them in Python |
| `02_viz.py` | `pvalue` `pcond` `pextrem` `getinfo` | `pvalue` turns HDR into numpy; everything else is matplotlib |
| `03_pointintime.py` | `gensky` `gendaylit` `rtrace -I` `rpict` `evalglare` | The core loop: sky + geometry → octree → numbers and pictures |
| `04_annual.py` | `rfluxmtx` `gendaymtx` `dctimestep` | Split geometry from weather; a year becomes a matrix multiply |
| `05_spectral.py` | `genssky` `rtrace -co+ -cs N` | Carry N wavelength bands instead of 3 |
| `06_extra.py` | `genBSDF` `wrapBSDF` `checkBSDF` `aBSDF` | Simulate a shade + IGU assembly into one measured-style BSDF |

---

## Files

```
wsvenv.py        shared paths, quality presets, render helper   (don't edit)
wsviz.py         all the plotting                               (don't edit)
data/            EPW weather, sensor grid, grid metadata
igsdb_product_*.json   measured glazing records, used by 01_model.py
E Screen 1% Pearl-Grey.xml   measured Klems BSDF of a Mermet solar
                 screen, used by 06_extra.py
model/           .rad files written by 01_model.py
views/           .vf view files written by 01_model.py
checkpoints/     pre-computed fallbacks
scratch/         octrees, matrices, intermediate files
out/             every PNG lands here
```

## The model

A south-facing single-sided office, 3.6 m wide × 8 m deep × 3.0 m high, in
New York City. It is deliberately deep so daylight falls off strongly from
front to back — roughly 4000 lux at the window to 90 lux at the back wall on a
clear March afternoon. That gradient is the thread running through blocks 3, 4
and 5.

Weather is `data/NewYork_TMY3.epw`, a full 8760-hour TMY3 file.

## Known pyradiance gotchas

These bit us while writing the workshop; see `cheatsheet.md` for the full list.

- `pr.render()` returns a **transposed** image — use `rpict` (we wrap it as
  `wsvenv.render_view()`).
- `Scene` surfaces must be **file paths**, not `Primitive` objects, or
  `render()` cannot compute its bounding box.
- `rpict` accepts `-cs` but ignores it — hyperspectral output needs
  `vwrays | rtrace -co+`.
- `falsecolor` and `genskyvec` are **not** bundled.
- A `BSDF`/`aBSDF` up vector that is **parallel to the surface normal** does
  not raise an error — it silently returns near-zero scattering.
- `pr.generate_bsdf()` needs an explicit `dim=SamplingBox(...)`, and the
  device geometry must be **larger than the sampling box** or oblique rays
  leak past its edges and inflate the transmittance.
- Anything pyradiance does not wrap (e.g. `checkBSDF`) is still in
  `pr.BINPATH` and one `subprocess.run` away.
