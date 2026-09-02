"""
wsvenv.py -- shared paths and small conveniences for the pyradiance workshop.

Every workshop script starts with `from wsvenv import *`, which gives you:

    HERE, DATA, MODEL, VIEWS, CKPT, OUT   -- Path objects for the workshop dirs
    NPROC                                 -- sensible parallelism for this machine
    banner(), step(), timed()             -- console output helpers
    checkpoint()                          -- "use my result, or fall back to the shipped one"

You never need to edit this file.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = HERE / "model"
VIEWS = HERE / "views"
CKPT = HERE / "checkpoints"
OUT = HERE / "out"
SCRATCH = HERE / "scratch"

for _d in (DATA, MODEL, VIEWS, CKPT, OUT, SCRATCH):
    _d.mkdir(exist_ok=True)

# Scene.build() and the ambient cache write to the *current* directory, so pin
# it. This makes the scripts work no matter where you launch them from.
os.chdir(HERE)

# Radiance's own tools parallelise with -n; render() ignores nproc on Windows.
NPROC = min(8, os.cpu_count() or 1)
IS_WINDOWS = os.name == "nt"


# ---------------------------------------------------------------------------
#  Render quality presets
# ---------------------------------------------------------------------------
# Radiance has a lot of knobs. These three sets cover the workshop. The ones
# that matter most:
#   -ab  ambient bounces      how many times light may bounce diffusely
#   -ad  ambient divisions    samples per bounce  (noise vs time)
#   -aa  ambient accuracy     0 disables interpolation (slow but exact)
#   -ps  pixel sample spacing 1 = every pixel
# These are tuned so that no single workshop render exceeds about a minute on
# a laptop. They are NOT publication settings -- for real work push -ab to 5+,
# -ad into the tens of thousands, and -aa down to 0.1 or 0.
QUALITY = {
    "draft": ["-ab", "1", "-ad", "256", "-as", "0", "-aa", "0.3",
              "-ar", "32", "-ps", "8", "-lw", "1e-2"],
    "fast": ["-ab", "2", "-ad", "800", "-as", "200", "-aa", "0.2",
             "-ar", "96", "-ps", "4", "-lw", "1e-3"],
    "good": ["-ab", "3", "-ad", "1400", "-as", "350", "-aa", "0.17",
             "-ar", "160", "-ps", "2", "-lw", "2e-4"],
}

# Sensor-grid (rtrace -I) parameters. A grid of 90 points is far cheaper than
# a megapixel image, so we can afford to be more generous here than in QUALITY.
GRID_PARAMS = ["-ab", "3", "-ad", "1024", "-as", "256", "-aa", "0.15",
               "-ar", "128", "-lw", "1e-4"]


def render_view(octree, view, xres=800, yres=600, quality="fast", extra=None):
    """Render a view with rpict.

    We call rpict directly rather than pyradiance's render() convenience
    wrapper, because rpict behaves identically on every platform whereas
    render() is silently single-threaded on Windows, and because rpict takes
    the ray parameters verbatim instead of routing them through `rad`'s
    quality presets.

    (render() used to also swap width and height -- it read the resolution
    back from `vwrays -d`, which prints "-x W -y H", using the token order of
    `getinfo -d`, which prints "-Y H +X W". That is fixed in the vendored
    pyradiance; see cheatsheet.md.)

    `view` may be a .vf path, a View object, or a list of view arguments.
    """
    import pyradiance as pr

    if isinstance(view, (str, Path)):
        view = pr.get_view_args(pr.viewfile(str(view)))
    elif not isinstance(view, (list, tuple)):
        view = pr.get_view_args(view)
    params = list(QUALITY[quality]) + list(extra or [])
    return pr.rpict(view, str(octree), xres=xres, yres=yres, params=params)


def banner(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def step(msg: str) -> None:
    print(f"  -> {msg}", flush=True)


@contextmanager
def timed(label: str):
    """Print how long a block took. Radiance timings are the whole point here."""
    t0 = time.time()
    print(f"  ... {label}", end="", flush=True)
    try:
        yield
    finally:
        print(f"  [{time.time() - t0:.1f}s]", flush=True)


def checkpoint(path: Path, name: str) -> Path:
    """Return `path` if you produced it, else the shipped fallback in checkpoints/.

    This is the workshop's safety net: if a step fails or you run a script out of
    order, you still get a usable input instead of a traceback.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    fallback = CKPT / name
    if fallback.exists():
        print(f"  !! {path.name} missing -- using shipped checkpoint {fallback.name}")
        return fallback
    raise FileNotFoundError(
        f"Neither {path} nor the checkpoint {fallback} exists.\n"
        f"   Run the earlier scripts first, or re-download the workshop bundle."
    )


GEOMETRY_FILES = ["room.rad", "glazing.rad", "frame.rad", "blinds.rad",
                  "furniture.rad", "macbeth.rad", "exterior.rad"]


def model_dir() -> Path:
    """Where the .rad files live: your own if 01 built them, else the shipped set.

    This is what stops a problem in 01_model.py from ending someone's workshop.
    """
    if all((MODEL / f).exists() for f in GEOMETRY_FILES + ["materials.rad"]):
        return MODEL
    fallback = CKPT / "model"
    if all((fallback / f).exists() for f in GEOMETRY_FILES + ["materials.rad"]):
        print("  !! model/ incomplete -- using the shipped model from checkpoints/")
        return fallback
    raise FileNotFoundError("No usable model. Run 01_model.py first.")


def scene_files() -> tuple[str, list[str]]:
    """(materials, [geometry...]) as strings, ready to hand to oconv."""
    d = model_dir()
    return str(d / "materials.rad"), [str(d / f) for f in GEOMETRY_FILES]


def require(*scripts: str) -> None:
    """Friendly reminder about run order."""
    missing = [s for s in scripts if not (HERE / s).exists()]
    if missing:
        sys.exit(f"Missing workshop scripts: {missing}")


def env_report() -> None:
    import pyradiance as pr

    print(f"  python      {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"  pyradiance  {getattr(pr, '__version__', 'unknown')}")
    print(f"  binaries    {pr.BINPATH}")
    print(f"  nproc       {NPROC}" + ("  (render() is single-threaded on Windows)" if IS_WINDOWS else ""))
