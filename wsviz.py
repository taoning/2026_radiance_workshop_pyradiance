"""
wsviz.py -- visualization helpers for the pyradiance workshop.

Radiance ships a `falsecolor` shell script, but pyradiance does NOT bundle it
(only the pvalue/pcond/pfilt binaries). That is fine -- matplotlib is a better
tool for this job anyway, and it keeps everything in Python.

The whole module rests on one idea:

    pvalue turns an HDR image into raw floats, and numpy takes it from there.

Nothing here calls plt.show(): every function writes a PNG into out/. That
avoids GUI-backend problems (a real hazard in a fresh venv on Windows) and
means you can keep a file browser open next to your editor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no GUI backend required
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pyradiance as pr
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from wsvenv import OUT, SCRATCH

# Radiance's RGB -> luminance weights. This exact triple shows up all over
# Radiance (rmtxop -c, ximage, etc.); it converts radiance [W/sr/m2] to
# luminance [cd/m2], or irradiance [W/m2] to illuminance [lux].
LUM = np.array([47.4, 119.9, 11.6])


# --------------------------------------------------------------------------
# HDR  ->  numpy
# --------------------------------------------------------------------------
def hdr_to_array(hdr: bytes | str | Path, nchan: int = 3) -> np.ndarray:
    """Read a Radiance HDR (bytes or path) into a (rows, cols, nchan) float array.

    `nchan` is 3 for ordinary RGB pictures and N for a hyperspectral picture
    rendered with -co+ -cs N.
    """
    xres, yres = pr.get_image_dimensions(hdr)  # note the order: (width, height)
    raw = pr.pvalue(hdr, header=False, resstr=False, outform="f")
    arr = np.frombuffer(raw, dtype=np.single)
    expected = xres * yres * nchan
    if arr.size != expected:
        raise ValueError(
            f"expected {expected} floats for a {xres}x{yres}x{nchan} image, "
            f"got {arr.size}. Wrong nchan?"
        )
    # Radiance scans row by row, so rows (y) is the FIRST axis.
    return arr.reshape(yres, xres, nchan)


def luminance(arr: np.ndarray) -> np.ndarray:
    """(rows, cols, 3) radiance -> (rows, cols) luminance in cd/m2."""
    return arr[:, :, :3] @ LUM


# --------------------------------------------------------------------------
# Image plots
# --------------------------------------------------------------------------
def as_path(hdr: bytes | str | Path, stem: str = "_tmp") -> Path:
    """Some pyradiance wrappers (pcond, for one) take only a path, while most
    also accept bytes. Spill bytes to a file so callers don't have to care."""
    if isinstance(hdr, (str, Path)):
        return Path(hdr)
    p = SCRATCH / f"{stem}.hdr"
    p.write_bytes(hdr)
    return p


def to_srgb(arr: np.ndarray, key: float = 0.25) -> np.ndarray:
    """Auto-exposure + Reinhard tone compression + sRGB gamma.

    A daylit room is the hard case for exposure: the window can be 10,000x
    brighter than the back wall. Scaling to a high percentile just makes the
    interior black (try it). So instead we:

      1. anchor the exposure on the LOG-AVERAGE luminance, which ignores the
         handful of blown-out window pixels,
      2. compress with Reinhard's x/(1+x), which rolls the window highlights
         off smoothly instead of clipping them,
      3. apply the sRGB transfer curve.

    Colour is preserved throughout, unlike pcond -h. Raise `key` to brighten.
    """
    rgb = np.maximum(np.asarray(arr[:, :, :3], dtype=float), 0.0)
    lum = luminance(rgb) / LUM.sum()
    log_avg = np.exp(np.mean(np.log(lum + 1e-6)))
    scaled = rgb * (key / max(log_avg, 1e-9))
    compressed = scaled / (1.0 + scaled)
    return np.where(compressed <= 0.0031308,
                    compressed * 12.92,
                    1.055 * compressed ** (1 / 2.4) - 0.055)


def save_srgb(hdr: bytes | str | Path, name: str, title: str = "", key: float = 0.25) -> Path:
    """Exposure-mapped, colour-preserving PNG. Use this for the pretty pictures."""
    return _imsave(to_srgb(hdr_to_array(hdr), key=key), name, title, cbar=None)


def save_tonemap(
    hdr: bytes | str | Path,
    name: str,
    title: str = "",
    contour: np.ndarray | None = None,
    levels=None,
    sigma: float = 2.0,
) -> Path:
    """Human-eye tonemap via pcond -h, optionally with iso-value contours.

    pcond models human visual response: local adaptation, veiling glare, loss
    of acuity and of colour in the dark. Compare it with save_srgb() -- the
    desaturation you see in the shadows is deliberate and physiological, not
    a bug.

    Because pcond has already destroyed the physical values, the greys are
    NOT quantitative. Pass `contour=` the ORIGINAL luminance array to draw
    labelled iso-luminance lines on top: the picture then shows what the eye
    does, and the lines show what the numbers are.

    Note pcond takes a PATH only, unlike most pyradiance functions.
    """
    toned = pr.pcond(as_path(hdr, name), human=True)
    arr = hdr_to_array(toned)
    img = np.clip(arr, 0, 1) ** (1 / 2.2)
    return _imsave(img, name, title, cbar=None,
                   contour=contour, levels=levels, sigma=sigma)


def save_falsecolor(
    lum: np.ndarray,
    name: str,
    title: str = "",
    label: str = "luminance [cd/m$^2$]",
    vmin: float = 1.0,
    vmax: float | None = None,
    log: bool = True,
    cmap: str = "turbo",
) -> Path:
    """Falsecolor an (rows, cols) scalar field. Log scale by default, because
    luminance in a daylit room spans four or five orders of magnitude."""
    vmax = float(np.percentile(lum, 99.5)) if vmax is None else vmax
    vmax = max(vmax, vmin * 10)
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else None
    return _imsave(
        np.clip(lum, vmin, vmax) if log else lum,
        name,
        title,
        cbar=label,
        cmap=cmap,
        norm=norm,
        vmax=None if log else vmax,
    )


# Decade / half-decade ladder. Luminance is log-distributed, so evenly spaced
# linear levels would put every line inside the window and none in the room.
CONTOUR_LEVELS = [100, 300, 1000, 3000, 10000]


def _imsave(img, name, title, cbar, cmap=None, norm=None, vmax=None,
            contour=None, levels=None, sigma=2.0, clabel=True) -> Path:
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8 * h / w + (0.6 if title else 0)))
    m = ax.imshow(img, cmap=cmap, norm=norm, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()

    if contour is not None:
        field = np.asarray(contour, dtype=float)
        if field.shape != (h, w):
            raise ValueError(
                f"contour field is {field.shape} but the image is {(h, w)}; "
                "contour and imshow share array-index coordinates, so they "
                "must be the same grid."
            )
        # Smooth in LOG space. rpict sampling noise turns raw contours into
        # spaghetti; but a linear blur also lets the ~10,000 cd/m2 window bleed
        # across its frame and drags the 1000 cd/m2 line out onto the wall.
        # The log-space blur is a geometric mean, and the lines stay put.
        sm = 10.0 ** gaussian_filter(np.log10(np.maximum(field, 1e-3)), sigma)

        lv = [v for v in (CONTOUR_LEVELS if levels is None else levels)
              if sm.min() < v < sm.max()]
        if lv:
            cs = ax.contour(sm, levels=lv, colors="w", linewidths=0.7, alpha=0.9)
            # Outline the lines so they read over both the blown-out window and
            # the dark back wall.
            stroke = [pe.withStroke(linewidth=1.8, foreground="black")]
            for coll in cs.collections if hasattr(cs, "collections") else [cs]:
                coll.set_path_effects(stroke)
            if clabel:
                lbls = ax.clabel(cs, fmt=lambda v: f"{v:,.0f}", fontsize=6, inline=True)
                for t in lbls:
                    t.set_path_effects(stroke)

    if title:
        ax.set_title(title, fontsize=11)
    if cbar:
        fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03, label=cbar)
    return _save(fig, name)


def save_profile(lum: np.ndarray, name: str, title: str = "") -> Path:
    """Horizontal luminance profile through the middle of an image."""
    row = lum[lum.shape[0] // 2]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.semilogy(row, lw=1.0)
    ax.set_xlabel("pixel column")
    ax.set_ylabel("luminance [cd/m$^2$]")
    ax.set_title(title or "luminance profile, image mid-height")
    ax.grid(alpha=0.3)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Sensor-grid plots
# --------------------------------------------------------------------------
def save_grid(
    values: np.ndarray,
    shape: tuple[int, int],
    name: str,
    title: str = "",
    label: str = "illuminance [lux]",
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
) -> Path:
    """Plan view of a workplane sensor grid.

    `shape` is (nrows_y, ncols_x). Row 0 is the window end of the room, so we
    flip the array to put the window at the bottom of the plot, the way a
    floor plan is normally drawn.
    """
    grid = np.asarray(values).reshape(shape)
    fig, ax = plt.subplots(figsize=(4.2, 8.2))
    m = ax.imshow(
        np.flipud(grid), cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", aspect="equal",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("room width")
    ax.set_ylabel("distance from window  ->")
    ax.set_xticks([])
    ax.set_yticks([])
    if annotate:
        flipped = np.flipud(grid)
        for (i, j), v in np.ndenumerate(flipped):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="w" if v < flipped.max() * 0.6 else "k")
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04, label=label)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Annual plots
# --------------------------------------------------------------------------
def save_annual_heatmap(
    series: np.ndarray, name: str, title: str = "", label: str = "illuminance [lux]",
    vmax: float | None = None,
) -> Path:
    """8760-value hourly series -> the classic day-of-year x hour-of-day carpet plot."""
    series = np.asarray(series).ravel()
    if series.size != 8760:
        raise ValueError(f"expected 8760 hourly values, got {series.size}")
    carpet = series.reshape(365, 24).T  # (hour, day)
    fig, ax = plt.subplots(figsize=(11, 4))
    m = ax.imshow(
        carpet, aspect="auto", origin="lower", cmap="inferno",
        vmax=vmax or np.percentile(series, 99), extent=[0, 365, 0, 24],
    )
    ax.set_xlabel("day of year")
    ax.set_ylabel("hour of day")
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_title(title, fontsize=11)
    fig.colorbar(m, ax=ax, pad=0.02, label=label)
    return _save(fig, name)


def save_udi_bars(
    lux: np.ndarray, name: str, occupied: np.ndarray | None = None,
    bins=(100, 300, 3000), title: str = "",
) -> Path:
    """Useful Daylight Illuminance: stacked bars, one per sensor row.

    lux: (nsensors, ntimesteps). Bins are <100 / 100-300 / 300-3000 / >3000 lux.
    """
    lux = np.asarray(lux)
    if occupied is not None:
        lux = lux[:, occupied]
    lo, mid, hi = bins
    frac = np.stack([
        (lux < lo).mean(1),
        ((lux >= lo) & (lux < mid)).mean(1),
        ((lux >= mid) & (lux < hi)).mean(1),
        (lux >= hi).mean(1),
    ]) * 100

    names = [f"< {lo}", f"{lo}-{mid}", f"{mid}-{hi}", f"> {hi}"]
    colors = ["#2b3a55", "#4c8dae", "#7fbf7b", "#d94801"]
    x = np.arange(lux.shape[0])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    bottom = np.zeros_like(x, dtype=float)
    for f, n, c in zip(frac, names, colors):
        ax.bar(x, f, bottom=bottom, color=c, label=f"{n} lux", width=1.0)
        bottom += f
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("sensor index  (0 = nearest the window)")
    ax.set_ylabel("% of occupied hours")
    ax.set_title(title or "Useful Daylight Illuminance")
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.06), frameon=False)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Spectral plots
# --------------------------------------------------------------------------
def save_band_grid(cube: np.ndarray, wavelengths, name: str, title: str = "") -> Path:
    """Show every band of a hyperspectral render as its own small image.

    All panels share one LOG colour scale, so you can compare bands against
    each other. Log is essential here: the window is thousands of times
    brighter than the back wall, and on a linear scale the room is just black.
    """
    n = cube.shape[2]
    ncol = int(np.ceil(np.sqrt(n)))          # 9 bands -> a tidy 3x3
    nrow = int(np.ceil(n / ncol))
    vmax = float(np.percentile(cube, 99.9))
    vmin = max(float(np.percentile(cube[cube > 0], 5)), vmax / 1e4)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.5 * nrow), squeeze=False)
    im = None
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        ax.set_axis_off()
        if k < n:
            im = ax.imshow(np.clip(cube[:, :, k], vmin, vmax), cmap="magma", norm=norm)
            ax.set_title(f"{wavelengths[k]:.0f} nm", fontsize=9)
    fig.suptitle(title or "per-band radiance", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label="spectral radiance [W/sr/m$^2$/band]")
    return _save(fig, name)


def save_spectrum(
    wavelengths, spectra: dict[str, np.ndarray], name: str, title: str = "",
    ylabel: str = "radiance [W/sr/m$^2$/band]",
) -> Path:
    """Spectral curves for a handful of picked pixels."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for lbl, s in spectra.items():
        ax.plot(wavelengths, s, marker="o", ms=3.5, lw=1.4, label=lbl)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "spectral radiance at selected pixels")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, name)


# --------------------------------------------------------------------------
def _save(fig, name: str) -> Path:
    path = OUT / (name if name.endswith(".png") else name + ".png")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"     wrote {path.relative_to(OUT.parent)}")
    return path
