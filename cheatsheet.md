# pyradiance cheat sheet

## The one convention

Almost every pyradiance function **accepts a path or bytes, and returns bytes**.

```python
sky   = pr.gensky(datetime(2024, 3, 21, 13), latitude=40.8, longitude=74.0, timezone=75)
octree = pr.oconv("materials.rad", "room.rad", stdin=sky)   # bytes in
img   = pr.rpict(view_args, "office.oct", xres=800, yres=600)
open("out.hdr", "wb").write(img)                            # bytes out
```

Nothing is written to disk unless you write it. This lets you pipe stages
together in memory, exactly like the shell pipelines in the Radiance manual.

## Where things live

```python
pr.BINPATH              # .../site-packages/pyradiance/bin   (91 executables)
pr.BINPATH.parent/"lib" # .../lib  (127 .cal/.dat files, on RAYPATH)
```

pyradiance sets `RAYPATH` and prepends `BINPATH` to `PATH` at import, so a
system Radiance install will not interfere.

Useful `.cal` files that ship on RAYPATH: `reinhartb.cal`, `klems_full.cal`,
`macbeth.cal`, `metals.cal`, `stdrefl.cal`, `cieresp.cal`, `noise.cal`,
`perezlum.cal`, `disk2square.cal`.

## Numbers worth memorising

```python
LUM = [47.4, 119.9, 11.6]     # RGB radiance -> luminance (cd/m2)
                              # RGB irradiance -> illuminance (lux)
```

## Core recipes

**HDR to numpy**
```python
xres, yres = pr.get_image_dimensions(hdr)          # returns (WIDTH, HEIGHT)
raw = pr.pvalue(hdr, header=False, resstr=False, outform="f")
arr = np.frombuffer(raw, np.single).reshape(yres, xres, 3)   # rows first!
```

**Illuminance on a sensor grid**
```python
rays = b"1.0 2.0 0.8 0 0 1\n"                      # x y z  dx dy dz
raw  = pr.rtrace(rays, "office.oct", irradiance=True, outform="f",
                 outspec="v", nproc=8, header=False, params=["-ab","3","-ad","1024"])
lux  = np.frombuffer(raw, np.single).reshape(-1,3) @ LUM
```

**Annual, two-phase**
```python
D   = pr.rfluxmtx(receiver="skyrecv.rad", rays=points, octree="nosky.oct",
                  params=["-I+","-y","90","-ab","3","-faf"])
S   = pr.gendaymtx("weather.epw", mfactor=1, outform="d")     # EPW read directly
out = pr.dctimestep(dc_path, S, outform="d")
body = pr.getinfo(out, strip_header=True)                     # drop the text header
```

**Matrix files** are a text header, a blank line, then raw numbers. Always
`pr.getinfo(..., strip_header=True)` before `np.frombuffer`.

**Hyperspectral image** (rpict cannot do this)
```python
d = pr.vwrays(view=vargs, dimensions=True, xres=X, yres=Y).decode().split()
X, Y = int(d[1]), int(d[3])                       # vwrays prints "-x W -y H"
rays = pr.vwrays(view=vargs, outform="f", xres=X, yres=Y)
cube = np.frombuffer(pr.rtrace(rays, oct, inform="f", outform="f", outspec="v",
        params=["-ab","2","-co+","-cs","9","-cw","390","770"]),
        np.single).reshape(Y, X, 9)
```
Bands come back **longest wavelength first**.

## Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `-cs 9` silently gives 3 channels | `rpict` ignores spectral flags | Use `vwrays \| rtrace -co+ -cs N` |
| `rpict: command line error at '-co+'` | `-co+` is rtrace-only | Drop it from rpict calls |
| `fatal - unexpected EOF in header` from `dctimestep` | You passed `header=False` to `gendaymtx` | dctimestep needs the headers to get dimensions |
| `gensky() got multiple values for 'latitude'` | First positional arg is a `datetime`, not month/day/hour | `pr.gensky(datetime(...), latitude=...)` |
| `Scene(sid)` raises `ValueError` on a path | `Scene` builds both `{sid}.oct` and `m{sid}.oct` in the CWD | Use a bare scene id and `os.chdir()` |
| Sensor reads 0 lux | It is buried inside furniture | Cast a ray up with `outspec="L"` and check the distance |
| `AttributeError: 'xyRGB' has no attribute 'r'` | `pextrem` returns uppercase `.R .G .B` | Use `.R` |

## Patched in this workshop's pyradiance

The vendored copy in `.venv/` carries fixes for a batch of upstream bugs. If you
`pip install pyradiance` fresh you will meet them again:

- `pr.render()` swapped width and height (parsed `vwrays -d` with `getinfo -d`
  token order), forced `-ab 0` over the quality preset, shelled out to POSIX
  `sort`, and leaked its temp OPT file.
- `getbbox()` silently dropped `Primitive` inputs, so a `Scene` built from
  `Primitive` objects gave `rad: bad value for variable 'ZONE'`.
- `get_image_dimensions()` raised `ValueError` for a path input.
- `getinfo(strip_header=True)` was ignored for path inputs.
- `pcond()` refused bytes; `pcomb(header=)` was inverted; `pcompos(ncols=)`
  dropped every input picture.
- `rtrace(trace_exclude=)`, `rmtxop(reflectance=)`, `bsdf2klems(maxlobes=)` and
  the `.sir` path of `bsdf2ttree()` all built invalid command lines.
- `RAYPATH` was overwritten at import instead of prepended.

## Not bundled

Seven wrappers point at binaries that are not shipped: `falsecolor`,
`rcode_depth`, `rcode_ident`, `rcode_norm`, `rcollate`, `rsensor`, `vwright`.
Also absent: `genskyvec`, `genBSDF`, `objview`. Calling them now raises a
`FileNotFoundError` that says so. Use matplotlib for falsecolour; use
`gendaymtx` where a tutorial calls for `genskyvec`.

## Platform notes

- The in-process `RtraceSimulManager` / `RcontribSimulManager` classes are
  **POSIX only** — they are not built on Windows.
- `pr.render()` forces `nproc=1` on Windows. `rpict` is single-threaded
  everywhere, so the workshop uses it for consistent behaviour.
- `rtrace`, `rfluxmtx` and `rcontrib` do take `nproc` / `-n` on all platforms.

## Ray parameters, briefly

| Flag | Meaning | Workshop values |
|---|---|---|
| `-ab` | diffuse interreflection bounces | 0 draft, 3 normal, 5+ real work |
| `-ad` | ambient divisions (samples per bounce) | 512–4096 |
| `-as` | super-samples in high-variance directions | ~ `-ad`/4 |
| `-aa` | ambient accuracy; 0 disables interpolation | 0.1–0.2 |
| `-ar` | ambient resolution | 64–512 |
| `-lw` | lowest ray weight worth following | 1e-3 … 1e-5 |
| `-ps` | pixel sample spacing; 1 = every pixel | 1 for fine geometry |
| `-I+` | return irradiance, not radiance | sensor grids |
