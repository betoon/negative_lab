# Negative Lab Developer Manual

## Digital Darkroom architecture (0.4)

`digital_darkroom.py` contains side-effect-free NumPy/OpenCV operations for
perspective rectification, curves, clipping diagnostics, mask construction,
healing/cloning, anchor discovery, consistency analysis, infrared cleanup,
exposure fusion, film character, contact sheets, capture assessment, and archival
manifests. `NegativeRecipe` stores geometry, curves, mask specifications, and film
character settings, preserving backward compatibility through filtered dataclass
loading. Both previews and exports pass through the same `apply_recipe` pipeline.

Masks are persisted as compact normalized vector descriptions rather than pixel
bitmaps. This keeps roll projects portable and allows masks to scale with output
resolution. Healing is content-aware OpenCV inpainting; cloning uses a stored source
offset. Automatic anchor and consistency calculations operate on proxies in the GUI,
while the functions themselves accept arbitrary float RGB arrays for testability.

## Architecture

The application separates the PySide6 interface from the numerical core:

- `models.py`: versioned roll, frame, anchor, profile, and recipe dataclasses.
- `io.py`: linear RAW and conventional image input plus high-bit-depth output.
- `conversion.py`: robust sampling, anchor diagnostics, density conversion, and recipes.
- `workflow.py`: synchronization, framing, analysis, profiles, and batch processing.
- `calibration.py`: dark/flat correction, DCP/matrix import, IT8 fitting, and output ICC.
- `restoration.py`: fading, dust, sprocket masks, local corrections, and trichrome.
- `app.py`: PhotoLab-style interface, tools, persistence, worker thread, and diagnostics.

## Density convention

Clear unexposed film base has high scanner transmission and maps to black in the
positive. Dense fully exposed leader has low transmission and maps to white. For
channel `c`, density is `log10(base[c] / pixel[c])`; two-point output divides by
`log10(base[c] / dense[c])`. Inputs are clamped away from zero and output is clipped
only after the density/profile operation.

The explicit clear-base/dense-leader names avoid ambiguous photographic “black point”
and “white point” labels. Anchor samples store normalized position, radius, robust
median, spread, and source path for reproducibility.

## Precision and color

The working arrays are RGB float32. Integer inputs are normalized by their dtype
range. RAW decode disables automatic brightening and gamma and requests raw output
color with neutral multipliers. A future color-management milestone must explicitly
characterize camera/scanner input and the internal working space; wide-gamut tagging
must never be substituted for a real transform.

## Persistence

`.nlroll.json` stores all settings by value, including anchor measurements and the
selected profile. This prevents later profile-library changes from silently altering
an existing roll. Unknown recipe fields are ignored for forward compatibility.

## Background work and errors

Batch conversion runs in a QObject moved to QThread. Widget changes remain in the GUI
thread. Each frame is independently guarded and recorded as successful or failed.
Anchor invalidity stops the batch before files are written.

## Testing

Run `pytest`. Tests cover robust anchor sampling, conversion endpoints, serialization,
selective synchronization, frame detection, bounded recipe output, 16-bit TIFF batch
output, and report generation. Future calibrated color work should add reference
targets, delta-E evaluation, CPU/GPU equivalence, preview/export equivalence, and
golden-output tests.

## Licensing boundary

This is an original implementation. Do not copy FreeCCR AGPL source, kernels, assets,
or interface code into this project without first making an explicit licensing
decision. General photographic mathematics and independently designed workflows may
be implemented with original code and tests.

## Calibration and managed output

Capture calibration precedes anchor sampling and negative conversion. Dark subtraction
is clamped at zero. Flat correction uses a broad Gaussian-smoothed, dark-subtracted
field normalized by its channel medians; gain is bounded to prevent division artifacts.
The input 3×3 matrix is then applied in float32. Changing any of these inputs invalidates
previous anchors in the GUI.

DNG/DCP ColorMatrix maps XYZ D50 to camera coordinates. The matrix-only importer
inverts it, applies Bradford D50-to-D65 adaptation, converts XYZ to linear sRGB, and
normalizes neutral exposure. It deliberately ignores DCP look and hue/saturation tables.
IT8 fitting is least squares over paired measured/reference RGB rows and exposes its
error statistics rather than claiming a valid fit from labels alone.

Built-in sRGB export generates and embeds an ICC profile. Arbitrary ICC export uses
Pillow/LittleCMS and currently quantizes through RGB8; the processing report records
that limitation. The scene-linear result receives the standard piecewise sRGB OETF
for preview/export before ICC handling. TIFF output stores ICC tag 34675 and physical
resolution tags, keeping the displayed preview and default export encoding consistent.

## Restoration ordering

Restoration runs after the primary positive adjustments: fading correction, dust
inpainting, local radial adjustments, and optional border artifact masking. These are
interpretive stages and remain per-frame recipe fields. Trichrome is a separate explicit
export that registers grayscale records to the green reference using affine ECC and
records every matrix and score.

## Performance and release operations

`estimate_frame_memory()` models several float32 RGB working copies plus masks and is
intentionally conservative. `configure_performance()` sets OpenCV's thread count and
OpenCL preference, then reports requested/available/enabled states. It does not claim
that NumPy density or recipe stages use the GPU.

`DiskPreviewCache` hashes the resolved source path, modification time, and complete
calibration settings. Cached `.npy` arrays are memory-mapped. The cache contains
calibrated source pixels and remains in a local temporary directory; diagnostic ZIPs
never include it.

`OutputOptions` stores presets and explicit output behavior. Existing-output protection
raises a per-frame failure unless overwrite was enabled. Web resizing uses area
interpolation. Failed-frame continuation belongs to `PerformanceOptions`.

QSettings stores recent project paths. Periodic recovery uses the normal versioned
RollProject JSON schema. Diagnostic ZIPs contain JSON state, system facts, history,
and logs but no source pixels. The close handler cancels and waits for an active worker
before destroying its QThread.
