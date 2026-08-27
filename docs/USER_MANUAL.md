# Negative Lab User Manual

## 1. Purpose

Negative Lab converts color-negative scans into consistent positive images. Its
recommended workflow uses physical references captured with the roll rather than
choosing different histogram endpoints for every photograph.

## 2. Preparing scans

Use the same manual exposure, aperture, light source, camera/scanner settings, and
color-correction settings for every frame in the session. Disable automatic exposure,
brightness, contrast, and color correction. Sixteen-bit linear TIFF or RAW provides
the best density headroom. Include clear unexposed film base and a uniformly dense,
fully exposed leader or tail.

## 3. Starting a roll

Use **Import** or **Folder** to load one roll. Yellow frame names have warnings;
red names could not be decoded. Keep different film rolls and scanning sessions in
separate `.nlroll.json` projects.

## 4. Sampling anchors

Open a frame containing clear film base, select **Sample Clear Film Base**, and click
a clean area. Then open a dense leader/tail frame, select **Sample Dense Leader**, and
click a uniform area. The radius determines how large a region is summarized. The
application uses a trimmed median and reports variation, reducing sensitivity to
one dust particle or defective pixel.

Review Anchor Quality before conversion. Resample when a channel is clipped, the
clear base is not brighter than the dense leader, density separation is weak, or
the selected region is uneven.

## 5. Film profiles and preview

The default Custom two-point profile is the safest starting point. Other profiles
apply restrained per-channel density slopes. Profiles are starting interpretations,
not laboratory measurements. Save a custom profile only after validating it across
multiple frames and neutral references.

The Develop tab stores adjustments independently for each frame. The source negative
is never overwritten. Enable **Show scanned negative** to compare with the input.
Use Ctrl+mouse-wheel to zoom and Fit to restore the whole-image view.

## 6. Automatic framing

Automatic framing estimates the exposed image rectangle inside dark holder or film
borders. Apply it to the current frame first and inspect the result before processing
the whole roll. Very bright borders, panoramic frames, or irregular masks may require
manual crop support in a future release.

## 7. Synchronization

Keep the source frame current and select targets using Ctrl/Shift-click. Choose Tone,
Color, or Geometry and select **Copy Current Groups to Selected**. Unselected groups
remain untouched.

## 8. Saving and converting

Save the roll project before a long session. **Convert Roll to 16-bit TIFF** performs
full-resolution conversion in a background thread and writes a JSON report beside
the results. Failed frames are recorded in the report without corrupting source files.

## 9. Diagnostics

The Diagnostics tab records imports, source dimensions and bit depth, clipping,
anchors and sample variation, frame-boundary confidence, synchronization, batch
progress, errors, and output results. Save the log when reporting a problem.

## 10. Limitations

## 11. Calibration workspace

Capture a dark frame with the lens covered at matching exposure, gain/ISO, and sensor
temperature. Capture a flat field of the bare, evenly illuminated scanning light
through the same lens, aperture, focus position, and holder. Choose the dark first and
the flat second. The application subtracts the dark from both scan and flat, smooths
the flat to avoid copying its noise, and divides out spatial illumination variation.

Changing a dark, flat, or input color profile invalidates existing density anchors.
This is intentional: anchors must be sampled from the newly calibrated pixel values.

Matrix JSON profiles contain a name and a 3×3 camera/scanner-to-working-RGB matrix.
DCP import reads ColorMatrix1 or ColorMatrix2 and performs a matrix-only D50-to-D65,
linear-sRGB approximation. DCP look tables and hue/saturation maps are not applied.
IT8/CGATS fitting currently requires measured `RGB_R/G/B` and explicit reference
`REF_R/G/B` columns. It reports patch count, RMS error, and maximum error.

The default output is sRGB with an embedded generated sRGB profile. A user ICC can be
selected for compatibility export; the Diagnostics report warns that this LittleCMS
path is currently 8-bit internally. Keep 16-bit sRGB TIFF as the archival default
until the custom high-bit-depth ICC path is completed.

## 12. Restoration workspace

- **Fading correction** moves channel medians toward neutrality and restores restrained
  contrast. It is interpretive and should be compared with known scene references.
- **Dust removal** identifies compact local outliers and uses surrounding pixels to
  fill them. Review fine texture at 100%; large defects need manual restoration.
- **Sprocket/holder mask** suppresses extreme border-connected components. It can
  mistake bright lamps or deep shadows touching an edge for film artifacts.
- **Local correction** stores a radial exposure dab at the clicked image position.
  Corrections are nondestructive and saved in the frame recipe.
- **Compare Film Profiles** renders the current frame through every built-in profile.
- **Trichrome** expects exactly three monochrome records selected in red, green, blue
  order. ECC affine registration aligns them before channel assembly.

## 13. Remaining limitations

Manual crop handles, full curves, infrared scanner dust channels, complete DCP look
tables, high-bit-depth arbitrary ICC transforms, and GPU processing remain future
work. Calibration improves consistency but does not turn an unmeasured camera-scanning
system into a certified colorimeter.

## 14. Performance and output

The Performance & Output tab estimates peak memory for the current full-resolution
frame and compares it with available RAM. It warns before conversion when the configured
percentage is exceeded. The estimate is conservative and decoder caches can add usage.

- **Preview longest side** affects interactive display only; exports remain full size
  unless Resize says otherwise.
- **CPU threads** configures OpenCV. Automatic lets OpenCV choose.
- **OpenCL** requests acceleration only for compatible OpenCV operations. Diagnostics
  distinguish requested, available, and actually enabled states. NumPy stays on CPU.
- **Disk-backed preview cache** stores calibrated previews in the system temporary
  directory. Clear it after changing calibration files outside Negative Lab.
- **Continue after a failed frame** records an error and proceeds with the roll.

Archival produces full-size 16-bit TIFF. Web produces a 50% JPEG. Proof produces a
full-size JPEG. Custom exposes format, bit depth, resize, and overwrite behavior.
Overwrite is disabled by default so a second conversion cannot silently replace work.

## 15. Recovery and release operations

Negative Lab writes an automatic recovery project every 30 seconds while a project
has unsaved changes. At the next launch it offers to restore that project. Recovery
contains paths and settings, not source pixels. Normal saves populate Recent Projects.

Images, folders, and `.nlroll.json` projects can be dragged onto the main window.
The Operations tab records important calibration, synchronization, framing, batch,
warning, and error events. A diagnostic ZIP contains project settings, system facts,
operation history, and the Diagnostics log. It excludes source pixels, although local
file paths and metadata may be present.

## 16. Portable build

Create `.venv`, install `requirements-dev.txt` plus optional RAW requirements, and
run `build_portable.bat`. The portable folder appears under `dist\NegativeLab`.
Test RAW decoding, ICC export, and a representative roll on the target computer.

## 17. Digital Darkroom

The Digital Darkroom tab contains nondestructive geometry, curves, clipping, film
character, restoration, and capture-analysis tools. Blue and red clipping overlays
are diagnostic display colors and never become part of normal exports.

Automatic Anchor Discovery scans reduced-resolution copies for uniform bright and
dark regions. Inspect its candidates: a pale subject or dark photograph can resemble
film leader. Roll Consistency reports suggestions without normalizing intentional
lighting automatically.

Infrared cleanup requires a matching infrared scan and is unsuitable for many
silver-rich black-and-white films. Multi-exposure fusion expects bracketed scans of
one stationary negative and records alignment scores in Diagnostics.

## 18. Virtual Light Table and archiving

The Virtual Light Table renders the roll as positive thumbnails. Contact sheets are
presentation files. Archival manifests store the project recipe and SHA-256 checksums
of every available source; checksums detect changes but do not replace backups.

Select a thumbnail to assign zero to five stars, a color label, a rejection flag,
and searchable production notes. Choose **Apply Metadata** before selecting another
frame. Double-clicking a thumbnail returns directly to that frame in the main Develop
view. All culling metadata is stored in the `.nlroll.json` project.

Use the minimum-rating selector to isolate stronger frames and turn off **Show
rejected frames** while making selects. Filtering changes only the Light Table view;
it never deletes a source or removes a frame from the project.

## 20. Visual mask editing

Choose an operation and place a brush mask with one click, or choose the gradient
tool and click its start and end points. **Show cyan mask overlay** displays the
combined active mask without changing normal output. The mask selector can disable
or delete one mask without disturbing the others. Disabled masks remain saved in
the project and can be restored later.

## 21. Preview and inspection

Use Fit for composition and 100%, 200%, or 400% for critical inspection. Moving the
pointer over the preview reports its displayed pixel coordinates and normalized
linear RGB values. **Before / After** opens the calibrated scanned negative and the
current conversion side by side. **Full Screen** provides a distraction-free view;
press Escape to close it.

The RGB Histogram displays red, green, and blue channel distributions. Its clipping
summary uses the same conservative thresholds as the preview warning overlay. A
histogram can reveal clipping and color imbalance, but it cannot determine whether
the intended photograph should be bright, dark, or strongly colored.

## 22. Archive and proof workflow

Contact sheets can be written as JPEG, TIFF, or printable PDF. Enable
**accepted frames only** to omit rejected frames without removing them from the roll.
The PDF uses the project DPI metadata and includes frame numbers, filenames, ratings,
and the current positive previews.

The CSV roll catalog records every frame's source path, rating, color label,
rejection state, notes, and warnings. Spreadsheet programs can open this UTF-8 file.
The archival verifier recalculates every SHA-256 checksum in a previously exported
manifest and reports intact, changed, and missing sources. Verification is read-only;
it never repairs, moves, or deletes files.

## 19. Camera-Scanning Assistant

The assistant reports sharpness, nine-zone illumination uniformity,
corner-to-center brightness, and clipping percentages. Comparisons are most useful
with the same camera, lens, aperture, carrier, and negative type.
