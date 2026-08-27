# Negative Lab

Negative Lab is an original, offline, PhotoLab-style desktop application for
consistent color-negative conversion. It treats a complete film roll or fixed-
exposure scanning session as one project and uses clear-film-base and dense-leader
samples to map optical density into a positive image.

## Current features

### Digital Darkroom 0.4

- Virtual Light Table with positive roll contact view
- Fine rotation, crop foundation, and perspective rectification
- Nondestructive RGB/channel curves and clipping overlays
- Radial/linear masks plus healing and cloning processing engines
- Automatic anchor discovery and roll-wide consistency analysis
- Infrared-channel dust cleanup and ECC-aligned multi-exposure fusion
- Contact sheets, SHA-256 archival manifests, Film Character Designer, and
  camera-scan quality analysis

- RAW, TIFF, PNG, JPEG, BMP, and WebP import where supporting decoders are available
- Linear/no-auto-brightness RAW decode
- Robust multi-pixel clear-base and dense-leader sampling
- Two-point logarithmic density conversion
- Film-profile library plus JSON profile import/export
- PhotoLab-style filmstrip, large preview, controls, and diagnostics
- Per-frame exposure, temperature, tint, contrast, saturation, shadows, highlights, and gamma
- Selective Tone, Color, and Geometry synchronization
- Automatic film-frame boundary detection
- Roll projects with complete anchor, profile, frame, crop, and recipe persistence
- Background batch conversion to 16-bit TIFF
- Per-frame clipping/bit-depth diagnostics and JSON processing reports
- Matching dark-frame subtraction and smooth flat-field correction
- Matrix profiles, DCP ColorMatrix import, and paired-RGB IT8/CGATS fitting
- Embedded sRGB or user-supplied output ICC profiles with DPI metadata
- Fading correction, dust detection/inpainting, and sprocket/holder masking
- Nondestructive local exposure corrections and film-profile comparison
- ECC-aligned red/green/blue trichrome assembly
- Full-resolution memory estimation and available-RAM warnings
- Configurable OpenCV CPU/OpenCL behavior with capability reporting
- Disk-backed calibrated-preview cache
- Archival, web, proof, and custom output presets
- Automatic recovery points, recent projects, and drag-and-drop
- Processing history and exportable diagnostic packages

## Install and run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-optional.txt  # recommended for RAW files
python main.py
```

## Capture requirements

Use fixed manual exposure, disable automatic scanner correction, prefer 16-bit
linear files, and keep a clear unexposed film-base region plus a fully exposed
dense leader/tail in the roll scans. Do not mix separate scan sessions in one project.

This project implements original code and does not contain FreeCCR source code.

## Portable Windows build

Install `requirements-dev.txt` in `.venv`, then run `build_portable.bat`. The result
is placed in `dist\NegativeLab`. RAW support is bundled when rawpy is installed;
otherwise the build remains usable for TIFF, PNG, and JPEG workflows.
