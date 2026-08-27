# Negative Lab Packaging

## Windows portable build

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -r requirements-optional.txt
build_portable.bat
```

The folder `dist\NegativeLab` is portable. Test it on a clean Windows account before
release. The build script detects whether rawpy is installed and bundles it only when
available.

## Release checklist

- Run `pytest` and the GUI smoke test.
- Import TIFF, JPEG, and representative RAW camera formats.
- Verify dark/flat calibration and resample anchors afterward.
- Export archival TIFF, Web JPEG, embedded sRGB, and a custom ICC proof.
- Confirm cancel, failed-frame recovery, overwrite protection, and recovery restore.
- Inspect the diagnostic package to ensure no source pixels are included.
- Check the portable build on a computer without Python installed.
