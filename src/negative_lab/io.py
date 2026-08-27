from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import tifffile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
                    ".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf"}
RAW_EXTENSIONS = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf"}


def _to_float(array):
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    maximum = float(np.nanmax(array)) if array.size else 1.0
    return np.clip(array.astype(np.float32) / (maximum if maximum > 1.5 else 1.0), 0, 1)


def read_linear(path: str):
    """Read RGB float32. RAW is decoded without auto-brightness and with linear gamma."""
    suffix = Path(path).suffix.lower(); metadata = {"path": str(path), "source_extension": suffix}
    if suffix in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("RAW support requires rawpy. Install requirements-optional.txt.") from exc
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                                  use_camera_wb=False, user_wb=[1, 1, 1, 1],
                                  output_color=rawpy.ColorSpace.raw)
            metadata.update({"raw": True, "bit_depth": 16, "camera_white_level": int(raw.white_level)})
            return _to_float(rgb), metadata
    if suffix in {".tif", ".tiff"}:
        array = tifffile.imread(path)
        if array.ndim == 2: array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[-1] > 3: array = array[..., :3]
        metadata.update({"raw": False, "bit_depth": int(array.dtype.itemsize * 8), "dtype": str(array.dtype)})
        return _to_float(array), metadata
    data = np.fromfile(path, np.uint8); bgr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if bgr is None: raise ValueError(f"Could not decode {path}")
    if bgr.ndim == 2: bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 4: bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    else: bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    metadata.update({"raw": False, "bit_depth": int(bgr.dtype.itemsize * 8), "dtype": str(bgr.dtype)})
    return _to_float(bgr), metadata


def write_image(path: str, rgb: np.ndarray, bit_depth=16, icc_profile: bytes | None = None, dpi: int = 300):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); image = np.clip(rgb, 0, 1)
    if target.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(target, np.rint(image * (65535 if bit_depth == 16 else 255)).astype(np.uint16 if bit_depth == 16 else np.uint8),
                         photometric="rgb", resolution=(dpi,dpi), resolutionunit="INCH",
                         extratags=[(34675,"B",len(icc_profile),icc_profile,False)] if icc_profile else None,
                         metadata={"Software":"Negative Lab","OutputColorSpace":"ICC" if icc_profile else "Unprofiled RGB"})
    else:
        bgr = cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(target.suffix or ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok: raise RuntimeError(f"Encoder failed for {target}")
        encoded.tofile(str(target))
