from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import cv2
import numpy as np
import tifffile

from .io import read_linear
from .models import CalibrationOptions


@lru_cache(maxsize=8)
def _calibration_image(path: str):
    return read_linear(path)[0] if path else None


def apply_capture_calibration(image: np.ndarray, options: CalibrationOptions):
    """Apply dark subtraction, flat division, then a characterized RGB matrix."""
    out=np.asarray(image,np.float32).copy(); warnings=[]
    dark=_calibration_image(options.dark_frame_path) if options.dark_frame_path else None
    flat=_calibration_image(options.flat_field_path) if options.flat_field_path else None
    if dark is not None:
        if dark.shape!=out.shape: dark=cv2.resize(dark,(out.shape[1],out.shape[0]),interpolation=cv2.INTER_AREA)
        out=np.maximum(out-dark,0)
    if flat is not None:
        if flat.shape!=out.shape: flat=cv2.resize(flat,(out.shape[1],out.shape[0]),interpolation=cv2.INTER_AREA)
        if dark is not None: flat=np.maximum(flat-dark,1e-6)
        smooth=np.empty_like(flat)
        sigma=max(8,min(out.shape[:2])/40)
        for c in range(3):smooth[...,c]=cv2.GaussianBlur(flat[...,c],(0,0),sigma)
        med=np.median(smooth.reshape(-1,3),axis=0);gain=med[None,None,:]/np.maximum(smooth,med[None,None,:]*.05)
        out*=np.clip(gain,.25,4)
    matrix=np.asarray(options.input_matrix,np.float32)
    if matrix.shape==(3,3): out=np.einsum("...c,dc->...d",out,matrix)
    else:warnings.append("Input calibration matrix was invalid and skipped")
    return np.clip(out,0,1),warnings


def flat_field_diagnostics(flat: np.ndarray, dark: np.ndarray | None = None):
    work=np.maximum(flat-(dark if dark is not None else 0),0); means=np.mean(work,axis=(0,1));
    corner=np.mean(np.concatenate((work[:max(1,work.shape[0]//10)].reshape(-1,3),work[-max(1,work.shape[0]//10):].reshape(-1,3))),axis=0)
    return {"channel_means":means.tolist(),"corner_to_mean":(corner/np.maximum(means,1e-6)).tolist(),"clipped_high":float(np.mean(flat>=.999)),"clipped_low":float(np.mean(flat<=1/65535))}


def load_dcp_matrix(path: str):
    """Approximate camera-to-linear-sRGB from a TIFF-based DCP ColorMatrix.

    DNG ColorMatrix maps XYZ(D50) to camera coordinates, so it is inverted and
    chromatically adapted before use. Full DCP hue/saturation/look tables are not
    silently approximated here.
    """
    with tifffile.TiffFile(path) as tif:
        tags=tif.pages[0].tags
        for code,name in ((50721,"ColorMatrix1"),(50722,"ColorMatrix2")):
            tag=tags.get(code)
            if tag is None:continue
            values=np.asarray(tag.value,dtype=np.float64).reshape(-1)
            if values.size==9:
                xyz_to_camera=values.reshape(3,3)
                camera_to_xyz_d50=np.linalg.inv(xyz_to_camera)
                # Bradford D50 → D65, then XYZ(D65) → linear sRGB.
                d50_to_d65=np.array([[.9555766,-.0230393,.0631636],[-.0282895,1.0099416,.0210077],[.0122982,-.0204830,1.3299098]])
                xyz_to_srgb=np.array([[3.2404542,-1.5371385,-.4985314],[-.9692660,1.8760108,.0415560],[.0556434,-.2040259,1.0572252]])
                matrix=xyz_to_srgb@d50_to_d65@camera_to_xyz_d50
                # Normalize neutral response to avoid profile-dependent exposure jumps.
                neutral=matrix@np.ones(3);matrix=matrix/max(float(np.mean(neutral)),1e-6)
                return matrix.tolist(),name+" (matrix-only approximation)"
    raise ValueError("The DCP does not contain a readable 3×3 ColorMatrix1 or ColorMatrix2")


def load_matrix_json(path: str):
    import json
    data=json.loads(Path(path).read_text(encoding="utf-8"));matrix=np.asarray(data.get("matrix",data),float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all():raise ValueError("Profile must contain a finite 3×3 matrix")
    return matrix.tolist(),str(data.get("name",Path(path).stem)) if isinstance(data,dict) else Path(path).stem


def fit_it8_matrix(measured_rgb, reference_rgb):
    measured=np.asarray(measured_rgb,float);reference=np.asarray(reference_rgb,float)
    if measured.shape!=reference.shape or measured.ndim!=2 or measured.shape[1]!=3 or len(measured)<6:raise ValueError("IT8 fit needs at least six paired RGB patches")
    matrix=np.linalg.lstsq(measured,reference,rcond=None)[0].T
    predicted=measured@matrix.T;rmse=float(np.sqrt(np.mean((predicted-reference)**2)))
    return matrix.tolist(),{"patches":len(measured),"rmse":rmse,"max_error":float(np.max(np.abs(predicted-reference)))}


def parse_it8_pairs(path: str):
    """Parse simple CGATS rows containing RGB_R/G/B and LAB or XYZ-derived RGB reference columns."""
    lines=Path(path).read_text(encoding="utf-8",errors="replace").splitlines();fields=[];rows=[];inside=False
    for raw in lines:
        line=raw.strip()
        if line=="BEGIN_DATA_FORMAT":inside="format";continue
        if line=="END_DATA_FORMAT":inside=False;continue
        if line=="BEGIN_DATA":inside="data";continue
        if line=="END_DATA":inside=False;continue
        if inside=="format":fields.extend(line.split())
        elif inside=="data":rows.append(line.split())
    required=("RGB_R","RGB_G","RGB_B")
    if not all(x in fields for x in required):raise ValueError("CGATS file needs RGB_R, RGB_G, and RGB_B columns")
    # Accept reference columns named REF_R/G/B; this avoids pretending Lab-to-RGB
    # conversion is valid without the target illuminant and observer metadata.
    refs=("REF_R","REF_G","REF_B")
    if not all(x in fields for x in refs):raise ValueError("CGATS file needs explicit REF_R, REF_G, and REF_B reference columns")
    idx=lambda names:[fields.index(x) for x in names];mi,ri=idx(required),idx(refs)
    measured=[];reference=[]
    for row in rows:
        try:measured.append([float(row[i]) for i in mi]);reference.append([float(row[i]) for i in ri])
        except (ValueError,IndexError):continue
    scale=max(np.max(measured),np.max(reference),1);return np.asarray(measured)/scale,np.asarray(reference)/scale


def transform_output(image: np.ndarray, options: CalibrationOptions):
    """Encode for export. Custom ICC uses Pillow/LittleCMS compatibility conversion."""
    linear=np.clip(image,0,1)
    rgb=np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055).astype(np.float32)
    space=options.output_space;icc=None;warnings=[]
    if options.output_icc_path:
        try:
            from PIL import Image,ImageCms
            srgb=ImageCms.createProfile("sRGB");target=ImageCms.getOpenProfile(options.output_icc_path)
            pil=Image.fromarray(np.rint(rgb*255).astype(np.uint8),"RGB");converted=ImageCms.profileToProfile(pil,srgb,target,outputMode="RGB")
            rgb=np.asarray(converted,np.float32)/255;icc=Path(options.output_icc_path).read_bytes();warnings.append("Custom ICC compatibility transform used an 8-bit LittleCMS path")
        except Exception as exc:warnings.append(f"Custom ICC transform failed: {exc}")
    elif space=="sRGB":
        try:
            from PIL import ImageCms
            icc=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        except Exception as exc:warnings.append(f"sRGB ICC generation failed: {exc}")
    return rgb,icc,warnings
