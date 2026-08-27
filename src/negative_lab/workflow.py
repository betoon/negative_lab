from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import cv2
import numpy as np

from .conversion import anchor_diagnostics, apply_recipe, convert_negative
from .calibration import apply_capture_calibration, transform_output
from .io import read_linear, write_image
from .models import FilmProfile, NegativeRecipe, RollProject
from .performance import configure_performance, estimate_frame_memory

SYNC_GROUPS = {
    "Tone": ("exposure","contrast","shadows","highlights","gamma"),
    "Color": ("temperature","tint","saturation"),
    "Geometry": ("crop","rotation"),
    "Everything": tuple(asdict(NegativeRecipe()).keys()),
}


def sync_recipes(project: RollProject, source_index: int, targets, groups):
    source=project.frames[source_index].recipe
    keys=set(k for group in groups for k in SYNC_GROUPS[group])
    for index in targets:
        for key in keys: setattr(project.frames[index].recipe,key,getattr(source,key))


def detect_frame_bounds(image: np.ndarray):
    """Estimate the exposed image rectangle inside dark holder/border regions."""
    gray=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    smooth=cv2.GaussianBlur(gray,(0,0),max(2,min(gray.shape)/300))
    dark=float(np.percentile(smooth,5)); bright=float(np.percentile(smooth,90))
    threshold=dark+(bright-dark)*.25
    mask=(smooth>threshold).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return [0,0,1,1],0.0
    contour=max(contours,key=cv2.contourArea); x,y,w,h=cv2.boundingRect(contour); ih,iw=gray.shape
    confidence=float(cv2.contourArea(contour)/max(w*h,1))
    return [x/iw,y/ih,w/iw,h/ih],confidence


def analyze_frame(image, metadata):
    warnings=[]; low=float(np.mean(image<=1/65535)); high=float(np.mean(image>=.999))
    if high>.001:warnings.append(f"{high*100:.2f}% of channel values are clipped high")
    if low>.001:warnings.append(f"{low*100:.2f}% of channel values are clipped low")
    if metadata.get("bit_depth",16)<14:warnings.append("Input has less than 14-bit precision; density headroom is limited")
    return {"dimensions":[image.shape[1],image.shape[0]],"bit_depth":metadata.get("bit_depth"),"clipped_low":low,"clipped_high":high,"warnings":warnings}


def save_profile(profile: FilmProfile, path): Path(path).write_text(json.dumps(asdict(profile),indent=2),encoding="utf-8")
def load_profile(path): return FilmProfile(**json.loads(Path(path).read_text(encoding="utf-8")))


def batch_convert(project: RollProject, output_directory: str, bit_depth=None, progress=None, cancelled=None):
    metrics,warnings=anchor_diagnostics(project.clear_base,project.dense_leader)
    if not metrics.get("valid"): raise ValueError("Invalid roll anchors: "+" ".join(warnings))
    acceleration=configure_performance(project.performance);output=Path(output_directory);output.mkdir(parents=True,exist_ok=True);report={"project":project.name,"anchors":metrics,"warnings":warnings,"performance":acceleration,"frames":[]}
    bit_depth=int(bit_depth or project.output.bit_depth);extension=".jpg" if project.output.format.upper()=="JPEG" else ".tif"
    active=[f for f in project.frames if f.recipe.enabled]
    for i,frame in enumerate(active):
        if cancelled and cancelled(): raise InterruptedError("Batch conversion cancelled")
        if progress: progress(int(i*100/max(len(active),1)),f"Converting {Path(frame.path).name}")
        try:
            image,meta=read_linear(frame.path);image,calibration_warnings=apply_capture_calibration(image,project.calibration);result=apply_recipe(convert_negative(image,project.clear_base,project.dense_leader,project.film_profile,project.mode),frame.recipe);result,icc,color_warnings=transform_output(result,project.calibration)
            if project.output.resize_percent!=100:
                scale=project.output.resize_percent/100;result=cv2.resize(result,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LANCZOS4)
            path=output/(Path(frame.path).stem+project.output.filename_suffix+extension)
            if path.exists() and not project.output.overwrite_existing:raise FileExistsError(f"Output exists and overwrite is disabled: {path}")
            write_image(str(path),result,bit_depth,icc if project.calibration.embed_icc else None,project.calibration.dpi)
            report["frames"].append({"source":frame.path,"output":str(path),"analysis":analyze_frame(image,meta),"calibration_warnings":calibration_warnings,"color_warnings":color_warnings,"status":"ok"})
        except Exception as exc:
            report["frames"].append({"source":frame.path,"status":"failed","error":str(exc)})
            if not project.performance.recover_failed_frames:raise
    if progress:progress(100,"Complete")
    report_path=output/(project.name.replace(" ","_")+"_conversion_report.json"); report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
