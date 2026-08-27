from __future__ import annotations

import cv2
import numpy as np
from .models import AnchorSample, FilmProfile, NegativeRecipe


def sample_anchor(image: np.ndarray, x: float, y: float, radius: int = 9, source_path="") -> AnchorSample:
    h, w = image.shape[:2]; px=int(np.clip(x,0,1)*(w-1)); py=int(np.clip(y,0,1)*(h-1)); r=max(1,int(radius))
    patch=np.asarray(image[max(0,py-r):min(h,py+r+1),max(0,px-r):min(w,px+r+1),:3],np.float32)
    pixels=patch.reshape(-1,3); lo=np.percentile(pixels,10,axis=0); hi=np.percentile(pixels,90,axis=0)
    trimmed=np.clip(pixels,lo,hi); median=np.median(trimmed,axis=0); spread=np.median(np.abs(trimmed-median),axis=0)*1.4826
    return AnchorSample(median.tolist(),spread.tolist(),source_path,float(x),float(y),r,True)


def anchor_diagnostics(clear_base: AnchorSample, dense_leader: AnchorSample):
    warnings=[]; metrics={"valid":False,"density_range":[],"base_spread":clear_base.spread,"leader_spread":dense_leader.spread}
    if not clear_base.valid or not dense_leader.valid:
        warnings.append("Sample both clear film base and dense exposed leader."); return metrics,warnings
    base=np.asarray(clear_base.rgb,float); dense=np.asarray(dense_leader.rgb,float)
    if np.any(base <= dense): warnings.append("Clear-base transmission must exceed dense-leader transmission in every channel.")
    if np.any(base > .995): warnings.append("Clear-base sample is clipped or extremely close to clipping.")
    if np.any(dense < 1/65535): warnings.append("Dense-leader sample contains clipped black values.")
    density=np.log10(np.maximum(base,1e-6)/np.maximum(dense,1e-6)); metrics["density_range"]=density.tolist()
    if np.any(density < .25): warnings.append("Anchor separation is small; conversion may amplify noise and color errors.")
    if max(clear_base.spread+dense_leader.spread) > .03: warnings.append("An anchor region is uneven; sample a cleaner, larger area.")
    metrics["valid"] = not any("must exceed" in w or "Sample both" in w for w in warnings)
    metrics["quality_score"] = float(np.clip(np.min(density)/1.5,0,1) * np.clip(1-max(clear_base.spread+dense_leader.spread)*10,0,1))
    return metrics,warnings


def convert_negative(image: np.ndarray, clear_base: AnchorSample, dense_leader: AnchorSample,
                     profile: FilmProfile | None = None, mode="two_point"):
    base=np.maximum(np.asarray(clear_base.rgb,np.float32),1e-6); pixels=np.maximum(np.asarray(image,np.float32),1e-6)
    density=np.log10(base[None,None,:]/pixels)
    profile=profile or FilmProfile()
    if mode == "two_point":
        dense=np.maximum(np.asarray(dense_leader.rgb,np.float32),1e-6)
        span=np.maximum(np.log10(base/dense),1e-4); positive=density/span[None,None,:]
    else:
        positive=density*np.asarray(profile.slope_rgb,np.float32)[None,None,:]
    positive *= np.asarray(profile.orange_mask_compensation,np.float32)[None,None,:]
    return np.clip(positive,0,1).astype(np.float32)


def apply_recipe(image: np.ndarray, recipe: NegativeRecipe):
    out=np.asarray(image,np.float32).copy()
    if recipe.rotation % 4: out=np.rot90(out,recipe.rotation%4).copy()
    if recipe.crop:
        h,w=out.shape[:2]; x,y,cw,ch=recipe.crop; out=out[int(y*h):max(int((y+ch)*h),int(y*h)+1),int(x*w):max(int((x+cw)*w),int(x*w)+1)]
    out *= 2.0**recipe.exposure
    # Relative warm/cool and green/magenta controls, intentionally independent of RAW decode WB.
    temp=recipe.temperature/100.0; tint=recipe.tint/100.0
    out[...,0]*=1+temp*.25+tint*.08; out[...,1]*=1-tint*.16; out[...,2]*=1-temp*.25+tint*.08
    lum=np.mean(out,axis=2)
    shadow_mask=np.clip((.5-lum)*2,0,1)[...,None]; highlight_mask=np.clip((lum-.5)*2,0,1)[...,None]
    out += shadow_mask*(recipe.shadows/100)*.35; out += highlight_mask*(recipe.highlights/100)*.35
    out=(out-.5)*(1+recipe.contrast/100)+.5
    gray=np.mean(out,axis=2,keepdims=True); out=gray+(out-gray)*(1+recipe.saturation/100)
    if abs(recipe.gamma-1)>1e-5: out=np.maximum(out,0)**(1/max(recipe.gamma,.05))
    from .restoration import apply_local_adjustments, correct_fading, remove_dust, sprocket_content_mask
    out=correct_fading(np.clip(out,0,1),getattr(recipe,"fade_correction",0))
    out,_=remove_dust(out,getattr(recipe,"dust_removal",0),getattr(recipe,"dust_max_radius",10))
    out=apply_local_adjustments(out,getattr(recipe,"local_adjustments",[]) or [])
    if getattr(recipe,"sprocket_mask",False):
        mask=sprocket_content_mask(out);out*=mask[...,None].astype(np.float32)/255
    return np.clip(out,0,1).astype(np.float32)


def preview_convert(image, project, recipe, max_dimension=1800):
    scale=min(1.0,max_dimension/max(image.shape[:2])); work=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else image
    return apply_recipe(convert_negative(work,project.clear_base,project.dense_leader,project.film_profile,project.mode),recipe)
