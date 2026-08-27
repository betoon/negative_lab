import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

from negative_lab.conversion import anchor_diagnostics, apply_recipe, convert_negative, sample_anchor
from negative_lab.io import read_linear
from negative_lab.models import AnchorSample, FrameRecord, NegativeRecipe, RollProject
from negative_lab.workflow import batch_convert, detect_frame_bounds, sync_recipes
from negative_lab.calibration import apply_capture_calibration, fit_it8_matrix, transform_output
from negative_lab.models import CalibrationOptions
from negative_lab.restoration import (apply_local_adjustments, correct_fading, detect_dust,
                                      remove_dust, sprocket_content_mask, trichrome_merge)
from negative_lab.performance import (DiskPreviewCache, configure_performance,
                                      estimate_frame_memory, human_bytes, system_diagnostics)
from negative_lab.digital_darkroom import (apply_curve, apply_mask_stack, camera_scan_assessment, combined_mask,
    clipping_overlay, contact_sheet, discover_anchor_candidates, fuse_exposures,
    infrared_clean, linear_gradient_mask, perspective_crop, radial_mask,
    rgb_histogram, roll_consistency)


def negative_gradient(h=80,w=120):
    x=np.linspace(.08,.9,w,dtype=np.float32); return np.repeat(np.repeat(x[None,:,None],h,axis=0),3,axis=2)


def anchors():
    return AnchorSample([.9,.8,.7],[.001]*3,valid=True),AnchorSample([.08,.07,.06],[.001]*3,valid=True)


def test_robust_anchor_sampling_rejects_single_outlier():
    image=np.full((80,100,3),(.8,.7,.6),np.float32);image[40,50]=1
    sample=sample_anchor(image,.5,.5,8)
    assert np.allclose(sample.rgb,[.8,.7,.6],atol=.01) and sample.valid


def test_anchor_diagnostics_and_density_endpoints():
    base,dense=anchors();metrics,warnings=anchor_diagnostics(base,dense)
    assert metrics["valid"] and metrics["quality_score"]>0 and not warnings
    image=np.array([[[.9,.8,.7],[.08,.07,.06]]],np.float32)
    out=convert_negative(image,base,dense)
    assert np.allclose(out[0,0],0,atol=1e-5) and np.allclose(out[0,1],1,atol=1e-5)


def test_recipe_and_project_round_trip(tmp_path):
    base,dense=anchors();p=RollProject("Roll 1",[FrameRecord("a.tif",NegativeRecipe(exposure=.5),rating=4,color_label="Green",notes="keeper")],base,dense)
    path=tmp_path/"roll.nlroll.json";p.save(path);loaded=RollProject.load(path)
    assert loaded.name=="Roll 1" and loaded.frames[0].recipe.exposure==.5 and loaded.clear_base.valid
    assert loaded.frames[0].rating==4 and loaded.frames[0].color_label=="Green" and loaded.frames[0].notes=="keeper"


def test_group_sync_is_selective():
    p=RollProject(frames=[FrameRecord("a",NegativeRecipe(exposure=1,temperature=20)),FrameRecord("b")])
    sync_recipes(p,0,[1],["Tone"])
    assert p.frames[1].recipe.exposure==1 and p.frames[1].recipe.temperature==0


def test_automatic_framing_finds_light_rectangle():
    image=np.zeros((100,140,3),np.float32);image[10:90,20:120]=.5
    crop,confidence=detect_frame_bounds(image)
    assert abs(crop[0]-20/140)<.03 and abs(crop[1]-10/100)<.03 and confidence>.9


def test_recipe_output_is_finite_and_bounded():
    result=apply_recipe(negative_gradient(),NegativeRecipe(exposure=.5,contrast=20,saturation=15,gamma=1.2))
    assert result.dtype==np.float32 and np.isfinite(result).all() and result.min()>=0 and result.max()<=1


def test_batch_conversion_writes_tiff_and_report(tmp_path):
    source=tmp_path/"negative.tif";tifffile.imwrite(source,(negative_gradient()*65535).astype(np.uint16),photometric="rgb")
    base,dense=anchors();project=RollProject("Test Roll",[FrameRecord(str(source))],base,dense)
    output=tmp_path/"output";report=batch_convert(project,str(output))
    restored=tifffile.imread(output/"negative_positive.tif")
    assert restored.dtype==np.uint16 and restored.shape==(80,120,3)
    assert report["frames"][0]["status"]=="ok" and (output/"Test_Roll_conversion_report.json").exists()


def test_dark_flat_and_matrix_calibration(tmp_path):
    dark=np.full((40,50,3),.05,np.float32);gradient=np.linspace(.45,.8,50,dtype=np.float32);flat=np.repeat(np.repeat(gradient[None,:,None],40,axis=0),3,axis=2);source=flat*.5+dark
    dp=tmp_path/"dark.tif";fp=tmp_path/"flat.tif";tifffile.imwrite(dp,(dark*65535).astype(np.uint16));tifffile.imwrite(fp,(flat*65535).astype(np.uint16))
    options=CalibrationOptions(str(dp),str(fp),[[1,0,0],[0,1,0],[0,0,1]])
    result,warnings=apply_capture_calibration(source,options)
    assert not warnings and result.shape==source.shape
    assert float(np.std(result[...,0]))<float(np.std((source-dark)[...,0]))*.5


def test_it8_matrix_fit_recovers_known_transform():
    rng=np.random.default_rng(3);measured=rng.uniform(.1,.9,(24,3));known=np.array([[1.05,-.03,0],[.01,.96,.02],[0,.04,.92]]);reference=measured@known.T
    matrix,stats=fit_it8_matrix(measured,reference)
    assert np.allclose(matrix,known,atol=1e-8) and stats["rmse"]<1e-10


def test_restoration_tools_and_local_adjustment():
    image=np.full((100,120,3),(.7,.45,.3),np.float32);image[50,60]=0
    faded=correct_fading(image,80);mask=detect_dust(image,90,8);clean,mask2=remove_dust(image,90,8)
    local=apply_local_adjustments(image,[{"x":.5,"y":.5,"radius":.2,"exposure":1,"hardness":.6}])
    assert np.ptp(np.median(faded.reshape(-1,3),axis=0))<np.ptp(np.median(image.reshape(-1,3),axis=0))
    assert mask.shape==mask2.shape==image.shape[:2] and clean[50,60].mean()>image[50,60].mean()
    assert local[50,60].mean()>=image[50,60].mean()


def test_sprocket_mask_and_trichrome_alignment():
    image=np.full((80,100,3),.5,np.float32);image[:8,:8]=1;mask=sprocket_content_mask(image)
    base=np.zeros((80,100,3),np.float32);cv2.circle(base,(50,40),12,(1,1,1),-1);shift=cv2.warpAffine(base,np.float32([[1,0,2],[0,1,-1]]),(100,80))
    merged,transforms=trichrome_merge(shift,base,shift)
    assert mask[2,2]==0 and mask[40,50]==255
    assert merged.shape==base.shape and len(transforms)==3


def test_srgb_output_generates_profile():
    image=np.full((20,30,3),.5,np.float32);result,icc,warnings=transform_output(image,CalibrationOptions())
    assert result.shape==image.shape and isinstance(icc,(bytes,bytearray)) and len(icc)>100


def test_performance_options_round_trip_and_memory_estimate(tmp_path):
    project=RollProject();project.performance.cpu_threads=2;project.performance.preview_dimension=1200;project.output.preset="Web";project.output.format="JPEG";project.output.resize_percent=50
    path=tmp_path/"performance.nlroll.json";project.save(path);restored=RollProject.load(path)
    assert restored.performance.cpu_threads==2 and restored.output.format=="JPEG" and restored.output.resize_percent==50
    estimate=estimate_frame_memory(6000,4000,True,True)
    assert estimate>6000*4000*12 and "GiB" in human_bytes(estimate)


def test_disk_preview_cache_and_system_diagnostics(tmp_path):
    cache=DiskPreviewCache(True,str(tmp_path/"cache"));image=np.full((12,16,3),.25,np.float32);cache.put("sample.tif","signature",image);restored=cache.get("sample.tif","signature")
    assert np.allclose(restored,image)
    diagnostics=system_diagnostics();configuration=configure_performance(type("Options",(),{"cpu_threads":0,"use_opencl":False})())
    assert "platform" in diagnostics and configuration["opencl_enabled"] is False


def test_web_preset_batch_resize_and_overwrite_recovery(tmp_path):
    source=tmp_path/"negative.tif";tifffile.imwrite(source,(negative_gradient()*65535).astype(np.uint16),photometric="rgb");base,dense=anchors();project=RollProject("Web Roll",[FrameRecord(str(source))],base,dense);project.output.format="JPEG";project.output.bit_depth=8;project.output.resize_percent=50
    output=tmp_path/"web";first=batch_convert(project,str(output));saved=output/"negative_positive.jpg";decoded=cv2.imread(str(saved))
    assert first["frames"][0]["status"]=="ok" and decoded.shape[:2]==(40,60)
    second=batch_convert(project,str(output));assert second["frames"][0]["status"]=="failed" and "overwrite" in second["frames"][0]["error"]

def test_geometry_curves_masks_and_clipping():
    image=negative_gradient(60,90);rectified,matrix=perspective_crop(image,[[.05,.05],[.95,.05],[.95,.95],[.05,.95]])
    curved=apply_curve(image,[[0,0],[.5,.65],[1,1]]);overlay,counts=clipping_overlay(np.concatenate((np.zeros((10,5,3)),np.ones((10,5,3))),axis=1))
    assert rectified.ndim==3 and matrix.shape==(3,3) and curved.mean()>image.mean()
    assert counts=={"shadow_pixels":50,"highlight_pixels":50} and overlay.shape==(10,10,3)
    histogram,stats=rgb_histogram(image);assert histogram.shape==(3,256) and histogram.sum()==image.shape[0]*image.shape[1]*3 and stats["pixels"]==image.shape[0]*image.shape[1]
    assert radial_mask(image.shape,(.5,.5),.4).max()==1 and linear_gradient_mask(image.shape,(0,0),(1,0))[30,-1]>.9
    masked=apply_mask_stack(image,[{"type":"brush","center":[.5,.5],"radius":.2,"operation":"exposure","amount":1}]);assert masked[30,45].mean()>image[30,45].mean()
    specs=[{"type":"gradient","start":[0,0],"end":[1,0],"enabled":True},{"type":"brush","center":[.5,.5],"radius":.2,"enabled":False}];combined=combined_mask(image.shape,specs);assert combined[30,-1]>.9 and combined.shape==image.shape[:2]

def test_anchor_discovery_and_roll_consistency():
    bright=np.full((80,100,3),.9,np.float32);dark=np.full((80,100,3),.1,np.float32)
    found=discover_anchor_candidates([bright,dark],["bright","dark"]);report=roll_consistency([bright,dark])
    assert found["clear_base"][0]["path"]=="bright" and found["dense_leader"][0]["path"]=="dark"
    assert len(report["frames"])==2 and report["frames"][0]["suggested_exposure_ev"]<0

def test_infrared_fusion_contact_sheet_and_capture_assessment():
    image=np.full((80,100,3),.5,np.float32);ir=image.copy();ir[40,50]=1;clean,mask=infrared_clean(image,ir,2)
    shifted=cv2.warpAffine(image,np.float32([[1,0,1],[0,1,0]]),(100,80));fused,report=fuse_exposures([image*.7,shifted])
    sheet=contact_sheet([image,fused],["one","two"],2,thumb=(80,60));assessment=camera_scan_assessment(image)
    assert mask.shape==image.shape[:2] and clean.shape==image.shape and len(report["alignment_scores"])==2
    assert sheet.shape[1]>160 and "illumination_uniformity" in assessment
