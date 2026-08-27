"""Advanced, nondestructive roll and digital-darkroom operations."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import cv2
import numpy as np

def perspective_crop(image,corners,output_size=None):
    h,w=image.shape[:2];src=np.asarray(corners,np.float32)*np.asarray([w-1,h-1],np.float32)
    if src.shape!=(4,2):raise ValueError("Perspective crop requires four [x, y] corners")
    top,bottom=np.linalg.norm(src[1]-src[0]),np.linalg.norm(src[2]-src[3]);left,right=np.linalg.norm(src[3]-src[0]),np.linalg.norm(src[2]-src[1])
    ow,oh=output_size or (max(1,round(max(top,bottom))),max(1,round(max(left,right))))
    dst=np.float32([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]]);matrix=cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(image,matrix,(ow,oh),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT),matrix

def apply_curve(image,points,channel="RGB"):
    pts=sorted((float(x),float(y)) for x,y in points)
    if not pts or pts[0][0]>0:pts.insert(0,(0.,0.))
    if pts[-1][0]<1:pts.append((1.,1.))
    x,y=np.asarray(pts,np.float32).T;lut=np.interp(np.linspace(0,1,4096),x,y).astype(np.float32);out=np.asarray(image,np.float32).copy();indices=np.clip(np.rint(out*4095),0,4095).astype(np.int32)
    channels=range(min(3,out.shape[2])) if channel.upper()=="RGB" else [{"R":0,"G":1,"B":2}[channel.upper()]]
    for c in channels:out[...,c]=lut[indices[...,c]]
    return np.clip(out,0,1)

def clipping_overlay(image,shadow=.002,highlight=.998):
    out=np.asarray(image,np.float32).copy();low=np.all(out<=shadow,axis=2);high=np.any(out>=highlight,axis=2);out[low]=(0.12,.35,1);out[high]=(1,.12,.15)
    return out,{"shadow_pixels":int(low.sum()),"highlight_pixels":int(high.sum())}

def rgb_histogram(image,bins=256):
    data=np.asarray(image,np.float32)[...,:3];hist=np.stack([np.histogram(data[...,c],bins=bins,range=(0,1))[0] for c in range(3)]).astype(np.int64)
    return hist,{"pixels":int(data.shape[0]*data.shape[1]),"shadow_clipped":int(np.all(data<=.002,axis=2).sum()),"highlight_clipped":int(np.any(data>=.998,axis=2).sum())}

def radial_mask(shape,center,radius,feather=.5):
    h,w=shape[:2];yy,xx=np.mgrid[0:h,0:w];distance=np.sqrt((xx-center[0]*w)**2+(yy-center[1]*h)**2)/max(radius*min(h,w),1)
    return np.clip((1-distance)/max(feather,.001),0,1).astype(np.float32)

def linear_gradient_mask(shape,start,end):
    h,w=shape[:2];yy,xx=np.mgrid[0:h,0:w];p=np.dstack((xx/w,yy/h));a=np.asarray(start,float);v=np.asarray(end,float)-a
    return np.clip(((p-a)@v)/max(float(v@v),1e-9),0,1).astype(np.float32)

def heal_or_clone(image,mask,source_offset=None):
    u8=np.clip(image*255,0,255).astype(np.uint8);m=(np.clip(mask,0,1)*255).astype(np.uint8)
    if source_offset is None:
        bgr=cv2.inpaint(cv2.cvtColor(u8,cv2.COLOR_RGB2BGR),m,3,cv2.INPAINT_TELEA);return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    dx,dy=map(int,source_offset);source=np.roll(u8,(dy,dx),(0,1));alpha=(m.astype(np.float32)/255)[...,None]
    return (u8*(1-alpha)+source*alpha).astype(np.float32)/255

def apply_mask_stack(image,specs):
    out=np.asarray(image,np.float32).copy()
    for spec in specs or []:
        if not spec.get("enabled",True):continue
        kind=spec.get("type","radial")
        if kind in ("radial","brush"):mask=radial_mask(out.shape,spec.get("center",[.5,.5]),float(spec.get("radius",.1)),float(spec.get("feather",.5)))
        elif kind=="gradient":mask=linear_gradient_mask(out.shape,spec.get("start",[0,0]),spec.get("end",[1,0]))
        else:continue
        operation=spec.get("operation","exposure")
        if operation=="exposure":out*=2**(float(spec.get("amount",0))*mask[...,None])
        elif operation=="heal":out=heal_or_clone(out,mask)
        elif operation=="clone":out=heal_or_clone(out,mask,spec.get("offset",[20,0]))
    return np.clip(out,0,1)

def combined_mask(shape,specs):
    result=np.zeros(shape[:2],np.float32)
    for spec in specs or []:
        if not spec.get("enabled",True):continue
        if spec.get("type","radial") in ("radial","brush"):mask=radial_mask(shape,spec.get("center",[.5,.5]),float(spec.get("radius",.1)),float(spec.get("feather",.5)))
        elif spec.get("type")=="gradient":mask=linear_gradient_mask(shape,spec.get("start",[0,0]),spec.get("end",[1,0]))
        else:continue
        result=np.maximum(result,mask)
    return result

def discover_anchor_candidates(images,paths=None,patch_fraction=.08,limit=8):
    candidates=[];paths=paths or [str(i) for i in range(len(images))]
    for fi,image in enumerate(images):
        work=cv2.resize(np.asarray(image,np.float32),(256,256),interpolation=cv2.INTER_AREA);size=max(8,int(256*patch_fraction))
        for y in range(0,257-size,size):
            for x in range(0,257-size,size):
                patch=work[y:y+size,x:x+size,:3];mean=patch.mean((0,1));spread=float(np.mean(np.std(patch,(0,1))))
                candidates.append({"frame":fi,"path":paths[fi],"x":(x+size/2)/256,"y":(y+size/2)/256,"rgb":mean.tolist(),"luminance":float(mean.mean()),"spread":spread})
    uniform=sorted(candidates,key=lambda c:c["spread"])[:max(limit*20,limit)]
    return {"clear_base":sorted(uniform,key=lambda c:c["luminance"],reverse=True)[:limit],"dense_leader":sorted(uniform,key=lambda c:c["luminance"])[:limit]}

def roll_consistency(images):
    stats=[]
    for i,image in enumerate(images):
        rgb=np.median(np.asarray(image,np.float32).reshape(-1,3),axis=0);stats.append({"frame":i,"median_rgb":rgb.tolist(),"luminance":float(np.mean(rgb))})
    if not stats:return {"frames":[],"target_luminance":0}
    target=float(np.median([s["luminance"] for s in stats]));target_rgb=np.median([s["median_rgb"] for s in stats],axis=0)
    for s in stats:
        s["suggested_exposure_ev"]=float(np.clip(np.log2(max(target,1e-6)/max(s["luminance"],1e-6)),-2,2));rgb=np.asarray(s["median_rgb"]);s["color_deviation"]=float(np.linalg.norm(rgb/max(rgb.mean(),1e-6)-target_rgb/max(target_rgb.mean(),1e-6)))
    return {"frames":stats,"target_luminance":target,"target_rgb":target_rgb.tolist()}

def infrared_defect_mask(infrared,threshold=3.5):
    gray=np.mean(infrared[...,:3],axis=2) if infrared.ndim==3 else np.asarray(infrared,np.float32);gray=np.asarray(gray,np.float32);smooth=cv2.medianBlur(gray,5);residual=np.abs(gray-smooth);med=np.median(residual);mad=np.median(np.abs(residual-med))*1.4826
    return cv2.morphologyEx((residual>med+threshold*max(mad,1e-6)).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))

def infrared_clean(image,infrared,threshold=3.5):
    mask=infrared_defect_mask(infrared,threshold);return heal_or_clone(image,mask/255),mask

def fuse_exposures(images):
    if len(images)<2:raise ValueError("Select at least two exposures")
    reference=np.asarray(images[len(images)//2],np.float32);aligned=[];scores=[];refgray=cv2.cvtColor(np.clip(reference*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    for image in images:
        work=np.asarray(image,np.float32);warp=np.eye(2,3,dtype=np.float32);score=1.
        try:
            gray=cv2.cvtColor(np.clip(work*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);score,warp=cv2.findTransformECC(refgray.astype(np.float32)/255,gray.astype(np.float32)/255,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,80,1e-5));work=cv2.warpAffine(work,warp,(reference.shape[1],reference.shape[0]),flags=cv2.INTER_LANCZOS4|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
        except cv2.error:pass
        aligned.append(np.clip(work,0,1));scores.append(float(score))
    fused=cv2.createMergeMertens().process([np.clip(x*255,0,255).astype(np.uint8) for x in aligned])
    return np.clip(fused,0,1).astype(np.float32),{"alignment_scores":scores}

def film_character(image,toe=0.,shoulder=0.,grain=0.,seed=0):
    out=np.asarray(image,np.float32).copy();lum=np.mean(out,axis=2,keepdims=True);out+=(1-lum)*float(toe)/100*.12;out-=lum*float(shoulder)/100*.12
    if grain:out+=np.random.default_rng(seed).normal(0,float(grain)/1000,out.shape[:2])[...,None]*(.4+.6*np.sqrt(np.clip(lum,0,1)))
    return np.clip(out,0,1)

def camera_scan_assessment(image):
    gray=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);h,w=gray.shape;cells=np.array([[gray[y*h//3:(y+1)*h//3,x*w//3:(x+1)*w//3].mean() for x in range(3)] for y in range(3)]);sharp=float(cv2.Laplacian(gray,cv2.CV_32F).var());corners=float(np.mean([cells[0,0],cells[0,2],cells[2,0],cells[2,2]]));center=float(cells[1,1])
    return {"sharpness":sharp,"illumination_uniformity":float(1-np.std(cells)/max(np.mean(cells),1)),"corner_to_center":corners/max(center,1),"clipped_black_percent":float(np.mean(gray<=1)*100),"clipped_white_percent":float(np.mean(gray>=254)*100)}

def contact_sheet(images,labels,columns=5,thumb=(320,220),margin=24):
    rows=max(1,(len(images)+columns-1)//columns);cellw,cellh=thumb[0]+margin*2,thumb[1]+margin*2+28;sheet=np.full((rows*cellh,columns*cellw,3),.08,np.float32)
    for i,(image,label) in enumerate(zip(images,labels)):
        r,c=divmod(i,columns);work=np.asarray(image,np.float32);scale=min(thumb[0]/work.shape[1],thumb[1]/work.shape[0]);view=cv2.resize(work,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA);y=r*cellh+margin;x=c*cellw+margin;sheet[y:y+view.shape[0],x:x+view.shape[1]]=view
    drawable=np.clip(sheet*255,0,255).astype(np.uint8)
    for i,label in enumerate(labels[:len(images)]):
        r,c=divmod(i,columns);cv2.putText(drawable,str(label),(c*cellw+margin,r*cellh+cellh-15),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1,cv2.LINE_AA)
    return drawable.astype(np.float32)/255

def archival_manifest(project,output_path):
    records=[]
    for frame in project.frames:
        path=Path(frame.path);digest=""
        if path.exists():
            h=hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
            digest=h.hexdigest()
        records.append({"frame":frame.frame_number,"path":str(path),"sha256":digest})
    manifest={"project":project.to_dict(),"sources":records};Path(output_path).write_text(json.dumps(manifest,indent=2),encoding="utf-8");return manifest
