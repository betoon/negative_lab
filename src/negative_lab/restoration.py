from __future__ import annotations

import cv2
import numpy as np


def correct_fading(image: np.ndarray, strength: float):
    if strength<=0:return image
    amount=np.clip(strength/100,0,1);out=np.asarray(image,np.float32);means=np.median(out.reshape(-1,3),axis=0);target=float(np.mean(means));gains=np.clip(target/np.maximum(means,1e-4),.5,2)
    corrected=out*gains[None,None,:];lum=np.mean(corrected,axis=2,keepdims=True);corrected=(corrected-.5)*(1+amount*.18)+.5;corrected+=(.5-lum)*amount*.06
    return np.clip(out*(1-amount)+corrected*amount,0,1)


def detect_dust(image: np.ndarray, strength=50, max_radius=10):
    gray=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);background=cv2.medianBlur(gray,max(5,int(max_radius)*2+1 if int(max_radius)%2==0 else int(max_radius)*2+1));res=cv2.absdiff(gray,background)
    med=float(np.median(res));mad=float(np.median(np.abs(res-med)))*1.4826;threshold=med+(7-strength/20)*max(mad,1);binary=(res>threshold).astype(np.uint8)*255
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary);mask=np.zeros_like(gray)
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if 1<=area<=np.pi*max_radius*max_radius and max(w,h)<=max_radius*2:mask[labels==i]=255
    return cv2.dilate(mask,np.ones((3,3),np.uint8))


def remove_dust(image: np.ndarray, strength=50, max_radius=10):
    if strength<=0:return image,np.zeros(image.shape[:2],np.uint8)
    mask=detect_dust(image,strength,max_radius);u8=np.clip(image*255,0,255).astype(np.uint8);clean=cv2.inpaint(cv2.cvtColor(u8,cv2.COLOR_RGB2BGR),mask,3,cv2.INPAINT_TELEA);return cv2.cvtColor(clean,cv2.COLOR_BGR2RGB).astype(np.float32)/255,mask


def sprocket_content_mask(image: np.ndarray):
    """Reject border-connected bright/dark components typical of sprockets/holders."""
    gray=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);h,w=gray.shape;border=max(3,int(min(h,w)*.12));mask=np.ones((h,w),np.uint8)*255
    extreme=((gray>np.percentile(gray,98))|(gray<np.percentile(gray,2))).astype(np.uint8)*255
    n,labels,stats,_=cv2.connectedComponentsWithStats(extreme)
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if x<border or y<border or x+bw>w-border or y+bh>h-border:mask[labels==i]=0
    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))


def apply_local_adjustments(image: np.ndarray, adjustments):
    out=np.asarray(image,np.float32).copy();h,w=out.shape[:2];yy,xx=np.mgrid[0:h,0:w]
    for item in adjustments or []:
        cx=float(item.get("x",.5))*w;cy=float(item.get("y",.5))*h;r=max(1,float(item.get("radius",.1))*min(h,w));mask=np.clip(1-np.sqrt((xx-cx)**2+(yy-cy)**2)/r,0,1)**max(.2,float(item.get("hardness",.5)))
        out*=2**(float(item.get("exposure",0))*mask[...,None]);sat=float(item.get("saturation",0))/100;gray=np.mean(out,axis=2,keepdims=True);out=gray+(out-gray)*(1+sat*mask[...,None])
    return np.clip(out,0,1)


def trichrome_merge(red: np.ndarray, green: np.ndarray, blue: np.ndarray):
    frames=[];reference=cv2.cvtColor(np.clip(green*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    transforms=[]
    for image in (red,green,blue):
        gray=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);warp=np.eye(2,3,dtype=np.float32);score=1.0
        try:score,warp=cv2.findTransformECC(reference.astype(np.float32)/255,gray.astype(np.float32)/255,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6));gray=cv2.warpAffine(gray,warp,(reference.shape[1],reference.shape[0]),flags=cv2.INTER_LANCZOS4|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
        except cv2.error:pass
        frames.append(gray.astype(np.float32)/255);transforms.append({"score":float(score),"matrix":warp.tolist()})
    return np.dstack((frames[0],frames[1],frames[2])),transforms
