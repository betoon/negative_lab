from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import tempfile

import cv2
import numpy as np


def configure_performance(options):
    threads=int(options.cpu_threads or 0);cv2.setNumThreads(threads)
    requested=bool(options.use_opencl);available=bool(cv2.ocl.haveOpenCL())
    try:cv2.ocl.setUseOpenCL(requested and available)
    except Exception:pass
    return {"cpu_threads_requested":threads,"opencv_threads":int(cv2.getNumThreads()),"opencl_requested":requested,"opencl_available":available,"opencl_enabled":bool(cv2.ocl.useOpenCL())}


def available_memory_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def estimate_frame_memory(width: int, height: int, calibration=True, restoration=True):
    # source + calibrated + converted + adjusted/output + temporary masks/gain
    rgb_float=width*height*3*4;copies=4+(2 if calibration else 0)+(1 if restoration else 0)
    return int(rgb_float*copies+width*height*4)


def human_bytes(value):
    value=float(value)
    for unit in ("B","KiB","MiB","GiB","TiB"):
        if value<1024 or unit=="TiB":return f"{value:.1f} {unit}"
        value/=1024


class DiskPreviewCache:
    def __init__(self, enabled=True, directory=""):
        self.enabled=enabled;self.root=Path(directory) if directory else Path(tempfile.gettempdir())/"negative_lab_preview_cache"
        if enabled:self.root.mkdir(parents=True,exist_ok=True)
    def _path(self,source,signature):
        key=hashlib.sha256((str(Path(source).resolve())+"|"+signature).encode()).hexdigest();return self.root/(key+".npy")
    def get(self,source,signature):
        if not self.enabled:return None
        path=self._path(source,signature)
        try:return np.load(path,mmap_mode="r") if path.exists() else None
        except Exception:return None
    def put(self,source,signature,array):
        if not self.enabled:return
        path=self._path(source,signature);np.save(path,np.asarray(array,np.float32),allow_pickle=False)
    def clear(self):
        if self.root.exists() and self.root.name=="negative_lab_preview_cache":shutil.rmtree(self.root,ignore_errors=True)


def system_diagnostics():
    return {"platform":platform.platform(),"python":platform.python_version(),"cpu_logical":os.cpu_count(),"opencv":cv2.__version__,"opencl_available":bool(cv2.ocl.haveOpenCL()),"available_memory":available_memory_bytes()}
