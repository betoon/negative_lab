from __future__ import annotations

import copy
import json
import logging
import os
import platform
import sys
import traceback
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, QSize, QSettings, QStandardPaths, QEvent
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QInputDialog,
    QSplitter, QTabWidget, QToolBar, QVBoxLayout, QWidget, QSpacerItem, QSizePolicy)

from . import __version__
from .conversion import anchor_diagnostics, preview_convert, sample_anchor
from .calibration import (apply_capture_calibration, flat_field_diagnostics,
    fit_it8_matrix, load_dcp_matrix, load_matrix_json, parse_it8_pairs, transform_output)
from .io import IMAGE_EXTENSIONS, read_linear, write_image
from .models import FilmProfile, FrameRecord, RollProject
from .restoration import trichrome_merge
from .performance import (DiskPreviewCache, available_memory_bytes, configure_performance,
                          estimate_frame_memory, human_bytes, system_diagnostics)
from .workflow import (SYNC_GROUPS, analyze_frame, batch_convert, detect_frame_bounds,
                       load_profile, save_profile, sync_recipes)
from .digital_darkroom import (archival_manifest, camera_scan_assessment, clipping_overlay,
    combined_mask, contact_sheet, discover_anchor_candidates, fuse_exposures, infrared_clean,
    export_roll_catalog, rgb_histogram, roll_consistency, verify_archival_manifest)

LOG=logging.getLogger("negative_lab")


class ImageCanvas(QScrollArea):
    sampled=Signal(float,float)
    hovered=Signal(float,float)
    def __init__(self):
        super().__init__(); self.label=QLabel("Import a roll to begin"); self.label.setAlignment(Qt.AlignCenter); self.setWidget(self.label); self.setWidgetResizable(True); self.array=None; self.zoom=1.0;self.setMouseTracking(True);self.viewport().setMouseTracking(True);self.label.setMouseTracking(True);self.label.installEventFilter(self)
    def set_image(self,image):
        self.array=image
        if image is None:self.label.setText("Import a roll to begin");return
        data=np.clip(image*255,0,255).astype(np.uint8); self.qimage=QImage(data.data,data.shape[1],data.shape[0],data.strides[0],QImage.Format_RGB888).copy(); self.fit()
    def _refresh(self):
        if self.array is not None:
            size=self.qimage.size()*self.zoom; self.label.setPixmap(QPixmap.fromImage(self.qimage).scaled(size,Qt.KeepAspectRatio,Qt.SmoothTransformation)); self.label.resize(size)
    def fit(self):
        if self.array is not None:self.zoom=max(.05,min(self.viewport().width()/self.array.shape[1],self.viewport().height()/self.array.shape[0]));self._refresh()
    def set_zoom(self,value):
        if self.array is not None:self.zoom=max(.05,min(8,float(value)));self._refresh()
    def wheelEvent(self,event):
        if event.modifiers()&Qt.ControlModifier and self.array is not None:self.zoom=max(.05,min(8,self.zoom*(1.2 if event.angleDelta().y()>0 else 1/1.2)));self._refresh();event.accept()
        else:super().wheelEvent(event)
    def mousePressEvent(self,event):
        if self.array is not None and event.button()==Qt.LeftButton:
            pix=self.label.pixmap(); pw=pix.width() if pix else self.label.width(); ph=pix.height() if pix else self.label.height()
            left=max(0,(self.label.width()-pw)/2); top=max(0,(self.label.height()-ph)/2)
            x=(event.position().x()+self.horizontalScrollBar().value()-left)/max(pw,1); y=(event.position().y()+self.verticalScrollBar().value()-top)/max(ph,1)
            if 0<=x<=1 and 0<=y<=1:self.sampled.emit(float(x),float(y))
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self.array is not None:
            pix=self.label.pixmap();pw=pix.width() if pix else self.label.width();ph=pix.height() if pix else self.label.height();left=max(0,(self.label.width()-pw)/2);top=max(0,(self.label.height()-ph)/2);x=(event.position().x()+self.horizontalScrollBar().value()-left)/max(pw,1);y=(event.position().y()+self.verticalScrollBar().value()-top)/max(ph,1)
            if 0<=x<=1 and 0<=y<=1:self.hovered.emit(float(x),float(y))
        super().mouseMoveEvent(event)
    def eventFilter(self,obj,event):
        if obj is self.label and event.type()==QEvent.MouseMove and self.array is not None:
            pix=self.label.pixmap();pw=pix.width() if pix else self.label.width();ph=pix.height() if pix else self.label.height();left=max(0,(self.label.width()-pw)/2);top=max(0,(self.label.height()-ph)/2);x=(event.position().x()-left)/max(pw,1);y=(event.position().y()-top)/max(ph,1)
            if 0<=x<=1 and 0<=y<=1:self.hovered.emit(float(x),float(y))
        return super().eventFilter(obj,event)


class BatchWorker(QObject):
    progress=Signal(int,str); finished=Signal(object); failed=Signal(str)
    def __init__(self,project,directory):super().__init__();self.project=copy.deepcopy(project);self.directory=directory;self.cancelled=False
    def cancel(self):self.cancelled=True
    def run(self):
        try:self.finished.emit(batch_convert(self.project,self.directory,None,self.progress.emit,lambda:self.cancelled))
        except Exception:self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.project=RollProject(); self.current_image=None; self.current_meta={}; self.sample_mode=""; self.project_path=""; self._dirty=False;self.setAcceptDrops(True);self.settings=QSettings("Brian E. Toon","Negative Lab");self.operation_history=[];self.preview_cache=DiskPreviewCache(True)
        self.setWindowTitle(f"Negative Lab {__version__} — Physics-oriented Film Conversion"); self.resize(1500,930); self.setMinimumSize(1000,680)
        self._build(); self._actions(); self._shortcuts(); self._log("SYSTEM",json.dumps(system_diagnostics()));self.autosave_timer=QTimer(self);self.autosave_timer.timeout.connect(self.autosave);self.autosave_timer.start(30000);QTimer.singleShot(0,self.offer_recovery)

    def _build(self):
        self.roll_list=QListWidget(); self.roll_list.setSelectionMode(QAbstractItemView.ExtendedSelection); self.roll_list.setIconSize(QSize(86,64)); self.roll_list.currentRowChanged.connect(self.load_frame)
        left=QWidget(); lv=QVBoxLayout(left); title=QLabel("ROLL / SCAN SESSION");title.setObjectName("heading");lv.addWidget(title);lv.addWidget(self.roll_list,1)
        row=QHBoxLayout(); add=QPushButton("＋ Images");add.clicked.connect(self.import_images);folder=QPushButton("▣ Folder");folder.clicked.connect(self.import_folder);remove=QPushButton("− Remove");remove.clicked.connect(self.remove_frames)
        for b in (add,folder,remove):row.addWidget(b)
        lv.addLayout(row); self.roll_summary=QLabel("No frames loaded");self.roll_summary.setObjectName("card");self.roll_summary.setWordWrap(True);lv.addWidget(self.roll_summary)
        self.canvas=ImageCanvas();self.canvas.sampled.connect(self.canvas_sampled);self.canvas.hovered.connect(self.update_pixel_status)
        center=QWidget();cv=QVBoxLayout(center); compare=QHBoxLayout();self.show_original=QCheckBox("Show scanned negative");self.show_original.toggled.connect(self.render);fit=QPushButton("Fit");fit.clicked.connect(self.canvas.fit);compare.addWidget(self.show_original);compare.addWidget(fit)
        for label,zoom in (("100%",1),("200%",2),("400%",4)):
            button=QPushButton(label);button.clicked.connect(lambda _checked=False,z=zoom:self.canvas.set_zoom(z));compare.addWidget(button)
        side=QPushButton("Before / After");side.clicked.connect(self.open_comparison);hist=QPushButton("Histogram");hist.clicked.connect(self.open_histogram);full=QPushButton("Full Screen");full.clicked.connect(self.open_fullscreen);compare.addWidget(side);compare.addWidget(hist);compare.addWidget(full);compare.addStretch();cv.addLayout(compare);cv.addWidget(self.canvas,1);self.pixel_status=QLabel("Move over the image for pixel coordinates and RGB values. Ctrl+wheel also zooms.");cv.addWidget(self.pixel_status)
        self.controls=QTabWidget();self._conversion_tab();self._adjust_tab();self._darkroom_tab();self._light_table_tab();self._calibration_tab();self._restoration_tab();self._workflow_tab();self._performance_tab();self._operations_tab();self._diagnostics_tab()
        split=QSplitter();split.addWidget(left);split.addWidget(center);split.addWidget(self.controls);split.setSizes([280,850,360]);self.setCentralWidget(split)
        self.progress=QProgressBar();self.progress.hide();self.statusBar().addPermanentWidget(self.progress)

    def _conversion_tab(self):
        page=QScrollArea();page.setWidgetResizable(True);body=QWidget();v=QVBoxLayout(body);page.setWidget(body)
        guide=QLabel("1. Load one roll captured at fixed exposure.  2. Sample clear unexposed film base.  3. Sample a dense fully exposed leader.  4. Review anchor quality, then convert.");guide.setObjectName("card");guide.setWordWrap(True);v.addWidget(guide)
        anchors=QGroupBox("Density Anchors");f=QFormLayout(anchors);self.radius=QSpinBox();self.radius.setRange(2,80);self.radius.setValue(9);f.addRow("Sample radius",self.radius)
        self.base_button=QPushButton("Sample Clear Film Base");self.base_button.clicked.connect(lambda:self.arm_sample("base"));f.addRow(self.base_button);self.base_value=QLabel("Not sampled");f.addRow("Clear base RGB",self.base_value)
        self.dense_button=QPushButton("Sample Dense Leader");self.dense_button.clicked.connect(lambda:self.arm_sample("dense"));f.addRow(self.dense_button);self.dense_value=QLabel("Not sampled");f.addRow("Dense leader RGB",self.dense_value);v.addWidget(anchors)
        profile=QGroupBox("Film Profile");pf=QFormLayout(profile);self.profile_combo=QComboBox();self.profile_combo.addItems([p.name for p in FilmProfile.builtins()]);self.profile_combo.currentIndexChanged.connect(self.profile_changed);pf.addRow("Profile",self.profile_combo)
        pr=QHBoxLayout();load=QPushButton("Load…");load.clicked.connect(self.load_film_profile);save=QPushButton("Save Current…");save.clicked.connect(self.save_film_profile);pr.addWidget(load);pr.addWidget(save);pf.addRow(pr);self.profile_description=QLabel(FilmProfile.builtins()[0].description);self.profile_description.setWordWrap(True);pf.addRow(self.profile_description);v.addWidget(profile)
        self.anchor_quality=QLabel("Anchor quality has not been measured.");self.anchor_quality.setObjectName("card");self.anchor_quality.setWordWrap(True);v.addWidget(self.anchor_quality);v.addStretch();self.controls.addTab(page,"Negative")

    def _adjust_tab(self):
        page=QScrollArea();page.setWidgetResizable(True);body=QWidget();form=QFormLayout(body);page.setWidget(body);self.adjust={}
        for key,label,lo,hi,default,decimals in [("exposure","Exposure (EV)",-4,4,0,2),("temperature","Temperature",-100,100,0,0),("tint","Tint",-100,100,0,0),("contrast","Contrast",-100,100,0,0),("saturation","Saturation",-100,100,0,0),("shadows","Shadows",-100,100,0,0),("highlights","Highlights",-100,100,0,0),("gamma","Gamma",.2,3,1,2)]:
            s=QDoubleSpinBox();s.setRange(lo,hi);s.setDecimals(decimals);s.setSingleStep(.1 if decimals else 1);s.setValue(default);s.valueChanged.connect(lambda value,k=key:self.adjust_changed(k,value));form.addRow(label,s);self.adjust[key]=s
        self.controls.addTab(page,"Develop")

    def _darkroom_tab(self):
        page=QScrollArea();page.setWidgetResizable(True);body=QWidget();v=QVBoxLayout(body);page.setWidget(body)
        geometry=QGroupBox("Geometry Studio");gf=QFormLayout(geometry);self.fine_rotation=QDoubleSpinBox();self.fine_rotation.setRange(-45,45);self.fine_rotation.setDecimals(2);self.fine_rotation.valueChanged.connect(lambda x:self.darkroom_value("fine_rotation",x));gf.addRow("Fine rotation",self.fine_rotation);self.crop_controls={}
        for key,label,default in (("x","Crop left %",0),("y","Crop top %",0),("w","Crop width %",100),("h","Crop height %",100)):
            s=QDoubleSpinBox();s.setRange(0 if key in "xy" else 1,100);s.setValue(default);s.setDecimals(1);s.valueChanged.connect(self.manual_crop_changed);gf.addRow(label,s);self.crop_controls[key]=s
        left=QPushButton("Rotate 90° Left");left.clicked.connect(lambda:self.rotate_quarter(-1));right=QPushButton("Rotate 90° Right");right.clicked.connect(lambda:self.rotate_quarter(1));auto=QPushButton("Detect Crop");auto.clicked.connect(self.detect_current_frame);perspective=QPushButton("Edit Perspective Corners…");perspective.clicked.connect(self.edit_perspective)
        for b in (left,right,auto,perspective):gf.addRow(b)
        v.addWidget(geometry)
        tone=QGroupBox("Curves & Exposure Warnings");tf=QVBoxLayout(tone);curve=QPushButton("Apply Gentle S-Curve");curve.clicked.connect(self.apply_s_curve);edit_curve=QPushButton("Edit RGB Curve Points…");edit_curve.clicked.connect(self.edit_curve);reset=QPushButton("Reset Curves");reset.clicked.connect(self.reset_curves);self.clipping=QCheckBox("Show blue shadow / red highlight clipping");self.clipping.toggled.connect(lambda x:self.darkroom_value("show_clipping",x));tf.addWidget(curve);tf.addWidget(edit_curve);tf.addWidget(reset);tf.addWidget(self.clipping);v.addWidget(tone)
        masks=QGroupBox("Masks, Healing & Cloning");mf=QFormLayout(masks);self.mask_operation=QComboBox();self.mask_operation.addItems(["Exposure","Heal","Clone"]);mf.addRow("Operation",self.mask_operation);self.mask_radius=QSpinBox();self.mask_radius.setRange(5,1000);self.mask_radius.setValue(120);mf.addRow("Brush / radial radius",self.mask_radius);self.mask_amount=QDoubleSpinBox();self.mask_amount.setRange(-4,4);self.mask_amount.setSingleStep(.1);self.mask_amount.setValue(.5);mf.addRow("Exposure amount",self.mask_amount);brush=QPushButton("Place Brush / Radial Mask on Image");brush.clicked.connect(lambda:self.arm_sample("mask"));gradient=QPushButton("Draw Two-Point Gradient on Image");gradient.clicked.connect(lambda:self.arm_sample("gradient_start"));clear_masks=QPushButton("Clear Frame Masks");clear_masks.clicked.connect(self.clear_masks)
        self.mask_list=QComboBox();self.mask_list.currentIndexChanged.connect(self.mask_selection_changed);mf.addRow("Frame masks",self.mask_list);self.mask_enabled=QCheckBox("Selected mask enabled");self.mask_enabled.toggled.connect(self.mask_enabled_changed);mf.addRow(self.mask_enabled);self.show_mask_overlay=QCheckBox("Show cyan mask overlay");self.show_mask_overlay.toggled.connect(self.render);mf.addRow(self.show_mask_overlay);remove_mask=QPushButton("Remove Selected Mask");remove_mask.clicked.connect(self.remove_selected_mask);self.mask_summary=QLabel("No masks on this frame");self.mask_summary.setWordWrap(True)
        for b in (brush,gradient,remove_mask,clear_masks):mf.addRow(b)
        mf.addRow(self.mask_summary)
        v.addWidget(masks)
        character=QGroupBox("Film Character Designer");cf=QFormLayout(character);self.film_controls={}
        for key,label in (("film_toe","Toe / lifted shadows"),("film_shoulder","Shoulder / highlight rolloff"),("film_grain","Natural grain")):
            s=QDoubleSpinBox();s.setRange(0,100);s.valueChanged.connect(lambda value,k=key:self.darkroom_value(k,value));cf.addRow(label,s);self.film_controls[key]=s
        v.addWidget(character)
        advanced=QGroupBox("Restoration & Capture");av=QVBoxLayout(advanced)
        for label,slot in (("Automatic Anchor Discovery…",self.auto_discover_anchors),("Analyze Roll Consistency…",self.analyze_roll_consistency),("Clean with Infrared Channel…",self.clean_with_infrared),("Fuse Bracketed Scans…",self.multi_exposure_fusion),("Camera-Scan Quality Check",self.camera_scan_check)):
            b=QPushButton(label);b.clicked.connect(slot);av.addWidget(b)
        v.addWidget(advanced);v.addStretch();self.controls.addTab(page,"Digital Darkroom")

    def _light_table_tab(self):
        page=QScrollArea();page.setWidgetResizable(True);body=QWidget();v=QVBoxLayout(body);page.setWidget(body);guide=QLabel("Create a positive contact view of the complete roll, then export proof sheets and checksum-backed archival records.");guide.setObjectName("card");guide.setWordWrap(True);v.addWidget(guide)
        self.contact_accepted_only=QCheckBox("Proof sheets include accepted frames only");self.contact_accepted_only.setChecked(False);v.addWidget(self.contact_accepted_only)
        for label,slot in (("Open Virtual Light Table…",self.open_light_table),("Export Contact Sheet (JPEG, TIFF, or PDF)…",self.export_contact_sheet),("Export Roll Catalog CSV…",self.export_roll_catalog_csv),("Export Archival Manifest…",self.export_archival_manifest),("Verify Archival Manifest…",self.verify_manifest)):
            b=QPushButton(label);b.setMinimumHeight(42);b.clicked.connect(slot);v.addWidget(b)
        v.addStretch();self.controls.addTab(page,"Virtual Light Table")

    def _workflow_tab(self):
        page=QScrollArea();page.setWidgetResizable(True);body=QWidget();v=QVBoxLayout(body);page.setWidget(body)
        framing=QGroupBox("Automatic Framing");fv=QVBoxLayout(framing);detect=QPushButton("Detect Frame Bounds on Current");detect.clicked.connect(self.detect_current_frame);allb=QPushButton("Detect and Apply to Entire Roll");allb.clicked.connect(self.detect_all_frames);fv.addWidget(detect);fv.addWidget(allb);v.addWidget(framing)
        sync=QGroupBox("Synchronize Selected Frames");sv=QVBoxLayout(sync);self.sync_checks={}
        for name in ("Tone","Color","Geometry"):c=QCheckBox(name);c.setChecked(name!="Geometry");sv.addWidget(c);self.sync_checks[name]=c
        b=QPushButton("Copy Current Groups to Selected");b.clicked.connect(self.sync_selected);sv.addWidget(b);v.addWidget(sync)
        batch=QGroupBox("Batch Conversion");bv=QVBoxLayout(batch);self.batch_button=QPushButton("Convert Roll to 16-bit TIFF…");self.batch_button.setObjectName("primary");self.batch_button.clicked.connect(self.start_batch);self.cancel_button=QPushButton("Cancel");self.cancel_button.setEnabled(False);self.cancel_button.clicked.connect(self.cancel_batch);bv.addWidget(self.batch_button);bv.addWidget(self.cancel_button);v.addWidget(batch);v.addStretch();self.controls.addTab(page,"Roll Workflow")

    def _calibration_tab(self):
        page=QScrollArea();page.setWidgetResizable(True);body=QWidget();v=QVBoxLayout(body);page.setWidget(body)
        capture=QGroupBox("Capture Calibration");f=QFormLayout(capture);self.dark_label=QLabel("Not selected");self.flat_label=QLabel("Not selected");dark=QPushButton("Choose Dark Frame…");dark.clicked.connect(self.choose_dark);flat=QPushButton("Choose Flat Field…");flat.clicked.connect(self.choose_flat);f.addRow(dark,self.dark_label);f.addRow(flat,self.flat_label);v.addWidget(capture)
        profile=QGroupBox("Camera / Scanner Characterization");pf=QVBoxLayout(profile);dcp=QPushButton("Load DCP Color Matrix…");dcp.clicked.connect(self.load_dcp);matrix=QPushButton("Load 3×3 Matrix JSON…");matrix.clicked.connect(self.load_matrix);it8=QPushButton("Fit Matrix from IT8/CGATS Pairs…");it8.clicked.connect(self.load_it8);self.input_profile_label=QLabel(self.project.calibration.input_profile_name);self.input_profile_label.setWordWrap(True)
        for widget in (dcp,matrix,it8,self.input_profile_label):pf.addWidget(widget)
        v.addWidget(profile)
        output=QGroupBox("Managed Output");of=QFormLayout(output);self.output_space=QComboBox();self.output_space.addItems(["sRGB","Custom ICC"]);self.output_space.currentTextChanged.connect(self.output_space_changed);of.addRow("Output space",self.output_space);self.output_icc_label=QLabel("Built-in sRGB");icc=QPushButton("Choose Custom Output ICC…");icc.clicked.connect(self.choose_output_icc);of.addRow(icc,self.output_icc_label);self.embed_icc=QCheckBox("Embed ICC profile");self.embed_icc.setChecked(True);self.embed_icc.toggled.connect(lambda value:self.set_calibration_value("embed_icc",bool(value)));of.addRow(self.embed_icc);self.output_dpi=QSpinBox();self.output_dpi.setRange(36,2400);self.output_dpi.setValue(300);self.output_dpi.valueChanged.connect(lambda value:self.set_calibration_value("dpi",int(value)));of.addRow("Resolution metadata",self.output_dpi);v.addWidget(output);v.addStretch();self.controls.addTab(page,"Calibration")

    def _restoration_tab(self):
        page=QScrollArea();page.setWidgetResizable(True);body=QWidget();f=QFormLayout(body);page.setWidget(body)
        self.fade=QDoubleSpinBox();self.fade.setRange(0,100);self.fade.valueChanged.connect(lambda value:self.restoration_changed("fade_correction",value));f.addRow("Fading correction",self.fade)
        self.dust=QDoubleSpinBox();self.dust.setRange(0,100);self.dust.valueChanged.connect(lambda value:self.restoration_changed("dust_removal",value));f.addRow("Dust removal",self.dust)
        self.dust_radius=QSpinBox();self.dust_radius.setRange(2,50);self.dust_radius.setValue(10);self.dust_radius.valueChanged.connect(lambda value:self.restoration_changed("dust_max_radius",value));f.addRow("Largest dust radius",self.dust_radius)
        self.sprocket=QCheckBox("Mask sprockets and holder artifacts");self.sprocket.toggled.connect(lambda value:self.restoration_changed("sprocket_mask",value));f.addRow(self.sprocket)
        self.local_radius=QSpinBox();self.local_radius.setRange(5,500);self.local_radius.setValue(100);f.addRow("Local brush radius",self.local_radius);self.local_exposure=QDoubleSpinBox();self.local_exposure.setRange(-4,4);self.local_exposure.setSingleStep(.1);f.addRow("Local exposure",self.local_exposure);local=QPushButton("Add Local Correction on Image");local.clicked.connect(lambda:self.arm_sample("local"));f.addRow(local);clear=QPushButton("Clear Local Corrections");clear.clicked.connect(self.clear_local);f.addRow(clear)
        compare=QPushButton("Compare Film Profiles…");compare.clicked.connect(self.compare_profiles);f.addRow(compare);tri=QPushButton("Build Trichrome from Three Exposures…");tri.clicked.connect(self.build_trichrome);f.addRow(tri);note=QLabel("Restoration is interpretive. Keep the original scans and review dust, fading, sprocket, and trichrome results at 100%.");note.setWordWrap(True);note.setObjectName("card");f.addRow(note);self.controls.addTab(page,"Restoration")

    def _diagnostics_tab(self):
        page=QWidget();self.diagnostic_page=page;v=QVBoxLayout(page);self.diagnostics=QPlainTextEdit();self.diagnostics.setReadOnly(True);self.diagnostics.setLineWrapMode(QPlainTextEdit.NoWrap);v.addWidget(self.diagnostics,1);row=QHBoxLayout();clear=QPushButton("Clear");clear.clicked.connect(self.diagnostics.clear);copyb=QPushButton("Copy All");copyb.clicked.connect(lambda:QApplication.clipboard().setText(self.diagnostics.toPlainText()));save=QPushButton("Save Log…");save.clicked.connect(self.save_log)
        for b in (clear,copyb,save):row.addWidget(b)
        row.addStretch();v.addLayout(row);self.controls.addTab(page,">_ Diagnostics")

    def _performance_tab(self):
        page=QScrollArea();page.setWidgetResizable(True);body=QWidget();v=QVBoxLayout(body);page.setWidget(body)
        perf=QGroupBox("Performance");f=QFormLayout(perf);self.preview_dimension=QSpinBox();self.preview_dimension.setRange(600,6000);self.preview_dimension.setSingleStep(200);self.preview_dimension.setValue(1800);self.preview_dimension.valueChanged.connect(lambda x:self.set_performance("preview_dimension",x));f.addRow("Preview longest side",self.preview_dimension)
        self.cpu_threads=QSpinBox();self.cpu_threads.setRange(0,max(1,os.cpu_count() or 1));self.cpu_threads.setSpecialValueText("Automatic");self.cpu_threads.valueChanged.connect(lambda x:self.set_performance("cpu_threads",x));f.addRow("CPU threads",self.cpu_threads);self.opencl=QCheckBox("Use OpenCL where OpenCV supports it");self.opencl.setChecked(True);self.opencl.toggled.connect(lambda x:self.set_performance("use_opencl",x));f.addRow(self.opencl);self.disk_cache=QCheckBox("Disk-backed calibrated preview cache");self.disk_cache.setChecked(True);self.disk_cache.toggled.connect(lambda x:self.set_performance("disk_preview_cache",x));f.addRow(self.disk_cache);self.recover_frames=QCheckBox("Continue after a failed frame");self.recover_frames.setChecked(True);self.recover_frames.toggled.connect(lambda x:self.set_performance("recover_failed_frames",x));f.addRow(self.recover_frames);clear=QPushButton("Clear Preview Cache");clear.clicked.connect(self.clear_preview_cache);f.addRow(clear);self.memory_label=QLabel("Load a frame to estimate memory.");self.memory_label.setObjectName("card");self.memory_label.setWordWrap(True);f.addRow(self.memory_label);v.addWidget(perf)
        output=QGroupBox("Output Presets");of=QFormLayout(output);self.output_preset=QComboBox();self.output_preset.addItems(["Archival","Web","Proof","Custom"]);self.output_preset.currentTextChanged.connect(self.apply_output_preset);of.addRow("Preset",self.output_preset);self.output_format=QComboBox();self.output_format.addItems(["TIFF","JPEG"]);self.output_format.currentTextChanged.connect(lambda x:self.set_output("format",x));of.addRow("Format",self.output_format);self.output_bits=QComboBox();self.output_bits.addItems(["16","8"]);self.output_bits.currentTextChanged.connect(lambda x:self.set_output("bit_depth",int(x)));of.addRow("Bit depth",self.output_bits);self.output_resize=QSpinBox();self.output_resize.setRange(10,400);self.output_resize.setSuffix("%");self.output_resize.setValue(100);self.output_resize.valueChanged.connect(lambda x:self.set_output("resize_percent",x));of.addRow("Resize",self.output_resize);self.output_overwrite=QCheckBox("Overwrite existing outputs");self.output_overwrite.toggled.connect(lambda x:self.set_output("overwrite_existing",x));of.addRow(self.output_overwrite);self.output_description=QLabel("16-bit TIFF with embedded profile and full dimensions.");self.output_description.setWordWrap(True);of.addRow(self.output_description);v.addWidget(output);v.addStretch();self.controls.addTab(page,"Performance & Output")

    def _operations_tab(self):
        page=QWidget();v=QVBoxLayout(page);self.history_list=QListWidget();v.addWidget(self.history_list,1);row=QHBoxLayout();recent=QPushButton("Recent Projects…");recent.clicked.connect(self.open_recent_project);package=QPushButton("Create Diagnostic Package…");package.clicked.connect(self.create_diagnostic_package);recovery=QPushButton("Save Recovery Point Now");recovery.clicked.connect(self.autosave)
        for b in (recent,package,recovery):row.addWidget(b)
        row.addStretch();v.addLayout(row);self.controls.addTab(page,"Operations")

    def _actions(self):
        bar=QToolBar("Main");bar.setMovable(False);self.addToolBar(bar)
        for label,slot in [("＋ Import",self.import_images),("▣ Folder",self.import_folder),("◇ New Roll",self.new_project),("◆ Open Roll",self.open_project),("▣ Save Roll",self.save_project),("◎ Sample Base",lambda:self.arm_sample("base")),("● Sample Leader",lambda:self.arm_sample("dense")),("↻ Convert Preview",self.render),("⇩ Convert Roll",self.start_batch),(">_ Diagnostics",lambda:self.controls.setCurrentWidget(self.diagnostic_page))]:
            a=QAction(label,self);a.triggered.connect(slot);bar.addAction(a)

    def _shortcuts(self):
        for key,slot in (("Ctrl+O",self.import_images),("Ctrl+S",self.save_project),("Ctrl+Shift+O",self.open_project),("Ctrl+E",self.start_batch),("F12",lambda:self.controls.setCurrentWidget(self.diagnostic_page))):QShortcut(QKeySequence(key),self,slot)

    def _log(self,level,message):
        stamp=__import__('datetime').datetime.now().strftime('%H:%M:%S.%f')[:-3];self.diagnostics.appendPlainText(f"[{stamp}] {level:<8} {message}")
        if hasattr(self,"history_list") and level in {"PROJECT","BATCH","SUCCESS","CALIBRATION","PROFILE","SYNC","FRAMING","TRICHROME","WARNING","ERROR"}:
            entry=f"{stamp}  {level}: {str(message).splitlines()[0][:180]}";self.operation_history.append(entry);self.history_list.addItem(entry);self.history_list.scrollToBottom()
    def current_record(self):
        i=self.roll_list.currentRow();return self.project.frames[i] if 0<=i<len(self.project.frames) else None
    def mark_dirty(self):self._dirty=True
    def import_images(self):
        paths,_=QFileDialog.getOpenFileNames(self,"Import one roll or scan session","","Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp *.dng *.nef *.cr2 *.cr3 *.arw *.orf *.rw2 *.raf)");self.add_paths(paths)
    def import_folder(self):
        folder=QFileDialog.getExistingDirectory(self,"Import roll folder");self.add_paths([str(p) for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS] if folder else [])
    def add_paths(self,paths):
        existing={x.path for x in self.project.frames}
        for path in paths:
            if path in existing:continue
            rec=FrameRecord(path,frame_number=len(self.project.frames)+1);self.project.frames.append(rec);item=QListWidgetItem(f"{rec.frame_number:02d}  {Path(path).name}");item.setToolTip(path)
            try:
                image,meta=read_linear(path);thumb=cv2.resize(image,(86,64),interpolation=cv2.INTER_AREA);d=np.clip(thumb*255,0,255).astype(np.uint8);item.setIcon(QIcon(QPixmap.fromImage(QImage(d.data,86,64,d.strides[0],QImage.Format_RGB888).copy())))
                rec.warnings=analyze_frame(image,meta)["warnings"]
                if rec.warnings:item.setForeground(QColor("#fbbf24"))
            except Exception as exc:rec.warnings=[str(exc)];item.setForeground(QColor("#f87171"));self._log("ERROR",f"{path}: {exc}")
            self.roll_list.addItem(item);existing.add(path);self._log("IMPORT",path)
        self.roll_summary.setText(f"{len(self.project.frames)} frames in {self.project.name}\nUse only one roll captured with fixed exposure.");self.mark_dirty()
        if self.roll_list.count() and self.roll_list.currentRow()<0:self.roll_list.setCurrentRow(0)
    def remove_frames(self):
        rows=sorted({self.roll_list.row(x) for x in self.roll_list.selectedItems()},reverse=True)
        for row in rows:self.project.frames.pop(row);self.roll_list.takeItem(row)
        self.mark_dirty()
    def load_frame(self,index):
        if not 0<=index<len(self.project.frames):return
        try:
            path=self.project.frames[index].path;signature=json.dumps(asdict(self.project.calibration),sort_keys=True)+f"|{Path(path).stat().st_mtime_ns}";cached=self.preview_cache.get(path,signature)
            if cached is not None:self.current_image=np.asarray(cached);self.current_meta={"cached":True,"bit_depth":16};cal_warnings=[];self._log("CACHE",f"Loaded calibrated preview cache: {path}")
            else:
                source,self.current_meta=read_linear(path);self.current_image,cal_warnings=apply_capture_calibration(source,self.project.calibration);self.preview_cache.put(path,signature,self.current_image)
            self._load_adjustments();self.render();analysis=analyze_frame(self.current_image,self.current_meta);self.update_memory_estimate();self._log("FRAME",json.dumps({"path":path,"calibration_warnings":cal_warnings,**analysis}))
        except Exception as exc:self._log("ERROR",traceback.format_exc());QMessageBox.critical(self,"Load failed",str(exc))
    def _load_adjustments(self):
        rec=self.current_record()
        if not rec:return
        for key,widget in self.adjust.items():widget.blockSignals(True);widget.setValue(getattr(rec.recipe,key));widget.blockSignals(False)
        for widget,value in ((self.fade,rec.recipe.fade_correction),(self.dust,rec.recipe.dust_removal),(self.dust_radius,rec.recipe.dust_max_radius),(self.sprocket,rec.recipe.sprocket_mask)):
            widget.blockSignals(True);widget.setValue(value) if hasattr(widget,"setValue") else widget.setChecked(value);widget.blockSignals(False)
        darkroom_widgets=[(self.fine_rotation,rec.recipe.fine_rotation),(self.clipping,rec.recipe.show_clipping)]
        darkroom_widgets.extend((self.film_controls[k],getattr(rec.recipe,k)) for k in self.film_controls)
        for widget,value in darkroom_widgets:
            widget.blockSignals(True);widget.setValue(value) if hasattr(widget,"setValue") else widget.setChecked(value);widget.blockSignals(False)
        crop=rec.recipe.crop or [0,0,1,1]
        for widget,value in zip(self.crop_controls.values(),crop):widget.blockSignals(True);widget.setValue(float(value)*100);widget.blockSignals(False)
        self.refresh_mask_controls()
    def adjust_changed(self,key,value):
        rec=self.current_record()
        if rec:setattr(rec.recipe,key,float(value));self.mark_dirty();self.preview_timer_start()
    def preview_timer_start(self):
        if not hasattr(self,"preview_timer"):self.preview_timer=QTimer(self);self.preview_timer.setSingleShot(True);self.preview_timer.timeout.connect(self.render)
        self.preview_timer.start(100)
    def render(self):
        rec=self.current_record()
        if self.current_image is None or not rec:return
        if self.show_original.isChecked():result=transform_output(self.current_image,self.project.calibration)[0]
        elif not self.project.clear_base.valid or (self.project.mode=="two_point" and not self.project.dense_leader.valid):result=transform_output(self.current_image,self.project.calibration)[0]
        else:result=preview_convert(self.current_image,self.project,rec.recipe,self.project.performance.preview_dimension)
        if not self.show_original.isChecked() and self.project.clear_base.valid and (self.project.mode!="two_point" or self.project.dense_leader.valid):result=transform_output(result,self.project.calibration)[0]
        if getattr(rec.recipe,"show_clipping",False) and not self.show_original.isChecked():
            result,counts=clipping_overlay(result);self.pixel_status.setText(f"Clipping overlay — shadows: {counts['shadow_pixels']:,}, highlights: {counts['highlight_pixels']:,}")
        if self.show_mask_overlay.isChecked() and rec.recipe.masks and not self.show_original.isChecked():
            mask=combined_mask(result.shape,rec.recipe.masks)[...,None];result=result*(1-mask*.42)+np.asarray([0,.95,1],np.float32)*mask*.42
        self.last_render=np.asarray(result,np.float32);self.canvas.set_image(self.last_render);self._update_anchor_quality()
    def update_pixel_status(self,x,y):
        image=getattr(self.canvas,"array",None)
        if image is None:return
        h,w=image.shape[:2];px=min(w-1,max(0,int(round(x*(w-1)))));py=min(h-1,max(0,int(round(y*(h-1)))));rgb=image[py,px,:3];self.pixel_status.setText(f"x {px:,}   y {py:,}   R {rgb[0]:.5f}   G {rgb[1]:.5f}   B {rgb[2]:.5f}   ({self.canvas.zoom*100:.0f}%)")
    def open_comparison(self):
        if self.current_image is None or not hasattr(self,"last_render"):return
        before=transform_output(self.current_image,self.project.calibration)[0];dialog=QDialog(self);dialog.setWindowTitle("Before / After Comparison");dialog.resize(1400,820);layout=QVBoxLayout(dialog);row=QHBoxLayout()
        for title,image in (("Scanned negative",before),("Converted result",self.last_render)):
            column=QVBoxLayout();heading=QLabel(title);heading.setObjectName("heading");column.addWidget(heading);view=ImageCanvas();view.set_image(image);column.addWidget(view,1);row.addLayout(column,1)
        layout.addLayout(row,1);buttons=QDialogButtonBox(QDialogButtonBox.Close);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons);dialog.exec()
    def open_fullscreen(self):
        if not hasattr(self,"last_render"):return
        dialog=QDialog(self);dialog.setWindowTitle("Negative Lab Full-Screen Preview — Esc to close");dialog.showFullScreen();layout=QVBoxLayout(dialog);view=ImageCanvas();view.set_image(self.last_render);layout.addWidget(view);QShortcut(QKeySequence("Esc"),dialog,dialog.reject);dialog.exec()
    def open_histogram(self):
        if not hasattr(self,"last_render"):return
        hist,stats=rgb_histogram(self.last_render);canvas=np.full((360,760,3),9,np.uint8);maximum=max(float(np.percentile(hist,99.5)),1);colors=((255,51,51),(51,255,89),(64,115,255))
        for channel,color in enumerate(colors):
            values=np.clip(hist[channel]/maximum,0,1);points=np.asarray([[i*759/(len(values)-1),340-values[i]*320] for i in range(len(values))],np.int32);cv2.polylines(canvas,[points],False,color,2,cv2.LINE_AA)
        dialog=QDialog(self);dialog.setWindowTitle("RGB Histogram and Clipping");dialog.resize(820,520);layout=QVBoxLayout(dialog);view=ImageCanvas();view.set_image(canvas.astype(np.float32)/255);layout.addWidget(view,1);summary=QLabel(f"Pixels: {stats['pixels']:,}    Shadow clipped: {stats['shadow_clipped']:,}    Highlight clipped: {stats['highlight_clipped']:,}");summary.setObjectName("card");layout.addWidget(summary);buttons=QDialogButtonBox(QDialogButtonBox.Close);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons);dialog.exec()
    def arm_sample(self,mode):
        if self.current_image is None:QMessageBox.information(self,"No image","Load a frame containing the required film area first.");return
        self.sample_mode=mode
        if mode in ("base","dense"):self.show_original.setChecked(True)
        self.statusBar().showMessage("Click a clean clear-film-base area" if mode=="base" else ("Click a uniformly dense fully exposed leader area" if mode=="dense" else "Click the center of the local correction"))
    def canvas_sampled(self,x,y):
        if not self.sample_mode:return
        if self.sample_mode=="mask":
            rec=self.current_record();operation=self.mask_operation.currentText().lower();rec.recipe.masks.append({"name":f"{operation.title()} Brush {len(rec.recipe.masks)+1}","enabled":True,"type":"brush","center":[x,y],"radius":self.mask_radius.value()/max(min(self.current_image.shape[:2]),1),"feather":.65,"operation":operation,"amount":self.mask_amount.value(),"offset":[self.mask_radius.value(),0]});self.refresh_mask_controls();self.sample_mode="";self.mark_dirty();self.render();self._log("MASK",f"{operation} mask at ({x:.3f},{y:.3f})");return
        if self.sample_mode=="gradient_start":self._gradient_start=[x,y];self.sample_mode="gradient_end";self.statusBar().showMessage("Click the gradient end point");return
        if self.sample_mode=="gradient_end":
            rec=self.current_record();operation=self.mask_operation.currentText().lower();rec.recipe.masks.append({"name":f"{operation.title()} Gradient {len(rec.recipe.masks)+1}","enabled":True,"type":"gradient","start":self._gradient_start,"end":[x,y],"operation":operation,"amount":self.mask_amount.value(),"offset":[self.mask_radius.value(),0]});self.sample_mode="";self.refresh_mask_controls();self.mark_dirty();self.render();self._log("MASK",f"{operation} gradient {self._gradient_start} to {[x,y]}");return
        if self.sample_mode=="local":
            rec=self.current_record();rec.recipe.local_adjustments.append({"x":x,"y":y,"radius":self.local_radius.value()/max(min(self.current_image.shape[:2]),1),"exposure":self.local_exposure.value(),"saturation":0,"hardness":.6});self._log("LOCAL",f"frame={rec.frame_number} x={x:.4f} y={y:.4f} exposure={self.local_exposure.value():.2f}");self.sample_mode="";self.mark_dirty();self.render();return
        sample=sample_anchor(self.current_image,x,y,self.radius.value(),self.current_record().path)
        if self.sample_mode=="base":self.project.clear_base=sample
        else:self.project.dense_leader=sample
        self._log("ANCHOR",f"{self.sample_mode} RGB={sample.rgb} spread={sample.spread} source={sample.source_path} xy=({x:.4f},{y:.4f})");self.sample_mode="";self.mark_dirty();self._update_anchor_labels();self.render()
    def _update_anchor_labels(self):
        fmt=lambda a:", ".join(f"{x:.5f}" for x in a.rgb) if a.valid else "Not sampled";self.base_value.setText(fmt(self.project.clear_base));self.dense_value.setText(fmt(self.project.dense_leader))
    def _update_anchor_quality(self):
        metrics,warnings=anchor_diagnostics(self.project.clear_base,self.project.dense_leader);score=metrics.get("quality_score",0);text=f"Anchor quality: {score*100:.0f}%" if metrics.get("valid") else "Anchors incomplete or invalid"
        if warnings:text+="\n• "+"\n• ".join(warnings)
        self.anchor_quality.setText(text)
    def profile_changed(self,index):
        profiles=FilmProfile.builtins()
        if 0<=index<len(profiles):self.project.film_profile=copy.deepcopy(profiles[index]);self.profile_description.setText(self.project.film_profile.description);self.mark_dirty();self.render()
    def load_film_profile(self):
        path,_=QFileDialog.getOpenFileName(self,"Load Film Profile","","Negative Lab Film Profile (*.nlfilm.json *.json)")
        if path:
            self.project.film_profile=load_profile(path);index=self.profile_combo.findText(self.project.film_profile.name)
            if index<0:self.profile_combo.addItem(self.project.film_profile.name);index=self.profile_combo.count()-1
            self.profile_combo.blockSignals(True);self.profile_combo.setCurrentIndex(index);self.profile_combo.blockSignals(False);self.profile_description.setText(self.project.film_profile.description);self._log("PROFILE",f"Loaded {path}");self.mark_dirty();self.render()
    def save_film_profile(self):
        path,_=QFileDialog.getSaveFileName(self,"Save Film Profile",self.project.film_profile.name+".nlfilm.json","Negative Lab Film Profile (*.nlfilm.json)")
        if path:save_profile(self.project.film_profile,path);self._log("PROFILE",f"Saved {path}")
    def restoration_changed(self,key,value):
        rec=self.current_record()
        if rec:setattr(rec.recipe,key,value);self.mark_dirty();self.preview_timer_start()
    def clear_local(self):
        rec=self.current_record()
        if rec:rec.recipe.local_adjustments=[];self.mark_dirty();self.render()

    def darkroom_value(self,key,value):
        rec=self.current_record()
        if rec:setattr(rec.recipe,key,value);self.mark_dirty();self.preview_timer_start()
    def rotate_quarter(self,direction):
        rec=self.current_record()
        if rec:rec.recipe.rotation=(rec.recipe.rotation+direction)%4;self.mark_dirty();self.render()
    def manual_crop_changed(self):
        rec=self.current_record()
        if rec:
            values=[self.crop_controls[k].value()/100 for k in ("x","y","w","h")];values[2]=min(values[2],1-values[0]);values[3]=min(values[3],1-values[1]);rec.recipe.crop=None if np.allclose(values,[0,0,1,1]) else values;self.mark_dirty();self.preview_timer_start()
    def edit_perspective(self):
        rec=self.current_record()
        if not rec:return
        current=rec.recipe.perspective_corners or [[0,0],[1,0],[1,1],[0,1]];text=", ".join(str(v) for point in current for v in point);value,ok=QInputDialog.getText(self,"Perspective corners","Normalized TL x,y, TR x,y, BR x,y, BL x,y",text=text)
        if ok:
            try:
                numbers=[float(x.strip()) for x in value.split(",")]
                if len(numbers)!=8 or any(x<0 or x>1 for x in numbers):raise ValueError
                rec.recipe.perspective_corners=[numbers[i:i+2] for i in range(0,8,2)];self.mark_dirty();self.render();self._log("GEOMETRY",f"Perspective corners {rec.recipe.perspective_corners}")
            except ValueError:QMessageBox.warning(self,"Invalid corners","Enter eight comma-separated values from 0 to 1.")
    def apply_s_curve(self):
        rec=self.current_record()
        if rec:rec.recipe.curves={"RGB":[[0,0],[.22,.16],[.5,.5],[.78,.84],[1,1]]};self.mark_dirty();self.render()
    def edit_curve(self):
        rec=self.current_record()
        if not rec:return
        current=rec.recipe.curves.get("RGB",[[0,0],[1,1]]);text="; ".join(f"{x:g},{y:g}" for x,y in current);value,ok=QInputDialog.getText(self,"Edit RGB Curve","Control points as input,output pairs separated by semicolons",text=text)
        if ok:
            try:
                points=[[float(v) for v in pair.split(",")] for pair in value.split(";")];
                if len(points)<2 or any(len(p)!=2 or min(p)<0 or max(p)>1 for p in points):raise ValueError
                rec.recipe.curves["RGB"]=sorted(points);self.mark_dirty();self.render();self._log("CURVE",str(rec.recipe.curves["RGB"]))
            except ValueError:QMessageBox.warning(self,"Invalid curve","Use entries such as 0,0; 0.25,0.2; 0.75,0.8; 1,1")
    def reset_curves(self):
        rec=self.current_record()
        if rec:rec.recipe.curves={"RGB":[[0,0],[1,1]]};self.mark_dirty();self.render()
    def add_gradient_mask(self):
        rec=self.current_record()
        if rec:rec.recipe.masks.append({"type":"gradient","start":[0,0],"end":[1,0],"operation":self.mask_operation.currentText().lower(),"amount":self.mask_amount.value(),"offset":[self.mask_radius.value(),0]});self.mask_summary.setText(f"{len(rec.recipe.masks)} nondestructive mask(s) on this frame");self.mark_dirty();self.render()
    def clear_masks(self):
        rec=self.current_record()
        if rec:rec.recipe.masks=[];self.refresh_mask_controls();self.mark_dirty();self.render()
    def refresh_mask_controls(self):
        rec=self.current_record();self.mask_list.blockSignals(True);self.mask_list.clear()
        if rec:
            for index,mask in enumerate(rec.recipe.masks):self.mask_list.addItem(mask.get("name",f"{mask.get('operation','Mask').title()} {index+1}"))
        self.mask_list.blockSignals(False);self.mask_summary.setText(f"{len(rec.recipe.masks)} nondestructive mask(s) on this frame" if rec and rec.recipe.masks else "No masks on this frame");self.mask_selection_changed(self.mask_list.currentIndex())
    def mask_selection_changed(self,index):
        rec=self.current_record();enabled=bool(rec and 0<=index<len(rec.recipe.masks));self.mask_enabled.blockSignals(True);self.mask_enabled.setEnabled(enabled);self.mask_enabled.setChecked(bool(rec.recipe.masks[index].get("enabled",True)) if enabled else False);self.mask_enabled.blockSignals(False)
    def mask_enabled_changed(self,value):
        rec=self.current_record();index=self.mask_list.currentIndex()
        if rec and 0<=index<len(rec.recipe.masks):rec.recipe.masks[index]["enabled"]=bool(value);self.mark_dirty();self.render()
    def remove_selected_mask(self):
        rec=self.current_record()
        index=self.mask_list.currentIndex()
        if rec and 0<=index<len(rec.recipe.masks):rec.recipe.masks.pop(index);self.refresh_mask_controls();self.mark_dirty();self.render()
    def _roll_proxies(self,limit=None):
        images=[];paths=[]
        for rec in self.project.frames[:limit]:
            try:
                image,_=read_linear(rec.path);scale=min(1,900/max(image.shape[:2]));images.append(cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else image);paths.append(rec.path)
            except Exception as exc:self._log("WARNING",f"Skipped {rec.path}: {exc}")
        return images,paths
    def auto_discover_anchors(self):
        images,paths=self._roll_proxies()
        if not images:return
        result=discover_anchor_candidates(images,paths);base=result["clear_base"][0];dense=result["dense_leader"][0]
        answer=QMessageBox.question(self,"Anchor candidates",f"Best clear base: frame {base['frame']+1}, RGB {np.round(base['rgb'],4)}\nBest dense leader: frame {dense['frame']+1}, RGB {np.round(dense['rgb'],4)}\n\nApply these candidates?",QMessageBox.Yes|QMessageBox.No)
        if answer==QMessageBox.Yes:
            from .models import AnchorSample
            self.project.clear_base=AnchorSample(base["rgb"],[base["spread"]]*3,base["path"],base["x"],base["y"],9,True);self.project.dense_leader=AnchorSample(dense["rgb"],[dense["spread"]]*3,dense["path"],dense["x"],dense["y"],9,True);self._update_anchor_labels();self.mark_dirty();self.render();self._log("ANCHOR","Automatic candidates approved")
    def analyze_roll_consistency(self):
        images,_=self._roll_proxies();report=roll_consistency(images);lines=[f"Frame {x['frame']+1}: suggested {x['suggested_exposure_ev']:+.2f} EV, color deviation {x['color_deviation']:.3f}" for x in report["frames"]];self._log("CONSISTENCY",json.dumps(report));QMessageBox.information(self,"Roll consistency","\n".join(lines[:30]) or "No readable frames")
    def clean_with_infrared(self):
        if self.current_image is None:return
        path,_=QFileDialog.getOpenFileName(self,"Choose matching infrared channel","","Images (*.tif *.tiff *.png)")
        if not path:return
        try:
            ir,_=read_linear(path);ir=cv2.resize(ir,(self.current_image.shape[1],self.current_image.shape[0]));clean,mask=infrared_clean(self.current_image,ir);target,_=QFileDialog.getSaveFileName(self,"Save infrared-cleaned scan","infrared_cleaned.tif","TIFF (*.tif)")
            if target:write_image(target,clean,16);self._log("INFRARED",json.dumps({"source":path,"defect_pixels":int(np.count_nonzero(mask)),"output":target}))
        except Exception as exc:QMessageBox.critical(self,"Infrared cleanup failed",str(exc))
    def multi_exposure_fusion(self):
        paths,_=QFileDialog.getOpenFileNames(self,"Choose bracketed scans of one negative","","Images (*.tif *.tiff *.png *.jpg *.jpeg)")
        if len(paths)<2:return
        try:
            images=[read_linear(p)[0] for p in paths];shape=images[len(images)//2].shape[:2];images=[cv2.resize(x,(shape[1],shape[0])) if x.shape[:2]!=shape else x for x in images];result,report=fuse_exposures(images);target,_=QFileDialog.getSaveFileName(self,"Save fused scan","fused_scan.tif","TIFF (*.tif)")
            if target:write_image(target,result,16);self._log("FUSION",json.dumps({"sources":paths,"output":target,**report}))
        except Exception as exc:QMessageBox.critical(self,"Fusion failed",str(exc))
    def camera_scan_check(self):
        if self.current_image is None:return
        report=camera_scan_assessment(self.current_image);self._log("CAPTURE",json.dumps(report));QMessageBox.information(self,"Camera-Scan Assistant","\n".join(f"{k.replace('_',' ').title()}: {v:.3f}" for k,v in report.items()))
    def _positive_proxies(self,accepted_only=False):
        images,labels,indices=[],[],[]
        for index,rec in enumerate(self.project.frames):
            if accepted_only and rec.rejected:continue
            try:
                source,_=read_linear(rec.path);images.append(preview_convert(source,self.project,rec.recipe,700) if self.project.clear_base.valid and self.project.dense_leader.valid else source);stars="★"*rec.rating if rec.rating else "Unrated";labels.append(f"{rec.frame_number:02d}  {Path(rec.path).name}  {stars}");indices.append(index)
            except Exception as exc:self._log("WARNING",f"Light table skipped {rec.path}: {exc}")
        return images,labels,indices
    def open_light_table(self):
        images,labels,indices=self._positive_proxies()
        if not images:return
        dialog=QDialog(self);dialog.setWindowTitle("Virtual Light Table — Rate, Label, Cull, and Annotate");dialog.resize(1350,850);layout=QVBoxLayout(dialog);filters=QHBoxLayout();filters.addWidget(QLabel("Minimum rating"));minimum_rating=QComboBox();minimum_rating.addItems(["All","1+","2+","3+","4+","5 only"]);filters.addWidget(minimum_rating);show_rejected=QCheckBox("Show rejected frames");show_rejected.setChecked(True);filters.addWidget(show_rejected);filter_count=QLabel();filters.addWidget(filter_count);filters.addStretch();layout.addLayout(filters);split=QSplitter();table=QListWidget();table.setViewMode(QListWidget.IconMode);table.setResizeMode(QListWidget.Adjust);table.setIconSize(QSize(180,130));table.setGridSize(QSize(205,185));table.setSelectionMode(QAbstractItemView.SingleSelection)
        for proxy_index,(image,label_text) in enumerate(zip(images,labels)):
            index=indices[proxy_index];data=np.clip(image*255,0,255).astype(np.uint8);q=QImage(data.data,data.shape[1],data.shape[0],data.strides[0],QImage.Format_RGB888).copy();rec=self.project.frames[index];stars="★"*rec.rating+"☆"*(5-rec.rating);item=QListWidgetItem(QIcon(QPixmap.fromImage(q)),f"{label_text}\n{stars}  {rec.color_label}"+("  [REJECT]" if rec.rejected else ""));item.setData(Qt.UserRole,index);item.setData(Qt.UserRole+1,proxy_index);item.setToolTip(rec.notes or rec.path);item.setForeground(QColor("#f87171") if rec.rejected else QColor("#e7eef9"));table.addItem(item)
        editor=QWidget();form=QFormLayout(editor);rating=QComboBox();rating.addItems(["0 — Unrated","1 ★","2 ★★","3 ★★★","4 ★★★★","5 ★★★★★"]);label=QComboBox();label.addItems(["None","Red","Yellow","Green","Blue","Purple"]);rejected=QCheckBox("Reject this frame");notes=QPlainTextEdit();notes.setPlaceholderText("Frame notes, printing instructions, subject information…");notes.setMaximumHeight(180);form.addRow("Rating",rating);form.addRow("Color label",label);form.addRow(rejected);form.addRow("Notes",notes);applyb=QPushButton("Apply Metadata");applyb.setObjectName("primary");form.addRow(applyb);openb=QPushButton("Open Selected in Develop");form.addRow(openb);info=QLabel("Double-click a thumbnail to return to that frame. Ratings, labels, rejection flags, and notes are saved in the roll project.");info.setObjectName("card");info.setWordWrap(True);form.addRow(info);form.addItem(QSpacerItem(1,1,QSizePolicy.Minimum,QSizePolicy.Expanding));split.addWidget(table);split.addWidget(editor);split.setSizes([1000,320]);layout.addWidget(split,1)
        def load_metadata(row):
            if row<0:return
            rec=self.project.frames[table.item(row).data(Qt.UserRole)];rating.setCurrentIndex(rec.rating);label.setCurrentText(rec.color_label);rejected.setChecked(rec.rejected);notes.setPlainText(rec.notes)
        def apply_metadata():
            item=table.currentItem()
            if not item:return
            index=item.data(Qt.UserRole);proxy_index=item.data(Qt.UserRole+1);rec=self.project.frames[index];rec.rating=rating.currentIndex();rec.color_label=label.currentText();rec.rejected=rejected.isChecked();rec.notes=notes.toPlainText();stars="★"*rec.rating+"☆"*(5-rec.rating);item.setText(f"{labels[proxy_index]}\n{stars}  {rec.color_label}"+("  [REJECT]" if rec.rejected else ""));item.setForeground(QColor("#f87171") if rec.rejected else QColor("#e7eef9"));item.setToolTip(rec.notes or rec.path);self.mark_dirty();self._log("LIGHTTABLE",f"frame={rec.frame_number} rating={rec.rating} label={rec.color_label} rejected={rec.rejected}")
            apply_filter()
        def apply_filter():
            threshold=minimum_rating.currentIndex();visible=0
            for row in range(table.count()):
                item=table.item(row);rec=self.project.frames[item.data(Qt.UserRole)];hide=(threshold>0 and rec.rating<threshold) or (rec.rejected and not show_rejected.isChecked());item.setHidden(hide);visible+=not hide
            filter_count.setText(f"{visible} of {table.count()} frames shown")
        def open_selected():
            item=table.currentItem()
            if item:self.roll_list.setCurrentRow(item.data(Qt.UserRole));dialog.accept()
        table.currentRowChanged.connect(load_metadata);table.itemDoubleClicked.connect(lambda _item:open_selected());applyb.clicked.connect(apply_metadata);openb.clicked.connect(open_selected);minimum_rating.currentIndexChanged.connect(apply_filter);show_rejected.toggled.connect(apply_filter);buttons=QDialogButtonBox(QDialogButtonBox.Close);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons);table.setCurrentRow(0);apply_filter();dialog.exec()
    def export_contact_sheet(self):
        images,labels,_indices=self._positive_proxies(self.contact_accepted_only.isChecked())
        if not images:return
        path,_=QFileDialog.getSaveFileName(self,"Export Contact Sheet","contact_sheet.pdf","PDF (*.pdf);;JPEG (*.jpg);;TIFF (*.tif)")
        if path:
            sheet=contact_sheet(images,labels,5)
            if Path(path).suffix.lower()==".pdf":
                from PIL import Image
                Image.fromarray(np.clip(sheet*255,0,255).astype(np.uint8)).save(path,"PDF",resolution=self.project.calibration.dpi)
            else:write_image(path,sheet,16 if Path(path).suffix.lower() in (".tif",".tiff") else 8)
            self._log("CONTACT",f"{path} accepted_only={self.contact_accepted_only.isChecked()}")
    def export_roll_catalog_csv(self):
        path,_=QFileDialog.getSaveFileName(self,"Export Roll Catalog","negative_lab_roll_catalog.csv","CSV (*.csv)")
        if path:self._log("CATALOG",json.dumps(export_roll_catalog(self.project,path)))
    def export_archival_manifest(self):
        path,_=QFileDialog.getSaveFileName(self,"Export Archival Manifest","negative_lab_archive.json","JSON (*.json)")
        if path:archival_manifest(self.project,path);self._log("ARCHIVE",path)
    def verify_manifest(self):
        path,_=QFileDialog.getOpenFileName(self,"Verify Archival Manifest","","Negative Lab archive (*.json)")
        if not path:return
        try:
            report=verify_archival_manifest(path);counts=report["counts"];self._log("VERIFY",json.dumps(report));message=f"Verified sources: {counts['ok']}\nChanged sources: {counts['changed']}\nMissing sources: {counts['missing']}";(QMessageBox.information if report["ok"] else QMessageBox.warning)(self,"Archive verification",message)
        except Exception as exc:QMessageBox.critical(self,"Verification failed",str(exc))
    def choose_dark(self):
        path,_=QFileDialog.getOpenFileName(self,"Choose Matching Dark Frame","","Images (*.tif *.tiff *.png *.jpg *.jpeg *.dng *.nef *.cr2 *.arw)")
        if path:self.project.calibration.dark_frame_path=path;self.dark_label.setText(Path(path).name);self._log("CALIBRATION",f"Dark frame: {path}");self.invalidate_anchors_for_calibration();self.reload_current()
    def choose_flat(self):
        path,_=QFileDialog.getOpenFileName(self,"Choose Matching Flat Field","","Images (*.tif *.tiff *.png *.jpg *.jpeg *.dng *.nef *.cr2 *.arw)")
        if path:
            self.project.calibration.flat_field_path=path;self.flat_label.setText(Path(path).name)
            try:flat,_=read_linear(path);dark=read_linear(self.project.calibration.dark_frame_path)[0] if self.project.calibration.dark_frame_path else None;self._log("CALIBRATION","Flat diagnostics: "+json.dumps(flat_field_diagnostics(flat,dark)))
            except Exception as exc:self._log("WARNING",f"Flat diagnostics failed: {exc}")
            self.invalidate_anchors_for_calibration();self.reload_current()
    def reload_current(self):
        self.mark_dirty();row=self.roll_list.currentRow()
        if row>=0:self.load_frame(row)
    def _set_input_matrix(self,matrix,name,path):
        self.project.calibration.input_matrix=matrix;self.project.calibration.input_profile_name=name;self.project.calibration.input_profile_path=path;self.input_profile_label.setText(name);self._log("COLOR",f"Input matrix {name}: {matrix}");self.invalidate_anchors_for_calibration();self.reload_current()
    def invalidate_anchors_for_calibration(self):
        if self.project.clear_base.valid or self.project.dense_leader.valid:
            self.project.clear_base.valid=False;self.project.dense_leader.valid=False;self._update_anchor_labels();self._log("WARNING","Calibration changed; prior density anchors were invalidated and must be sampled again")
    def load_dcp(self):
        path,_=QFileDialog.getOpenFileName(self,"Load DCP Profile","","DNG Camera Profile (*.dcp)")
        if path:
            try:matrix,name=load_dcp_matrix(path);self._set_input_matrix(matrix,f"{Path(path).name} — {name}",path)
            except Exception as exc:QMessageBox.critical(self,"DCP error",str(exc))
    def load_matrix(self):
        path,_=QFileDialog.getOpenFileName(self,"Load Matrix Profile","","Matrix JSON (*.json)")
        if path:
            try:matrix,name=load_matrix_json(path);self._set_input_matrix(matrix,name,path)
            except Exception as exc:QMessageBox.critical(self,"Matrix error",str(exc))
    def load_it8(self):
        path,_=QFileDialog.getOpenFileName(self,"Load IT8/CGATS Paired RGB Data","","CGATS text (*.txt *.ti3 *.cgats)")
        if path:
            try:measured,reference=parse_it8_pairs(path);matrix,stats=fit_it8_matrix(measured,reference);self._set_input_matrix(matrix,f"IT8 fit — {Path(path).name}",path);self._log("IT8",json.dumps(stats));QMessageBox.information(self,"IT8 fit",f"Patches: {stats['patches']}\nRMSE: {stats['rmse']:.6f}\nMaximum error: {stats['max_error']:.6f}")
            except Exception as exc:QMessageBox.critical(self,"IT8 error",str(exc))
    def set_calibration_value(self,key,value):setattr(self.project.calibration,key,value);self.mark_dirty()
    def output_space_changed(self,text):self.set_calibration_value("output_space",text)
    def choose_output_icc(self):
        path,_=QFileDialog.getOpenFileName(self,"Choose Output ICC Profile","","ICC profiles (*.icc *.icm)")
        if path:self.project.calibration.output_icc_path=path;self.project.calibration.output_space="Custom ICC";self.output_space.setCurrentText("Custom ICC");self.output_icc_label.setText(Path(path).name);self._log("COLOR",f"Output ICC: {path}");self.mark_dirty()
    def compare_profiles(self):
        rec=self.current_record()
        if self.current_image is None or not rec or not self.project.clear_base.valid or not self.project.dense_leader.valid:QMessageBox.information(self,"Profiles","Load a frame and sample valid anchors first.");return
        dialog=QDialog(self);dialog.setWindowTitle("Film Profile Comparison");dialog.resize(1100,750);layout=QVBoxLayout(dialog);tabs=QTabWidget();layout.addWidget(tabs,1)
        for profile in FilmProfile.builtins():
            candidate=copy.deepcopy(self.project);candidate.film_profile=profile;view=ImageCanvas();developed=preview_convert(self.current_image,candidate,rec.recipe,1200);view.set_image(transform_output(developed,candidate.calibration)[0]);tabs.addTab(view,profile.name)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons);dialog.exec()
    def build_trichrome(self):
        paths,_=QFileDialog.getOpenFileNames(self,"Choose Red, Green, and Blue Records in That Order","","Images (*.tif *.tiff *.png *.jpg *.jpeg)")
        if len(paths)!=3:
            if paths:QMessageBox.information(self,"Trichrome","Select exactly three monochrome records in red, green, blue order.")
            return
        try:
            images=[read_linear(path)[0] for path in paths];result,transforms=trichrome_merge(*images);target,_=QFileDialog.getSaveFileName(self,"Save Trichrome","trichrome.tif","TIFF (*.tif *.tiff)")
            if target:write_image(target,result,16);self._log("TRICHROME",json.dumps({"sources":paths,"transforms":transforms,"output":target}))
        except Exception as exc:self._log("ERROR",traceback.format_exc());QMessageBox.critical(self,"Trichrome failed",str(exc))
    def detect_current_frame(self):
        rec=self.current_record()
        if rec and self.current_image is not None:rec.recipe.crop,confidence=detect_frame_bounds(self.current_image);self._log("FRAMING",f"{rec.path} crop={rec.recipe.crop} confidence={confidence:.3f}");self._load_adjustments();self.render();self.mark_dirty()
    def detect_all_frames(self):
        for i,rec in enumerate(self.project.frames):
            try:image,_=read_linear(rec.path);rec.recipe.crop,confidence=detect_frame_bounds(image);self._log("FRAMING",f"frame={i+1} confidence={confidence:.3f} crop={rec.recipe.crop}")
            except Exception as exc:self._log("WARNING",f"Framing skipped {rec.path}: {exc}")
        self.render();self.mark_dirty()
    def sync_selected(self):
        source=self.roll_list.currentRow();targets=sorted({self.roll_list.row(x) for x in self.roll_list.selectedItems()}-{source});groups=[n for n,c in self.sync_checks.items() if c.isChecked()]
        if source<0 or not targets or not groups:QMessageBox.information(self,"Nothing to synchronize","Keep the source frame current, select target frames with Ctrl/Shift-click, and choose at least one group.");return
        sync_recipes(self.project,source,targets,groups);self._log("SYNC",f"source={source+1} targets={[x+1 for x in targets]} groups={groups}");self.mark_dirty()
    def start_batch(self):
        if not self.project.frames:return
        required=self.current_memory_estimate();available=available_memory_bytes()
        if available and required>available*self.project.performance.memory_warning_percent/100:
            answer=QMessageBox.question(self,"High memory estimate",f"Estimated peak per frame: {human_bytes(required)}\nCurrently available RAM: {human_bytes(available)}\n\nContinue?",QMessageBox.Yes|QMessageBox.No)
            if answer!=QMessageBox.Yes:return
        directory=QFileDialog.getExistingDirectory(self,"Choose converted-roll output folder",self.project.output_directory)
        if not directory:return
        self.project.output_directory=directory;self._batch_thread=QThread();self._batch_worker=BatchWorker(self.project,directory);self._batch_worker.moveToThread(self._batch_thread);self._batch_thread.started.connect(self._batch_worker.run);self._batch_worker.progress.connect(self.batch_progress);self._batch_worker.finished.connect(self.batch_finished);self._batch_worker.failed.connect(self.batch_failed);self._batch_worker.finished.connect(self._batch_thread.quit);self._batch_worker.failed.connect(self._batch_thread.quit);self._batch_thread.finished.connect(self._batch_worker.deleteLater);self._batch_thread.finished.connect(self._batch_thread.deleteLater);self.batch_button.setEnabled(False);self.cancel_button.setEnabled(True);self.progress.show();self._batch_thread.start();self._log("BATCH",f"Started {len(self.project.frames)} frames → {directory}")
    def cancel_batch(self):
        if hasattr(self,"_batch_worker"):self._batch_worker.cancel();self._log("CANCEL","Batch cancellation requested")
    def batch_progress(self,value,text):self.progress.setValue(value);self.statusBar().showMessage(text);self._log("PROGRESS",f"{value}% {text}")
    def batch_finished(self,report):self.batch_button.setEnabled(True);self.cancel_button.setEnabled(False);self.progress.hide();self._log("SUCCESS",json.dumps(report));QMessageBox.information(self,"Roll complete",f"Converted roll saved to:\n{self.project.output_directory}")
    def batch_failed(self,error):self.batch_button.setEnabled(True);self.cancel_button.setEnabled(False);self.progress.hide();self._log("ERROR",error);QMessageBox.critical(self,"Conversion stopped",error.splitlines()[-1])
    def set_performance(self,key,value):
        setattr(self.project.performance,key,value);configure_performance(self.project.performance);self.preview_cache=DiskPreviewCache(self.project.performance.disk_preview_cache,self.project.performance.cache_directory);self.mark_dirty();self.update_memory_estimate()
    def set_output(self,key,value):setattr(self.project.output,key,value);self.output_preset.blockSignals(True);self.output_preset.setCurrentText("Custom");self.output_preset.blockSignals(False);self.mark_dirty()
    def apply_output_preset(self,name):
        presets={"Archival":("TIFF",16,100,False,"16-bit TIFF, full dimensions, embedded profile, no overwrite."),"Web":("JPEG",8,50,False,"High-quality 8-bit JPEG at 50% for convenient sharing."),"Proof":("JPEG",8,100,False,"Full-size JPEG proof while preserving archival originals.")}
        if name not in presets:return
        fmt,bits,resize,overwrite,description=presets[name];self.project.output.preset=name;self.project.output.format=fmt;self.project.output.bit_depth=bits;self.project.output.resize_percent=resize;self.project.output.overwrite_existing=overwrite
        for widget,value in ((self.output_format,fmt),(self.output_bits,str(bits)),(self.output_resize,resize),(self.output_overwrite,overwrite)):
            widget.blockSignals(True);widget.setCurrentText(value) if hasattr(widget,"setCurrentText") else (widget.setValue(value) if hasattr(widget,"setValue") else widget.setChecked(value));widget.blockSignals(False)
        self.output_description.setText(description);self.mark_dirty()
    def current_memory_estimate(self):
        if self.current_image is None:return 0
        return estimate_frame_memory(self.current_image.shape[1],self.current_image.shape[0],bool(self.project.calibration.dark_frame_path or self.project.calibration.flat_field_path),True)
    def update_memory_estimate(self):
        if not hasattr(self,"memory_label"):return
        estimate=self.current_memory_estimate();available=available_memory_bytes();config=configure_performance(self.project.performance);self.memory_label.setText((f"Estimated peak per full-resolution frame: {human_bytes(estimate)}\n" if estimate else "Load a frame to estimate memory.\n")+(f"Available RAM: {human_bytes(available)}\n" if available else "")+f"CPU/OpenCL: {json.dumps(config)}")
    def clear_preview_cache(self):self.preview_cache.clear();self.preview_cache=DiskPreviewCache(self.project.performance.disk_preview_cache,self.project.performance.cache_directory);self._log("CACHE","Preview cache cleared")
    def recovery_path(self):
        root=Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation));root.mkdir(parents=True,exist_ok=True);return root/"recovery.nlroll.autosave"
    def autosave(self):
        if self._dirty and self.project.frames:
            try:self.project.save(self.recovery_path());self._log("RECOVERY",f"Recovery point saved: {self.recovery_path()}")
            except Exception as exc:self._log("WARNING",f"Recovery save failed: {exc}")
    def offer_recovery(self):
        path=self.recovery_path()
        if path.exists() and QMessageBox.question(self,"Recovery available","Negative Lab found an automatic recovery project. Open it?",QMessageBox.Yes|QMessageBox.No)==QMessageBox.Yes:
            try:self.project=RollProject.load(path);self.project_path="";self.roll_list.clear();[self.roll_list.addItem(f"{r.frame_number:02d}  {Path(r.path).name}") for r in self.project.frames];self._apply_project_ui();self.roll_list.setCurrentRow(0 if self.project.frames else -1);self._dirty=True;self._log("RECOVERY",f"Recovered {path}")
            except Exception as exc:self._log("ERROR",f"Recovery failed: {exc}")
    def remember_project(self,path):
        recent=[path]+[p for p in self.settings.value("recent_projects",[],list) if p!=path and Path(p).exists()];self.settings.setValue("recent_projects",recent[:10])
    def open_recent_project(self):
        recent=[p for p in self.settings.value("recent_projects",[],list) if Path(p).exists()]
        if not recent:QMessageBox.information(self,"Recent projects","No saved recent projects.");return
        path,ok=QInputDialog.getItem(self,"Recent projects","Project",recent,0,False)
        if ok and path:self.load_project_path(path)
    def load_project_path(self,path):
        try:self.project=RollProject.load(path);self.project_path=path;self.roll_list.clear();[self.roll_list.addItem(f"{r.frame_number:02d}  {Path(r.path).name}") for r in self.project.frames];self._apply_project_ui();self.roll_summary.setText(f"{len(self.project.frames)} frames in {self.project.name}");self.roll_list.setCurrentRow(0 if self.project.frames else -1);self._dirty=False;self.remember_project(path);self._log("PROJECT",f"Loaded {path}")
        except Exception as exc:QMessageBox.critical(self,"Project error",str(exc))
    def create_diagnostic_package(self):
        path,_=QFileDialog.getSaveFileName(self,"Save Diagnostic Package","negative_lab_diagnostics.zip","ZIP (*.zip)")
        if not path:return
        manifest={"version":__version__,"system":system_diagnostics(),"project":self.project.to_dict(),"history":self.operation_history,"privacy":"No source image pixels included; paths and metadata may be present."}
        try:
            with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:archive.writestr("diagnostics.json",json.dumps(manifest,indent=2));archive.writestr("debug_console.log",self.diagnostics.toPlainText());archive.writestr("README.txt",manifest["privacy"])
            QApplication.clipboard().setText(path);self._log("DIAGNOSTIC",f"Saved package and copied path: {path}")
        except Exception as exc:QMessageBox.critical(self,"Diagnostic error",str(exc))
    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls():event.acceptProposedAction()
    def dropEvent(self,event):
        paths=[Path(url.toLocalFile()) for url in event.mimeData().urls()];projects=[p for p in paths if p.is_file() and p.name.lower().endswith((".nlroll.json",".json"))];images=[str(p) for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        for folder in [p for p in paths if p.is_dir()]:images.extend(str(f) for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS)
        if projects:self.load_project_path(str(projects[0]))
        if images:self.add_paths(images)
        event.acceptProposedAction()
    def new_project(self):
        self.project=RollProject();self.project_path="";self.roll_list.clear();self.current_image=None;self.canvas.set_image(None);self._apply_project_ui();self._dirty=False
    def save_project(self):
        path=self.project_path
        if not path:path,_=QFileDialog.getSaveFileName(self,"Save Roll Project",self.project.name+".nlroll.json","Negative Lab Roll (*.nlroll.json)")
        if path:self.project.save(path);self.project_path=path;self._dirty=False;self.remember_project(path);self._log("PROJECT",f"Saved {path}")
    def open_project(self):
        path,_=QFileDialog.getOpenFileName(self,"Open Roll Project","","Negative Lab Roll (*.nlroll.json *.json)")
        if not path:return
        self.load_project_path(path)
    def _apply_project_ui(self):
        self._update_anchor_labels();cal=self.project.calibration
        self.dark_label.setText(Path(cal.dark_frame_path).name if cal.dark_frame_path else "Not selected");self.flat_label.setText(Path(cal.flat_field_path).name if cal.flat_field_path else "Not selected");self.input_profile_label.setText(cal.input_profile_name);self.output_icc_label.setText(Path(cal.output_icc_path).name if cal.output_icc_path else "Built-in sRGB")
        for widget,value in ((self.output_space,cal.output_space),(self.output_dpi,cal.dpi),(self.embed_icc,cal.embed_icc)):
            widget.blockSignals(True);widget.setCurrentText(value) if hasattr(widget,"setCurrentText") else (widget.setValue(value) if hasattr(widget,"setValue") else widget.setChecked(value));widget.blockSignals(False)
        index=self.profile_combo.findText(self.project.film_profile.name)
        if index<0:self.profile_combo.addItem(self.project.film_profile.name);index=self.profile_combo.count()-1
        self.profile_combo.blockSignals(True);self.profile_combo.setCurrentIndex(index);self.profile_combo.blockSignals(False);self.profile_description.setText(self.project.film_profile.description)
        perf=self.project.performance;out=self.project.output
        self.preview_cache=DiskPreviewCache(perf.disk_preview_cache,perf.cache_directory)
        for widget,value in ((self.preview_dimension,perf.preview_dimension),(self.cpu_threads,perf.cpu_threads),(self.opencl,perf.use_opencl),(self.disk_cache,perf.disk_preview_cache),(self.recover_frames,perf.recover_failed_frames),(self.output_preset,out.preset),(self.output_format,out.format),(self.output_bits,str(out.bit_depth)),(self.output_resize,out.resize_percent),(self.output_overwrite,out.overwrite_existing)):
            widget.blockSignals(True);widget.setCurrentText(value) if hasattr(widget,"setCurrentText") else (widget.setValue(value) if hasattr(widget,"setValue") else widget.setChecked(value));widget.blockSignals(False)
        self.update_memory_estimate()
    def save_log(self):
        path,_=QFileDialog.getSaveFileName(self,"Save Diagnostics","negative_lab.log","Log (*.log *.txt)")
        if path:Path(path).write_text(self.diagnostics.toPlainText(),encoding="utf-8")
    def closeEvent(self,event):
        if hasattr(self,"_batch_thread") and self._batch_thread.isRunning():
            answer=QMessageBox.question(self,"Conversion running","Cancel the running conversion and close?",QMessageBox.Yes|QMessageBox.No)
            if answer!=QMessageBox.Yes:event.ignore();return
            self._batch_worker.cancel();self._batch_thread.quit();self._batch_thread.wait(5000)
        if self._dirty:
            answer=QMessageBox.question(self,"Unsaved roll","Save this roll project before closing?",QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel)
            if answer==QMessageBox.Cancel:event.ignore();return
            if answer==QMessageBox.Yes:self.save_project()
        event.accept()


STYLE="""
QMainWindow,QWidget{background:#101827;color:#e7eef9;font-family:'Segoe UI';font-size:10pt}QToolBar{background:#17243a;border-bottom:1px solid #334a68;padding:6px;spacing:4px}QToolButton{color:#dbeafe;padding:7px;border-radius:6px;font-weight:600}QToolButton:hover{background:#28517a}QTabWidget::pane{border:1px solid #334a68}QTabBar::tab{background:#17243a;color:#9fb3ca;padding:9px 15px}QTabBar::tab:selected{background:#0f7893;color:white}QListWidget,QScrollArea,QPlainTextEdit{background:#0b1321;border:1px solid #334a68;border-radius:7px}QListWidget::item{padding:6px;border-bottom:1px solid #20314b}QListWidget::item:selected{background:#0f7893}QGroupBox{border:1px solid #3b506c;border-radius:7px;margin-top:10px;padding-top:8px;font-weight:700}QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 5px}QPushButton{background:#2a425f;color:white;border:1px solid #4c6685;border-radius:7px;padding:8px;font-weight:600}QPushButton:hover{background:#35658d;border-color:#5ed4ef}QPushButton#primary{background:#0789a7;border-color:#68e1f5;font-size:11pt}QComboBox,QSpinBox,QDoubleSpinBox{background:#1d2b40;color:white;border:1px solid #526681;border-radius:5px;padding:5px}QLabel#heading{color:#67e8f9;font-size:13pt;font-weight:800}QLabel#card{background:#142b48;color:#cfe7ff;border-left:4px solid #22b8d6;border-radius:6px;padding:9px}QProgressBar{text-align:center;background:#1d2b40;border:1px solid #526681;border-radius:5px}QProgressBar::chunk{background:#16a394}QToolTip{background:#f8fafc;color:#111827;border:1px solid #22b8d6}
"""


def main():
    logging.basicConfig(level=logging.INFO);app=QApplication.instance() or QApplication(sys.argv);app.setApplicationName("Negative Lab");app.setStyleSheet(STYLE);window=MainWindow();window.show();return app.exec()


if __name__=="__main__":raise SystemExit(main())
