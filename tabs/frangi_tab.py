import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, 
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox, 
                             QCheckBox, QComboBox, QFrame, QStyleOption, QStyle, QSizePolicy,
                             QHBoxLayout)

from components.ui_help import HELP_FRANGI, show_help
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from skimage.filters import frangi, apply_hysteresis_threshold
from core.root_graph_builder import extract_skeleton
from scipy.ndimage import label, distance_transform_edt

def fixed_frangi(image, sigmas=range(1, 10, 1), **kwargs):
    # Create an empty array to hold our maximum values
    filtered_max = np.zeros_like(image, dtype=float)
    
    # Loop through the scales ourselves
    for sigma in sigmas:
        # Call the original frangi, but force it to look at only ONE scale
        current_result = frangi(image, sigmas=[sigma], **kwargs)
        
        # Combine by keeping the brightest pixels
        filtered_max = np.maximum(filtered_max, current_result)
    
    return filtered_max


def _build_filled_overlay_static(rgb_crop, binary_mask, multiclass_crop, class_colors):
    overlay = np.zeros_like(rgb_crop)
    for cid, color in class_colors.items():
        if cid != 0:
            overlay[(binary_mask > 0) & (multiclass_crop == cid)] = color[:3]
    alpha = 0.6
    out_image = rgb_crop.copy()
    active = binary_mask > 0
    for c in range(3):
        out_image[active, c] = (rgb_crop[active, c] * (1 - alpha) + overlay[active, c] * alpha).astype(np.uint8)
    return out_image


def _calculate_mapping_static(
    combined_mask, x1, y1, x2, y2, original_mask, current_cls_crop,
    untouched_mask_crop, target_classes,
):
    full_mask = np.zeros_like(original_mask)
    full_mask[y1:y2, x1:x2] = combined_mask

    new_multiclass_crop = np.zeros_like(combined_mask)
    new_multiclass_crop[untouched_mask_crop] = current_cls_crop[untouched_mask_crop]

    refined_area = (combined_mask > 0) & (~untouched_mask_crop)
    existing_targets = refined_area & np.isin(current_cls_crop, target_classes)
    new_multiclass_crop[existing_targets] = current_cls_crop[existing_targets]

    holes = refined_area & (new_multiclass_crop == 0)
    if np.any(holes):
        valid_targets = np.zeros_like(combined_mask)
        valid_targets[existing_targets] = current_cls_crop[existing_targets]
        if np.any(existing_targets):
            _, indices = distance_transform_edt(~existing_targets, return_indices=True)
            new_multiclass_crop[holes] = valid_targets[indices[0], indices[1]][holes]
        else:
            fallback = target_classes[0] if target_classes else 1
            new_multiclass_crop[holes] = fallback

    ys_c, xs_c = np.where(combined_mask)
    if len(xs_c) > 0:
        cx1, cy1 = int(xs_c.min()), int(ys_c.min())
        cx2, cy2 = int(xs_c.max()), int(ys_c.max())
        tight_patch = new_multiclass_crop[cy1:cy2 + 1, cx1:cx2 + 1]
        return full_mask, tight_patch, x1 + cx1, y1 + cy1, new_multiclass_crop

    return full_mask, None, 0, 0, new_multiclass_crop


def compute_frangi_proposal(snapshot):
    """Heavy Frangi pipeline; safe to run off the UI thread."""
    gray_crop = snapshot["gray_crop"]
    mask_crop = snapshot["mask_crop"]
    current_cls_crop = snapshot["current_cls_crop"]
    target_mask_crop = snapshot["target_mask_crop"]
    untouched_mask_crop = snapshot["untouched_mask_crop"]
    search_mask = snapshot["search_mask"]
    original_mask = snapshot["original_mask"]
    x1, y1, x2, y2 = snapshot["x1"], snapshot["y1"], snapshot["x2"], snapshot["y2"]
    params = snapshot["params"]
    class_colors = snapshot["class_colors"]

    if params["centerline_correction"]:
        frangi_scales = np.arange(
            params["min_sigma"], params["max_sigma"] + 1, params["sigma_step"]
        )
        vesselness = fixed_frangi(gray_crop, black_ridges=params["roots_dark"], sigmas=frangi_scales)
        vesselness[search_mask == 0] = 0
    else:
        vesselness = np.zeros(gray_crop.shape, dtype=np.float32)

    vessel_8u = (vesselness * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(vessel_8u, cv2.COLORMAP_JET)
    heatmap_rgb_base = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    ch, cw = heatmap_rgb_base.shape[:2]
    pad_w = 40
    heatmap_rgb = np.zeros((ch, cw + pad_w, 3), dtype=np.uint8)
    heatmap_rgb[:, :cw] = heatmap_rgb_base

    cb_w = 10
    cb_h = max(1, min(150, ch - 4))
    cb_x = cw + 5
    cb_y = (ch - cb_h) // 2

    grad = np.linspace(255, 0, cb_h, dtype=np.uint8).reshape(-1, 1)
    grad_color = cv2.applyColorMap(np.tile(grad, (1, cb_w)), cv2.COLORMAP_JET)
    grad_rgb = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)

    heatmap_rgb[cb_y:cb_y + cb_h, cb_x:cb_x + cb_w] = grad_rgb
    cv2.rectangle(heatmap_rgb, (cb_x, cb_y), (cb_x + cb_w, cb_y + cb_h), (255, 255, 255), 1)
    cv2.putText(heatmap_rgb, "1.0", (cb_x + cb_w + 5, cb_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(heatmap_rgb, "0.0", (cb_x + cb_w + 5, cb_y + cb_h), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    kernel = np.ones((3, 3), np.uint8)
    if params["centerline_correction"]:
        binary_vessels = apply_hysteresis_threshold(
            vesselness, params["faint_sens"], params["strong_conf"]
        )
        labeled_frangi, num_frangi = label(binary_vessels, structure=np.ones((3, 3)))
        sizes = np.bincount(labeled_frangi.ravel())
        sizes[0] = 0

        core_fragments = np.zeros_like(binary_vessels, dtype=np.uint8)
        for i in range(1, num_frangi + 1):
            if sizes[i] >= params["min_part"]:
                core_fragments[labeled_frangi == i] = 1

        labeled_valid, num_valid = label(core_fragments, structure=np.ones((3, 3)))

        if num_valid <= 1:
            proposed_blob = core_fragments.copy()
        else:
            bridge_kernel = np.ones((3, 3), np.uint8)
            seen_dilations = np.zeros_like(core_fragments)
            bridge_seeds = np.zeros_like(core_fragments)

            for i in range(1, num_valid + 1):
                comp_mask = (labeled_valid == i).astype(np.uint8)
                dilated_comp = cv2.dilate(comp_mask, bridge_kernel, iterations=params["bridge_gaps"])
                intersection = cv2.bitwise_and(dilated_comp, seen_dilations)
                bridge_seeds = cv2.bitwise_or(bridge_seeds, intersection)
                seen_dilations = cv2.bitwise_or(seen_dilations, dilated_comp)

            if np.sum(bridge_seeds) > 0:
                local_bridges = cv2.dilate(bridge_seeds, bridge_kernel, iterations=params["bridge_gaps"])
                local_bridges = cv2.bitwise_and(local_bridges, search_mask)
                proposed_blob = cv2.bitwise_or(core_fragments, local_bridges)
            else:
                proposed_blob = core_fragments.copy()
    else:
        proposed_blob = target_mask_crop.copy().astype(np.uint8)

    proposed_skeleton = extract_skeleton(proposed_blob)[0].astype(np.uint8)
    proposed_mask_raw = cv2.dilate(proposed_skeleton, kernel, iterations=params["final_thick"])
    combined_raw = cv2.bitwise_or(proposed_mask_raw, untouched_mask_crop.astype(np.uint8))

    if not params["allow_disconnected"] and np.sum(combined_raw) > 0:
        labeled_prop, num_prop = label(combined_raw, structure=np.ones((3, 3)))
        if num_prop > 0:
            prop_sizes = np.bincount(labeled_prop.ravel())
            prop_sizes[0] = 0
            combined_proposed_mask = (labeled_prop == prop_sizes.argmax()).astype(np.uint8)
        else:
            combined_proposed_mask = combined_raw
    else:
        combined_proposed_mask = combined_raw

    proposed_full_mask, proposed_tight_patch, proposed_x, proposed_y, prop_mc = \
        _calculate_mapping_static(
            combined_proposed_mask, x1, y1, x2, y2, original_mask,
            current_cls_crop, untouched_mask_crop, params["target_classes"],
        )

    old_mc = np.zeros_like(mask_crop)
    old_mc[mask_crop > 0] = current_cls_crop[mask_crop > 0]
    rgb_base_canvas = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
    rgb_old = _build_filled_overlay_static(rgb_base_canvas, mask_crop, old_mc, class_colors)
    rgb_proposed = _build_filled_overlay_static(
        rgb_base_canvas, combined_proposed_mask, prop_mc, class_colors
    )

    return {
        "rgb_base_canvas": rgb_base_canvas,
        "rgb_old": rgb_old,
        "heatmap_rgb": heatmap_rgb,
        "rgb_proposed": rgb_proposed,
        "proposed_full_mask": proposed_full_mask,
        "proposed_tight_patch": proposed_tight_patch,
        "proposed_x": proposed_x,
        "proposed_y": proposed_y,
    }


class FrangiWorker(QThread):
    finished = pyqtSignal(int, object)
    error = pyqtSignal(str)

    def __init__(self, generation, snapshot):
        super().__init__()
        self.generation = generation
        self.snapshot = snapshot

    def run(self):
        try:
            result = compute_frangi_proposal(self.snapshot)
            self.finished.emit(self.generation, result)
        except Exception as e:
            self.error.emit(str(e))


class _NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


# --- FIXED: Upgraded to QWidget with manual paintEvent to stop UI bouncing ---
class AspectRatioLabel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(10, 10)
        self._pixmap = None
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.setStyleSheet("background-color: transparent;") 

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update() 

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)
        
        if self._pixmap and not self._pixmap.isNull():
            rect = self.contentsRect()
            scaled_pixmap = self._pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = int((rect.width() - scaled_pixmap.width()) / 2)
            y = int((rect.height() - scaled_pixmap.height()) / 2)
            p.drawPixmap(x, y, scaled_pixmap)

# ==========================================
# 1. THE MAIN VIEWPORT (CANVAS)
# ==========================================
class FrangiCanvasTab(QWidget):
    """Handles the heavy processing and 4-grid visual display."""
    FRANGI_LOADING_MSG = (
        "Computing root-strength map…\n"
        "If this takes too long, reduce the min/max root radius range or increase the radius step."
    )

    def __init__(self, model, main_window=None):
        super().__init__()
        self.model = model
        self.main_window = main_window
        self._frangi_worker = None
        self._frangi_generation = 0
        self.active_workers = set()
        
        # Processing Parameters 
        self.p_search_range = 0
        self.p_bridge_gaps = 1
        self.p_faint_sens = 0.15  
        self.p_strong_conf = 0.50 
        self.p_min_part = 5
        self.p_final_thick = 1
        self.p_min_sigma = 1     
        self.p_max_sigma = 5    
        self.p_sigma_step = 1    
        
        self.p_roots_dark = False
        self.p_allow_disconnected = False
        self.p_channel_mode = "Red-Blue Avg (RB)"
        self.p_smooth_mode = "Medium"
        self.p_clahe_mode = "None"  
        self.p_target_classes = [1, 2]
        
        self.p_centerline_correction = True
        
        self.proposed_full_mask = None
        self.proposed_tight_patch = None
        self.proposed_x = 0; self.proposed_y = 0
        
        self.init_ui()
        
        self.model.register_data_callback(self.on_data_changed)
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.info_label = QLabel(
            "Select a plant. Compare panels. Apply if the refinement looks better."
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 5px; background-color: #eee; padding: 5px;")
        layout.addWidget(self.info_label)
        
        self.grid_layout = QGridLayout()
        
        self.img_label_orig = AspectRatioLabel()
        self.img_label_old = AspectRatioLabel()
        self.img_label_vesselness = AspectRatioLabel()
        self.img_label_proposed = AspectRatioLabel()
        self.images = [self.img_label_orig, self.img_label_old, self.img_label_vesselness, self.img_label_proposed]

        lbl_orig_title = QLabel("Processed Grayscale")
        lbl_old_title = QLabel("Current Segmentation")
        lbl_vesselness_title = QLabel("Root-strength map")
        lbl_proposed_title = QLabel("Proposed Refinement")
        self.titles = [lbl_orig_title, lbl_old_title, lbl_vesselness_title, lbl_proposed_title]
        
        for t in self.titles:
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 5px;")

        self.set_display_mode(is_tall=True)
        layout.addLayout(self.grid_layout, stretch=1)
        self.clear_to_black()

    def set_display_mode(self, is_tall):
        for w in self.titles + self.images:
            self.grid_layout.removeWidget(w)

        if is_tall:
            for i in range(4):
                self.grid_layout.addWidget(self.titles[i], 0, i)
                self.grid_layout.addWidget(self.images[i], 1, i)
        else:
            for i in range(4):
                row = (i // 2) * 2
                col = i % 2
                self.grid_layout.addWidget(self.titles[i], row, col)
                self.grid_layout.addWidget(self.images[i], row + 1, col)

    def clear_to_black(self):
        black_pixmap = QPixmap(100, 100)
        black_pixmap.fill(QColor("black"))
        for lbl in self.images:
            lbl.setPixmap(black_pixmap)

    def on_data_changed(self):
        if getattr(self, "_suppress_regen", False):
            return
        if self.isVisible():
            self.generate_and_display_proposal()

    def on_selection_changed(self):
        if not self.model.active_uid:
            self.info_label.setText("No plant selected. Select a plant from the plant list.")
            self.clear_to_black()
            return
            
        self.info_label.setText(
            f"Plant {self.model.active_uid} — compare panels, then Apply to this plant if needed."
        )
        if self.isVisible():
            self.generate_and_display_proposal()

    def showEvent(self, event):
        super().showEvent(event)
        self.on_selection_changed()

    def _build_filled_overlay(self, rgb_crop, binary_mask, multiclass_crop):
        return _build_filled_overlay_static(
            rgb_crop, binary_mask, multiclass_crop, self.model.class_colors
        )

    def _get_current_multiclass_crop(self, x1, y1, x2, y2, uid):
        cls_crop = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        patch_data = self.model._get_class_patch(uid)
        
        if patch_data is not None:
            patch, px, py = patch_data
            patch_h, patch_w = patch.shape
            
            g_y1 = max(y1, py); g_y2 = min(y2, py + patch_h)
            g_x1 = max(x1, px); g_x2 = min(x2, px + patch_w)
            
            if g_y1 < g_y2 and g_x1 < g_x2:
                c_y1 = g_y1 - y1; c_y2 = g_y2 - y1
                c_x1 = g_x1 - x1; c_x2 = g_x2 - x1
                p_y1 = g_y1 - py; p_y2 = g_y2 - py
                p_x1 = g_x1 - px; p_x2 = g_x2 - px
                cls_crop[c_y1:c_y2, c_x1:c_x2] = patch[p_y1:p_y2, p_x1:p_x2]
                
        return cls_crop

    def _calculate_mapping(self, combined_mask, x1, y1, x2, y2, original_mask, current_cls_crop, untouched_mask_crop):
        return _calculate_mapping_static(
            combined_mask, x1, y1, x2, y2, original_mask,
            current_cls_crop, untouched_mask_crop, self.p_target_classes,
        )

    def _params_snapshot(self):
        return {
            "search_range": self.p_search_range,
            "bridge_gaps": self.p_bridge_gaps,
            "faint_sens": self.p_faint_sens,
            "strong_conf": self.p_strong_conf,
            "min_part": self.p_min_part,
            "final_thick": self.p_final_thick,
            "min_sigma": self.p_min_sigma,
            "max_sigma": self.p_max_sigma,
            "sigma_step": self.p_sigma_step,
            "roots_dark": self.p_roots_dark,
            "allow_disconnected": self.p_allow_disconnected,
            "centerline_correction": self.p_centerline_correction,
            "target_classes": list(self.p_target_classes),
        }

    def _on_frangi_finished(self, generation, result):
        if self.main_window:
            self.main_window.hide_loading()
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()
        if worker is self._frangi_worker:
            self._frangi_worker = None
        if generation != self._frangi_generation:
            return
        self._apply_proposal_result(result)

    def _on_frangi_error(self, message):
        if self.main_window:
            self.main_window.hide_loading()
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()
        if worker is self._frangi_worker:
            self._frangi_worker = None
        if hasattr(worker, "generation") and worker.generation != self._frangi_generation:
            return
        if self.main_window:
            self.main_window._on_thread_error(message)

    def _apply_proposal_result(self, result):
        self.proposed_full_mask = result["proposed_full_mask"]
        self.proposed_tight_patch = result["proposed_tight_patch"]
        self.proposed_x = result["proposed_x"]
        self.proposed_y = result["proposed_y"]
        self.img_label_orig.setPixmap(self.numpy_to_qpixmap(result["rgb_base_canvas"]))
        self.img_label_old.setPixmap(self.numpy_to_qpixmap(result["rgb_old"]))
        self.img_label_vesselness.setPixmap(self.numpy_to_qpixmap(result["heatmap_rgb"]))
        self.img_label_proposed.setPixmap(self.numpy_to_qpixmap(result["rgb_proposed"]))

    def generate_and_display_proposal(self):
        uid = self.model.active_uid
        raw_image = self.model.raw_image

        if uid is None or uid not in self.model.masks or raw_image is None:
            self.clear_to_black()
            return

        original_mask = self.model.masks[uid]
        x, y, w, h = self.model.bboxes.get(uid, (0, 0, 0, 0))
        if w == 0 or h == 0:
            self.clear_to_black()
            return

        self.set_display_mode(is_tall=(h > w * 1.2))

        pad = 15
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(raw_image.shape[1], x + w + pad), min(raw_image.shape[0], y + h + pad)

        img_crop_rgb = raw_image[y1:y2, x1:x2].copy()
        mask_crop = original_mask[y1:y2, x1:x2].copy()

        R = img_crop_rgb[:, :, 0].astype(np.float32)
        G = img_crop_rgb[:, :, 1].astype(np.float32)
        B = img_crop_rgb[:, :, 2].astype(np.float32)

        if "Red-Blue" in self.p_channel_mode:
            gray_crop = ((R + B) / 2.0).astype(np.uint8)
        elif "Standard" in self.p_channel_mode:
            gray_crop = cv2.cvtColor(img_crop_rgb, cv2.COLOR_RGB2GRAY)
        elif "Red" in self.p_channel_mode:
            gray_crop = R.astype(np.uint8)
        elif "Green" in self.p_channel_mode:
            gray_crop = G.astype(np.uint8)
        else:
            gray_crop = B.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if self.p_clahe_mode == "Before Smoothing":
            gray_crop = clahe.apply(gray_crop)

        if self.p_smooth_mode == "Light":
            gray_crop = cv2.bilateralFilter(gray_crop, d=3, sigmaColor=20, sigmaSpace=20)
        elif self.p_smooth_mode == "Medium":
            gray_crop = cv2.bilateralFilter(gray_crop, d=5, sigmaColor=25, sigmaSpace=25)
        elif self.p_smooth_mode == "High":
            gray_crop = cv2.bilateralFilter(gray_crop, d=7, sigmaColor=35, sigmaSpace=35)

        if self.p_clahe_mode == "After Smoothing":
            gray_crop = clahe.apply(gray_crop)

        current_cls_crop = self._get_current_multiclass_crop(x1, y1, x2, y2, uid)
        target_mask_crop = (mask_crop > 0) & np.isin(current_cls_crop, self.p_target_classes)
        untouched_mask_crop = (mask_crop > 0) & (~np.isin(current_cls_crop, self.p_target_classes))

        kernel = np.ones((3, 3), np.uint8)
        if self.p_search_range > 0:
            search_mask = cv2.dilate(target_mask_crop.astype(np.uint8), kernel, iterations=self.p_search_range)
        else:
            search_mask = target_mask_crop.copy().astype(np.uint8)

        snapshot = {
            "gray_crop": gray_crop,
            "mask_crop": mask_crop,
            "current_cls_crop": current_cls_crop,
            "target_mask_crop": target_mask_crop,
            "untouched_mask_crop": untouched_mask_crop,
            "search_mask": search_mask,
            "original_mask": original_mask.copy(),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "params": self._params_snapshot(),
            "class_colors": dict(self.model.class_colors),
        }

        if self.main_window is None:
            self._apply_proposal_result(compute_frangi_proposal(snapshot))
            return

        self._frangi_generation += 1
        generation = self._frangi_generation

        self.main_window.show_loading(self.FRANGI_LOADING_MSG)

        worker = FrangiWorker(generation, snapshot)
        self._frangi_worker = worker
        self.active_workers.add(worker)
        worker.finished.connect(self._on_frangi_finished)
        worker.error.connect(self._on_frangi_error)
        worker.start()

    def accept_proposal(self):
        uid = self.model.active_uid
        if not uid or self.proposed_tight_patch is None: return
        
        # Save proposed states to local variables
        proposed_full_mask = self.proposed_full_mask
        proposed_tight_patch = self.proposed_tight_patch
        proposed_x = self.proposed_x
        proposed_y = self.proposed_y
        
        # Clear proposed states immediately to prevent duplicate application on double click
        self.proposed_full_mask = None
        self.proposed_tight_patch = None
        
        self.model.save_state(uid)
        
        self.model.masks[uid] = proposed_full_mask
        self.model.class_patches[uid] = (proposed_tight_patch, proposed_x, proposed_y)
            
        self._suppress_regen = True
        self.model.callbacks_muted = True
        try:
            self.model.update_metadata_for_uid(uid)
            self.model.dirty = True
        finally:
            self.model.callbacks_muted = False
        self.model._notify_data_changed()
        applied = self.img_label_proposed._pixmap
        if applied is not None and not applied.isNull():
            self.img_label_old.setPixmap(applied)
        self._suppress_regen = False

    def numpy_to_qpixmap(self, img_array):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

# ==========================================
# 2. THE SIDEBAR TOOL PANEL
# ==========================================
class FrangiToolPanel(QWidget):
    """Handles the UI sliders and buttons inside the Sidebar StackedWidget."""
    def __init__(self, shared_model, canvas_tab: FrangiCanvasTab):
        super().__init__()
        self.model = shared_model
        self.canvas_tab = canvas_tab
        
        # --- NEW: Initialize the debounce timer ---
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.execute_push_params)
        
        self.init_ui()
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Main layout with very tight margins
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Helper functions for inputs
        def make_spin(val, vmin, vmax, step=1):
            sb = _NoWheelSpinBox()
            sb.setRange(vmin, vmax)
            sb.setSingleStep(step)
            sb.setValue(val)
            sb.setMaximumWidth(65) # Widened slightly to use horizontal space
            return sb

        def make_dspin(val, vmin, vmax, step=0.01):
            sb = _NoWheelDoubleSpinBox()
            sb.setRange(vmin, vmax)
            sb.setSingleStep(step)
            sb.setValue(val)
            sb.setMaximumWidth(65)
            return sb

        def separator():
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            return line

        # --- Header ---
        frangi_header = QHBoxLayout()
        frangi_header.setContentsMargins(0, 0, 0, 0)
        frangi_header.addWidget(QLabel("<b>Root Cleanup Tools</b>"))
        frangi_header.addStretch()
        btn_help = QPushButton("Help")
        btn_help.clicked.connect(lambda: show_help(self, "Centerline Help", HELP_FRANGI))
        frangi_header.addWidget(btn_help)
        main_layout.addLayout(frangi_header)
        main_layout.addWidget(separator())

        # --- Section 1: Modes (Expanded Labels & Tooltips) ---
        mode_grid = QGridLayout()
        mode_grid.setContentsMargins(0, 0, 0, 0)
        mode_grid.setVerticalSpacing(2)
        
        self.chk_correction = QCheckBox("Apply Correction")
        self.chk_correction.setChecked(self.canvas_tab.p_centerline_correction)
        self.chk_correction.setToolTip("On: Redraws roots using Frangi heatmap.\nOff: Simply reshapes existing mask to uniform width.")
        
        self.chk_dark = QCheckBox("Dark Roots")
        self.chk_dark.setChecked(self.canvas_tab.p_roots_dark)
        self.chk_dark.setToolTip("On: For infrared/backlight (dark roots, light background).\nOff: For scanned images (light roots, dark agar).")
        
        self.chk_disconnected = QCheckBox("Keep Disconnected Pieces")
        self.chk_disconnected.setChecked(self.canvas_tab.p_allow_disconnected)
        self.chk_disconnected.setToolTip("On: Keep all floating root pieces.\nOff: Discard floaters and keep only the main connected root system.")
        
        mode_grid.addWidget(self.chk_correction, 0, 0)
        mode_grid.addWidget(self.chk_dark, 0, 1)
        mode_grid.addWidget(self.chk_disconnected, 1, 0, 1, 2)
        main_layout.addLayout(mode_grid)

        param_grid = QGridLayout()
        param_grid.setContentsMargins(0, 0, 0, 0)
        param_grid.setHorizontalSpacing(8)
        param_grid.setVerticalSpacing(2)

        self.sp_min_sigma = make_spin(self.canvas_tab.p_min_sigma, 1, 40)
        self.sp_min_sigma.setToolTip("Minimum root radius (pixels).\nRaise this in high resolution images.")
        self.sp_max_sigma = make_spin(self.canvas_tab.p_max_sigma, 2, 80)
        self.sp_max_sigma.setToolTip(
            "Maximum root radius to detect (in pixels).\n"
            "Increase this for high-resolution images or very thick roots."
        )
        self.sp_sigma_step = make_spin(self.canvas_tab.p_sigma_step, 1, 10)
        self.sp_sigma_step.setToolTip(
            "Step between root radius scales in the Frangi filter.\n"
            "Increase to speed up processing on large radius ranges."
        )
        self.sp_thick = make_spin(self.canvas_tab.p_final_thick, 1, 10)
        self.sp_thick.setToolTip(
            "Final root mask width (pixels).\n"
            "Use 1 for standard resolution, 2 or 3 for high-resolution images."
        )
        self.sp_f_high = make_dspin(self.canvas_tab.p_strong_conf, 0.0, 1.0)
        self.sp_f_high.setToolTip(
            "Strict threshold (0-1) for solid root tissue.\n"
            "Raise to drop background noise; lower if the root core is missing."
        )
        self.sp_f_low = make_dspin(self.canvas_tab.p_faint_sens, 0.0, 1.0)
        self.sp_f_low.setToolTip(
            "Relaxed threshold (0-1).\n"
            "Recovers faint lateral tips that physically connect to core roots (Hysteresis)."
        )
        self.sp_search = make_spin(self.canvas_tab.p_search_range, 0, 50)
        self.sp_search.setToolTip("Pixels to look outside the current mask boundary for missing root segments.")
        self.sp_bridge = make_spin(self.canvas_tab.p_bridge_gaps, 0, 50)
        self.sp_bridge.setToolTip(
            "Maximum gap (pixels) to jump across broken heatmap segments.\n"
            "Raise to connect 'dotted' roots."
        )

        param_grid.addWidget(QLabel("Min Root Radius:"), 0, 0)
        param_grid.addWidget(self.sp_min_sigma, 0, 1)
        param_grid.addWidget(QLabel("Max Root Radius:"), 0, 2)
        param_grid.addWidget(self.sp_max_sigma, 0, 3)

        param_grid.addWidget(QLabel("Radius Step:"), 1, 0)
        param_grid.addWidget(self.sp_sigma_step, 1, 1)
        param_grid.addWidget(QLabel("Final Width:"), 1, 2)
        param_grid.addWidget(self.sp_thick, 1, 3)

        param_grid.addWidget(QLabel("Search Range:"), 2, 0)
        param_grid.addWidget(self.sp_search, 2, 1)
        param_grid.addWidget(QLabel("Bridge Gaps:"), 2, 2)
        param_grid.addWidget(self.sp_bridge, 2, 3)

        param_grid.addWidget(QLabel("Core Thresh:"), 3, 0)
        param_grid.addWidget(self.sp_f_high, 3, 1)
        param_grid.addWidget(QLabel("Faint Thresh:"), 3, 2)
        param_grid.addWidget(self.sp_f_low, 3, 3)

        main_layout.addLayout(param_grid)
        main_layout.addWidget(separator())

        # --- Section 3: Image Adjustments (Expanded Labels & Tooltips) ---
        main_layout.addWidget(QLabel("<b>Image Adjustments</b>"))
        img_grid = QGridLayout()
        img_grid.setContentsMargins(0, 0, 0, 0)
        img_grid.setHorizontalSpacing(8)
        img_grid.setVerticalSpacing(2)
        
        self.cb_channel = QComboBox()
        self.cb_channel.addItems(["Red-Blue Avg", "Standard Avg", "Red", "Green", "Blue"])
        self.cb_channel.setCurrentText(self.canvas_tab.p_channel_mode)
        self.cb_channel.setToolTip("Color channel used for grayscale conversion.\n'Red-Blue Avg' works best for typical scanned plates.")
        
        self.cb_clahe = QComboBox()
        self.cb_clahe.addItems(["None", "Before Smoothing", "After Smoothing"])
        self.cb_clahe.setCurrentText(self.canvas_tab.p_clahe_mode)
        self.cb_clahe.setToolTip("Local contrast boost (CLAHE) to fix uneven lighting or shadows.")
        
        self.cb_smooth = QComboBox()
        self.cb_smooth.addItems(["None", "Light", "Medium", "High"])
        self.cb_smooth.setCurrentText(self.canvas_tab.p_smooth_mode)
        self.cb_smooth.setToolTip("Blurs the image to reduce noise before generating the root heatmap.")
        
        img_grid.addWidget(QLabel("Image Channel:"), 0, 0)
        img_grid.addWidget(self.cb_channel, 0, 1)
        img_grid.addWidget(QLabel("Contrast Enhancement:"), 1, 0)
        img_grid.addWidget(self.cb_clahe, 1, 1)
        img_grid.addWidget(QLabel("Smooth Filter:"), 2, 0)
        img_grid.addWidget(self.cb_smooth, 2, 1)
        main_layout.addLayout(img_grid)
        main_layout.addWidget(separator())

        # --- Section 4: Target Classes ---
        main_layout.addWidget(QLabel("<b>Classes to Refine</b>"))
        class_grid = QGridLayout()
        class_grid.setContentsMargins(0, 0, 0, 0)
        class_grid.setHorizontalSpacing(4)
        class_grid.setVerticalSpacing(2)
        
        self.class_checkboxes = {}
        class_defs = [
            (1, "Main Root"), (2, "Lateral Root"), (3, "Seed"),
            (4, "Hypocotyl"), (5, "Leaves/Aerial"), (6, "Petiole"),
        ]
        for i, (cid, name) in enumerate(class_defs):
            chk = QCheckBox(name)
            chk.setChecked(cid in self.canvas_tab.p_target_classes)
            chk.setToolTip("Checked: Refines this root class.\nUnchecked: Preserves current mask exactly without changes.")
            chk.stateChanged.connect(self.push_params)
            self.class_checkboxes[cid] = chk
            class_grid.addWidget(chk, i // 2, i % 2)
        main_layout.addLayout(class_grid)

        # --- Connect Signals ---
        self.sp_search.valueChanged.connect(self.push_params)
        self.sp_bridge.valueChanged.connect(self.push_params)
        self.sp_f_low.valueChanged.connect(self.push_params)
        self.sp_f_high.valueChanged.connect(self.push_params)
        self.sp_thick.valueChanged.connect(self.push_params)
        self.chk_dark.stateChanged.connect(self.push_params)
        self.chk_disconnected.stateChanged.connect(self.push_params)
        self.cb_channel.currentIndexChanged.connect(self.push_params)
        self.cb_clahe.currentIndexChanged.connect(self.push_params)
        self.cb_smooth.currentIndexChanged.connect(self.push_params)
        self.chk_correction.stateChanged.connect(self.push_params)
        self.sp_min_sigma.valueChanged.connect(self.push_params)
        self.sp_max_sigma.valueChanged.connect(self.push_params)
        self.sp_sigma_step.valueChanged.connect(self.push_params)

        # --- Bottom Stretch & Button ---
        main_layout.addStretch(1)
        
        self.btn_accept = QPushButton("Apply to this plant")
        self.btn_accept.setToolTip("Apply the proposed refinement to this plant.\nRemember to Save Progress afterwards to commit changes to disk.")
        self.btn_accept.clicked.connect(self.canvas_tab.accept_proposal)
        main_layout.addWidget(self.btn_accept)

        self.toggle_buttons(False)

    def push_params(self):
        """Intercepts the UI signal and restarts the countdown timer."""
        self.debounce_timer.start(300)

    def execute_push_params(self):
        """Passes all UI states down to the Canvas Tab and triggers a regeneration."""
        self.canvas_tab.p_search_range = self.sp_search.value()
        self.canvas_tab.p_bridge_gaps = self.sp_bridge.value()
        self.canvas_tab.p_faint_sens = self.sp_f_low.value()
        self.canvas_tab.p_strong_conf = self.sp_f_high.value()
        self.canvas_tab.p_final_thick = self.sp_thick.value()
        self.canvas_tab.p_centerline_correction = self.chk_correction.isChecked()
        self.canvas_tab.p_roots_dark = self.chk_dark.isChecked()
        self.canvas_tab.p_allow_disconnected = self.chk_disconnected.isChecked()
        self.canvas_tab.p_channel_mode = self.cb_channel.currentText()
        self.canvas_tab.p_clahe_mode = self.cb_clahe.currentText()
        self.canvas_tab.p_smooth_mode = self.cb_smooth.currentText()
        self.canvas_tab.p_min_sigma = self.sp_min_sigma.value()
        self.canvas_tab.p_max_sigma = self.sp_max_sigma.value()
        self.canvas_tab.p_sigma_step = self.sp_sigma_step.value()
                
        targets = [cid for cid, chk in self.class_checkboxes.items() if chk.isChecked()]
        self.canvas_tab.p_target_classes = targets
        
        if self.model.active_uid:
            self.canvas_tab.generate_and_display_proposal()
            
    def toggle_buttons(self, enabled):
        self.btn_accept.setEnabled(enabled)

    def on_selection_changed(self):
        self.toggle_buttons(self.model.active_uid is not None)