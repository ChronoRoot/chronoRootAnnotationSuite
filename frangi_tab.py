import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, 
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox, 
                             QCheckBox, QComboBox, QFormLayout, QStyleOption, QStyle, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
from PyQt5.QtCore import Qt

from skimage.filters import frangi, apply_hysteresis_threshold
from skimage.morphology import skeletonize
from scipy.ndimage import label, distance_transform_edt

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
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Processing Parameters 
        self.p_search_range = 0
        self.p_bridge_gaps = 1
        self.p_faint_sens = 0.05  
        self.p_strong_conf = 0.20 
        self.p_min_part = 5
        self.p_final_thick = 1
        
        self.p_roots_dark = False
        self.p_allow_disconnected = False
        self.p_channel_mode = "Red-Blue Avg (RB)"
        self.p_smooth_mode = "Medium"
        self.p_clahe_mode = "None"  
        
        self.proposed_full_mask = None
        self.proposed_tight_patch = None
        self.proposed_x = 0; self.proposed_y = 0
        
        self.init_ui()
        
        self.model.register_data_callback(self.on_data_changed)
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.info_label = QLabel("Select a plant from the list to begin refinement.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 5px; background-color: #eee; padding: 5px;")
        layout.addWidget(self.info_label)
        
        self.grid_layout = QGridLayout()
        
        self.img_label_orig = AspectRatioLabel()
        self.img_label_old = AspectRatioLabel()
        self.img_label_vesselness = AspectRatioLabel()
        self.img_label_proposed = AspectRatioLabel()
        self.images = [self.img_label_orig, self.img_label_old, self.img_label_vesselness, self.img_label_proposed]

        lbl_orig_title = QLabel("1. Processed Grayscale")
        lbl_old_title = QLabel("2. Current Segmentation")
        lbl_vesselness_title = QLabel("3. Raw Vesselness Heatmap")
        lbl_proposed_title = QLabel("4. Proposed Refinement")
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
        if self.isVisible(): self.generate_and_display_proposal()

    def on_selection_changed(self):
        if not self.model.active_uid:
            self.info_label.setText("No plant selected. Select a plant from the sidebar.")
            self.clear_to_black()
            return
            
        self.info_label.setText(f"Processing Plant UID: {self.model.active_uid}")
        if self.isVisible(): self.generate_and_display_proposal()

    def showEvent(self, event):
        super().showEvent(event)
        self.on_selection_changed()

    def _build_filled_overlay(self, rgb_crop, binary_mask, multiclass_crop):
        overlay = np.zeros_like(rgb_crop)
        for cid, color in self.model.class_colors.items():
            if cid != 0: overlay[(binary_mask > 0) & (multiclass_crop == cid)] = color[:3]
        alpha = 0.6
        out_image = rgb_crop.copy()
        active = binary_mask > 0
        for c in range(3):
            out_image[active, c] = (rgb_crop[active, c] * (1 - alpha) + overlay[active, c] * alpha).astype(np.uint8)
        return out_image

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

    def _calculate_mapping(self, centered_mask, x1, y1, x2, y2, original_mask, current_cls_crop):
        full_mask = np.zeros_like(original_mask)
        full_mask[y1:y2, x1:x2] = centered_mask
        
        new_multiclass_crop = np.zeros_like(centered_mask)
        new_multiclass_crop[centered_mask > 0] = current_cls_crop[centered_mask > 0]
        
        holes = (centered_mask > 0) & (new_multiclass_crop == 0)
        if np.any(holes):
            valid = new_multiclass_crop > 0
            if np.any(valid):
                _, indices = distance_transform_edt(~valid, return_indices=True)
                new_multiclass_crop[holes] = new_multiclass_crop[indices[0], indices[1]][holes]
            else: new_multiclass_crop[holes] = 1 
                
        ys_c, xs_c = np.where(centered_mask)
        if len(xs_c) > 0:
            cx1, cy1 = int(xs_c.min()), int(ys_c.min())
            cx2, cy2 = int(xs_c.max()), int(ys_c.max())
            tight_patch = new_multiclass_crop[cy1:cy2+1, cx1:cx2+1]
            return full_mask, tight_patch, x1 + cx1, y1 + cy1, new_multiclass_crop
        return full_mask, None, 0, 0, new_multiclass_crop

    def generate_and_display_proposal(self):
        uid = self.model.active_uid
        raw_image = self.model.raw_image
        
        if uid is None or uid not in self.model.masks or raw_image is None: 
            self.clear_to_black()
            return
            
        original_mask = self.model.masks[uid]
        x, y, w, h = self.model.bboxes.get(uid, (0,0,0,0))
        if w == 0 or h == 0: 
            self.clear_to_black()
            return
            
        self.set_display_mode(is_tall=(h > w * 1.2))
            
        pad = 15 
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(raw_image.shape[1], x + w + pad), min(raw_image.shape[0], y + h + pad)
        
        img_crop_rgb = raw_image[y1:y2, x1:x2]
        mask_crop = original_mask[y1:y2, x1:x2]

        R = img_crop_rgb[:,:,0].astype(np.float32)
        G = img_crop_rgb[:,:,1].astype(np.float32)
        B = img_crop_rgb[:,:,2].astype(np.float32)

        if "Red-Blue" in self.p_channel_mode: gray_crop = ((R + B) / 2.0).astype(np.uint8)
        elif "Standard" in self.p_channel_mode: gray_crop = cv2.cvtColor(img_crop_rgb, cv2.COLOR_RGB2GRAY)
        elif "Red" in self.p_channel_mode: gray_crop = R.astype(np.uint8)
        elif "Green" in self.p_channel_mode: gray_crop = G.astype(np.uint8)
        else: gray_crop = B.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if self.p_clahe_mode == "Before Smoothing": gray_crop = clahe.apply(gray_crop)

        if self.p_smooth_mode == "Light": gray_crop = cv2.bilateralFilter(gray_crop, d=3, sigmaColor=20, sigmaSpace=20)
        elif self.p_smooth_mode == "Medium": gray_crop = cv2.bilateralFilter(gray_crop, d=5, sigmaColor=25, sigmaSpace=25)
        elif self.p_smooth_mode == "High": gray_crop = cv2.bilateralFilter(gray_crop, d=7, sigmaColor=35, sigmaSpace=35)

        if self.p_clahe_mode == "After Smoothing": gray_crop = clahe.apply(gray_crop)
            
        current_cls_crop = self._get_current_multiclass_crop(x1, y1, x2, y2, uid)
        
        kernel = np.ones((3, 3), np.uint8)
        if self.p_search_range > 0: search_mask = cv2.dilate(mask_crop, kernel, iterations=self.p_search_range)
        else: search_mask = mask_crop.copy()
        
        vesselness = frangi(gray_crop, black_ridges=self.p_roots_dark, sigmas=(1, 10, 1))
        vesselness[search_mask == 0] = 0 
        
        vesselness_clipped = np.clip(vesselness, 0.0, 0.5)
        vesselness_norm = vesselness_clipped / 0.5
        
        # --- FIXED: Mathematical bounds for colorbar to prevent shape mismatch crashes ---
        vessel_8u = (vesselness_norm * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(vessel_8u, cv2.COLORMAP_JET)
        heatmap_rgb_base = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        ch, cw = heatmap_rgb_base.shape[:2]
        pad_w = 40
        heatmap_rgb = np.zeros((ch, cw + pad_w, 3), dtype=np.uint8)
        heatmap_rgb[:, :cw] = heatmap_rgb_base
        
        cb_w = 10
        # Dynamically clamp the colorbar height to guarantee it is strictly smaller than the image height
        cb_h = max(1, min(150, ch - 4)) 
        cb_x = cw + 5
        cb_y = (ch - cb_h) // 2 

        grad = np.linspace(255, 0, cb_h, dtype=np.uint8).reshape(-1, 1)
        grad_color = cv2.applyColorMap(np.tile(grad, (1, cb_w)), cv2.COLORMAP_JET)
        grad_rgb = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)
        
        heatmap_rgb[cb_y:cb_y+cb_h, cb_x:cb_x+cb_w] = grad_rgb
        cv2.rectangle(heatmap_rgb, (cb_x, cb_y), (cb_x+cb_w, cb_y+cb_h), (255, 255, 255), 1)
        
        cv2.putText(heatmap_rgb, "0.5", (cb_x + cb_w + 5, cb_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(heatmap_rgb, "0.0", (cb_x + cb_w + 5, cb_y + cb_h), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        # --------------------------------------------------------------------------------

        binary_vessels = apply_hysteresis_threshold(vesselness_norm, self.p_faint_sens, self.p_strong_conf)
        labeled_frangi, num_frangi = label(binary_vessels, structure=np.ones((3,3)))
        sizes = np.bincount(labeled_frangi.ravel()); sizes[0] = 0
        
        core_fragments = np.zeros_like(binary_vessels, dtype=np.uint8)
        for i in range(1, num_frangi + 1):
            if sizes[i] >= self.p_min_part: core_fragments[labeled_frangi == i] = 1
                
        labeled_valid, num_valid = label(core_fragments, structure=np.ones((3,3)))

        if num_valid <= 1: 
            proposed_blob = core_fragments.copy()
        else:
            bridge_kernel = np.ones((3, 3), np.uint8)
            seen_dilations = np.zeros_like(core_fragments)
            bridge_seeds = np.zeros_like(core_fragments)
            
            for i in range(1, num_valid + 1):
                comp_mask = (labeled_valid == i).astype(np.uint8)
                dilated_comp = cv2.dilate(comp_mask, bridge_kernel, iterations=self.p_bridge_gaps)
                intersection = cv2.bitwise_and(dilated_comp, seen_dilations)
                bridge_seeds = cv2.bitwise_or(bridge_seeds, intersection)
                seen_dilations = cv2.bitwise_or(seen_dilations, dilated_comp)
                
            if np.sum(bridge_seeds) > 0:
                local_bridges = cv2.dilate(bridge_seeds, bridge_kernel, iterations=self.p_bridge_gaps)
                local_bridges = cv2.bitwise_and(local_bridges, search_mask) 
                proposed_blob = cv2.bitwise_or(core_fragments, local_bridges)
            else: 
                proposed_blob = core_fragments.copy()
                
        proposed_skeleton = skeletonize(proposed_blob).astype(np.uint8)
        proposed_mask_raw = cv2.dilate(proposed_skeleton, kernel, iterations=self.p_final_thick)
        
        if not self.p_allow_disconnected and np.sum(proposed_mask_raw) > 0:
            labeled_prop, num_prop = label(proposed_mask_raw, structure=np.ones((3,3)))
            if num_prop > 0:
                prop_sizes = np.bincount(labeled_prop.ravel()); prop_sizes[0] = 0
                proposed_mask = (labeled_prop == prop_sizes.argmax()).astype(np.uint8)
            else: proposed_mask = proposed_mask_raw
        else: proposed_mask = proposed_mask_raw

        self.proposed_full_mask, self.proposed_tight_patch, self.proposed_x, self.proposed_y, prop_mc = \
            self._calculate_mapping(proposed_mask, x1, y1, x2, y2, original_mask, current_cls_crop)

        old_mc = np.zeros_like(mask_crop)
        old_mc[mask_crop > 0] = current_cls_crop[mask_crop > 0]

        rgb_base_canvas = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB) 

        rgb_old = self._build_filled_overlay(rgb_base_canvas, mask_crop, old_mc)
        rgb_proposed = self._build_filled_overlay(rgb_base_canvas, proposed_mask, prop_mc)

        self.img_label_orig.setPixmap(self.numpy_to_qpixmap(rgb_base_canvas))
        self.img_label_old.setPixmap(self.numpy_to_qpixmap(rgb_old))
        self.img_label_vesselness.setPixmap(self.numpy_to_qpixmap(heatmap_rgb))
        self.img_label_proposed.setPixmap(self.numpy_to_qpixmap(rgb_proposed))

    def accept_proposal(self):
        uid = self.model.active_uid
        if not uid or self.proposed_tight_patch is None: return
        
        self.model.save_state(uid)
        
        self.model.masks[uid] = self.proposed_full_mask
        self.model.class_patches[uid] = (self.proposed_tight_patch, self.proposed_x, self.proposed_y)
            
        self.model.update_metadata_for_uid(uid)
        self.model.dirty = True
        self.model._notify_data_changed()
        self.generate_and_display_proposal()

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
        
        self.init_ui()
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # --- Form Layout for Numeric Parameters ---
        form = QFormLayout()
        
        def make_spin(val, vmin, vmax, step=1):
            sb = QSpinBox(); sb.setRange(vmin, vmax); sb.setSingleStep(step); sb.setValue(val)
            return sb
            
        def make_dspin(val, vmin, vmax, step=0.01):
            sb = QDoubleSpinBox(); sb.setRange(vmin, vmax); sb.setSingleStep(step); sb.setValue(val)
            return sb

        self.sp_search = make_spin(self.canvas_tab.p_search_range, 0, 50)
        self.sp_bridge = make_spin(self.canvas_tab.p_bridge_gaps, 0, 50)
        self.sp_min_part = make_spin(self.canvas_tab.p_min_part, 1, 500)
        self.sp_f_low = make_dspin(self.canvas_tab.p_faint_sens, 0.0, 1.0)
        self.sp_f_high = make_dspin(self.canvas_tab.p_strong_conf, 0.0, 1.0)
        self.sp_thick = make_spin(self.canvas_tab.p_final_thick, 1, 10)
        
        form.addRow("Search Range (px):", self.sp_search)
        form.addRow("Bridge Gaps (px):", self.sp_bridge)
        form.addRow("Ignore Specks < (px):", self.sp_min_part)
        form.addRow("Low Thresh (Faint):", self.sp_f_low)
        form.addRow("High Thresh (Core):", self.sp_f_high)        
        form.addRow("Final Thick:", self.sp_thick)
        layout.addLayout(form)
        
        # --- Checkboxes ---
        self.chk_dark = QCheckBox("Dark Roots")
        self.chk_dark.setChecked(self.canvas_tab.p_roots_dark)
        layout.addWidget(self.chk_dark)
        
        self.chk_disconnected = QCheckBox("Allow Disconnected Parts")
        self.chk_disconnected.setChecked(self.canvas_tab.p_allow_disconnected)
        layout.addWidget(self.chk_disconnected)
        
        layout.addSpacing(5)
        
        # --- Dropdowns ---
        layout.addWidget(QLabel("Image Mode:"))
        self.cb_channel = QComboBox()
        self.cb_channel.addItems(["Red-Blue Avg (RB)", "Standard Avg", "Red Channel", "Green Channel", "Blue Channel"])
        self.cb_channel.setCurrentText(self.canvas_tab.p_channel_mode)
        layout.addWidget(self.cb_channel)

        layout.addWidget(QLabel("Contrast Enhancement:"))
        self.cb_clahe = QComboBox()
        self.cb_clahe.addItems(["None", "Before Smoothing", "After Smoothing"])
        self.cb_clahe.setCurrentText(self.canvas_tab.p_clahe_mode)
        layout.addWidget(self.cb_clahe)
        
        layout.addWidget(QLabel("Smoothing:"))
        self.cb_smooth = QComboBox()
        self.cb_smooth.addItems(["None", "Light", "Medium", "High"])
        self.cb_smooth.setCurrentText(self.canvas_tab.p_smooth_mode)
        layout.addWidget(self.cb_smooth)

        # --- Connections ---
        self.sp_search.valueChanged.connect(self.push_params)
        self.sp_bridge.valueChanged.connect(self.push_params)
        self.sp_min_part.valueChanged.connect(self.push_params)
        self.sp_f_low.valueChanged.connect(self.push_params)
        self.sp_f_high.valueChanged.connect(self.push_params)
        self.sp_thick.valueChanged.connect(self.push_params)
        self.chk_dark.stateChanged.connect(self.push_params)
        self.chk_disconnected.stateChanged.connect(self.push_params)
        self.cb_channel.currentIndexChanged.connect(self.push_params)
        self.cb_clahe.currentIndexChanged.connect(self.push_params)
        self.cb_smooth.currentIndexChanged.connect(self.push_params)
        
        layout.addStretch()

        # --- Accept Button ---
        self.btn_accept = QPushButton("Accept Refinement")
        self.btn_accept.setStyleSheet("background-color: #cceeff; font-weight: bold; color: black; padding: 15px; font-size: 14px;")
        self.btn_accept.clicked.connect(self.canvas_tab.accept_proposal)
        
        layout.addWidget(self.btn_accept)
        self.toggle_buttons(False)

    def push_params(self):
        """Passes all UI states down to the Canvas Tab and triggers a regeneration."""
        self.canvas_tab.p_search_range = self.sp_search.value()
        self.canvas_tab.p_bridge_gaps = self.sp_bridge.value()
        self.canvas_tab.p_min_part = self.sp_min_part.value()
        self.canvas_tab.p_faint_sens = self.sp_f_low.value()
        self.canvas_tab.p_strong_conf = self.sp_f_high.value()
        self.canvas_tab.p_final_thick = self.sp_thick.value()
        self.canvas_tab.p_roots_dark = self.chk_dark.isChecked()
        self.canvas_tab.p_allow_disconnected = self.chk_disconnected.isChecked()
        self.canvas_tab.p_channel_mode = self.cb_channel.currentText()
        self.canvas_tab.p_clahe_mode = self.cb_clahe.currentText()
        self.canvas_tab.p_smooth_mode = self.cb_smooth.currentText()
        
        if self.model.active_uid:
            self.canvas_tab.generate_and_display_proposal()

    def toggle_buttons(self, enabled):
        self.btn_accept.setEnabled(enabled)

    def on_selection_changed(self):
        self.toggle_buttons(self.model.active_uid is not None)