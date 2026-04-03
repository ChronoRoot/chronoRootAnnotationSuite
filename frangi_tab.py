import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QLabel, QSizePolicy, QSpinBox, 
                             QDoubleSpinBox, QCheckBox, QFrame, QMessageBox, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from skimage.filters import frangi, apply_hysteresis_threshold
from skimage.morphology import skeletonize
from scipy.ndimage import label, distance_transform_edt

class AspectRatioLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(100, 100)
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_scaled_pixmap()

    def resizeEvent(self, event):
        self.update_scaled_pixmap()
        super().resizeEvent(event)

    def update_scaled_pixmap(self):
        if self._pixmap and not self.size().isEmpty():
            super().setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            super().setPixmap(QPixmap())

class FrangiTab(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Internal State
        self.current_uids = []
        self.current_index = 0
        self.current_uid = None
        self.last_image_path = None
        
        # Friendly Default Parameters
        self.p_search_range = 0
        self.p_bridge_gaps = 1
        self.p_faint_sens = 0.10
        self.p_strong_conf = 0.40
        self.p_min_part = 20
        self.p_final_thick = 1
        
        # Checkbox & Mode States
        self.p_roots_dark = False
        self.p_allow_disconnected = False
        self.p_channel_mode = "Red-Blue Avg (RB)"
        self.p_smooth_mode = "Medium"
        
        self.strict_full_mask = None
        self.strict_tight_patch = None
        self.strict_x = 0; self.strict_y = 0
        
        self.hybrid_full_mask = None
        self.hybrid_tight_patch = None
        self.hybrid_x = 0; self.hybrid_y = 0
        
        self.init_ui()
        self.model.register_callback(self.on_model_update)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Parameter Control Bar (2 Rows) ---
        param_frame = QFrame()
        param_layout_main = QVBoxLayout(param_frame)
        param_layout_main.setContentsMargins(5, 5, 5, 5)
        
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        
        def make_spin(val, vmin, vmax, step=1):
            sb = QSpinBox(); sb.setRange(vmin, vmax); sb.setSingleStep(step); sb.setValue(val)
            return sb
            
        def make_dspin(val, vmin, vmax, step=0.05):
            sb = QDoubleSpinBox(); sb.setRange(vmin, vmax); sb.setSingleStep(step); sb.setValue(val)
            return sb

        # Controls Setup
        self.sp_search = make_spin(self.p_search_range, 0, 50) 
        self.sp_bridge = make_spin(self.p_bridge_gaps, 0, 50)
        self.sp_min_part = make_spin(self.p_min_part, 1, 500)
        self.sp_f_low = make_dspin(self.p_faint_sens, 0.0, 1.0)
        self.sp_f_high = make_dspin(self.p_strong_conf, 0.0, 1.0)
        self.sp_thick = make_spin(self.p_final_thick, 1, 10)
        
        self.chk_dark = QCheckBox("Dark Roots")
        self.chk_dark.setChecked(self.p_roots_dark)
        
        self.chk_disconnected = QCheckBox("Allow Disconnected Parts")
        self.chk_disconnected.setChecked(self.p_allow_disconnected)
        
        self.cb_channel = QComboBox()
        self.cb_channel.addItems(["Red-Blue Avg (RB)", "Standard Avg", "Red Channel", "Green Channel", "Blue Channel"])
        self.cb_channel.setCurrentText(self.p_channel_mode)

        self.cb_smooth = QComboBox()
        self.cb_smooth.addItems(["None", "Light", "Medium", "High"])
        self.cb_smooth.setCurrentText(self.p_smooth_mode)

        # Connect ALL parameters to live-update the proposals instantly
        self.sp_search.valueChanged.connect(self.refresh_proposals)
        self.sp_bridge.valueChanged.connect(self.refresh_proposals)
        self.sp_min_part.valueChanged.connect(self.refresh_proposals)
        self.sp_f_low.valueChanged.connect(self.refresh_proposals)
        self.sp_f_high.valueChanged.connect(self.refresh_proposals)
        self.sp_thick.valueChanged.connect(self.refresh_proposals)
        
        self.cb_channel.currentIndexChanged.connect(self.refresh_proposals)
        self.cb_smooth.currentIndexChanged.connect(self.refresh_proposals)
        self.chk_dark.stateChanged.connect(self.refresh_proposals)
        self.chk_disconnected.stateChanged.connect(self.refresh_proposals)

        # Row 1: Shapes & Thresholds
        row1.addWidget(QLabel("Search Range (px):"))
        row1.addWidget(self.sp_search)
        row1.addWidget(QLabel(" | Bridge Gaps (px):"))
        row1.addWidget(self.sp_bridge)
        row1.addWidget(QLabel(" | Ignore Specks < (px):"))
        row1.addWidget(self.sp_min_part)
        row1.addWidget(QLabel(" | Faint Sens.:"))
        row1.addWidget(self.sp_f_low)
        row1.addWidget(QLabel(" | Strong Conf.:"))
        row1.addWidget(self.sp_f_high)
        row1.addWidget(QLabel(" | Final Thick:"))
        row1.addWidget(self.sp_thick)
        row1.addStretch()

        # Row 2: Visual Adjustments & Execution
        row2.addWidget(QLabel("Image Mode:"))
        row2.addWidget(self.cb_channel)
        row2.addWidget(QLabel(" | Smoothing:"))
        row2.addWidget(self.cb_smooth)
        row2.addWidget(QLabel(" | "))
        row2.addWidget(self.chk_dark)
        row2.addWidget(QLabel(" | "))
        row2.addWidget(self.chk_disconnected)
        row2.addStretch()
        
        param_layout_main.addLayout(row1)
        param_layout_main.addLayout(row2)
        layout.addWidget(param_frame)

        # --- Info Label ---
        self.info_label = QLabel("Open a file from the browser to begin refinement.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 5px;")
        layout.addWidget(self.info_label)
        
        # --- Dynamic Grid Display Area ---
        self.grid_layout = QGridLayout()
        
        self.img_label_orig = AspectRatioLabel()
        self.img_label_old = AspectRatioLabel()
        self.img_label_strict = AspectRatioLabel()
        self.img_label_hybrid = AspectRatioLabel()
        self.images = [self.img_label_orig, self.img_label_old, self.img_label_strict, self.img_label_hybrid]

        lbl_orig_title = QLabel("1. Processed Grayscale Canvas")
        lbl_old_title = QLabel("2. Current Segmentation")
        lbl_strict_title = QLabel("3. Proposed: STRICT")
        lbl_hybrid_title = QLabel("4. Proposed: SMART HYBRID")
        self.titles = [lbl_orig_title, lbl_old_title, lbl_strict_title, lbl_hybrid_title]
        
        for t in self.titles:
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 5px;")

        self.set_display_mode(is_tall=True)
        layout.addLayout(self.grid_layout, stretch=1)
        
        # --- Action & Navigation Buttons ---
        btn_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("⬅ Previous Plant")
        self.btn_prev.setMinimumHeight(50)
        self.btn_prev.clicked.connect(self.prev_plant)
        
        self.btn_accept_strict = QPushButton("Accept STRICT")
        self.btn_accept_strict.setStyleSheet("background-color: #ccffcc; font-size: 14px; font-weight: bold; color: black;")
        self.btn_accept_strict.setMinimumHeight(50)
        self.btn_accept_strict.clicked.connect(lambda: self.accept_proposal('strict'))
        
        self.btn_accept_hybrid = QPushButton("Accept SMART HYBRID")
        self.btn_accept_hybrid.setStyleSheet("background-color: #cceeff; font-size: 14px; font-weight: bold; color: black;")
        self.btn_accept_hybrid.setMinimumHeight(50)
        self.btn_accept_hybrid.clicked.connect(lambda: self.accept_proposal('hybrid'))

        self.btn_next = QPushButton("Next Plant ➡\n(Keep Current)")
        self.btn_next.setMinimumHeight(50)
        self.btn_next.clicked.connect(self.next_plant)
        
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_accept_strict)
        btn_layout.addWidget(self.btn_accept_hybrid)
        btn_layout.addWidget(self.btn_next)
        
        layout.addLayout(btn_layout)
        self.toggle_buttons(False)

    def toggle_buttons(self, enabled):
        self.btn_accept_strict.setEnabled(enabled)
        self.btn_accept_hybrid.setEnabled(enabled)
        self.btn_prev.setEnabled(enabled and self.current_index > 0)
        self.btn_next.setEnabled(enabled and self.current_index < len(self.current_uids) - 1)

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

    def on_model_update(self):
        if getattr(self, 'model', None) is None or not self.model.image_path:
            return

        is_new_image = (self.model.image_path != self.last_image_path)
        self.last_image_path = self.model.image_path
        self.current_uids = sorted(list(self.model.masks.keys()))

        if is_new_image:
            self.current_index = 0
            if self.isVisible():
                self.load_current_plant()
        else:
            if not self.isVisible():
                # A change was made in the other tab. Flag it to reset to 0 when we switch back.
                self._external_change_pending = True
            else:
                # We are in the Frangi tab (e.g., just accepted a proposal). Keep our current plant selected!
                if self.current_uid in self.current_uids:
                    self.current_index = self.current_uids.index(self.current_uid)
                else:
                    self.current_index = 0

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, 'model', None) or not self.model.image_path:
            return
            
        self.last_image_path = self.model.image_path
        self.current_uids = sorted(list(self.model.masks.keys()))
        
        # Check if we flagged an external change from the other tab
        if getattr(self, '_external_change_pending', False):
            self.current_index = 0
            self._external_change_pending = False
        else:
            if self.current_uid not in self.current_uids:
                self.current_index = 0
            else:
                self.current_index = self.current_uids.index(self.current_uid)
            
        self.load_current_plant()

    def prev_plant(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_plant()

    def next_plant(self):
        if self.current_index < len(self.current_uids) - 1:
            self.current_index += 1
            self.load_current_plant()
        else:
            QMessageBox.information(self, "Finished", "You have reached the last plant in this image.")

    def load_current_plant(self):
        if not self.current_uids:
            self.info_label.setText("✅ No plants available in this image.")
            for lbl in self.images: lbl.setPixmap(QPixmap())
            self.toggle_buttons(False)
            return

        if self.current_index >= len(self.current_uids):
            self.current_index = len(self.current_uids) - 1
            
        self.current_uid = self.current_uids[self.current_index]
        self.info_label.setText(f"Processing Plant UID: {self.current_uid} (Plant {self.current_index + 1} of {len(self.current_uids)})")
        
        self.toggle_buttons(True)
        self.generate_and_display_proposal()

    def refresh_proposals(self):
        if self.current_uid is None: return
        self.p_search_range = self.sp_search.value()
        self.p_bridge_gaps = self.sp_bridge.value()
        self.p_faint_sens = self.sp_f_low.value()
        self.p_strong_conf = self.sp_f_high.value()
        self.p_min_part = self.sp_min_part.value()
        self.p_final_thick = self.sp_thick.value()
        
        self.p_roots_dark = self.chk_dark.isChecked()
        self.p_allow_disconnected = self.chk_disconnected.isChecked()
        self.p_channel_mode = self.cb_channel.currentText()
        self.p_smooth_mode = self.cb_smooth.currentText()
        
        self.generate_and_display_proposal()

    def _build_filled_overlay(self, rgb_crop, binary_mask, multiclass_crop):
        overlay = np.zeros_like(rgb_crop)
        for cid, color in self.model.class_colors.items():
            if cid != 0:
                overlay[(binary_mask > 0) & (multiclass_crop == cid)] = color[:3]
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
            else:
                new_multiclass_crop[holes] = 1 
                
        ys_c, xs_c = np.where(centered_mask)
        if len(xs_c) > 0:
            cx1, cy1 = int(xs_c.min()), int(ys_c.min())
            cx2, cy2 = int(xs_c.max()), int(ys_c.max())
            tight_patch = new_multiclass_crop[cy1:cy2+1, cx1:cx2+1]
            return full_mask, tight_patch, x1 + cx1, y1 + cy1, new_multiclass_crop
        return full_mask, None, 0, 0, new_multiclass_crop

    def generate_and_display_proposal(self):
        raw_image = self.model.raw_image
        if self.current_uid not in self.model.masks: return
        original_mask = self.model.masks[self.current_uid]
        
        x, y, w, h = self.model.bboxes.get(self.current_uid, (0,0,0,0))
        if w == 0 or h == 0: 
            self.next_plant() 
            return
            
        self.set_display_mode(is_tall=(h > w * 1.2))
            
        pad = 15 
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(raw_image.shape[1], x + w + pad), min(raw_image.shape[0], y + h + pad)
        
        img_crop_rgb = raw_image[y1:y2, x1:x2]
        mask_crop = original_mask[y1:y2, x1:x2]

        # --- 1. Channel Extraction & Smoothing ---
        R = img_crop_rgb[:,:,0].astype(np.float32)
        G = img_crop_rgb[:,:,1].astype(np.float32)
        B = img_crop_rgb[:,:,2].astype(np.float32)

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

        # Dynamic Smoothing based on selection
        if self.p_smooth_mode == "Light":
            gray_crop = cv2.bilateralFilter(gray_crop, d=3, sigmaColor=20, sigmaSpace=20)
        elif self.p_smooth_mode == "Medium":
            gray_crop = cv2.bilateralFilter(gray_crop, d=5, sigmaColor=25, sigmaSpace=25)
        elif self.p_smooth_mode == "High":
            gray_crop = cv2.bilateralFilter(gray_crop, d=7, sigmaColor=35, sigmaSpace=35)
        
        current_cls_crop = self._get_current_multiclass_crop(x1, y1, x2, y2, self.current_uid)
        
        # --- 2. Safe Zone Dilation ---
        kernel = np.ones((3, 3), np.uint8)
        if self.p_search_range > 0:
            search_mask = cv2.dilate(mask_crop, kernel, iterations=self.p_search_range)
        else:
            search_mask = mask_crop.copy()
        
        # --- 3. Filtering ---
        vesselness = frangi(gray_crop, black_ridges=self.p_roots_dark)
        vesselness[search_mask == 0] = 0 
        vesselness_norm = vesselness / vesselness.max() if vesselness.max() > 0 else vesselness
        binary_vessels = apply_hysteresis_threshold(vesselness_norm, self.p_faint_sens, self.p_strong_conf)
        
        labeled_frangi, num_frangi = label(binary_vessels, structure=np.ones((3,3)))
        sizes = np.bincount(labeled_frangi.ravel()); sizes[0] = 0
        
        core_fragments = np.zeros_like(binary_vessels, dtype=np.uint8)
        for i in range(1, num_frangi + 1):
            if sizes[i] >= self.p_min_part:
                core_fragments[labeled_frangi == i] = 1
                
        labeled_valid, num_valid = label(core_fragments, structure=np.ones((3,3)))

        # --- OPTION 1: STRICT ---
        if num_valid > 0:
            strict_blob = core_fragments.copy()
        else:
            strict_blob = np.zeros_like(core_fragments)

        strict_skeleton = skeletonize(strict_blob).astype(np.uint8)
        strict_mask_raw = cv2.dilate(strict_skeleton, kernel, iterations=self.p_final_thick)

        # Check for disconnected parts AFTER final dilation
        if not self.p_allow_disconnected and np.sum(strict_mask_raw) > 0:
            labeled_strict, num_strict = label(strict_mask_raw, structure=np.ones((3,3)))
            if num_strict > 0:
                strict_sizes = np.bincount(labeled_strict.ravel()); strict_sizes[0] = 0
                strict_mask = (labeled_strict == strict_sizes.argmax()).astype(np.uint8)
            else:
                strict_mask = strict_mask_raw
        else:
            strict_mask = strict_mask_raw

        # --- OPTION 2: SMART HYBRID ---
        if num_valid <= 1:
            hybrid_mask = strict_mask.copy()
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
                hybrid_blob = cv2.bitwise_or(core_fragments, local_bridges)
            else:
                hybrid_blob = core_fragments.copy()
                
            hybrid_skeleton = skeletonize(hybrid_blob).astype(np.uint8)
            hybrid_mask_raw = cv2.dilate(hybrid_skeleton, kernel, iterations=self.p_final_thick)
            
            # Check for disconnected parts AFTER final dilation
            if not self.p_allow_disconnected and np.sum(hybrid_mask_raw) > 0:
                labeled_hybrid, num_hybrid = label(hybrid_mask_raw, structure=np.ones((3,3)))
                if num_hybrid > 0:
                    hybrid_sizes = np.bincount(labeled_hybrid.ravel()); hybrid_sizes[0] = 0
                    hybrid_mask = (labeled_hybrid == hybrid_sizes.argmax()).astype(np.uint8)
                else:
                    hybrid_mask = hybrid_mask_raw
            else:
                hybrid_mask = hybrid_mask_raw

        self.strict_full_mask, self.strict_tight_patch, self.strict_x, self.strict_y, strict_mc = \
            self._calculate_mapping(strict_mask, x1, y1, x2, y2, original_mask, current_cls_crop)
        self.hybrid_full_mask, self.hybrid_tight_patch, self.hybrid_x, self.hybrid_y, hybrid_mc = \
            self._calculate_mapping(hybrid_mask, x1, y1, x2, y2, original_mask, current_cls_crop)

        old_mc = np.zeros_like(mask_crop)
        old_mc[mask_crop > 0] = current_cls_crop[mask_crop > 0]

        # Convert our processed grayscale canvas into an RGB layout so we can paint colors on it safely!
        rgb_base_canvas = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB) 

        rgb_old = self._build_filled_overlay(rgb_base_canvas, mask_crop, old_mc)
        rgb_strict = self._build_filled_overlay(rgb_base_canvas, strict_mask, strict_mc)
        rgb_hybrid = self._build_filled_overlay(rgb_base_canvas, hybrid_mask, hybrid_mc)

        self.img_label_orig.setPixmap(self.numpy_to_qpixmap(rgb_base_canvas))
        self.img_label_old.setPixmap(self.numpy_to_qpixmap(rgb_old))
        self.img_label_strict.setPixmap(self.numpy_to_qpixmap(rgb_strict))
        self.img_label_hybrid.setPixmap(self.numpy_to_qpixmap(rgb_hybrid))

    def accept_proposal(self, choice):
        if choice == 'strict' and self.strict_tight_patch is not None:
            self.model.masks[self.current_uid] = self.strict_full_mask
            self.model.class_patches[self.current_uid] = (self.strict_tight_patch, self.strict_x, self.strict_y)
            self.model.update_metadata_for_uid(self.current_uid)
            self.model.dirty = True
            
        elif choice == 'hybrid' and self.hybrid_tight_patch is not None:
            self.model.masks[self.current_uid] = self.hybrid_full_mask
            self.model.class_patches[self.current_uid] = (self.hybrid_tight_patch, self.hybrid_x, self.hybrid_y)
            self.model.update_metadata_for_uid(self.current_uid)
            self.model.dirty = True
            
        self.model._notify_changed() 
        self.generate_and_display_proposal()

    def numpy_to_qpixmap(self, img_array):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)