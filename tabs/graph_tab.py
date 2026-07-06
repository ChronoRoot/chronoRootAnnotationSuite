import cv2
import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QFrame, QFormLayout, 
                             QComboBox, QGridLayout, QStyleOption, QStyle,
                             QListWidget, QListWidgetItem,
                             QHBoxLayout, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
from PyQt5.QtCore import Qt, pyqtSignal

from components.ui_help import HELP_GRAPH, show_help

# Import the isolated builder
from core.root_graph_builder import extract_skeleton, createGraph, graphInit

class AspectRatioLabel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(10, 10)
        self._pixmap = None
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")

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


class InteractiveGraphLabel(AspectRatioLabel):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if not self._pixmap or self._pixmap.isNull(): return
        
        rect = self.contentsRect()
        scaled_pixmap = self._pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        rw, rh = scaled_pixmap.width(), scaled_pixmap.height()
        ox = (rect.width() - rw) / 2
        oy = (rect.height() - rh) / 2
        
        mx, my = event.x(), event.y()
        
        if ox <= mx <= ox + rw and oy <= my <= oy + rh:
            scale_x = self._pixmap.width() / rw
            scale_y = self._pixmap.height() / rh
            
            ix = int((mx - ox) * scale_x)
            iy = int((my - oy) * scale_y)
            self.clicked.emit(ix, iy)


# ==========================================
# 1. THE MAIN VIEWPORT (CANVAS)
# ==========================================
class GraphCanvasTab(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self.p_thick = 1
        self.p_prune = 5
        self.interaction_mode = "START" 
        
        self.current_graph = None
        self.colored_skeleton = None
        self.last_processed_uid = None
        self.last_prune_val = -1
        
        self.user_start_node = None
        self.user_end_node = None
        self.user_waypoints = []
        
        self.crop_x = 0
        self.crop_y = 0
        self.base_rgb = None
        self.mc_skel_bin = None
        self.actual_base = None
        self.mask_crop = None
        self.p_target_classes = [1, 2]
        
        self.init_ui()
        self.model.register_data_callback(self.on_data_changed)
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.info_label = QLabel(
            "Check each plant before Finish. Confirm skeleton and main-root path look correct."
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 5px; background-color: #eee; padding: 5px;")
        layout.addWidget(self.info_label)
        
        grid = QGridLayout()
        grid.setSpacing(5)
        
        titles = [
            "RGB Image",
            "Root part labels",
            "Skeleton Overlay",
            "Recolored Preview",
            "Interactive Graph"
        ]
        
        for col, title in enumerate(titles):
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 13px; padding-bottom: 5px;")
            lbl.setFixedHeight(30)
            grid.addWidget(lbl, 0, col)
            
        self.lbl_orig = AspectRatioLabel()
        self.lbl_sem = AspectRatioLabel()
        self.lbl_skel = AspectRatioLabel()
        self.lbl_preview = AspectRatioLabel()
        self.lbl_graph = InteractiveGraphLabel()
        self.lbl_graph.clicked.connect(self.on_graph_clicked)
        
        grid.addWidget(self.lbl_orig, 1, 0)
        grid.addWidget(self.lbl_sem, 1, 1)
        grid.addWidget(self.lbl_skel, 1, 2)
        grid.addWidget(self.lbl_preview, 1, 3)
        grid.addWidget(self.lbl_graph, 1, 4)
        
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)
        layout.addLayout(grid, stretch=1)
        
        self.clear_to_black()

    def clear_graph_panels(self):
        black_pixmap = QPixmap(100, 100)
        black_pixmap.fill(QColor("black"))
        self.lbl_skel.setPixmap(black_pixmap)
        self.lbl_preview.setPixmap(black_pixmap)
        self.lbl_graph.setPixmap(black_pixmap)
        self.current_graph = None
        self.colored_skeleton = None

    def clear_to_black(self):
        black_pixmap = QPixmap(100, 100)
        black_pixmap.fill(QColor("black"))
        self.lbl_orig.setPixmap(black_pixmap)
        self.lbl_sem.setPixmap(black_pixmap)
        self.lbl_skel.setPixmap(black_pixmap)
        self.lbl_preview.setPixmap(black_pixmap)
        self.lbl_graph.setPixmap(black_pixmap)
        self.current_graph = None
        self.colored_skeleton = None

    def _mask_is_disconnected(self, mask):
        bin_mask = (mask > 0).astype(np.uint8)
        if np.sum(bin_mask) == 0:
            return False
        num_features, _ = cv2.connectedComponents(bin_mask, connectivity=8)
        return num_features > 2

    def refresh_graph_preview(self):
        """Regenerate skeleton preview when model prune or selection is ready."""
        if self.model.active_uid:
            self.generate_pipeline()

    def reset_user_nodes(self):
        self.user_start_node = None
        self.user_end_node = None
        self.user_waypoints = []
        if self.current_graph:
            self.update_graph_visuals()

    def on_data_changed(self):
        if self.isVisible(): self.generate_pipeline()

    def on_selection_changed(self):
        if not self.model.active_uid:
            self.info_label.setText("No plant selected. Select a plant from the plant list.")
            self.clear_to_black()
            return

        if self.isVisible():
            self.generate_pipeline()

    def showEvent(self, event):
        super().showEvent(event)
        self.on_selection_changed()

    def on_graph_clicked(self, x, y):
        if not self.current_graph: return
        
        global_x, global_y = x + self.crop_x, y + self.crop_y
        
        nodes = list(self.current_graph.nodes)
        nodes_arr = np.array(nodes)
        dists = np.linalg.norm(nodes_arr - np.array([global_x, global_y]), axis=1)
        nearest_idx = np.argmin(dists)
        
        if dists[nearest_idx] > 100: return 
            
        nearest_node = nodes[nearest_idx]
        
        if self.interaction_mode == "START":
            self.user_start_node = nearest_node
        elif self.interaction_mode == "END":
            self.user_end_node = nearest_node
        elif self.interaction_mode == "WAYPOINT":
            if nearest_node in self.user_waypoints:
                self.user_waypoints.remove(nearest_node)
            else:
                self.user_waypoints.append(nearest_node)
                
        self.update_graph_visuals()

    def generate_pipeline(self):
        uid = self.model.active_uid
        if uid is None or uid not in self.model.masks or self.model.raw_image is None: 
            self.clear_to_black()
            return
            
        if uid != self.last_processed_uid or self.p_prune != self.last_prune_val:
            self.user_start_node = None
            self.user_end_node = None
            self.user_waypoints = []
            self.last_processed_uid = uid
            self.last_prune_val = self.p_prune
        
        mask = self.model.masks[uid]
        if self._mask_is_disconnected(mask):
            self.info_label.setText(
                "Disconnected mask — merge or connect pieces in Annotation, "
                "or use Split disconnected fragments."
            )
            self.clear_graph_panels()
            return

        patch_data = self.model._get_class_patch(uid)
        if not patch_data: 
            self.clear_to_black()
            return
            
        patch, x_off, y_off = patch_data
        
        x, y, w, h = self.model.bboxes.get(uid, (0,0,0,0))
        if w == 0: 
            self.clear_to_black()
            return
        
        img_h, img_w = self.model.raw_image.shape[:2]
        
        pad = 20
        self.crop_x, self.crop_y = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
        
        self.mask_crop = mask[self.crop_y:y2, self.crop_x:x2]
        
        rgb_crop = self.model.raw_image[self.crop_y:y2, self.crop_x:x2].copy()
        self.base_rgb = rgb_crop.copy() 
        self.lbl_orig.setPixmap(self.numpy_to_qpixmap(rgb_crop))
        
        # --- NEW: Alpha Blended Semantic Overlay ---
        sem_vis = self.base_rgb.copy()
        overlay_sem = np.zeros_like(self.base_rgb)
        active_mask = np.zeros(self.base_rgb.shape[:2], dtype=bool)
        
        bin_crop = mask[y_off:y_off+patch.shape[0], x_off:x_off+patch.shape[1]]
        for cid, color in self.model.class_colors.items():
            if cid != 0:
                active = (patch == cid) & (bin_crop > 0)
                p_y, p_x = np.where(active)
                global_y, global_x = p_y + y_off, p_x + x_off
                valid = (global_y >= self.crop_y) & (global_y < y2) & (global_x >= self.crop_x) & (global_x < x2)
                
                vy, vx = global_y[valid] - self.crop_y, global_x[valid] - self.crop_x
                overlay_sem[vy, vx] = color[:3]
                active_mask[vy, vx] = True
                
        alpha = 0.5  # 50% transparency
        for c in range(3):
            sem_vis[active_mask, c] = (self.base_rgb[active_mask, c] * (1 - alpha) + overlay_sem[active_mask, c] * alpha).astype(np.uint8)
                
        self.lbl_sem.setPixmap(self.numpy_to_qpixmap(sem_vis))
        # -------------------------------------------
        
        root_bin_canvas = np.zeros((img_h, img_w), dtype=np.uint8)

        # ONLY extract the user-targeted classes for the skeletonizer
        for cid in self.p_target_classes:
            active = (patch == cid) & (bin_crop > 0)
            p_y, p_x = np.where(active)
            root_bin_canvas[p_y + y_off, p_x + x_off] = 1
                    
        full_skel, branches, endpoints, is_valid = extract_skeleton(
            root_bin_canvas, self.p_prune
        )
        
        # Skeleton drawn directly over full-brightness image
        skel_vis = self.base_rgb.copy()
        skel_crop = full_skel[self.crop_y:y2, self.crop_x:x2]
        skel_vis[skel_crop > 0] = [255, 255, 255]
        self.lbl_skel.setPixmap(self.numpy_to_qpixmap(skel_vis))
        
        if not is_valid:
            self.info_label.setText(
                "Cannot trace this plant. Check Main Root and Lateral Root labels in Annotation."
            )
            self.clear_graph_panels()
            return
            
        self.mc_skel_bin = np.zeros((img_h, img_w), dtype=np.uint8)
        for cid in self.p_target_classes: 
            active = (patch == cid) & (bin_crop > 0)
            p_y, p_x = np.where(active)
            self.mc_skel_bin[p_y + y_off, p_x + x_off] = cid
        self.mc_skel_bin[full_skel == 0] = 0

        skel_ys, skel_xs = np.where(full_skel > 0)
        top_idx = np.argmin(skel_ys)
        root_base = [skel_xs[top_idx], skel_ys[top_idx]]
        
        self.current_graph, self.actual_base, self.colored_skeleton = createGraph(full_skel.copy(), self.mc_skel_bin, root_base, endpoints, branches)
        
        if self.current_graph:
            for u, v, data in self.current_graph.edges(data=True):
                data['orig_root_type'] = data.get('root_type', 2)
        
        self.update_graph_visuals()

    def update_graph_visuals(self):
        if not self.current_graph: return
        
        # =================================================================
        # 1. FETCH BASE MATHEMATICAL STATE
        # Let the core logic determine the true biological start/end and 
        # apply the junction forgiveness traversal costs.
        # =================================================================
        self.current_graph = graphInit(self.current_graph)
        
        auto_start, auto_end = None, None
        
        # Extract the calculated seed and tip from the node attributes
        for node, data in self.current_graph.nodes(data=True):
            if data.get('type') == 'Ini':
                auto_start = node
            elif data.get('type') == 'FTip':
                auto_end = node

        # =================================================================
        # 2. APPLY USER UI OVERRIDES
        # =================================================================
        if self.user_start_node and self.user_start_node not in self.current_graph:
            self.user_start_node = None
        if self.user_end_node and self.user_end_node not in self.current_graph:
            self.user_end_node = None
            
        self.user_waypoints = [wp for wp in self.user_waypoints if wp in self.current_graph]
        
        start_node = self.user_start_node if self.user_start_node else auto_start
        end_node = self.user_end_node if self.user_end_node else auto_end

        # =================================================================
        # 3. INTERACTIVE PATHFINDING 
        # Uses the traversal_cost already set by graphInit_static!
        # =================================================================
        sorted_wps = sorted(self.user_waypoints, key=lambda n: n[1]) 
        targets = []
        if start_node: targets.append(start_node)
        targets.extend(sorted_wps)
        if end_node: targets.append(end_node)
        
        path_edges = set()
        if len(targets) >= 2:
            for i in range(len(targets) - 1):
                try:
                    sub_path = nx.shortest_path(
                        self.current_graph, 
                        source=targets[i], 
                        target=targets[i+1], 
                        weight='traversal_cost'
                    )
                    sub_edges = set(zip(sub_path[:-1], sub_path[1:]))
                    path_edges.update(sub_edges)
                except nx.NetworkXNoPath:
                    pass 
                    
        # Visually lock the path edges as Main Root (1), others as Lateral (2)
        for u, v, data in self.current_graph.edges(data=True):
            if (u, v) in path_edges or (v, u) in path_edges:
                self.current_graph.edges[u, v]['root_type'] = 1
            else:
                self.current_graph.edges[u, v]['root_type'] = 2
                
        # =================================================================
        # 4. RENDER GRAPH TO CANVAS
        # =================================================================
        graph_vis = self.base_rgb.copy()
        
        for u, v, data in self.current_graph.edges(data=True):
            r_type = data.get('root_type', 0)
            col = (255, 0, 0) if r_type == 1 else (0, 255, 0) if r_type == 2 else (150, 150, 150)
            pt1 = (int(u[0]) - self.crop_x, int(u[1]) - self.crop_y)
            pt2 = (int(v[0]) - self.crop_x, int(v[1]) - self.crop_y)
            cv2.line(graph_vis, pt1, pt2, col, 2)
            
        for node in self.current_graph.nodes:
            pt = (int(node[0]) - self.crop_x, int(node[1]) - self.crop_y)
            
            if node == start_node:
                cv2.circle(graph_vis, pt, 6, (0, 255, 0), -1)      # Green (Start)
            elif node == end_node:
                cv2.circle(graph_vis, pt, 6, (0, 0, 255), -1)      # Blue (End)
            elif node in self.user_waypoints:
                cv2.circle(graph_vis, pt, 6, (255, 255, 0), -1)    # Yellow (Waypoint)
            else:
                cv2.circle(graph_vis, pt, 3, (255, 255, 255), -1)  # White (Standard Node)
                
        self.lbl_graph.setPixmap(self.numpy_to_qpixmap(graph_vis))
        
        # =================================================================
        # 5. RENDER SEMANTIC PREVIEW
        # =================================================================
        preview_vis = self.base_rgb.copy() 
        overlay_prev = np.zeros_like(self.base_rgb)
        active_mask_prev = np.zeros(self.base_rgb.shape[:2], dtype=bool)
        
        recolored_semantic = self._calculate_recolored_segmentation()
        
        for cid, color in self.model.class_colors.items():
            if cid == 0: 
                continue 
                
            mask = recolored_semantic == cid
            if np.any(mask):
                overlay_prev[mask] = color[:3] 
                active_mask_prev[mask] = True
            
        alpha = 0.5 
        for c in range(3):
            preview_vis[active_mask_prev, c] = (
                self.base_rgb[active_mask_prev, c] * (1 - alpha) + 
                overlay_prev[active_mask_prev, c] * alpha
            ).astype(np.uint8)
            
        self.lbl_preview.setPixmap(self.numpy_to_qpixmap(preview_vis))

        uid = self.model.active_uid
        if self.user_start_node or self.user_end_node:
            self.info_label.setText(
                f"Plant {uid} — custom start/end nodes applied. "
                "Save root tracing when the path looks correct."
            )
        else:
            self.info_label.setText(
                f"Plant {uid} — if the main-root path looks wrong, set Start/End nodes below "
                "or adjust labels in Annotation."
            )

    def _calculate_recolored_segmentation(self):
        final_mc_skel = np.zeros_like(self.colored_skeleton)
        
        for u, v, data in self.current_graph.edges(data=True):
            e_col, r_type = data.get('color'), data.get('root_type')
            if e_col is not None and r_type is not None:
                final_mc_skel[self.colored_skeleton == e_col] = r_type

        kernel = np.ones((3, 3), np.uint8)
        dilated_canvas = np.zeros_like(final_mc_skel)

        # Define what gets drawn last (highest priority at the end of the list)
        draw_order = [6, 5, 4, 3, 2, 1] 
            
        # Sort the target classes based on your custom order
        ordered_targets = sorted(self.p_target_classes, key=lambda c: draw_order.index(c) if c in draw_order else -1)

        for cls in ordered_targets:
            cls_skel = (final_mc_skel == cls).astype(np.uint8)
            if np.sum(cls_skel) == 0: continue
            dilated = cv2.dilate(cls_skel, kernel, iterations=self.p_thick)
            dilated_canvas[dilated > 0] = cls

        h, w = self.mask_crop.shape
        dilated_crop = dilated_canvas[self.crop_y : self.crop_y + h, self.crop_x : self.crop_x + w]

        # --- THE MERGE LOGIC ---
        recolored_semantic = np.zeros_like(self.mask_crop)
        
        # 1. Fetch the original patch to salvage the non-targeted classes
        patch, x_off, y_off = self.model._get_class_patch(self.model.active_uid)
        
        # 2. Map the original patch to our current crop coordinates
        orig_mc_crop = np.zeros_like(self.mask_crop)
        p_h, p_w = patch.shape
        cy_start = max(0, y_off - self.crop_y)
        cx_start = max(0, x_off - self.crop_x)
        cy_end = min(h, cy_start + p_h)
        cx_end = min(w, cx_start + p_w)
        
        # Handle inner patch slicing if crop cuts off the patch
        py_start = max(0, self.crop_y - y_off)
        px_start = max(0, self.crop_x - x_off)
        py_end = py_start + (cy_end - cy_start)
        px_end = px_start + (cx_end - cx_start)
        
        orig_mc_crop[cy_start:cy_end, cx_start:cx_end] = patch[py_start:py_end, px_start:px_end]
        
        # 3. Copy over ANY class that was NOT in p_target_classes
        untargeted_mask = (self.mask_crop > 0) & (~np.isin(orig_mc_crop, self.p_target_classes))
        recolored_semantic[untargeted_mask] = orig_mc_crop[untargeted_mask]

        # 4. Handle unreached pixels (distance transform) ONLY for target classes
        unreached = (self.mask_crop > 0) & (dilated_crop == 0) & (~untargeted_mask)
        if np.any(unreached):
            valid = dilated_crop > 0
            if np.any(valid):
                _, indices = distance_transform_edt(~valid, return_indices=True)
                dilated_crop[unreached] = dilated_crop[indices[0], indices[1]][unreached]
                
        # 5. Overlay the new graph-calculated target classes
        recolored_semantic[dilated_crop > 0] = dilated_crop[dilated_crop > 0]
        
        return recolored_semantic

    def apply_graph_colors(self):
        uid = self.model.active_uid
        if not self.current_graph or self.colored_skeleton is None or not uid: return
        if uid not in self.model.masks: return
        
        self.user_start_node = None
        self.user_end_node = None
        self.user_waypoints = []
        
        original_mask = self.model.masks[uid]
        recolored_semantic = self._calculate_recolored_segmentation()

        self.model.save_state(uid)

        ys, xs = np.where(original_mask > 0)
        if len(xs) > 0:
            cx1, cx2 = int(xs.min()), int(xs.max())
            cy1, cy2 = int(ys.min()), int(ys.max())
            tight_patch = np.zeros((cy2 - cy1 + 1, cx2 - cx1 + 1), dtype=np.uint8)
            
            for ly, lx in zip(ys, xs):
                tight_patch[ly - cy1, lx - cx1] = recolored_semantic[ly - self.crop_y, lx - self.crop_x]
                
            self.model.class_patches[uid] = (tight_patch, cx1, cy1)
        else:
            self.model.masks.pop(uid, None)
            self.model.class_patches.pop(uid, None)

        self.model.update_metadata_for_uid(uid)
        self.model.dirty = True
        
        self.model._notify_data_changed()
        
    def numpy_to_qpixmap(self, img_array):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)


# ==========================================
# 2. THE SIDEBAR TOOL PANEL
# ==========================================
class GraphToolPanel(QWidget):
    def __init__(self, shared_model, canvas_tab: GraphCanvasTab):
        super().__init__()
        self.model = shared_model
        self.canvas_tab = canvas_tab
        
        self.init_ui()
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        graph_header = QHBoxLayout()
        graph_header.addWidget(QLabel("<b>Root tracing check</b>"))
        graph_header.addStretch()
        btn_help = QPushButton("Help")
        btn_help.setToolTip("Open root tracing check help")
        btn_help.clicked.connect(lambda: show_help(self, "Tracing Help", HELP_GRAPH))
        graph_header.addWidget(btn_help)
        layout.addLayout(graph_header)

        layout.addWidget(QLabel(
            "<span style='color:#555; font-size:11px;'>"
            "Legend: Green = start, Blue = end, Yellow = waypoint</span>"
        ))
        
        layout.addWidget(QLabel("<b>Graph Settings:</b>"))
        
        form = QFormLayout()
        self.sp_prune = QSpinBox()
        self.sp_prune.setRange(0, 10)
        self.sp_prune.setValue(self.canvas_tab.p_prune)
        self.sp_prune.valueChanged.connect(self.push_params)
        
        self.sp_thick = QSpinBox()
        self.sp_thick.setRange(1, 10)
        self.sp_thick.setValue(self.canvas_tab.p_thick)
        self.sp_thick.setToolTip("Thickness used when applying graph colors to the mask.")
        self.sp_thick.valueChanged.connect(self.push_params)

        self.sp_prune.setToolTip(
            "Plate-wide skeleton pruning (DPI-dependent). Lower if lateral roots disappear; "
            "raise if skeleton spurs/noise remain."
        )
        
        form.addRow("Plate Prune Iterations:", self.sp_prune)
        form.addRow("Dilation Base Thick:", self.sp_thick)
        layout.addLayout(form)
        
        layout.addSpacing(10)
        layout.addWidget(QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken))
        layout.addSpacing(10)

        layout.addWidget(QLabel("<b>Node Click Interaction:</b>"))
        
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Set Start Node (Green)", "Set End Node (Blue)", "Toggle Waypoint (Yellow)"])
        self.cb_mode.setToolTip("Click the interactive graph to place start, end, or waypoint nodes.")
        self.cb_mode.currentIndexChanged.connect(self.update_mode)
        layout.addWidget(self.cb_mode)
        
        self.btn_reset_nodes = QPushButton("Reset Nodes to Auto")
        self.btn_reset_nodes.setToolTip("Clear manual nodes and use automatic start/end detection.")
        self.btn_reset_nodes.clicked.connect(self.canvas_tab.reset_user_nodes)
        layout.addWidget(self.btn_reset_nodes)

        layout.addSpacing(10)
        layout.addWidget(QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken))
        layout.addSpacing(10)
        
        # --- Target Classes ---
        layout.addWidget(QLabel("<b>Classes to Graph:</b>"))
        self.list_target_classes = QListWidget()
        self.list_target_classes.setFixedHeight(100)

        classes = [(1, "Main Root"), (2, "Lateral Root"), (3, "Seed"),
                   (4, "Hypocotyl"), (5, "Leaves/Aerial"), (6, "Petiole")]

        for cid, name in classes:
            item = QListWidgetItem(f"{cid} - {name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if cid in self.canvas_tab.p_target_classes else Qt.Unchecked)
            item.setData(Qt.UserRole, cid)
            self.list_target_classes.addItem(item)

        self.list_target_classes.itemChanged.connect(self.push_params)
        layout.addWidget(self.list_target_classes)

        main_layout.addLayout(layout)

        # --- Apply Button (sticky at the bottom) ---
        self.btn_apply_mask = QPushButton("Save root tracing for this plant")
        self.btn_apply_mask.setToolTip("Commit the traced main/lateral colors to this plant's mask.")
        self.btn_apply_mask.setFixedHeight(32)
        self.btn_apply_mask.setStyleSheet(
            "background-color: #d4edda; font-weight: bold; color: #155724; font-size: 12px;"
        )
        self.btn_apply_mask.clicked.connect(self.canvas_tab.apply_graph_colors)
        
        main_layout.addWidget(self.btn_apply_mask)
        self.setFixedHeight(self.sizeHint().height())
        self.toggle_buttons(False)

    def update_mode(self):
        idx = self.cb_mode.currentIndex()
        if idx == 0: self.canvas_tab.interaction_mode = "START"
        elif idx == 1: self.canvas_tab.interaction_mode = "END"
        else: self.canvas_tab.interaction_mode = "WAYPOINT"

    def push_params(self):
        self.canvas_tab.p_prune = self.sp_prune.value()
        self.canvas_tab.p_thick = self.sp_thick.value()
        
        # Extract checked classes
        targets = []
        for i in range(self.list_target_classes.count()):
            item = self.list_target_classes.item(i)
            if item.checkState() == Qt.Checked:
                targets.append(item.data(Qt.UserRole))
                
        self.canvas_tab.p_target_classes = targets
    
        if self.model.active_uid:
            self.canvas_tab.generate_pipeline()

    def toggle_buttons(self, enabled):
        self.btn_apply_mask.setEnabled(enabled)

    def on_selection_changed(self):
        self.toggle_buttons(self.model.active_uid is not None)