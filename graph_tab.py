import cv2
import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QFrame, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

# Import the isolated builder
from root_graph_builder import extract_skeleton, createGraph

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


class ChronoRootTab(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.current_uids = []
        self.current_index = 0
        self.current_uid = None
        
        # State variables for the Bake step
        self.current_graph = None
        self.colored_skeleton = None
        
        self.init_ui()
        self.model.register_callback(self.on_model_update)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Parameter Bar ---
        param_frame = QFrame()
        p_layout = QHBoxLayout(param_frame)
        
        self.sp_prune = QSpinBox()
        self.sp_prune.setRange(0, 10)
        self.sp_prune.setValue(0) # Default to 0
        self.sp_prune.valueChanged.connect(self.generate_pipeline)
        
        p_layout.addWidget(QLabel("Prune Iterations (Roots):"))
        p_layout.addWidget(self.sp_prune)
        p_layout.addWidget(QFrame(frameShape=QFrame.VLine, frameShadow=QFrame.Sunken))

        # Apply Actions
        self.sp_thick = QSpinBox()
        self.sp_thick.setRange(1, 10)
        self.sp_thick.setValue(1) # Default to 1
        self.sp_thick.valueChanged.connect(self.generate_pipeline) 

        self.btn_apply_mask = QPushButton("Apply Graph Colors to Mask")
        self.btn_apply_mask.setStyleSheet("background-color: #d4edda; font-weight: bold; color: #155724;")
        self.btn_apply_mask.clicked.connect(self.apply_graph_colors)
        
        p_layout.addWidget(QLabel("Dilation Base Thickness:"))
        p_layout.addWidget(self.sp_thick)
        p_layout.addWidget(self.btn_apply_mask)
        p_layout.addStretch()
        layout.addWidget(param_frame)
        
        # --- Visual Layout (5 Static Columns) ---
        visual_layout = QHBoxLayout()
        visual_layout.setSpacing(5)
        
        self.lbl_orig = AspectRatioLabel()
        self.lbl_sem = AspectRatioLabel()
        self.lbl_skel = AspectRatioLabel()
        self.lbl_preview = AspectRatioLabel()
        self.lbl_graph = AspectRatioLabel()
        
        def make_col(title, widget):
            v = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
            v.addWidget(lbl)
            v.addWidget(widget)
            return v
            
        visual_layout.addLayout(make_col("1. RGB Image", self.lbl_orig))
        visual_layout.addLayout(make_col("2. Original Semantic", self.lbl_sem))
        visual_layout.addLayout(make_col("3. Skeleton Overlay", self.lbl_skel))
        visual_layout.addLayout(make_col("4. Recolored Preview", self.lbl_preview))
        visual_layout.addLayout(make_col("5. Network Graph", self.lbl_graph))
        
        layout.addLayout(visual_layout, stretch=1)
        
        # --- Navigation ---
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⬅ Previous Plant")
        self.btn_prev.clicked.connect(self.prev_plant)
        
        self.lbl_info = QLabel("Load an image to begin.")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        
        self.btn_next = QPushButton("Next Plant ➡")
        self.btn_next.clicked.connect(self.next_plant)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_info)
        nav_layout.addWidget(self.btn_next)
        
        layout.addLayout(nav_layout)

    def on_model_update(self):
        if getattr(self, 'model', None) is None or not self.model.image_path: return
        self.current_uids = sorted(list(self.model.masks.keys()))
        if self.isVisible(): self.load_current_plant()

    def showEvent(self, event):
        super().showEvent(event)
        self.on_model_update()

    def prev_plant(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_plant()

    def next_plant(self):
        if self.current_index < len(self.current_uids) - 1:
            self.current_index += 1
            self.load_current_plant()

    def load_current_plant(self):
        if not self.current_uids: return
        if self.current_index >= len(self.current_uids): self.current_index = 0
            
        self.current_uid = self.current_uids[self.current_index]
        self.lbl_info.setText(f"Plant UID: {self.current_uid} ({self.current_index + 1} of {len(self.current_uids)})")
        self.generate_pipeline()

    def generate_pipeline(self):
        if self.current_uid not in self.model.masks: return
        
        mask = self.model.masks[self.current_uid]
        patch_data = self.model._get_class_patch(self.current_uid)
        if not patch_data: return
        patch, x_off, y_off = patch_data
        
        x, y, w, h = self.model.bboxes.get(self.current_uid, (0,0,0,0))
        if w == 0: return
        
        img_h, img_w = self.model.raw_image.shape[:2]
        
        pad = 20
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
        
        # 1. RGB Crop & Dimmed Base
        rgb_crop = self.model.raw_image[y1:y2, x1:x2].copy()
        dim_rgb = rgb_crop.copy() // 2
        self.lbl_orig.setPixmap(self.numpy_to_qpixmap(rgb_crop))
        
        # 2. Semantic Context Overlay
        sem_vis = dim_rgb.copy()
        bin_crop = mask[y_off:y_off+patch.shape[0], x_off:x_off+patch.shape[1]]
        for cid, color in self.model.class_colors.items():
            if cid != 0:
                active = (patch == cid) & (bin_crop > 0)
                p_y, p_x = np.where(active)
                global_y, global_x = p_y + y_off, p_x + x_off
                valid = (global_y >= y1) & (global_y < y2) & (global_x >= x1) & (global_x < x2)
                sem_vis[global_y[valid] - y1, global_x[valid] - x1] = color[:3]
                
        self.lbl_sem.setPixmap(self.numpy_to_qpixmap(sem_vis))
        
        # --- Skeletonization ---
        root_bin_canvas = np.zeros((img_h, img_w), dtype=np.uint8)
        for cid in [1, 2]:
            active = (patch == cid) & (bin_crop > 0)
            p_y, p_x = np.where(active)
            root_bin_canvas[p_y + y_off, p_x + x_off] = 1
            
        full_skel, branches, endpoints, is_valid = extract_skeleton(root_bin_canvas, self.sp_prune.value())
        
        # 3. Skeleton Overlay
        skel_vis = dim_rgb.copy()
        skel_crop = full_skel[y1:y2, x1:x2]
        skel_vis[skel_crop > 0] = [255, 255, 255]
        self.lbl_skel.setPixmap(self.numpy_to_qpixmap(skel_vis))
        
        # 4 & 5. Graph and Preview
        if not is_valid:
            self.lbl_preview.setPixmap(QPixmap())
            self.lbl_graph.setPixmap(QPixmap())
            self.current_graph = None
            self.colored_skeleton = None
            return
            
        mc_skel_bin = np.zeros((img_h, img_w), dtype=np.uint8)
        for cid in [1, 2]: 
            active = (patch == cid) & (bin_crop > 0)
            p_y, p_x = np.where(active)
            mc_skel_bin[p_y + y_off, p_x + x_off] = cid
        mc_skel_bin[full_skel == 0] = 0

        skel_ys, skel_xs = np.where(full_skel > 0)
        top_idx = np.argmin(skel_ys)
        root_base = [skel_xs[top_idx], skel_ys[top_idx]]
        
        self.current_graph, actual_base, self.colored_skeleton = createGraph(full_skel.copy(), mc_skel_bin, root_base, endpoints, branches)
        
        # --- Fixed Semantic-Weighted Edge Labeling ---
        start_node = None
        end_node = None
        
        if self.current_graph:
            # 1. Find the biological "Seed" based ONLY on pixels labeled Main Root (1)
            # We look at the mc_skel_bin (which is masked by the skeleton)
            c1_ys, c1_xs = np.where(mc_skel_bin == 1)
            
            if len(c1_ys) > 0:
                # Topmost Class 1 pixel
                top_c1_idx = np.argmin(c1_ys)
                anchor_pt = (c1_xs[top_c1_idx], c1_ys[top_c1_idx])
                
                # Snap to the nearest existing graph node
                nodes_list = list(self.current_graph.nodes)
                nodes_arr = np.array(nodes_list)
                dists = np.linalg.norm(nodes_arr - np.array(anchor_pt), axis=1)
                start_node = nodes_list[np.argmin(dists)]
            else:
                # Fallback to actual_base if no Class 1 is found at all
                start_node = tuple(actual_base) if actual_base else None

            # 2. End node is still the bottom-most point of the whole system
            end_node = max(self.current_graph.nodes, key=lambda n: n[1]) 

            if start_node and end_node:
                # 3. Calculate Traversal Cost
                # We trust the model/manual labels: Class 1 is cheap, Class 2 is expensive
                for u, v, data in self.current_graph.edges(data=True):
                    phys_len = data.get('weight', 1.0)
                    # Get the root type assigned during createGraph (from mc_skel_bin)
                    orig_type = data.get('root_type', 2)
                    
                    if orig_type == 1:
                        data['traversal_cost'] = phys_len
                    else:
                        # 100x penalty to ensure we stay on Class 1 unless there's a gap
                        data['traversal_cost'] = phys_len * 100.0 
                        
                try:
                    # 4. Find the optimal semantic path
                    path = nx.shortest_path(self.current_graph, source=start_node, target=end_node, weight='traversal_cost')
                    path_edges = set(zip(path[:-1], path[1:]))
                    
                    # 5. Final Edge Labeling: Path is 1, Everything else is 2
                    for u, v, data in self.current_graph.edges(data=True):
                        if (u, v) in path_edges or (v, u) in path_edges:
                            self.current_graph.edges[u, v]['root_type'] = 1
                        else:
                            self.current_graph.edges[u, v]['root_type'] = 2
                            
                except nx.NetworkXNoPath:
                    # If disconnected, default everything to lateral (2)
                    for u, v, data in self.current_graph.edges(data=True):
                        self.current_graph.edges[u, v]['root_type'] = 2
                
        # 5. Network Graph Overlay (Static Drawing)
        graph_vis = dim_rgb.copy()
        if self.current_graph:
            for u, v, data in self.current_graph.edges(data=True):
                r_type = data.get('root_type', 0)
                col = (255, 0, 0) if r_type == 1 else (0, 255, 0) if r_type == 2 else (150, 150, 150)
                
                pt1 = (int(u[0]) - x1, int(u[1]) - y1)
                pt2 = (int(v[0]) - x1, int(v[1]) - y1)
                cv2.line(graph_vis, pt1, pt2, col, 2)
                
            for node in self.current_graph.nodes:
                pt = (int(node[0]) - x1, int(node[1]) - y1)
                cv2.circle(graph_vis, pt, 3, (255, 255, 255), -1)
                
            if start_node:
                cv2.circle(graph_vis, (int(start_node[0]) - x1, int(start_node[1]) - y1), 5, (0, 255, 0), -1)
            if end_node:
                cv2.circle(graph_vis, (int(end_node[0]) - x1, int(end_node[1]) - y1), 5, (0, 0, 255), -1)
                
        self.lbl_graph.setPixmap(self.numpy_to_qpixmap(graph_vis))
        
        # 4. Dilated/Distance Transform Recolored Preview
        preview_vis = dim_rgb.copy()
        
        recolored_semantic = self._calculate_recolored_segmentation(mask)
        
        cls_colors = {1: [255, 0, 0], 2: [0, 255, 0]}
        sem_crop = recolored_semantic[y1:y2, x1:x2]
        
        for cid in [1, 2]:
            preview_vis[sem_crop == cid] = cls_colors[cid]
            
        self.lbl_preview.setPixmap(self.numpy_to_qpixmap(preview_vis))


    def _calculate_recolored_segmentation(self, original_mask):
        final_mc_skel = np.zeros_like(self.colored_skeleton)
        
        if self.current_graph:
            for u, v, data in self.current_graph.edges(data=True):
                e_col, r_type = data.get('color'), data.get('root_type')
                if e_col is not None and r_type is not None:
                    final_mc_skel[self.colored_skeleton == e_col] = r_type

        thickness = self.sp_thick.value()
        kernel = np.ones((3, 3), np.uint8)
        dilated_canvas = np.zeros_like(final_mc_skel)

        # Priority Dilation: Draw Lateral (2) then Main (1) over it
        for cls in [2, 1]:
            cls_skel = (final_mc_skel == cls).astype(np.uint8)
            if np.sum(cls_skel) == 0: continue
            dilated = cv2.dilate(cls_skel, kernel, iterations=thickness)
            dilated_canvas[dilated > 0] = cls

        # Fill unreached parts of the original mask using Euclidean Distance Transform
        unreached = (original_mask > 0) & (dilated_canvas == 0)
        
        if np.any(unreached):
            valid = dilated_canvas > 0
            if np.any(valid):
                _, indices = distance_transform_edt(~valid, return_indices=True)
                dilated_canvas[unreached] = dilated_canvas[indices[0], indices[1]][unreached]
                
        recolored_semantic = np.zeros_like(original_mask)
        recolored_semantic[original_mask > 0] = dilated_canvas[original_mask > 0]
        
        return recolored_semantic


    def apply_graph_colors(self):
        if not self.current_graph or self.colored_skeleton is None: return
        uid = self.current_uid
        if uid not in self.model.masks: return
        
        original_mask = self.model.masks[uid]

        recolored_semantic = self._calculate_recolored_segmentation(original_mask)

        self.model.save_state(uid)

        ys, xs = np.where(original_mask > 0)
        if len(xs) > 0:
            cx1, cx2 = int(xs.min()), int(xs.max())
            cy1, cy2 = int(ys.min()), int(ys.max())
            tight_patch = np.zeros((cy2 - cy1 + 1, cx2 - cx1 + 1), dtype=np.uint8)
            
            for ly, lx in zip(ys, xs):
                tight_patch[ly - cy1, lx - cx1] = recolored_semantic[ly, lx]
                
            self.model.class_patches[uid] = (tight_patch, cx1, cy1)
        else:
            self.model.masks.pop(uid, None); self.model.class_patches.pop(uid, None)

        self.model.update_metadata_for_uid(uid)
        self.model.dirty = True
        self.model._notify_changed()

        self.generate_pipeline()
        QMessageBox.information(self, "Semantic Sync", f"Graph colors successfully applied to original mask geometry for Plant {uid}.")

    def numpy_to_qpixmap(self, img_array):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)