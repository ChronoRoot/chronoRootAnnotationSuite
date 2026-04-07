import os
import json
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import label, distance_transform_edt

# ==========================================
# CONSTANTS & HELPERS
# ==========================================
LINE_THICKNESS = 1               
LOGICAL_THICKNESS = 2            
MAX_HISTORY = 10

HIGH_CONTRAST_COLORS = [
    (255, 0, 0),    (0, 255, 0),    (0, 0, 255),    (255, 255, 0),
    (255, 0, 255),  (0, 255, 255),  (255, 165, 0),  (128, 0, 128),
    (0, 255, 127),  (255, 20, 147), (139, 69, 19),  (75, 0, 130)
]

def encode_rle(mask):
    pixels = mask.flatten()
    if len(pixels) == 0: return {"size": [0, 0], "counts": []}
    pixels = np.concatenate([pixels, [-1]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    counts = []
    vals = []
    prev = 0
    for idx in runs:
        vals.append(int(pixels[prev]))
        counts.append(int(idx - prev))
        prev = idx
        
    rle = []
    for v, c in zip(vals, counts):
        rle.extend([v, c])
    return {"size": mask.shape, "counts": rle}

def decode_rle(rle_data):
    h, w = rle_data["size"]
    counts = rle_data["counts"]
    pixels = []
    for i in range(0, len(counts), 2):
        val = counts[i]
        count = counts[i+1]
        pixels.extend([val] * count)
    return np.array(pixels, dtype=np.uint8).reshape((h, w))

# ==========================================
# HYBRID DATA MODEL (Pure Python API)
# ==========================================
class PlantImageModel:
    def __init__(self):
        self.image_path = None
        self.raw_image = None       
        self.masks = {} 
        
        self.max_id = 0
        self.color_map = {}     
        self.areas = {}    
        self.bboxes = {}            
        self.history = []
        self.dirty = False
        self.status = "pending" 
        self.original_multiclass = None
        self.class_patches = {}
        self.class_colors = {
            0: (0, 0, 0, 0),        
            1: (255, 0, 0, 255),    
            2: (0, 255, 0, 255),    
            3: (0, 0, 255, 255),    
            4: (255, 255, 0, 255),  
            5: (0, 255, 255, 255),  
            6: (255, 0, 255, 255)   
        }

        # --- NEW: Global Selection State ---
        self.selected_uids = set()
        self.active_uid = None
        
        # --- NEW: Split Callbacks ---
        self._data_callbacks = []       # For mask changes (paint, split, etc.)
        self._selection_callbacks = []  # For UI selection changes
        self.callbacks_muted = False 

    def register_data_callback(self, callback):
        self._data_callbacks.append(callback)
        
    def register_selection_callback(self, callback):
        self._selection_callbacks.append(callback)

    def _notify_data_changed(self):
        if self.callbacks_muted: return
        for callback in self._data_callbacks: callback()
        
    def _notify_selection_changed(self):
        if self.callbacks_muted: return
        for callback in self._selection_callbacks: callback()

    # (Optional helper to safely update selection)
    def set_selection(self, uids):
        self.selected_uids = set(uids)
        self.active_uid = list(self.selected_uids)[0] if len(self.selected_uids) == 1 else None
        self._notify_selection_changed()

    # --- FILE SYSTEM & IO (Completely Isolated) ---
    
    def scan_directory(self, folder_path):
        """Scans a directory and returns a formatted list of dicts for the GUI without exposing OS/JSON ops."""
        contents = []
        if not os.path.exists(folder_path): return contents
        
        items = sorted(os.listdir(folder_path))
        
        # 1. Folders
        for item_name in items:
            full_path = os.path.join(folder_path, item_name)
            if os.path.isdir(full_path):
                total, in_progress, completed = 0, 0, 0
                for root, _, files in os.walk(full_path):
                    for f in files:
                        if f.endswith('.nii.gz') and f.replace('.nii.gz', '.png') in files:
                            total += 1
                            json_path = os.path.join(root, f.replace('.nii.gz', '.json'))
                            if os.path.exists(json_path):
                                try:
                                    with open(json_path, 'r') as jf:
                                        if json.load(jf).get("status") == "completed":
                                            completed += 1
                                        else:
                                            in_progress += 1
                                except: pass
                contents.append({
                    "type": "dir", "name": item_name, "path": full_path,
                    "total": total, "in_progress": in_progress, "completed": completed
                })

        # 2. Files
        for item_name in items:
            if item_name.endswith('.nii.gz'):
                base = item_name.replace('.nii.gz', '')
                if (base + '.png') in items:
                    full_path = os.path.join(folder_path, item_name)
                    json_path = os.path.join(folder_path, base + ".json")
                    status = "Pending"
                    
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                st = json.load(f).get("status", "in_progress")
                                status = "Completed" if st == "completed" else "In Progress"
                        except: pass
                        
                    contents.append({
                        "type": "file", "name": base, "path": full_path, "status": status
                    })
        return contents

    def load_task(self, task_path, base_name):
        """Loads all required files and pre-caches heavy data."""
        
        # 1. ABSOLUTE MEMORY WIPE
        self.image_path = None
        self.raw_image = None       
        self.masks.clear()
        self.max_id = 0
        self.color_map.clear()  
        self.areas.clear()       
        self.bboxes.clear()            
        self.history.clear()
        self.dirty = False
        self.original_multiclass = None
        self.class_patches.clear()
        
        # 2. LOAD NEW DATA
        self.status = "pending"
        self.image_path = os.path.join(task_path, f"{base_name}.png")
        nii_path = os.path.join(task_path, f"{base_name}.nii.gz")
        json_path = os.path.join(task_path, f"{base_name}.json")
        
        img_bgr = cv2.imread(self.image_path)
        if img_bgr is None: raise FileNotFoundError(f"Missing PNG: {self.image_path}")
        self.raw_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = self.raw_image.shape[:2]
        
        if os.path.exists(json_path):
            self._load_mask_from_nifti(nii_path, (h, w))
            self._load_from_json(json_path, (h, w))
        else:
            self._load_instances_from_nifti(nii_path, (h, w))

        # --- NEW: Filter out spurious instances (width < 3 or height < 3) ---
        uids_to_remove = []
        for uid, mask in self.masks.items():
            coords = cv2.findNonZero(mask)
            if coords is None:
                uids_to_remove.append(uid)
            else:
                _, _, bw, bh = cv2.boundingRect(coords)
                # Discard if smaller than 3 pixels in any dimension (treat as noise)
                if bw < 3 or bh < 3:
                    uids_to_remove.append(uid)
                    
        for uid in uids_to_remove:
            self.masks.pop(uid, None)
            self.class_patches.pop(uid, None)
        # ---------------------------------------------------------------------

        self.max_id = max(self.masks.keys()) if self.masks else 0
        self.regenerate_metadata() 
        
        # 3. Warm up class patches for faster GUI response
        for uid in self.masks.keys():
            self._get_class_patch(uid)

    def clean_and_validate_masks(self):
        needs_metadata_update = False
        
        for uid, mask in list(self.masks.items()):
            if np.sum(mask) == 0: 
                continue
            
            # Use SciPy Label to find true 8-connected pixel components
            labeled_mask, num_features = label(mask, structure=np.ones((3,3)))
            
            valid_count = 0
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0 # Ignore the background (label 0)
                
                for i in range(1, num_features + 1):
                    # Discard spurious noise strictly by pixel count, not polygon area
                    if sizes[i] < 5:
                        mask[labeled_mask == i] = 0
                        self.dirty = True
                        needs_metadata_update = True
                    else:
                        valid_count += 1
                        
            # After cleaning, check if we still have multiple disconnected parts
            if valid_count > 1:
                return False, uid
                
        if needs_metadata_update:
            # Re-generate bounding boxes for any masks that had noise erased
            for uid in self.masks.keys():
                coords = cv2.findNonZero(self.masks[uid])
                if coords is not None:
                    self.bboxes[uid] = cv2.boundingRect(coords)
                    
        return True, None
    
    def reindex_instances(self):
        """Remaps all UIDs to be strictly continuous. Returns a mapping of {old_uid: new_uid}."""
        sorted_uids = sorted(self.masks.keys())
        if not sorted_uids: return {}
        
        # If they are already 1 to N perfectly, return a 1:1 mapping
        if sorted_uids == list(range(1, len(sorted_uids) + 1)):
            return {u: u for u in sorted_uids}

        mapping = {}
        new_masks, new_bboxes, new_areas, new_color_map, new_class_patches = {}, {}, {}, {}, {}
        
        for new_uid, old_uid in enumerate(sorted_uids, start=1):
            mapping[old_uid] = new_uid
            
            new_masks[new_uid] = self.masks[old_uid]
            if old_uid in self.bboxes: new_bboxes[new_uid] = self.bboxes[old_uid]
            if old_uid in self.areas: new_areas[new_uid] = self.areas[old_uid]
            
            new_color_map[new_uid] = HIGH_CONTRAST_COLORS[new_uid % len(HIGH_CONTRAST_COLORS)]
            
            if old_uid in self.class_patches:
                patch, x, y = self.class_patches[old_uid]
                new_class_patches[new_uid] = (patch, x, y)
                
        self.masks = new_masks
        self.bboxes = new_bboxes
        self.areas = new_areas
        self.color_map = new_color_map
        self.class_patches = new_class_patches
        self.max_id = len(sorted_uids)
        self.history.clear()
        
        return mapping

    def save_current_task(self, task_path, base_name, mark_finished=False):
        """Writes current mask data to disk. Completely insulates GUI from JSON/NIfTI."""
        if not self.masks: return False

        # 1. CLEANUP AND VALIDATE
        is_valid, bad_uid = self.clean_and_validate_masks()
        if not is_valid:
            raise ValueError(f"UID {bad_uid} consists of more than one connected component.\nPlease use the paint tool to connect the pieces, or the split tool to separate them into different IDs.")

        # 2. REINDEX PLANTS AND GET THE MAPPING
        mapping = self.reindex_instances()

        # 3. PROCEED WITH SAVING
        self.status = "completed" if mark_finished else "in_progress"
        
        data = self._export_coco()
        json_path = os.path.join(task_path, f"{base_name}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f)
            
        final_labels = self._flatten_multiclass()
        self.original_multiclass = final_labels
        
        if final_labels is not None:
            nii_path = os.path.join(task_path, f"{base_name}.nii.gz")
            new_nifti = nib.Nifti1Image(final_labels.astype(np.uint8).T, np.eye(4))
            nib.save(new_nifti, nii_path)
                
        self.dirty = False
        
        # Return the mapping dictionary so the GUI knows how to update itself!
        return mapping

    # --- INTERNAL LOAD/SAVE HELPERS ---
    
    def _load_mask_from_nifti(self, nii_path, shape):
        try:
            data = np.squeeze(nib.load(nii_path).get_fdata())
            data = data.T
            self.original_multiclass = data.astype(np.uint8)
        except Exception as e: print(f"NIfTI Error: {e}")
        
    def _load_instances_from_nifti(self, nii_path, shape):
        try:
            data = np.squeeze(nib.load(nii_path).get_fdata())
            data = data.T
            self.original_multiclass = data.astype(np.uint8)
            
            labeled_map, _ = label((data > 0).astype(np.int32), structure=np.ones((3,3)))
            for uid in np.unique(labeled_map):
                if uid == 0: continue
                self.masks[int(uid)] = (labeled_map == uid).astype(np.uint8)
        except Exception as e: print(f"NIfTI Error: {e}")

    def _load_from_json(self, json_path, shape):
        try:
            with open(json_path, 'r') as f: data = json.load(f)
            self.status = data.get("status", "in_progress")
            
            for ann in data.get('annotations', []):
                uid = ann['id']
                
                # --- BUGFIX: Prioritize lossless RLE to preserve holes ---
                if "instance_rle" in ann:
                    mask = decode_rle(ann["instance_rle"])
                else:
                    # Fallback for old saves (using the hole-filling polygons)
                    mask = np.zeros(shape, dtype=np.uint8)
                    for seg in ann.get('segmentation', []):
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 1)
                        
                self.masks[uid] = mask
                
                ys, xs = np.where(mask)
                if len(xs) > 0 and "semantic_rle" in ann:
                    self.class_patches[uid] = (decode_rle(ann["semantic_rle"]), int(xs.min()), int(ys.min()))
        except Exception as e: print(f"JSON Error: {e}")
    
    def _export_coco(self):
        output = {
            "info": {"status": self.status}, "status": self.status,
            "images": [{"id": 1, "file_name": os.path.basename(self.image_path), 
                        "width": self.raw_image.shape[1], "height": self.raw_image.shape[0]}],
            "annotations": [], "categories": [{"id": 1, "name": "Plant"}]
        }
        ann_id = 1
        for uid, mask in self.masks.items():
            if np.sum(mask) == 0: continue
            
            # We keep standard polygons for legacy readers, but they lose holes
            polys = [c.flatten().tolist() for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if c.shape[0] >= 3]
            
            ys, xs = np.where(mask)
            if len(xs) == 0: continue
            
            patch, _, _ = self._get_class_patch(uid)
            output["annotations"].append({
                "id": ann_id, "image_id": 1, "category_id": 1,
                "segmentation": polys, 
                "instance_rle": encode_rle(mask), # <--- BUGFIX: Lossless binary mask encoding
                "semantic_rle": encode_rle(patch), 
                "area": float(np.sum(mask)), "iscrowd": 0,
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max()-xs.min()), float(ys.max()-ys.min())]
            })
            ann_id += 1
        return output

    # --- GUI-FRIENDLY RENDER EXPORTS ---
    
    def _pack_for_gui(self, array, is_rgba=True):
        """Converts NumPy array to a GUI-safe tuple (bytes, width, height, bytes_per_line)."""
        if array is None: return None
        h, w = array.shape[:2]
        bpl = (4 if is_rgba else 3) * w
        # tobytes() creates a safe copy for the GUI to consume
        return (array.tobytes(), w, h, bpl)

    def get_raw_image_data(self):
        return self._pack_for_gui(self.raw_image, is_rgba=False)

    def get_overlay_data(self, selected_ids=[], opacity=0.35, isolate_uids=None):
        if self.raw_image is None: return None
        h, w = self.raw_image.shape[:2]
        
        render_idx = np.zeros((h, w), dtype=np.int32)
        target_ids = isolate_uids if isolate_uids is not None else self.masks.keys()
        
        for uid in target_ids:
            if uid not in self.bboxes: 
                continue
            x, y, bw, bh = self.bboxes[uid]
            
            # Extract only the small box where the plant actually exists
            mask_crop = self.masks[uid][y:y+bh, x:x+bw]
            render_crop = render_idx[y:y+bh, x:x+bw]
            
            # Apply the ID only to the active pixels in that small box
            render_crop[mask_crop > 0] = uid

        max_id_val = max(self.masks.keys()) if self.masks else 0
        lut = np.zeros((max_id_val + 1, 4), dtype=np.uint8)
        
        for uid in target_ids:
            if uid not in self.color_map: continue
            lut[uid] = [*self.color_map[uid], int(255 * opacity)]

        overlay = lut[render_idx]

        # Add 1px strictly-outside white border for selected instances
        kernel = np.ones((3, 3), dtype=np.uint8)
        
        for uid in selected_ids:
            if uid in self.masks and uid in self.bboxes:
                x, y, bw, bh = self.bboxes[uid]
                
                # Crop with a small pad to allow the outer boundary to exist
                pad = 2
                x1, y1 = max(0, x-pad), max(0, y-pad)
                x2, y2 = min(w, x+bw+pad), min(h, y+bh+pad)
                
                crop = self.masks[uid][y1:y2, x1:x2]
                
                # Expand the mask by exactly 1 pixel
                dilated = cv2.dilate(crop, kernel, iterations=1)
                
                # Keep only the newly expanded pixels (the outer ring)
                outer_border = (dilated > 0) & (crop == 0)
                
                # Paint only that outer ring pure white on the overlay
                overlay_crop = overlay[y1:y2, x1:x2]
                overlay_crop[outer_border] = [255, 255, 255, 255]

        return self._pack_for_gui(overlay)

    def get_binary_contours_data(self, uid):
        if uid not in self.masks: return None, 0, 0
        mask = self.masks[uid]
        x, y, w, h = self.bboxes[uid]
        
        # A small pad is all we need to allow the 1px boundary to exist
        pad = 2
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(mask.shape[1], x+w+pad), min(mask.shape[0], y+h+pad)
        crop = mask[y1:y2, x1:x2]
        
        # Expand the mask by exactly 1 pixel
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(crop, kernel, iterations=1)
        
        # Keep only the strictly outer ring
        outer_border = (dilated > 0) & (crop == 0)
        
        # Create a transparent RGBA canvas and paint the border pure white
        rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
        rgba[outer_border] = [255, 255, 255, 255]
        
        return self._pack_for_gui(rgba), x1, y1

    def get_class_overlay_data(self, uid, opacity=0.6):
        patch_data = self._get_class_patch(uid)
        if patch_data is None: return None, 0, 0
        patch, x1, y1 = patch_data
        
        rgba = np.zeros((patch.shape[0], patch.shape[1], 4), dtype=np.uint8)
        for cid, color in self.class_colors.items():
            if cid != 0: rgba[patch == cid] = [*color[:3], int(255 * opacity)]
            
        return self._pack_for_gui(rgba), x1, y1

    def get_full_class_overlay_data(self, selected_ids=[], opacity=0.6):
        if self.raw_image is None: return None
        h, w = self.raw_image.shape[:2]
        full_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 1. Paint the biological class colors
        for uid in self.masks.keys():
            patch_data = self._get_class_patch(uid)
            if not patch_data: continue
            patch, x, y = patch_data
            ph, pw = patch.shape
            
            roi_rgba = full_rgba[y:y+ph, x:x+pw]
            bin_roi = self.masks[uid][y:y+ph, x:x+pw]
            
            for cid, color in self.class_colors.items():
                if cid != 0: roi_rgba[(patch == cid) & (bin_roi > 0)] = [*color[:3], int(255 * opacity)]

        # 2. Add 1px strictly-outside white border for selected instances
        kernel = np.ones((3, 3), dtype=np.uint8)
        
        for uid in selected_ids:
            if uid in self.masks and uid in self.bboxes:
                x, y, bw, bh = self.bboxes[uid]
                
                pad = 2
                x1, y1 = max(0, x-pad), max(0, y-pad)
                x2, y2 = min(w, x+bw+pad), min(h, y+bh+pad)
                
                crop = self.masks[uid][y1:y2, x1:x2]
                dilated = cv2.dilate(crop, kernel, iterations=1)
                outer_border = (dilated > 0) & (crop == 0)
                
                overlay_crop = full_rgba[y1:y2, x1:x2]
                overlay_crop[outer_border] = [255, 255, 255, 255]
                
        return self._pack_for_gui(full_rgba)

    # --- METADATA & DATA MANAGEMENT ---
    
    def get_id_at(self, x, y):
        for uid in sorted(self.masks.keys(), reverse=True):
            mask = self.masks[uid]
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0: return uid
        return 0

    def update_metadata_for_uid(self, uid):
        if uid not in self.masks:
            self.bboxes.pop(uid, None)
            self.areas.pop(uid, None)
            return

        coords = cv2.findNonZero(self.masks[uid])
        if coords is not None:
            self.bboxes[uid] = cv2.boundingRect(coords)
            self.areas[uid] = int(np.sum(self.masks[uid] > 0)) # Compute total pixel area
            if uid not in self.color_map:
                self.color_map[uid] = HIGH_CONTRAST_COLORS[uid % len(HIGH_CONTRAST_COLORS)]
        else:
            self.bboxes.pop(uid, None)
            self.areas.pop(uid, None)
            self.masks.pop(uid, None)
        self._notify_data_changed()

    def regenerate_metadata(self):
        self.bboxes = {}
        self.areas = {}
        for uid, mask in self.masks.items():
            coords = cv2.findNonZero(mask)
            if coords is not None:
                self.bboxes[uid] = cv2.boundingRect(coords)
                self.areas[uid] = int(np.sum(mask > 0))
                if uid not in self.color_map:
                    self.color_map[uid] = HIGH_CONTRAST_COLORS[uid % len(HIGH_CONTRAST_COLORS)]
        self._notify_data_changed()

    # --- TOOLS (Operating on Specific Masks) ---
    def prepare_new_uid(self):
        return self.max_id + 1

    def commit_new_instance(self, uid, first_points, brush_size):
        # Target the new UID. It will record {uid: None}
        self.save_state(uid) 
        
        h, w = self.raw_image.shape[:2]
        self.masks[uid] = np.zeros((h, w), dtype=np.uint8)
        self.max_id = max(self.max_id, uid)
        
        self.apply_stroke(uid, first_points, brush_size, is_erase=False, record_undo=False)
        
        self.color_map[uid] = HIGH_CONTRAST_COLORS[uid % len(HIGH_CONTRAST_COLORS)]
        self.update_metadata_for_uid(uid)
        
        self.dirty = True
        self._notify_data_changed() 

    def merge_instances(self, ids):
        """Creates a brand new UID for the merged result and deletes the originals."""
        if len(ids) < 2: return None
        
        self.save_state(ids)
        
        # 1. Create the new ID
        self.max_id += 1
        new_id = self.max_id
        
        h, w = self.raw_image.shape[:2]
        merged_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 2. Combine all old masks and delete them
        for uid in ids:
            if uid in self.masks:
                merged_mask = cv2.bitwise_or(merged_mask, self.masks[uid])
                
                self.masks.pop(uid, None)
                self.bboxes.pop(uid, None)
                self.color_map.pop(uid, None)
                if uid in self.class_patches:
                    del self.class_patches[uid]

        # 3. Assign the combined mask to the new ID
        self.masks[new_id] = merged_mask
        self.update_metadata_for_uid(new_id)
        
        self.dirty = True
        self._notify_data_changed()
        
        # Return the new ID so the GUI knows what to select
        return new_id

    def split_instance(self, target_id, points):
        if target_id not in self.masks: return False
        self.save_state(target_uids=target_id)
        
        full_mask = self.masks[target_id]
        x, y, w, h = self.bboxes[target_id]
        pad = 20
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(full_mask.shape[1], x+w+pad), min(full_mask.shape[0], y+h+pad)
        roi = full_mask[y1:y2, x1:x2].copy()
        
        pts = np.array([[int(p[0]) - x1, int(p[1]) - y1] for p in points], np.int32)
        cv2.polylines(roi, [pts], isClosed=False, color=0, thickness=LOGICAL_THICKNESS)
            
        sub_labels, num_features = label(roi, structure=np.ones((3,3)))
        
        if num_features > 1:
            # --- NEW: Filter valid components to eliminate spurious cut pixels ---
            valid_components = []
            for i in range(1, num_features + 1):
                component_mask = (sub_labels == i).astype(np.uint8)
                coords = cv2.findNonZero(component_mask)
                if coords is not None:
                    _, _, bw, bh = cv2.boundingRect(coords)
                    # Only keep components larger than a 2x2 area
                    if bw >= 3 and bh >= 3:
                        valid_components.append(component_mask)
            
            if len(valid_components) > 1:
                # True Split: We got 2+ valid pieces. Delete original and create new IDs.
                self.masks.pop(target_id, None)
                self.bboxes.pop(target_id, None)

                for comp in valid_components:
                    self.max_id += 1
                    new_mask = np.zeros_like(full_mask)
                    new_mask[y1:y2, x1:x2] = comp
                    self.masks[self.max_id] = new_mask
                    self.update_metadata_for_uid(self.max_id) 
                    
                self.dirty = True
                return True
                
            elif len(valid_components) == 1:
                # Edge Case: The split cut off a speck, but only 1 valid plant remained.
                # Treat this as an "erase/trim" operation. Keep the original ID!
                new_mask = np.zeros_like(full_mask)
                new_mask[y1:y2, x1:x2] = valid_components[0]
                self.masks[target_id] = new_mask
                if target_id in self.class_patches: del self.class_patches[target_id]
                self.update_metadata_for_uid(target_id)
                self.dirty = True
                return True
                
            else:
                # Everything was tiny? Abort.
                self.undo() 
                return False
        else:
            self.undo() 
            return False

    def delete_instances(self, ids):
        if not ids: return
        
        self.save_state(ids)
        
        for uid in ids:
            self.masks.pop(uid, None)
            self.bboxes.pop(uid, None)
            if uid in self.class_patches:
                del self.class_patches[uid]
                
        self.dirty = True
        self._notify_data_changed()

    # --- STROKE RENDERER ---
    
    def apply_stroke(self, uid, points, brush_size, is_erase=False, record_undo=True):
        if not points or uid not in self.masks: return
        
        # 1. Capture targeted state BEFORE modifying pixels
        if record_undo:
            self.save_state(target_uids=uid)
            
        if uid in self.class_patches: 
            del self.class_patches[uid]
            
        mask = self.masks[uid]
        color = 0 if is_erase else 1
        circle_radius = max(0, (brush_size - 1) // 2)
        
        if len(points) > 1:
            pts = np.array([[int(p[0]), int(p[1])] for p in points], np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], isClosed=False, color=color, thickness=brush_size)
        
        for p in [points[0], points[-1]]:
            x, y = int(p[0]), int(p[1])
            if brush_size == 1:
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]: mask[y, x] = color
            elif brush_size == 2:
                y1, y2 = max(0, y), min(mask.shape[0], y+2)
                x1, x2 = max(0, x), min(mask.shape[1], x+2)
                mask[y1:y2, x1:x2] = color
            else:
                cv2.circle(mask, (x, y), circle_radius, color, -1)

        self.update_metadata_for_uid(uid)
        self.dirty = True

    def apply_class_stroke(self, uid, points, brush_size, class_id):
        if not points or class_id == 0 or uid not in self.masks: return 
        
        patch_data = self._get_class_patch(uid) 
        if not patch_data: return
        patch, x_off, y_off = patch_data
        
        h, w = patch.shape
        stroke_mask = np.zeros((h, w), dtype=np.uint8)
        circle_radius = max(0, (brush_size - 1) // 2)
        
        # 1. Draw the line if there was movement
        if len(points) > 1:
            local_pts = np.array([[int(p[0]) - x_off, int(p[1]) - y_off] for p in points], np.int32).reshape((-1, 1, 2))
            cv2.polylines(stroke_mask, [local_pts], False, 1, thickness=brush_size)
        
        # 2. Always draw start and end points
        for p in [points[0], points[-1]]:
            lx, ly = int(p[0]) - x_off, int(p[1]) - y_off
            
            # Exact pixel manipulation for small brushes
            if brush_size == 1:
                if 0 <= ly < h and 0 <= lx < w:
                    stroke_mask[ly, lx] = 1
            elif brush_size == 2:
                y1, y2 = max(0, ly), min(h, ly+2)
                x1, x2 = max(0, lx), min(w, lx+2)
                stroke_mask[y1:y2, x1:x2] = 1
            else:
                cv2.circle(stroke_mask, (lx, ly), circle_radius, 1, -1)
            
        bin_crop = self.masks[uid][y_off:y_off+h, x_off:x_off+w]
        patch[(stroke_mask > 0) & (bin_crop > 0)] = class_id
        
        self.class_patches[uid] = (patch, x_off, y_off)
        self.dirty = True
        self._notify_data_changed()

    def _get_class_patch(self, uid):
        if uid in self.class_patches: return self.class_patches[uid]
        if uid not in self.masks: return None
        mask = self.masks[uid]
        
        ys, xs = np.where(mask)
        if len(xs) == 0: return None
        x_ex, y_ex = int(xs.min()), int(ys.min())
        w_ex, h_ex = int(xs.max() - x_ex + 1), int(ys.max() - y_ex + 1)
        
        pad = 10 
        x1, y1 = max(0, x_ex - pad), max(0, y_ex - pad)
        x2, y2 = min(mask.shape[1], x_ex + w_ex + pad), min(mask.shape[0], y_ex + h_ex + pad)
        
        bin_crop = mask[y1:y2, x1:x2]
        cls_crop = self.original_multiclass[y1:y2, x1:x2].copy() if self.original_multiclass is not None else np.zeros_like(bin_crop)
            
        result = np.zeros_like(bin_crop)
        result[bin_crop > 0] = cls_crop[bin_crop > 0]
        holes = (bin_crop > 0) & (result == 0)
        
        if np.any(holes):
            valid = result > 0
            if np.any(valid):
                _, indices = distance_transform_edt(~valid, return_indices=True)
                result[holes] = result[indices[0], indices[1]][holes]
            else:
                result[holes] = 1 
                
        final_patch = result[(y_ex - y1):(y_ex - y1) + h_ex, (x_ex - x1):(x_ex - x1) + w_ex]
        self.class_patches[uid] = (final_patch, x_ex, y_ex)
        return final_patch, x_ex, y_ex

    def _flatten_multiclass(self):
        if self.raw_image is None: return None
        final_map = np.zeros(self.raw_image.shape[:2], dtype=np.uint8)
        
        for uid in sorted(self.masks.keys()):
            patch_data = self._get_class_patch(uid)
            if not patch_data: continue
            patch, x, y = patch_data
            ph, pw = patch.shape
            
            binary_roi = self.masks[uid][y:y+ph, x:x+pw]
            final_map[y:y+ph, x:x+pw][binary_roi > 0] = patch[binary_roi > 0]
            
        return final_map

    # --- FULLY OPTIMIZED UNDO/STATE ---
    def save_state(self, target_uids):
        """Saves only the affected masks, plus the exact max_id at this moment."""
        if isinstance(target_uids, int):
            target_uids = [target_uids]
            
        backup = {}
        for uid in target_uids:
            if uid in self.masks:
                backup[uid] = self.masks[uid].copy()
            else:
                backup[uid] = None 
                
        # We now explicitly record the max_id to guarantee perfect cleanup on Undo
        self.history.append({
            "masks": backup,
            "max_id": self.max_id 
        })
        
        if len(self.history) > MAX_HISTORY: 
            self.history.pop(0)

    def undo(self):
        if not self.history: return False
        last_state = self.history.pop()
        
        # 1. Restore the exact state of the targeted plants
        for uid, old_mask in last_state["masks"].items():
            if old_mask is not None:
                self.masks[uid] = old_mask
                self.update_metadata_for_uid(uid)
            else:
                self.masks.pop(uid, None)
                self.bboxes.pop(uid, None)
                self.color_map.pop(uid, None)
            
            if uid in self.class_patches: 
                del self.class_patches[uid]
                
        # 2. Bulletproof Cleanup: Obliterate ANY newly created IDs
        old_max_id = last_state["max_id"]
        current_uids = list(self.masks.keys())
        
        for uid in current_uids:
            if uid > old_max_id:
                self.masks.pop(uid, None)
                self.bboxes.pop(uid, None)
                self.color_map.pop(uid, None)
                if uid in self.class_patches: 
                    del self.class_patches[uid]
                
        self.max_id = old_max_id
        
        self.dirty = True
        self._notify_data_changed()
        return True