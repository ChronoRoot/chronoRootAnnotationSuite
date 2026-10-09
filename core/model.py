import os
import json
import re
import copy
import time
import threading

import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import label, distance_transform_edt

# --- Constants ---
LINE_THICKNESS = 1
LOGICAL_THICKNESS = 2
MAX_HISTORY = 10
SCHEMA_VERSION = 1
# Half-size of the box drawn around a seed placeholder, which has no mask of its own.
SEED_MARKER_RADIUS = 8

ANNOTATION_STATUSES = ("without_annotation", "pending", "in_progress", "completed")
ANALYSIS_STATUSES = ("not_applicable", "not_analyzed", "analyzed")

HIGH_CONTRAST_COLORS = [
    (255, 0, 0),    (0, 255, 0),    (0, 0, 255),    (255, 255, 0),
    (255, 0, 255),  (0, 255, 255),  (255, 165, 0),  (128, 0, 128),
    (0, 255, 127),  (255, 20, 147), (139, 69, 19),  (75, 0, 130),
]


# --- Task status vocabulary ---
def normalize_annotation_status(raw_status):
    """Map legacy/raw status strings to canonical annotation status."""
    if not raw_status:
        return "without_annotation"
    s = str(raw_status).lower().strip()
    if s in ("completed", "complete", "done"):
        return "completed"
    if s in ("in_progress", "inprogress", "active", "progress"):
        return "in_progress"
    if s in ("without_annotation", "without annotation", "no_annotation"):
        return "without_annotation"
    if s in ("pending",):
        return "pending"
    if s in ("missing", "none", ""):
        return "pending"
    return "in_progress"


def annotation_status_display(status):
    mapping = {
        "without_annotation": "Without Annotation",
        "pending": "Pending",
        "in_progress": "In Progress",
        "completed": "Completed",
    }
    return mapping.get(normalize_annotation_status(status), "In Progress")


def analysis_status_display(status):
    mapping = {
        "not_applicable": "—",
        "not_analyzed": "Not Analyzed",
        "analyzed": "Analyzed",
    }
    return mapping.get(status, "—")


# --- Per-plant metadata ---
def normalize_plant_meta(meta, uid):
    """
    Canonical per-plant record used by the model, the metadata table and the saved JSON.
    A seed placeholder has `germinated` False and a `seed_pos`: it holds a plate position
    so plant numbering stays intact, but it has no mask and no root system to measure.
    `ignore` marks a plant that did germinate but cannot be analyzed (e.g. contamination).
    """
    meta = meta or {}
    seed_pos = meta.get("seed_pos")
    return {
        "genotype": meta.get("genotype", ""),
        "plant_num": str(meta.get("plant_num", "")).strip() or str(uid),
        "germinated": bool(meta.get("germinated", True)),
        "ignore": bool(meta.get("ignore", False)),
        "seed_pos": [int(seed_pos[0]), int(seed_pos[1])] if seed_pos else None,
    }


# --- Analysis export paths (filesystem only; not stored in annotation JSON) ---
def canonical_metrics_name(base_name):
    return f"{base_name}_Metrics.json"


def canonical_topology_name(base_name):
    return f"{base_name}_Topology.rsml"


def metrics_path(directory, base_name):
    if not directory or not base_name:
        return None
    path = os.path.join(directory, canonical_metrics_name(base_name))
    return path if os.path.exists(path) else None


def topology_path(directory, base_name):
    if not directory or not base_name:
        return None
    path = os.path.join(directory, canonical_topology_name(base_name))
    return path if os.path.exists(path) else None


# --- Directory browse peek ---
PEEK_BYTES = 4096
_PEEK_CACHE_VERSION = 3
_file_peek_cache = {}
_peek_tls = threading.local()


def _peek_bucket():
    stats = getattr(_peek_tls, "stats", None)
    if stats is None:
        stats = {"calls": 0, "hits": 0, "misses": 0, "full_scans": 0, "ms": 0.0}
        _peek_tls.stats = stats
    return stats


def _agent_dbg(hypothesis_id, location, message, data):
    # #region agent log
    try:
        with open(
            "/home/IPS2/rgaggio/Documents/MultiInstance/chronoRootAnnotationSuite/.cursor/debug-46ecae.log",
            "a",
            encoding="utf-8",
        ) as _f:
            _f.write(json.dumps({
                "sessionId": "46ecae",
                "runId": "post-fix",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion


def _peek_cache_get(json_path):
    try:
        mtime = os.path.getmtime(json_path)
    except OSError:
        return None
    cached = _file_peek_cache.get(json_path)
    if cached and cached[0] == mtime and cached[1] == _PEEK_CACHE_VERSION:
        return cached[2]
    return None


def _peek_cache_set(json_path, summary):
    try:
        mtime = os.path.getmtime(json_path)
        _file_peek_cache[json_path] = (mtime, _PEEK_CACHE_VERSION, summary)
    except OSError:
        pass


def _detect_annotation_from_compact(compact):
    if '"annotation_status":"completed"' in compact or '"status":"completed"' in compact:
        return "completed"
    if '"annotation_status":"in_progress"' in compact:
        return "in_progress"
    if '"annotation_status":"missing"' in compact:
        return "in_progress"
    if '"status":"in_progress"' in compact or '"status":"pending"' in compact:
        return "in_progress"
    return "in_progress"


def _count_bytes_marker_in_file(json_path, marker):
    """Count occurrences of a byte marker without loading the full file into memory."""
    # #region agent log
    _peek_bucket()["full_scans"] += 1
    # #endregion
    if not marker:
        return 0
    overlap = max(len(marker) - 1, 0)
    tail = b""
    count = 0
    try:
        with open(json_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                data = tail + chunk
                count += data.count(marker)
                tail = data[-overlap:] if overlap else b""
    except OSError:
        return 0
    return count


def _parse_plant_count(json_path):
    """Fast plant count: summary field from header, else count mask markers in file."""
    try:
        with open(json_path, "rb") as f:
            head = f.read(PEEK_BYTES)
    except OSError:
        return 0

    compact_head = head.replace(b" ", b"").replace(b"\n", b"")
    match = re.search(rb'"plant_count"\s*:\s*(\d+)', compact_head)
    if match:
        return int(match.group(1))

    rle_count = _count_bytes_marker_in_file(json_path, b'"instance_rle"')
    if rle_count:
        return rle_count
    return _count_bytes_marker_in_file(json_path, b'"segmentation"')


def peek_task_summary(json_path, task_dir, base_name, out_dir=None, probe_analysis=False,
                       dir_names=None, metrics_names=None):
    """Fast header-only read for directory browsing. Never parses full JSON or NIfTI."""
    # #region agent log
    _t0 = time.perf_counter()
    _peek_bucket()["calls"] += 1
    # #endregion
    if dir_names is None:
        nii_path = os.path.join(task_dir, base_name + ".nii.gz")
        has_nii = os.path.exists(nii_path)
        has_json = os.path.exists(json_path)
    else:
        has_nii = (base_name + ".nii.gz") in dir_names
        has_json = (base_name + ".json") in dir_names

    result = {
        "annotation_status": "without_annotation",
        "plant_count": 0,
        "analysis_status": "not_applicable",
    }
    if not has_json:
        result["annotation_status"] = "pending" if has_nii else "without_annotation"
        # #region agent log
        _peek_bucket()["ms"] += (time.perf_counter() - _t0) * 1000.0
        # #endregion
        return result

    cached = _peek_cache_get(json_path)
    if cached is not None:
        # #region agent log
        _peek_bucket()["hits"] += 1
        # #endregion
        result["annotation_status"] = cached["annotation_status"]
        result["plant_count"] = cached["plant_count"]
    else:
        # #region agent log
        _peek_bucket()["misses"] += 1
        # #endregion
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                chunk = f.read(PEEK_BYTES)
            compact = chunk.replace(" ", "").replace("\n", "")
            result["annotation_status"] = _detect_annotation_from_compact(compact)
            result["plant_count"] = _parse_plant_count(json_path)
            _peek_cache_set(json_path, {
                "annotation_status": result["annotation_status"],
                "plant_count": result["plant_count"],
            })
        except OSError:
            # #region agent log
            _peek_bucket()["ms"] += (time.perf_counter() - _t0) * 1000.0
            # #endregion
            return result

    if probe_analysis:
        metrics_dir = out_dir if out_dir is not None else task_dir
        ann = normalize_annotation_status(result["annotation_status"])
        if ann != "completed":
            result["analysis_status"] = "not_applicable"
        else:
            metric_name = canonical_metrics_name(base_name)
            if metrics_names is not None:
                has_metrics = metric_name in metrics_names
            elif dir_names is not None:
                has_metrics = metric_name in dir_names
            else:
                has_metrics = bool(metrics_path(metrics_dir, base_name))
            result["analysis_status"] = "analyzed" if has_metrics else "not_analyzed"
    # #region agent log
    _peek_bucket()["ms"] += (time.perf_counter() - _t0) * 1000.0
    # #endregion
    return result


# --- RLE codec ---
def encode_rle(mask):
    """Vectorized RLE encoding using NumPy."""
    pixels = mask.ravel()
    if len(pixels) == 0: 
        return {"size": mask.shape, "counts": []}
    
    # Identify boolean array where transitions occur
    changes = np.concatenate(([True], pixels[1:] != pixels[:-1], [True]))
    
    # Get the integer indices of those transitions
    run_indices = np.where(changes)[0]
    
    # Extract the values and calculate the lengths (counts) of each run
    vals = pixels[run_indices[:-1]]
    counts = np.diff(run_indices)
    
    # Interleave values and counts into a single 1D array
    rle = np.empty(vals.size * 2, dtype=int)
    rle[0::2] = vals
    rle[1::2] = counts
    
    return {"size": mask.shape, "counts": rle.tolist()}

def decode_rle(rle_data):
    """Vectorized RLE decoding using NumPy."""
    h, w = rle_data["size"]
    counts_list = rle_data["counts"]
    
    if not counts_list: 
        return np.zeros((h, w), dtype=np.uint8)
    
    # Convert the Python list to a NumPy array for fast C-level slicing
    rle_arr = np.array(counts_list, dtype=np.uint32)
    
    # Slice out the alternating values and repetitions
    vals = rle_arr[0::2].astype(np.uint8)
    reps = rle_arr[1::2]
    
    # np.repeat reconstructs the 9-million pixel array in milliseconds
    pixels = np.repeat(vals, reps)
    
    return pixels.reshape((h, w))

# --- PlantImageModel ---
class PlantImageModel:
    """In-memory plant masks, annotation I/O, and editing tools. Analysis exports are separate files."""

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
        self.plants_meta = {}
        self.plate_meta = {}
        self.class_colors = {
            0: (0, 0, 0, 0),        
            1: (255, 0, 0, 255),    
            2: (0, 255, 0, 255),    
            3: (0, 0, 255, 255),    
            4: (255, 255, 0, 255),  
            5: (0, 255, 255, 255),  
            6: (255, 0, 255, 255)   
        }

        self.selected_uids = set()
        self.active_uid = None

        self._data_callbacks = []
        self._selection_callbacks = []
        self._history_callbacks = []
        self.callbacks_muted = False

    def register_data_callback(self, callback):
        self._data_callbacks.append(callback)

    def register_selection_callback(self, callback):
        self._selection_callbacks.append(callback)

    def register_history_callback(self, callback):
        self._history_callbacks.append(callback)

    def _notify_data_changed(self):
        if self.callbacks_muted:
            return
        for callback in self._data_callbacks:
            callback()

    def _notify_selection_changed(self):
        if self.callbacks_muted:
            return
        for callback in self._selection_callbacks:
            callback()

    def _notify_history_changed(self):
        if self.callbacks_muted:
            return
        for callback in self._history_callbacks:
            callback()

    def clear_history(self):
        if self.history:
            self.history.clear()
            self._notify_history_changed()

    def set_selection(self, uids):
        self.selected_uids = set(uids)
        self.active_uid = list(self.selected_uids)[0] if len(self.selected_uids) == 1 else None
        self._notify_selection_changed()

    # --- Plant registry ---
    def plant_uids(self):
        """Every plant on the plate: painted masks plus seed-only placeholders."""
        seeds = {
            uid for uid, meta in self.plants_meta.items()
            if meta.get("seed_pos") and uid not in self.masks
        }
        return sorted(set(self.masks) | seeds)

    def is_seed_placeholder(self, uid):
        """True for a plant that only holds a plate position (no mask to measure)."""
        return uid not in self.masks and bool((self.plants_meta.get(uid) or {}).get("seed_pos"))

    def _plant_bbox(self, uid):
        """Bbox from the plant mask, or a fixed marker box for a seed placeholder."""
        mask = self.masks.get(uid)
        if mask is not None:
            _x, _y, bw, bh = cv2.boundingRect(mask)
            if bw == 0 or bh == 0:
                return None
            return (_x, _y, bw, bh)

        pos = (self.plants_meta.get(uid) or {}).get("seed_pos")
        if not pos:
            return None
        x, y = int(pos[0]), int(pos[1])
        r = SEED_MARKER_RADIUS
        return (x - r, y - r, 2 * r, 2 * r)

    def add_seed_marker(self, x, y):
        """Register an undoable seed placeholder at (x, y)."""
        uid = self.prepare_new_uid()
        self.save_state(uid)
        self.max_id = max(self.max_id, uid)
        self.color_map[uid] = HIGH_CONTRAST_COLORS[uid % len(HIGH_CONTRAST_COLORS)]
        self.plants_meta[uid] = normalize_plant_meta(
            {"germinated": False, "seed_pos": [x, y]}, uid
        )
        self.update_metadata_for_uid(uid)
        self.dirty = True
        return uid

    # --- Task metadata ---
    def set_task_metadata(self, plants_meta=None, plate_meta=None):
        """Inject per-plant and plate metadata before save."""
        if plants_meta is not None:
            self.plants_meta = {
                int(uid): normalize_plant_meta(meta, uid)
                for uid, meta in plants_meta.items()
            }
        if plate_meta is not None:
            self.plate_meta = dict(plate_meta)

    def get_plants_metadata(self):
        return {
            uid: normalize_plant_meta(meta, uid)
            for uid, meta in self.plants_meta.items()
        }

    def sync_plant_numbers_to_uids(self):
        """Give every plant a default Plant # equal to its UID, keeping user edits."""
        for uid in self.plant_uids():
            entry = self.plants_meta.setdefault(uid, normalize_plant_meta({}, uid))
            if not str(entry.get("plant_num", "")).strip():
                entry["plant_num"] = str(uid)

    def get_plate_meta(self):
        return dict(self.plate_meta)

    def invalidate_peek_cache(self, json_path=None):
        """Clear browse peek cache after annotation save."""
        if json_path:
            _file_peek_cache.pop(json_path, None)
        else:
            _file_peek_cache.clear()

    # --- UID remapping ---
    def _remap_plants_meta(self, mapping):
        if not mapping or not self.plants_meta:
            return
        remapped = {}
        for old_uid, meta in self.plants_meta.items():
            new_uid = mapping.get(old_uid, old_uid)
            remapped[new_uid] = normalize_plant_meta(meta, new_uid)
        self.plants_meta = remapped

    def _remap_uid_collections(self, mapping):
        """Remap masks and related per-UID collections using old->new mapping."""
        if not mapping:
            return

        new_masks, new_bboxes, new_areas, new_color_map, new_class_patches = {}, {}, {}, {}, {}

        for old_uid, new_uid in mapping.items():
            # Seed placeholders have no mask but must keep their box and metadata.
            if old_uid in self.masks:
                new_masks[new_uid] = self.masks[old_uid]
            if old_uid in self.bboxes:
                new_bboxes[new_uid] = self.bboxes[old_uid]
            if old_uid in self.areas:
                new_areas[new_uid] = self.areas[old_uid]
            new_color_map[new_uid] = HIGH_CONTRAST_COLORS[new_uid % len(HIGH_CONTRAST_COLORS)]
            if old_uid in self.class_patches:
                new_class_patches[new_uid] = self.class_patches[old_uid]

        self.masks = new_masks
        self.bboxes = new_bboxes
        self.areas = new_areas
        self.color_map = new_color_map
        self.class_patches = new_class_patches
        self.max_id = max(mapping.values())
        self.clear_history()
        self._remap_plants_meta(mapping)

    # --- Directory scan ---
    def _reset_peek_timing(self):
        bucket = _peek_bucket()
        bucket["calls"] = 0
        bucket["hits"] = 0
        bucket["misses"] = 0
        bucket["full_scans"] = 0
        bucket["ms"] = 0.0

    def _peek_timing_snapshot(self, started, folder_path, extra):
        total_ms = (time.perf_counter() - started) * 1000.0
        bucket = _peek_bucket()
        data = {
            "folder": folder_path,
            "total_ms": round(total_ms, 1),
            "peek_ms": round(bucket["ms"], 1),
            "walk_other_ms": round(total_ms - bucket["ms"], 1),
            "peek_calls": bucket["calls"],
            "peek_hits": bucket["hits"],
            "peek_misses": bucket["misses"],
            "full_file_scans": bucket["full_scans"],
        }
        data.update(extra)
        return data

    def _metrics_name_set(self, output_dir, fixed_output):
        if not fixed_output or not output_dir:
            return None
        try:
            return set(os.listdir(output_dir))
        except OSError:
            return set()

    def _file_entry(self, root, base, image_path, metrics_dir, dir_names, metrics_names):
        summary = peek_task_summary(
            os.path.join(root, base + ".json"),
            root,
            base,
            out_dir=metrics_dir,
            probe_analysis=True,
            dir_names=dir_names,
            metrics_names=metrics_names,
        )
        ann = summary["annotation_status"]
        analysis = summary["analysis_status"]
        return {
            "type": "file",
            "name": base,
            "path": image_path,
            "annotation_status": ann,
            "status": annotation_status_display(ann),
            "plant_count": summary["plant_count"],
            "analysis_status": analysis,
            "analyzer_status": analysis_status_display(analysis),
        }

    def _empty_dir_record(self, name, full_path, stats_pending):
        return {
            "type": "dir",
            "name": name,
            "path": full_path,
            "total": 0,
            "without_annotation": 0,
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "analyzed": 0,
            "not_analyzed": 0,
            "stats_pending": stats_pending,
        }

    def _add_folder_status(self, bucket, entry):
        bucket["total"] += 1
        ann = entry["annotation_status"]
        if ann == "without_annotation":
            bucket["without_annotation"] += 1
        elif ann == "pending":
            bucket["pending"] += 1
        elif ann == "completed":
            bucket["completed"] += 1
            if entry["analysis_status"] == "analyzed":
                bucket["analyzed"] += 1
            elif entry["analysis_status"] == "not_analyzed":
                bucket["not_analyzed"] += 1
        else:
            bucket["in_progress"] += 1

    def scan_directory(self, folder_path, output_dir=None, fixed_output=False):
        """List one folder. Nested annotation totals are filled in later."""
        # #region agent log
        _scan_t0 = time.perf_counter()
        self._reset_peek_timing()
        # #endregion
        contents = []
        if not os.path.isdir(folder_path):
            return contents

        dir_names = []
        file_names = []
        try:
            with os.scandir(folder_path) as iterator:
                for entry in iterator:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dir_names.append(entry.name)
                        elif entry.is_file(follow_symlinks=False):
                            file_names.append(entry.name)
                    except OSError:
                        continue
        except OSError:
            return contents
        dir_names.sort()
        file_names.sort()
        name_set = set(file_names)

        effective_out = output_dir or folder_path
        file_metrics_dir = effective_out if fixed_output else folder_path
        metrics_names = self._metrics_name_set(effective_out, fixed_output)
        file_metrics_names = metrics_names if fixed_output else None

        for item_name in dir_names:
            contents.append(self._empty_dir_record(
                item_name, os.path.join(folder_path, item_name), stats_pending=True
            ))

        processed_bases = set()
        for item_name in file_names:
            if not item_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            base = os.path.splitext(item_name)[0]
            if base in processed_bases:
                continue
            processed_bases.add(base)
            contents.append(self._file_entry(
                folder_path,
                base,
                os.path.join(folder_path, item_name),
                file_metrics_dir,
                name_set,
                file_metrics_names,
            ))

        # #region agent log
        _agent_dbg("H1", "model.py:scan_directory", "folder scan", self._peek_timing_snapshot(
            _scan_t0, folder_path, {
                "mode": "shallow",
                "entries": len(contents),
                "dirs": len(dir_names),
                "files": len(processed_bases),
            },
        ))
        # #endregion
        return contents

    def index_dataset(self, folder_path, output_dir=None, fixed_output=False, cancel=None):
        """Subtree annotation totals for every directory under the dataset root.

        Returns None if cancelled. Keys are directory paths; values are the same
        count fields shown on a folder row, including images in nested folders.
        """
        # #region agent log
        _scan_t0 = time.perf_counter()
        self._reset_peek_timing()
        # #endregion
        if not os.path.isdir(folder_path):
            return {}

        effective_out = output_dir or folder_path
        metrics_names = self._metrics_name_set(effective_out, fixed_output)
        direct = {}
        child_dirs = {}

        def cancelled():
            return bool(cancel and cancel.get("stop"))

        def counts():
            return {
                "total": 0,
                "without_annotation": 0,
                "pending": 0,
                "in_progress": 0,
                "completed": 0,
                "analyzed": 0,
                "not_analyzed": 0,
            }

        for current, dirnames, files in os.walk(folder_path):
            if cancelled():
                # #region agent log
                _agent_dbg("H1", "model.py:index_dataset", "folder stats", self._peek_timing_snapshot(
                    _scan_t0, folder_path, {"mode": "dataset", "cancelled": True},
                ))
                # #endregion
                return None
            bucket = counts()
            direct[current] = bucket
            child_dirs[current] = [os.path.join(current, name) for name in dirnames]
            name_set = set(files)
            processed_bases = set()
            metrics_dir = effective_out if fixed_output else current
            child_metrics = metrics_names if fixed_output else None
            for filename in files:
                if cancelled():
                    # #region agent log
                    _agent_dbg("H1", "model.py:index_dataset", "folder stats", self._peek_timing_snapshot(
                        _scan_t0, folder_path, {"mode": "dataset", "cancelled": True},
                    ))
                    # #endregion
                    return None
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                base = os.path.splitext(filename)[0]
                if base in processed_bases:
                    continue
                processed_bases.add(base)
                entry = self._file_entry(
                    current, base, os.path.join(current, filename),
                    metrics_dir, name_set, child_metrics,
                )
                self._add_folder_status(bucket, entry)

        totals = {}

        def rollup(path):
            if path in totals:
                return totals[path]
            combined = counts()
            source = direct.get(path, combined)
            for key in combined:
                combined[key] = source.get(key, 0)
            for child in child_dirs.get(path, []):
                nested = rollup(child)
                for key in combined:
                    combined[key] += nested.get(key, 0)
            totals[path] = combined
            return combined

        rollup(folder_path)
        # #region agent log
        _agent_dbg("H1", "model.py:index_dataset", "folder stats", self._peek_timing_snapshot(
            _scan_t0, folder_path, {
                "mode": "dataset",
                "cancelled": False,
                "directories": len(totals),
            },
        ))
        # #endregion
        return totals

    # --- Load / save ---
    def load_task(self, task_path, base_name):
        # #region agent log
        _load_t0 = time.perf_counter()
        _phases = {}

        def _lap(name, started):
            _phases[name] = round((time.perf_counter() - started) * 1000.0, 1)
            return time.perf_counter()
        # #endregion
        if base_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(base_name)[0]
            
        # 1. ABSOLUTE MEMORY WIPE
        self.image_path = None
        self.raw_image = None       
        self.masks.clear()
        self.max_id = 0
        self.color_map.clear()  
        self.areas.clear()       
        self.bboxes.clear()            
        self.clear_history()
        self.dirty = False
        self.original_multiclass = None
        self.class_patches.clear()
        self.plants_meta.clear()
        self.plate_meta.clear()
        
        # Load new data
        self.status = "pending"
        self.image_path = None
        
        valid_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        
        for ext in valid_extensions:
            potential_path = os.path.join(task_path, f"{base_name}{ext}")
            if os.path.exists(potential_path):
                self.image_path = potential_path
                break
                
        if self.image_path is None:
            raise FileNotFoundError(f"Missing image for task '{base_name}' in '{task_path}'.")
                
        nii_path = os.path.join(task_path, f"{base_name}.nii.gz")
        json_path = os.path.join(task_path, f"{base_name}.json")
        
        # #region agent log
        _t = time.perf_counter()
        # #endregion
        img_bgr = cv2.imread(self.image_path)
        if img_bgr is None: raise FileNotFoundError(f"Missing image: {self.image_path}")
        self.raw_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = self.raw_image.shape[:2]
        # #region agent log
        _t = _lap("imread_ms", _t)
        # #endregion
        
        if os.path.exists(json_path):
            self._load_mask_from_nifti(nii_path, (h, w))
            # #region agent log
            _t = _lap("nifti_ms", _t)
            # #endregion
            self._load_from_json(json_path, (h, w))
            # #region agent log
            _t = _lap("json_ms", _t)
            # #endregion
        else:
            self._load_instances_from_nifti(nii_path, (h, w))
            # #region agent log
            _t = _lap("nifti_instances_ms", _t)
            _phases["json_ms"] = 0.0
            # #endregion

        # Filter out spurious instances (width < 3 or height < 3)
        uids_to_remove = []
        for uid, mask in self.masks.items():
            _x, _y, bw, bh = cv2.boundingRect(mask)
            # Discard empty masks and specks smaller than 3 pixels in either dimension.
            if bw < 3 or bh < 3:
                uids_to_remove.append(uid)
                    
        for uid in uids_to_remove:
            self.masks.pop(uid, None)
            self.class_patches.pop(uid, None)
        # ---------------------------------------------------------------------
        # #region agent log
        _t = _lap("noise_filter_ms", _t)
        _n_removed = len(uids_to_remove)
        # #endregion

        uids = self.plant_uids()
        self.max_id = max(uids) if uids else 0
        self.regenerate_metadata()
        # #region agent log
        _t = _lap("metadata_ms", _t)
        # #endregion
        self.sort_instances_spatially(row_tolerance=250)
        self.sync_plant_numbers_to_uids()
        # #region agent log
        _t = _lap("spatial_sort_ms", _t)
        _n_json_patch = 0
        _n_computed_patch = 0
        # #endregion
        
        # 3. Warm up class patches for faster GUI response
        for uid in self.masks.keys():
            json_crop = self.class_patches.pop(uid, None)
            if json_crop is not None:
                # #region agent log
                _n_json_patch += 1
                # #endregion
                self._sync_class_patch(uid, json_crop, None)
            else:
                # #region agent log
                _n_computed_patch += 1
                # #endregion
                self._get_class_patch(uid)
        # #region agent log
        _t = _lap("class_patch_ms", _t)
        _agent_dbg("H3", "model.py:load_task", "file load phases", {
            "base": base_name,
            "image": self.image_path,
            "image_bytes": os.path.getsize(self.image_path) if self.image_path and os.path.exists(self.image_path) else 0,
            "shape": [int(h), int(w)],
            "n_masks": len(self.masks),
            "n_removed": _n_removed,
            "n_json_patch": _n_json_patch,
            "n_computed_patch": _n_computed_patch,
            "total_ms": round((time.perf_counter() - _load_t0) * 1000.0, 1),
            **_phases,
        })
        # #endregion

    # --- Mask validation & spatial sort ---
    def clean_and_validate_masks(self):
        needs_metadata_update = False
        
        for uid, mask in list(self.masks.items()):
            if np.sum(mask) == 0: 
                continue
            
            # --- FASTER OPENCV CONNECTED COMPONENTS ---
            num_features, labeled_mask = cv2.connectedComponents(mask, connectivity=8)
            num_features -= 1 # OpenCV counts background (0) as a feature
            
            valid_count = 0
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0 # Ignore background
                
                for i in range(1, num_features + 1):
                    if sizes[i] < 5:
                        mask[labeled_mask == i] = 0
                        self.dirty = True
                        needs_metadata_update = True
                    else:
                        valid_count += 1
                        
            if valid_count > 1:
                return False, uid
                
        if needs_metadata_update:
            self.regenerate_metadata() # Run exact calculation only when cleaning finishes
                    
        return True, None
    
    def reindex_instances(self):
        """Remaps all UIDs to be strictly continuous. Returns a mapping of {old_uid: new_uid}."""
        sorted_uids = self.plant_uids()
        if not sorted_uids:
            return {}

        if sorted_uids == list(range(1, len(sorted_uids) + 1)):
            return {u: u for u in sorted_uids}

        mapping = {old_uid: new_uid for new_uid, old_uid in enumerate(sorted_uids, start=1)}
        self._remap_uid_collections(mapping)
        return mapping

    def sort_instances_spatially(self, row_tolerance=250):
        """
        Sorts instances Left-to-Right, grouped by horizontal rows.
        Seed placeholders take part through their marker box, so they keep their slot.
        Returns mapping {old_uid: new_uid} (empty if no plants or already ordered).
        """
        if not self.bboxes:
            return {}

        centroids = []
        for uid, bbox in self.bboxes.items():
            x, y, w, h = bbox
            cx, cy = x + (w / 2), y + (h / 2)
            centroids.append((uid, cx, cy))

        centroids.sort(key=lambda item: item[2])

        rows = []
        current_row = [centroids[0]]

        for item in centroids[1:]:
            _, _, cy = item
            last_cy = current_row[-1][2]

            if abs(cy - last_cy) <= row_tolerance:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]
        if current_row:
            rows.append(current_row)

        sorted_uids = []
        for row in rows:
            row.sort(key=lambda item: item[1])
            sorted_uids.extend([item[0] for item in row])

        if sorted_uids == list(range(1, len(sorted_uids) + 1)):
            return {u: u for u in sorted_uids}

        mapping = {old_uid: new_uid for new_uid, old_uid in enumerate(sorted_uids, start=1)}
        self._remap_uid_collections(mapping)
        return mapping
        
    # --- Load / save (write) ---
    def save_current_task(self, task_path, base_name, mark_finished=False):
        """Writes current mask data to disk. Completely insulates GUI from JSON/NIfTI."""
        if not self.plant_uids(): return False

        # 1. CLEANUP AND VALIDATE
        is_valid, bad_uid = self.clean_and_validate_masks()
        if not is_valid:
            raise ValueError(
                f"Plant {bad_uid} has disconnected pieces.\n"
                "Connect them in Annotation with Shape Paint, or split into separate plants."
            )

        # 2. Spatially order UIDs (left-to-right rows); preserve user plant numbers
        mapping = self.sort_instances_spatially(row_tolerance=250)
        self.sync_plant_numbers_to_uids()

        # 3. PROCEED WITH SAVING
        if mark_finished:
            self.status = "completed"
        else:
            self.status = "in_progress"
        
        data = self._build_annotation_document()
        json_path = os.path.join(task_path, f"{base_name}.json")
        tmp_path = json_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, json_path)
        self.invalidate_peek_cache(json_path)
            
        final_labels = self._flatten_multiclass()
        self.original_multiclass = final_labels
        
        if final_labels is not None:
            nii_path = os.path.join(task_path, f"{base_name}.nii.gz")
            new_nifti = nib.Nifti1Image(final_labels.astype(np.uint8).T, np.eye(4))
            nib.save(new_nifti, nii_path)
                
        self.dirty = False
        self.clear_history()
        
        # Return the mapping dictionary so the GUI knows how to update itself!
        return mapping

    # --- Annotation JSON helpers ---
    def _load_mask_from_nifti(self, nii_path, shape):
        try:
            data = np.squeeze(np.asanyarray(nib.load(nii_path).dataobj))
            data = data.T
            self.original_multiclass = data.astype(np.uint8)
        except Exception as e: print(f"NIfTI Error: {e}")
        
    def _load_instances_from_nifti(self, nii_path, shape):
        try:
            # --- FASTER: Bypass get_fdata() float64 casting ---
            data = np.squeeze(np.asanyarray(nib.load(nii_path).dataobj))
            data = data.T
            self.original_multiclass = data.astype(np.uint8)
            
            # --- FASTER: Swap SciPy for OpenCV ---
            bin_data = (data > 0).astype(np.uint8)
            num_features, labeled_map = cv2.connectedComponents(bin_data, connectivity=8)
            
            for uid in range(1, num_features):
                self.masks[int(uid)] = (labeled_map == uid).astype(np.uint8)
        except Exception as e: print(f"NIfTI Error: {e}")
    
    def _load_from_json(self, json_path, shape):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            summary = data.get("summary") or {}
            ann_raw = summary.get("annotation_status") or data.get("status", "in_progress")
            self.status = normalize_annotation_status(ann_raw)
            if self.status in ("without_annotation", "pending"):
                self.status = "in_progress"

            self.plate_meta = data.get("plate_meta") or {}
            self.plants_meta = {}

            for ann in data.get('annotations', []):
                uid = int(ann['id'])
                self.plants_meta[uid] = normalize_plant_meta(ann, uid)

                # A seed placeholder stores only its position, so there is nothing to decode.
                if "instance_rle" not in ann and not ann.get('segmentation'):
                    continue

                if "instance_rle" in ann:
                    mask = decode_rle(ann["instance_rle"])
                else:
                    mask = np.zeros(shape, dtype=np.uint8)
                    for seg in ann.get('segmentation', []):
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 1)

                self.masks[uid] = mask

                ys, xs = np.where(mask)
                if len(xs) > 0 and "semantic_rle" in ann:
                    self.class_patches[uid] = (
                        decode_rle(ann["semantic_rle"]), int(xs.min()), int(ys.min())
                    )
        except Exception as e:
            print(f"JSON Error: {e}")

    def _build_annotation_document(self):
        ann_status = normalize_annotation_status(self.status)
        if ann_status in ("without_annotation", "pending"):
            ann_status = "in_progress"

        output = {
            "schema_version": SCHEMA_VERSION,
            "summary": {
                "annotation_status": ann_status,
                "plant_count": len(self.masks),
            },
            "status": ann_status,
            "plate_meta": dict(self.plate_meta),
            "images": [{
                "id": 1,
                "file_name": os.path.basename(self.image_path),
                "width": self.raw_image.shape[1],
                "height": self.raw_image.shape[0],
            }],
            "categories": [{"id": 1, "name": "Plant"}],
            "annotations": [],
        }

        for uid in self.plant_uids():
            plant_meta = normalize_plant_meta(self.plants_meta.get(uid), uid)
            entry = {"id": int(uid), "image_id": 1, "category_id": 1, **plant_meta}

            mask = self.masks.get(uid)
            if mask is None:
                # Seed placeholder: plate position only, no pixels to encode.
                output["annotations"].append(entry)
                continue

            if np.sum(mask) == 0:
                continue

            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            patch, _, _ = self._get_class_patch(uid)
            output["annotations"].append({
                **entry,
                "instance_rle": encode_rle(mask),
                "semantic_rle": encode_rle(patch),
                "area": float(np.sum(mask)),
                "iscrowd": 0,
                "bbox": [
                    float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min()),
                ],
            })

        output["summary"]["plant_count"] = len(output["annotations"])
        return output

    # --- GUI render exports ---
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
        bin_crop = self.masks[uid][y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]]
        for cid, color in self.class_colors.items():
            if cid != 0: rgba[(patch == cid) & (bin_crop > 0)] = [*color[:3], int(255 * opacity)]
            
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

    # --- Instance metadata ---
    def get_id_at(self, x, y):
        for uid in sorted(self.masks.keys(), reverse=True):
            mask = self.masks[uid]
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0: return uid
        return 0

    def _refresh_bbox_for_uid(self, uid):
        """Recompute box, area and colour for one plant. Returns False if it has nothing left."""
        box = self._plant_bbox(uid)
        if box is None:
            self.bboxes.pop(uid, None)
            self.areas.pop(uid, None)
            return False

        mask = self.masks.get(uid)
        self.bboxes[uid] = box
        self.areas[uid] = int(cv2.countNonZero(mask)) if mask is not None else 0
        if uid not in self.color_map:
            self.color_map[uid] = HIGH_CONTRAST_COLORS[uid % len(HIGH_CONTRAST_COLORS)]
        return True

    def update_metadata_for_uid(self, uid):
        if not self._refresh_bbox_for_uid(uid):
            self.masks.pop(uid, None)
        self._notify_data_changed()

    def regenerate_metadata(self):
        self.bboxes = {}
        self.areas = {}
        for uid in self.plant_uids():
            self._refresh_bbox_for_uid(uid)
        self._notify_data_changed()

    # --- Editing tools ---
    def prepare_new_uid(self):
        return self.max_id + 1

    def commit_new_instance(self, uid, first_points, brush_size):
        # Target the new UID. It will record {uid: None}
        self.save_state(uid) 
        
        h, w = self.raw_image.shape[:2]
        self.masks[uid] = np.zeros((h, w), dtype=np.uint8)
        self.max_id = max(self.max_id, uid)
        self.color_map[uid] = HIGH_CONTRAST_COLORS[uid % len(HIGH_CONTRAST_COLORS)]

        self.apply_stroke(uid, first_points, brush_size, is_erase=False, record_undo=False)
        
        self.update_metadata_for_uid(uid)

        self.dirty = True
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
            valid_components = []
            for i in range(1, num_features + 1):
                component_mask = (sub_labels == i).astype(np.uint8)
                coords = cv2.findNonZero(component_mask)
                if coords is not None:
                    _, _, bw, bh = cv2.boundingRect(coords)
                    if bw >= 3 and bh >= 3:
                        valid_components.append(component_mask)
            
            if len(valid_components) > 1:
                self.masks.pop(target_id, None)
                self.bboxes.pop(target_id, None)

                for comp in valid_components:
                    self.max_id += 1
                    new_mask = np.zeros_like(full_mask)
                    new_mask[y1:y2, x1:x2] = comp
                    self.masks[self.max_id] = new_mask
                    self.update_metadata_for_uid(self.max_id) 
                    
                self.dirty = True
                self._notify_data_changed() 
                return True
                
            elif len(valid_components) == 1:
                new_mask = np.zeros_like(full_mask)
                new_mask[y1:y2, x1:x2] = valid_components[0]
                self.masks[target_id] = new_mask
                if target_id in self.class_patches: del self.class_patches[target_id]
                self.update_metadata_for_uid(target_id)
                self.dirty = True
                self._notify_data_changed() 
                return True
                
            else:
                self.undo() 
                return False
        else:
            self.undo() 
            return False

    def split_disconnected_components(self, target_id):
        """Scans a mask for disconnected parts and splits them automatically without a line cut."""
        if target_id not in self.masks: return False
        
        full_mask = self.masks[target_id]
        
        x, y, w, h = self.bboxes[target_id]
        pad = 2
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(full_mask.shape[1], x+w+pad), min(full_mask.shape[0], y+h+pad)
        roi = full_mask[y1:y2, x1:x2]
        
        # --- FASTER OPENCV CONNECTED COMPONENTS ---
        num_features, sub_labels = cv2.connectedComponents(roi, connectivity=8)
        num_features -= 1 # Ignore background
        
        valid_components = []
        if num_features > 1:
            for i in range(1, num_features + 1):
                component_mask = (sub_labels == i).astype(np.uint8)
                coords = cv2.findNonZero(component_mask)
                if coords is not None:
                    _, _, bw, bh = cv2.boundingRect(coords)
                    if bw >= 3 and bh >= 3:
                        valid_components.append(component_mask)
                        
        if len(valid_components) > 1:
            self.save_state(target_uids=target_id)
            self.masks.pop(target_id, None)
            self.bboxes.pop(target_id, None)
            if target_id in self.class_patches: del self.class_patches[target_id]
            
            for comp in valid_components:
                self.max_id += 1
                new_mask = np.zeros_like(full_mask)
                new_mask[y1:y2, x1:x2] = comp
                self.masks[self.max_id] = new_mask
                self.update_metadata_for_uid(self.max_id)
                
            self.dirty = True
            self._notify_data_changed()
            return True
            
        return False
    
    def delete_instances(self, ids):
        if not ids: return
        
        self.save_state(ids)
        
        for uid in ids:
            # Seed placeholders have no mask; their metadata snapshot restores them on undo.
            if uid not in self.masks:
                self.plants_meta.pop(uid, None)
            self.masks.pop(uid, None)
            self.bboxes.pop(uid, None)
            self.areas.pop(uid, None)
            if uid in self.class_patches:
                del self.class_patches[uid]
                
        self.dirty = True
        self._notify_data_changed()

    def apply_stroke(self, uid, points, brush_size, is_erase=False, record_undo=True):
        if not points or uid not in self.masks: return
        
        if record_undo:
            self.save_state(target_uids=uid)
            
        # Keep the live semantic edits as the source for the updated outline.
        previous_class_patch = self._get_class_patch(uid)
        mask = self.masks[uid]
        color = 0 if is_erase else 1
        circle_radius = max(0, (brush_size - 1) // 2)
        
        # 1. Draw the stroke
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

        # 2. FAST METADATA UPDATE
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        
        sx = max(0, min(xs) - brush_size)
        sy = max(0, min(ys) - brush_size)
        ex = min(mask.shape[1], max(xs) + brush_size)
        ey = min(mask.shape[0], max(ys) + brush_size)
        
        if uid in self.bboxes:
            ox, oy, ow, oh = self.bboxes[uid]
            if not is_erase:
                new_x = min(ox, sx)
                new_y = min(oy, sy)
                new_w = max(ox + ow, ex) - new_x
                new_h = max(oy + oh, ey) - new_y
                self.bboxes[uid] = (new_x, new_y, new_w, new_h)
        else:
            self.bboxes[uid] = (sx, sy, ex - sx, ey - sy)

        # 3. Preserve live class edits, clear erased pixels and fill new outline pixels.
        self._sync_class_patch(uid, previous_class_patch, None)

        self.dirty = True
        self._notify_data_changed()
        
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
        affected = (stroke_mask > 0) & (bin_crop > 0)
        if not np.any(affected) or not np.any(patch[affected] != class_id): return

        self.save_state(target_uids=uid)
        patch[affected] = class_id
        
        self.class_patches[uid] = (patch, x_off, y_off)
        self.dirty = True
        self._notify_data_changed()

    def _sync_class_patch(self, uid, source_patch, bounds):
        """Align live class labels to the tight outline and fill new holes."""
        box = self._plant_bbox(uid)
        if uid not in self.masks or box is None:
            self.class_patches.pop(uid, None)
            return None

        x, y, w, h = box
        bin_crop = self.masks[uid][y:y+h, x:x+w]
        if not np.any(bin_crop):
            self.class_patches.pop(uid, None)
            return None

        result = np.zeros_like(bin_crop, dtype=np.uint8)
        if source_patch is not None:
            source, source_x, source_y = source_patch
            overlap_x1 = max(x, source_x)
            overlap_y1 = max(y, source_y)
            overlap_x2 = min(x + w, source_x + source.shape[1])
            overlap_y2 = min(y + h, source_y + source.shape[0])

            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                result[
                    overlap_y1 - y:overlap_y2 - y,
                    overlap_x1 - x:overlap_x2 - x,
                ] = source[
                    overlap_y1 - source_y:overlap_y2 - source_y,
                    overlap_x1 - source_x:overlap_x2 - source_x,
                ]

        result[bin_crop == 0] = 0
        result[(result < 1) | (result > 6)] = 0
        holes = (bin_crop > 0) & (result == 0)
        if np.any(holes):
            valid = result > 0
            if np.any(valid):
                indices = distance_transform_edt(
                    ~valid, return_distances=False, return_indices=True
                )
                result[holes] = result[indices[0], indices[1]][holes]
            else:
                result[holes] = 3

        self.class_patches[uid] = (result, x, y)
        return result, x, y

    def _get_class_patch(self, uid):
        if uid in self.class_patches: return self.class_patches[uid]
        if uid not in self.masks: return None
        mask = self.masks[uid]

        ys, xs = np.where(mask)
        if len(xs) == 0: return None
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

        source_patch = None
        if self.original_multiclass is not None:
            source_patch = (self.original_multiclass[y:y+h, x:x+w], x, y)
        return self._sync_class_patch(uid, source_patch, (x, y, w, h))

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

    # --- Undo ---
    def save_state(self, target_uids):
        """Save the complete annotation state affected by one undoable action."""
        if isinstance(target_uids, int):
            target_uids = [target_uids]

        masks_backup = {}
        class_backup = {}
        metadata_backup = {}
        for uid in target_uids:
            masks_backup[uid] = self.masks[uid].copy() if uid in self.masks else None

            patch_data = self.class_patches.get(uid)
            if patch_data is None:
                class_backup[uid] = None
            else:
                patch, x, y = patch_data
                class_backup[uid] = (patch.copy(), x, y)

            metadata_backup[uid] = (
                copy.deepcopy(self.plants_meta[uid])
                if uid in self.plants_meta else None
            )

        self.history.append({
            "masks": masks_backup,
            "class_patches": class_backup,
            "plants_meta": metadata_backup,
            "max_id": self.max_id,
            "selected_uids": set(self.selected_uids),
            "active_uid": self.active_uid,
            "dirty": self.dirty,
        })

        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        self._notify_history_changed()

    def undo(self):
        if self.callbacks_muted or not self.history: return False
        last_state = self.history.pop()

        # 1. Restore the exact state of every UID affected by the action.
        for uid, old_mask in last_state["masks"].items():
            if old_mask is None:
                self.masks.pop(uid, None)
            else:
                self.masks[uid] = old_mask

            old_patch = last_state["class_patches"][uid]
            if old_patch is None:
                self.class_patches.pop(uid, None)
            else:
                patch, x, y = old_patch
                self.class_patches[uid] = (patch, x, y)

            old_meta = last_state["plants_meta"][uid]
            if old_meta is None:
                self.plants_meta.pop(uid, None)
            else:
                self.plants_meta[uid] = copy.deepcopy(old_meta)

        # 2. Remove every collection entry created after the snapshot.
        old_max_id = last_state["max_id"]
        current_uids = (
            set(self.masks) | set(self.bboxes) | set(self.areas) |
            set(self.color_map) | set(self.class_patches) | set(self.plants_meta)
        )
        for uid in current_uids:
            if uid <= old_max_id:
                continue
            self.masks.pop(uid, None)
            self.bboxes.pop(uid, None)
            self.areas.pop(uid, None)
            self.color_map.pop(uid, None)
            self.class_patches.pop(uid, None)
            self.plants_meta.pop(uid, None)

        # 3. Rebuild derived per-plant state from the restored source data.
        restored_uids = set(last_state["masks"])
        for uid in restored_uids:
            if uid in self.masks or self.is_seed_placeholder(uid):
                self._refresh_bbox_for_uid(uid)
            else:
                self.bboxes.pop(uid, None)
                self.areas.pop(uid, None)
                self.color_map.pop(uid, None)

        self.max_id = old_max_id
        valid_uids = set(self.plant_uids())
        self.selected_uids = {
            uid for uid in last_state["selected_uids"] if uid in valid_uids
        }
        old_active = last_state["active_uid"]
        if old_active in valid_uids:
            self.active_uid = old_active
        elif len(self.selected_uids) == 1:
            self.active_uid = next(iter(self.selected_uids))
        else:
            self.active_uid = None
        self.dirty = last_state["dirty"]

        self._notify_history_changed()
        self._notify_data_changed()
        self._notify_selection_changed()
        return True