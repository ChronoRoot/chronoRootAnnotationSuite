import numpy as np
import xml.etree.ElementTree as ET
import os
import json

from .metrics import compute_metrics
from .root_topology import (
    Topology,
    automatic_topology,
    build_graph,
    header_for,
    import_plant,
    measurement_geometry,
    restore_saved,
    plant_element,
    read_rsml_plants,
    set_rsml_pixel_scale,
)

METRICS_SCHEMA_VERSION = 2

METRICS_SCALAR_KEYS = (
    "uid",
    "genotype",
    "plant_num",
    "germinated",
    "mr_length_mm",
    "lr_length_mm",
    "lr_mean_mm",
    "lr_std_mm",
    "tr_length_mm",
    "lr_count",
    "lr_density_cm",
    "mr_over_tr_ratio",
    "hull_area_mm2",
    "hull_width_mm",
    "hull_height_mm",
    "root_density_mm_mm2",
    "aspect_ratio",
    "tip_angle_deg",
    "emergence_angle_deg",
    "hull_pts",
    "ini_pos",
    "ftip_pos",
)


def plant_scalar_record(data):
    """Serializable metrics-only plant record for session cache and JSON export."""
    record = {k: data[k] for k in METRICS_SCALAR_KEYS if k in data}
    record.setdefault("germinated", 1)
    return record


def zero_plant_record(uid, genotype, plant_num):
    """Record for a plant that never germinated: identifiers kept, every trait at 0."""
    record = dict.fromkeys(METRICS_SCALAR_KEYS, 0)
    record.update({
        "uid": uid,
        "genotype": genotype,
        "plant_num": plant_num,
        "germinated": 0,
        # Empty geometry keeps the plant out of the convex hull atlas.
        "hull_pts": [],
        "ini_pos": [],
        "ftip_pos": [],
    })
    return record


def scalar_metrics_dict(full_results):
    """Replace a full per-plant analysis dict with scalar-only records."""
    return {uid: plant_scalar_record(data) for uid, data in full_results.items()}


def prepare_plant_analysis(model, uid):
    """Copy the pixels one analysis needs so a worker never reads the live model."""
    if model.raw_image is None or uid not in model.masks:
        return None
    patch_data = model._get_class_patch(uid)
    if not patch_data:
        return None
    patch, x_off, y_off = patch_data
    patch = np.array(patch, copy=True)
    y2 = y_off + patch.shape[0]
    x2 = x_off + patch.shape[1]
    image_name = os.path.basename(model.image_path) if model.image_path else "unknown_plate.png"
    return {
        "patch": patch,
        "x_off": int(x_off),
        "y_off": int(y_off),
        "bin_crop": np.array(model.masks[uid][y_off:y2, x_off:x2], copy=True),
        "img_shape": model.raw_image.shape[:2],
        "image_name": image_name,
    }


def analyze_single_plant(model, uid, genotype, plant_num, cm_per_px, prune_iters=5):
    """Run skeleton/graph/metrics for one plant. Returns scalars plus viz fields, or None."""
    return analyze_prepared_plant(
        prepare_plant_analysis(model, uid), uid, genotype, plant_num, cm_per_px, prune_iters,
    )


def analyze_prepared_plant(
    prepared, uid, genotype, plant_num, cm_per_px, prune_iters=5,
    topology_record=None, decided=False, plant_xml=None,
):
    """Measure one plant from its edge-chain topology.

    An in-memory override wins. Otherwise a matching exported plant is mapped
    onto the current graph. If that map fails, the automatic tree is used.
    """
    if not prepared:
        return None
    built = build_graph(
        prepared["patch"], prepared["x_off"], prepared["y_off"],
        prepared["bin_crop"], prepared["img_shape"], prune_iters,
    )
    if built is None:
        return None
    graph, colored_skel, full_skel = built
    topology, warning, imported = _choose_topology(
        graph, colored_skel, full_skel, prepared["image_name"], uid, genotype,
        prune_iters, topology_record, decided, plant_xml,
    )
    if topology is None:
        return None

    x_off = prepared["x_off"]
    y_off = prepared["y_off"]
    patch = prepared["patch"]
    lateral_pts_list, mr_len_px, lr_len_px, main_points, individual_px = measurement_geometry(topology)
    metrics = compute_metrics(
        graph, prepared["bin_crop"], cm_per_px, lateral_pts_list, len(lateral_pts_list),
        length_px=(mr_len_px, lr_len_px), individual_px=individual_px,
    )
    hull_pts_list = []
    if len(metrics["_hull_pts_local"]) > 0:
        global_hull = metrics["_hull_pts_local"]
        global_hull[:, 0, 0] += x_off
        global_hull[:, 0, 1] += y_off
        hull_pts_list = global_hull.reshape(-1, 2).tolist()
    del metrics["_hull_pts_local"]

    colored_skel_crop = colored_skel[
        y_off:y_off + patch.shape[0], x_off:x_off + patch.shape[1]
    ]
    rsml_main_colors = [
        data.get("color", 0) for _u, _v, data in graph.edges(data=True) if data.get("root_type") == 10
    ]
    return {
        "uid": int(uid),
        "genotype": str(genotype),
        "plant_num": str(plant_num),
        "ini_pos": list(main_points[0]) if main_points else None,
        "ftip_pos": list(main_points[-1]) if main_points else None,
        **metrics,
        "rsml_xml": plant_element(topology, uid, genotype),
        "metadata_xml": header_for(prepared["image_name"], uid, genotype),
        "lateral_pts_list": lateral_pts_list,
        "hull_pts": hull_pts_list,
        "colored_skel_crop": colored_skel_crop,
        "crop_offset": (x_off, y_off),
        "semantic_patch": patch,
        "main_root_colors": rsml_main_colors,
        "graph_edges": [
            (u, v, 1 if data.get("root_type") == 10 else 2)
            for u, v, data in graph.edges(data=True)
        ],
        "topology_warning": warning,
        "imported_record": topology.to_record() if imported else None,
    }


def _choose_topology(
    graph, colored, full_skel, image_name, uid, genotype, prune,
    topology_record, decided, plant_xml,
):
    stale = None
    if topology_record:
        restored = restore_saved(
            graph, colored, full_skel, image_name, uid, genotype, prune, topology_record, plant_xml,
        )
        if restored is not None:
            return restored, None, False
        stale = f"Plant {uid}: the applied topology no longer matches this skeleton, so the automatic tracing was used."
    if not decided and plant_xml is not None and stale is None:
        imported = import_plant(graph, colored, plant_xml, prune)
        if imported is not None:
            return imported, None, True
        return automatic_topology(graph, colored, full_skel, image_name, uid, genotype, prune), (
            f"Plant {uid}: the exported RSML does not match this skeleton, so the automatic tracing was used."
        ), False
    try:
        return automatic_topology(graph, colored, full_skel, image_name, uid, genotype, prune), stale, False
    except Exception as exc:
        print(f"RSML generation failed for UID {uid}: {exc}")
        return None, str(exc), False


def extract_plate_metrics(model, plants_meta, cm_per_px, prune_iters=5, topology_path=None):
    results = {}
    warnings = []
    imported = {}
    saved = read_rsml_plants(topology_path)
    overrides = getattr(model, "topology_overrides", {})
    decided = getattr(model, "topology_decided", set())
    for p_meta in plants_meta:
        uid = int(p_meta["uid"])
        # Ignored plants (contamination, damage, ...) are left out of the results entirely.
        if p_meta.get("ignore", False):
            continue
        if not p_meta.get("germinated", True):
            results[uid] = zero_plant_record(uid, p_meta["genotype"], p_meta["plant_num"])
            continue
        full = analyze_prepared_plant(
            prepare_plant_analysis(model, uid), uid, p_meta["genotype"],
            p_meta["plant_num"], cm_per_px, prune_iters,
            topology_record=overrides.get(uid),
            decided=uid in decided,
            plant_xml=None if uid in decided or uid in overrides else saved.get(uid),
        )
        if not full:
            continue
        if full.get("topology_warning"):
            warnings.append(full["topology_warning"])
        if full.get("imported_record") is not None:
            imported[uid] = full["imported_record"]
        results[uid] = full
    return results, warnings, imported


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def _plant_match_key(plant):
    return (
        str(plant.get("plant_num", "")).strip(),
        str(plant.get("genotype", "")).strip().lower(),
    )


def _plant_num_key(plant):
    return str(plant.get("plant_num", "")).strip()


def _filter_scalar_plant(plant):
    return plant_scalar_record(plant)


def load_measurements_from_metrics_json(metrics_path, current_uids, plants_metadata=None):
    """
    Load a prior metrics export into scalar session records.
    Remaps plant UIDs when saved UIDs no longer match the loaded annotation model.
    Returns (cache_dict, warnings_list).
    """
    plants_metadata = plants_metadata or {}
    warnings = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    plants = payload.get("plants") or []
    if not plants:
        return {}, ["Metrics file contains no plant records."]

    uid_set = set(int(u) for u in current_uids)
    by_uid = {}
    unmatched = []

    for plant in plants:
        saved_uid = plant.get("uid")
        if saved_uid is not None and int(saved_uid) in uid_set:
            by_uid[int(saved_uid)] = _filter_scalar_plant(plant)
            continue
        unmatched.append(plant)

    if unmatched:
        meta_by_key = {
            _plant_match_key(meta): uid for uid, meta in plants_metadata.items()
        }
        meta_by_plant_num = {}
        for uid, meta in plants_metadata.items():
            pn = str(meta.get("plant_num", "")).strip()
            if pn and pn not in meta_by_plant_num:
                meta_by_plant_num[pn] = uid

        for plant in unmatched:
            new_uid = None
            pn = _plant_num_key(plant)
            if pn and pn in meta_by_plant_num:
                new_uid = int(meta_by_plant_num[pn])
            if new_uid is None:
                key = _plant_match_key(plant)
                if key in meta_by_key:
                    new_uid = int(meta_by_key[key])
            if new_uid is None:
                try:
                    pn_uid = int(pn)
                except (TypeError, ValueError):
                    pn_uid = None
                if pn_uid is not None and pn_uid in uid_set:
                    new_uid = pn_uid

            if new_uid is None:
                warnings.append(
                    f"Could not match plant #{plant.get('plant_num')} ({plant.get('genotype')}) "
                    f"to current annotations."
                )
                continue
            if new_uid in by_uid:
                warnings.append(
                    f"Duplicate match for plant #{plant.get('plant_num')} ({plant.get('genotype')})."
                )
                continue
            remapped = _filter_scalar_plant(plant)
            remapped["uid"] = new_uid
            by_uid[new_uid] = remapped
            if plant.get("uid") != new_uid:
                warnings.append(
                    f"Remapped plant #{plant.get('plant_num')} ({plant.get('genotype')}) "
                    f"from UID {plant.get('uid')} to UID {new_uid}."
                )

    missing_uids = {
        uid for uid in uid_set - set(by_uid.keys())
        if not (plants_metadata.get(uid) or {}).get("ignore", False)
    }
    if missing_uids:
        warnings.append(
            f"{len(missing_uids)} annotated plant(s) have no restored measurements."
        )

    extra_uids = set(by_uid.keys()) - uid_set
    for uid in extra_uids:
        by_uid.pop(uid, None)
        warnings.append(f"Dropped restored measurement for stale UID {uid}.")

    return by_uid, warnings


def export_rsml_and_json(out_dir, base_name, plate_meta, measurements_dict):
    master_rsml = ET.Element("rsml")

    # Non-germinated plants carry no RSML, so take the header from the first
    # measured plant rather than simply the first record.
    header = next(
        (d["metadata_xml"] for d in measurements_dict.values() if d.get("metadata_xml") is not None),
        None,
    )
    if header is not None:
        master_rsml.append(header)
    else:
        metadata = ET.SubElement(master_rsml, "metadata")
        ET.SubElement(metadata, "version").text = "1.0"

    scene = ET.SubElement(master_rsml, "scene")

    export_data = plate_meta.copy()
    export_data["metrics_schema_version"] = METRICS_SCHEMA_VERSION
    export_data["plants"] = []

    for uid, data in measurements_dict.items():
        export_data["plants"].append(plant_scalar_record(data))

        if "rsml_xml" in data and data["rsml_xml"] is not None:
            scene.append(data["rsml_xml"])

    set_rsml_pixel_scale(master_rsml, plate_meta.get("scale_cm_px"))

    json_path = os.path.join(out_dir, f"{base_name}_Metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, separators=(",", ":"), cls=NumpyEncoder)

    rsml_path = os.path.join(out_dir, f"{base_name}_Topology.rsml")
    tree = ET.ElementTree(master_rsml)

    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)

    tree.write(rsml_path, encoding="utf-8", xml_declaration=True)
