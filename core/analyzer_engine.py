import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import json

from .root_graph_builder import extract_skeleton, createGraph, graphInit
from .rsml import createTree
from .metrics import compute_metrics

METRICS_SCHEMA_VERSION = 2

METRICS_SCALAR_KEYS = (
    "uid",
    "genotype",
    "plant_num",
    "mr_length_mm",
    "lr_length_mm",
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
    return {k: data[k] for k in METRICS_SCALAR_KEYS if k in data}


def scalar_metrics_dict(full_results):
    """Replace a full per-plant analysis dict with scalar-only records."""
    return {uid: plant_scalar_record(data) for uid, data in full_results.items()}


def analyze_single_plant(model, uid, genotype, plant_num, cm_per_px):
    """
    Run skeleton/graph/metrics pipeline for one plant.
    Returns a full in-memory dict (scalars + transient viz fields) or None.
    """
    patch_data = model._get_class_patch(uid)
    if not patch_data:
        return None
    patch, x_off, y_off = patch_data

    bin_crop = model.masks[uid][y_off:y_off + patch.shape[0], x_off:x_off + patch.shape[1]]

    img_h, img_w = model.raw_image.shape[:2]
    root_bin = np.zeros((img_h, img_w), dtype=np.uint8)
    mc_skel = np.zeros((img_h, img_w), dtype=np.uint8)

    for cid in [1, 2]:
        active = (patch == cid) & (bin_crop > 0)
        p_y, p_x = np.where(active)
        root_bin[p_y + y_off, p_x + x_off] = 1
        mc_skel[p_y + y_off, p_x + x_off] = cid

    full_skel, branches, endpoints, is_valid = extract_skeleton(
        root_bin, *model.get_graph_params()
    )
    if not is_valid:
        return None

    skel_ys, skel_xs = np.where(full_skel > 0)
    raw_root_base = [skel_xs[np.argmin(skel_ys)], skel_ys[np.argmin(skel_ys)]]

    graph, actual_base, colored_skel = createGraph(
        full_skel.copy(), mc_skel, raw_root_base, endpoints, branches
    )

    try:
        graph = graphInit(graph, main_root_class=1)
    except Exception as e:
        print(f"Skipping UID {uid} - Graph Init failed: {e}")
        return None

    img_filename = os.path.basename(model.image_path) if model.image_path else "unknown_plate.png"
    mock_conf = {
        "fileKey": f"UID_{uid}_{genotype}",
        "sequenceLabel": "Static_Extraction",
        "Plant": f"Plant_{uid}_{genotype}",
    }

    try:
        rsml_tree, number_lateral_roots = createTree(
            mock_conf, 0, [img_filename], graph, full_skel, colored_skel
        )
    except Exception as e:
        print(f"RSML generation failed for UID {uid}: {e}")
        return None

    plant_xml = rsml_tree.find(".//plant")
    metadata_xml = rsml_tree.find(".//metadata")

    lateral_pts_list = []
    main_root_elem = plant_xml.find("./root[@label='mainRoot']")
    if main_root_elem is not None:
        for child in main_root_elem.findall("./root"):
            if child.get("label", "").startswith("lat_o1"):
                l_pts = [
                    [float(pt.attrib["x"]), float(pt.attrib["y"])]
                    for pt in child.findall("./geometry/polyline/point")
                ]
                if len(l_pts) >= 2:
                    lateral_pts_list.append(l_pts)

    metrics = compute_metrics(graph, bin_crop, cm_per_px, lateral_pts_list, number_lateral_roots)

    hull_pts_list = []
    if len(metrics["_hull_pts_local"]) > 0:
        global_hull = metrics["_hull_pts_local"]
        global_hull[:, 0, 0] += x_off
        global_hull[:, 0, 1] += y_off
        hull_pts_list = global_hull.reshape(-1, 2).tolist()

    del metrics["_hull_pts_local"]

    ini_pos = None
    ftip_pos = None
    for n, d in graph.nodes(data=True):
        if d.get("type") == "Ini":
            ini_pos = list(n)
        elif d.get("type") == "FTip":
            ftip_pos = list(n)

    colored_skel_crop = colored_skel[
        y_off:y_off + patch.shape[0], x_off:x_off + patch.shape[1]
    ]
    rsml_main_colors = [
        d.get("color", 0) for u, v, d in graph.edges(data=True) if d.get("root_type") == 10
    ]
    ui_edges = [
        (u, v, 1 if d.get("root_type") == 10 else 2) for u, v, d in graph.edges(data=True)
    ]

    return {
        "uid": int(uid),
        "genotype": str(genotype),
        "plant_num": str(plant_num),
        "ini_pos": ini_pos,
        "ftip_pos": ftip_pos,
        **metrics,
        "rsml_xml": plant_xml,
        "metadata_xml": metadata_xml,
        "lateral_pts_list": lateral_pts_list,
        "hull_pts": hull_pts_list,
        "colored_skel_crop": colored_skel_crop,
        "crop_offset": (x_off, y_off),
        "semantic_patch": patch,
        "main_root_colors": rsml_main_colors,
        "graph_edges": ui_edges,
    }


def extract_plate_metrics(model, plants_meta, cm_per_px):
    results = {}
    for p_meta in plants_meta:
        uid = p_meta["uid"]
        full = analyze_single_plant(
            model, uid, p_meta["genotype"], p_meta["plant_num"], cm_per_px
        )
        if full:
            results[uid] = full
    return results


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

    missing_uids = uid_set - set(by_uid.keys())
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

    first_uid = list(measurements_dict.keys())[0] if measurements_dict else None

    if first_uid and "metadata_xml" in measurements_dict[first_uid]:
        master_rsml.append(measurements_dict[first_uid]["metadata_xml"])
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

    json_path = os.path.join(out_dir, f"{base_name}_Metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, separators=(",", ":"), cls=NumpyEncoder)

    rsml_path = os.path.join(out_dir, f"{base_name}_Topology.rsml")
    tree = ET.ElementTree(master_rsml)

    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)

    tree.write(rsml_path, encoding="utf-8", xml_declaration=True)
