import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import json

# Import the fully updated static graph builder and original RSML creator
from .root_graph_builder import extract_skeleton, createGraph, graphInit
from .rsml import createTree
from .metrics import compute_metrics

def extract_plate_metrics(model, plants_meta, cm_per_px):
    results = {}
    img_filename = os.path.basename(model.image_path) if model.image_path else "unknown_plate.png"

    for p_meta in plants_meta:
        uid, genotype, plant_num = p_meta['uid'], p_meta['genotype'], p_meta['plant_num']
        
        patch_data = model._get_class_patch(uid)
        if not patch_data: continue
        patch, x_off, y_off = patch_data
        
        bin_crop = model.masks[uid][y_off:y_off+patch.shape[0], x_off:x_off+patch.shape[1]]
        
        # ==========================================
        # 1. BUILD SKELETON & GRAPH
        # ==========================================
        img_h, img_w = model.raw_image.shape[:2]
        root_bin = np.zeros((img_h, img_w), dtype=np.uint8)
        mc_skel = np.zeros((img_h, img_w), dtype=np.uint8)
        
        for cid in [1, 2]: 
            active = (patch == cid) & (bin_crop > 0)
            p_y, p_x = np.where(active)
            root_bin[p_y + y_off, p_x + x_off] = 1
            mc_skel[p_y + y_off, p_x + x_off] = cid
            
        full_skel, branches, endpoints, is_valid = extract_skeleton(root_bin, prune_iters=1)
        if not is_valid: continue
        
        skel_ys, skel_xs = np.where(full_skel > 0)
        raw_root_base = [skel_xs[np.argmin(skel_ys)], skel_ys[np.argmin(skel_ys)]]
        
        graph, actual_base, colored_skel = createGraph(full_skel.copy(), mc_skel, raw_root_base, endpoints, branches)
        
        try:
            graph = graphInit(graph, main_root_class=1)
        except Exception as e:
            print(f"Skipping UID {uid} - Graph Init failed: {e}")
            continue

        # ==========================================
        # 2. GENERATE RSML
        # ==========================================
        mock_conf = {
            'fileKey': f'UID_{uid}_{genotype}',
            'sequenceLabel': 'Static_Extraction',
            'Plant': f'Plant_{uid}_{genotype}',
        }
        
        try:
            rsml_tree, number_lateral_roots = createTree(mock_conf, 0, [img_filename], graph, full_skel, colored_skel)
        except Exception as e:
            print(f"RSML generation failed for UID {uid}: {e}")
            continue
            
        plant_xml = rsml_tree.find(".//plant")
        metadata_xml = rsml_tree.find(".//metadata")

        # Parse RSML for Angle Calculations
        lateral_pts_list = []
        main_root_elem = plant_xml.find("./root[@label='mainRoot']")
        if main_root_elem is not None:
            for child in main_root_elem.findall("./root"):
                if child.get('label', '').startswith('lat_o1'):
                    l_pts = [[float(pt.attrib['x']), float(pt.attrib['y'])] for pt in child.findall('./geometry/polyline/point')]
                    if len(l_pts) >= 2:
                        lateral_pts_list.append(l_pts)

        # ==========================================
        # 3. COMPUTE MANUSCRIPT METRICS
        # ==========================================
        metrics = compute_metrics(graph, bin_crop, cm_per_px, lateral_pts_list, number_lateral_roots)
        
        # Shift the localized convex hull back to global coordinates for the UI
        hull_pts_list = []
        if len(metrics["_hull_pts_local"]) > 0:
            global_hull = metrics["_hull_pts_local"]
            global_hull[:, 0, 0] += x_off
            global_hull[:, 0, 1] += y_off
            hull_pts_list = global_hull.reshape(-1, 2).tolist()
        
        # Cleanup the internal dict
        del metrics["_hull_pts_local"]
        
        # --- NEW: Extract Initial and Final Topologies ---
        ini_pos = None
        ftip_pos = None
        for n, d in graph.nodes(data=True):
            if d.get('type') == 'Ini': 
                ini_pos = list(n)
            elif d.get('type') == 'FTip': 
                ftip_pos = list(n)

        # ==========================================
        # 4. PACKAGE RESULTS
        # ==========================================
        colored_skel_crop = colored_skel[y_off:y_off+patch.shape[0], x_off:x_off+patch.shape[1]]
        rsml_main_colors = [d.get('color', 0) for u, v, d in graph.edges(data=True) if d.get('root_type') == 10]
        ui_edges = [(u, v, 1 if d.get('root_type') == 10 else 2) for u, v, d in graph.edges(data=True)]

        # Merge the computed metrics with the UI data elements
        results[uid] = {
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
            "graph_edges": ui_edges                    
        }

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


def _normalize_restored_plant(plant):
    """Convert JSON-serialized measurement fields back to in-memory types."""
    sp = plant.get("semantic_patch")
    if isinstance(sp, list):
        plant["semantic_patch"] = np.array(sp, dtype=np.uint8)

    csc = plant.get("colored_skel_crop")
    if isinstance(csc, list):
        plant["colored_skel_crop"] = np.array(csc, dtype=np.uint8)

    co = plant.get("crop_offset")
    if isinstance(co, (list, tuple)) and len(co) >= 2:
        plant["crop_offset"] = (int(co[0]), int(co[1]))

    return plant


def load_measurements_from_metrics_json(metrics_path, current_uids, plants_metadata=None):
    """
    Load a prior metrics export into an in-memory measurements_cache dict.
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
            by_uid[int(saved_uid)] = _normalize_restored_plant(dict(plant))
            continue
        unmatched.append(plant)

    if unmatched:
        meta_by_key = {
            _plant_match_key(meta): uid
            for uid, meta in plants_metadata.items()
        }
        for plant in unmatched:
            key = _plant_match_key(plant)
            new_uid = None
            if key in meta_by_key:
                new_uid = int(meta_by_key[key])
            else:
                # Legacy exports: plant_num often matched the spatial UID index.
                try:
                    pn_uid = int(str(plant.get("plant_num", "")).strip())
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
            remapped = _normalize_restored_plant(dict(plant))
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
    # 1. Initialize the Master XML Tree
    master_rsml = ET.Element('rsml')
    
    # 2. Steal the rich metadata header from the first successfully processed plant
    first_uid = list(measurements_dict.keys())[0] if measurements_dict else None
    
    if first_uid and "metadata_xml" in measurements_dict[first_uid]:
        master_rsml.append(measurements_dict[first_uid]["metadata_xml"])
    else:
        metadata = ET.SubElement(master_rsml, 'metadata')
        ET.SubElement(metadata, 'version').text = '1.0'
        
    # 3. Create the Master Scene
    scene = ET.SubElement(master_rsml, 'scene')

    export_data = plate_meta.copy()
    export_data["plants"] = []

    for uid, data in measurements_dict.items():
        # Strip heavy memory objects before JSON dump
        json_safe_data = {k: v for k, v in data.items() if k not in [
            "rsml_xml", "metadata_xml", "main_pts", "lateral_pts_list",
        ]}
        export_data["plants"].append(json_safe_data)
        
        # Plug the individual plant into the Master Scene
        if "rsml_xml" in data and data["rsml_xml"] is not None:
            scene.append(data["rsml_xml"]) 

    # 4. Save JSON (Using the Custom Encoder)
    json_path = os.path.join(out_dir, f"{base_name}_Metrics.json")
    with open(json_path, 'w') as f: 
        json.dump(export_data, f, indent=4, cls=NumpyEncoder)
    
    # 5. Save RSML
    rsml_path = os.path.join(out_dir, f"{base_name}_Topology.rsml")
    tree = ET.ElementTree(master_rsml)
    
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="\t", level=0)
        
    tree.write(rsml_path, encoding='utf-8', xml_declaration=True)