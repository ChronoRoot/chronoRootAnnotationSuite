import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import json

from .root_graph_builder import extract_skeleton, createGraph
from .rsml import completeRSML

def extract_plate_metrics(model, plants_meta, cm_per_px):
    results = {}

    for p_meta in plants_meta:
        uid, genotype, plant_num = p_meta['uid'], p_meta['genotype'], p_meta['plant_num']
        
        patch_data = model._get_class_patch(uid)
        if not patch_data: continue
        patch, x_off, y_off = patch_data
        
        bin_crop = model.masks[uid][y_off:y_off+patch.shape[0], x_off:x_off+patch.shape[1]]
        
        # 1. SPATIAL METRICS (Convex Hull)
        contours, _ = cv2.findContours(bin_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        biggest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(biggest_contour)
        
        area_hull_cm2 = float(cv2.contourArea(hull) * (cm_per_px ** 2))
        x, y, w, h = cv2.boundingRect(hull)
        width_cm, height_cm = float(w * cm_per_px), float(h * cm_per_px)
        
        global_hull = hull.copy()
        global_hull[:, 0, 0] += x_off
        global_hull[:, 0, 1] += y_off
        hull_pts_list = global_hull.reshape(-1, 2).tolist()
        
        # 1. Setup Data
        img_h, img_w = model.raw_image.shape[:2]
        
        # 2. Build Skeletons
        root_bin = np.zeros((img_h, img_w), dtype=np.uint8)
        mc_skel = np.zeros((img_h, img_w), dtype=np.uint8)
        
        for cid in [1, 2]: # Main/Lateral
            active = (patch == cid) & (bin_crop > 0)
            p_y, p_x = np.where(active)
            root_bin[p_y + y_off, p_x + x_off] = 1
            mc_skel[p_y + y_off, p_x + x_off] = cid
            
        full_skel, branches, endpoints, is_valid = extract_skeleton(root_bin, prune_iters=1)
        if not is_valid: continue
        
        # 3. Graph Construction
        skel_ys, skel_xs = np.where(full_skel > 0)
        root_base = [skel_xs[np.argmin(skel_ys)], skel_ys[np.argmin(skel_ys)]]
        
        # This graph contains the 'color' and 'root_type' attributes you set in UI
        graph, actual_base, colored_skel = createGraph(full_skel.copy(), mc_skel, root_base, endpoints, branches)
        
        # 4. REPLICATED LOGIC: Get Main Root Colors from Graph
        # In your UI, you set root_type=1 for main, 2 for lateral.
        main_root_colors = [d.get('color', 0) for u, v, d in graph.edges(data=True) if d.get('root_type') == 1]
        
        # 5. RSML Generation
        rsml_wrapper = ET.Element('rsml')
        scene_xml = ET.SubElement(rsml_wrapper, 'scene')
        plant_xml = ET.SubElement(scene_xml, 'plant', {'id': f'Plant_{uid}', 'label': p_meta['genotype']})
        
        # The 'colored_skel' acts as the 'skeleton_overlay'
        _, number_lateral_roots = completeRSML(colored_skel.copy(), actual_base, rsml_wrapper, main_root_colors)

        # 4. Strict 1st-Order Extraction
        # We must keep this strict extraction so that metrics calculation doesn't "zig-zag" across sub-branches
        main_pts = []
        lateral_pts_list = []
        
        main_root_elem = plant_xml.find("./root[@label='mainRoot']")
        if main_root_elem is not None:
            main_pts = [[int(float(pt.attrib['x'])), int(float(pt.attrib['y']))] 
                        for pt in main_root_elem.findall('./geometry/polyline/point')]
            
            for child in main_root_elem.findall("./root"):
                if child.get('label', '').startswith('lat_o1'):
                    l_pts = [[int(float(pt.attrib['x'])), int(float(pt.attrib['y']))] 
                             for pt in child.findall('./geometry/polyline/point')]
                    if len(l_pts) >= 2:
                        lateral_pts_list.append(l_pts)

        mr_len_cm = lenRoot(main_pts, cm_per_px)
        lr_len_cm = sum(lenRoot(pts, cm_per_px) for pts in lateral_pts_list)
        
        tip_angles = []
        emergence_angles = []
        for pts in lateral_pts_list:
            t_ang = tipAngle(pts)
            e_ang = emergenceAngle(pts, 0.2, cm_per_px)
            if not np.isnan(t_ang): tip_angles.append(t_ang)
            if not np.isnan(e_ang): emergence_angles.append(e_ang)

        colored_skel_crop = colored_skel[y_off:y_off+patch.shape[0], x_off:x_off+patch.shape[1]]
        
        # We still pull main_root_colors here solely so the UI dictionary doesn't break
        main_root_colors = [d.get('color', 0) for u, v, d in graph.edges(data=True) if d.get('root_type') == 1]

        results[uid] = {
            "uid": int(uid),
            "genotype": str(genotype),
            "plant_num": str(plant_num),
            "mr_length_cm": float(mr_len_cm),
            "lr_length_cm": float(lr_len_cm),
            "lr_count": int(number_lateral_roots),
            "hull_area_cm2": area_hull_cm2,
            "width_cm": width_cm,
            "height_cm": height_cm,
            "tip_angle_deg": float(np.mean(tip_angles)) if tip_angles else 0.0,
            "emergence_angle_deg": float(np.mean(emergence_angles)) if emergence_angles else 0.0,
            "rsml_xml": plant_xml,
            "main_pts": main_pts,  
            "lateral_pts_list": lateral_pts_list,
            "hull_pts": hull_pts_list,
            "colored_skel_crop": colored_skel_crop,
            "crop_offset": (x_off, y_off),
            "main_root_colors": main_root_colors,
            "graph_edges": [(u, v, d.get('root_type')) for u, v, d in graph.edges(data=True)]
        }

    return results

# Keep your crash-safe angle logic intact
def tipAngle(points):
    if len(points) < 2: return 0.0
    start, end = points[0], points[-1]
    hypotenuse = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    if hypotenuse == 0: return 0.0
    ratio = np.clip((end[1] - start[1]) / hypotenuse, -1.0, 1.0)
    return (np.arccos(ratio) * 180 / np.pi)

def emergenceAngle(points, distance_cm=0.2, pixel_size=0.04):
    num_points = len(points)
    if num_points < 2: return 0.0
    distance_pixels = max(1, int(distance_cm / pixel_size))
    start = points[0]
    end = points[min(distance_pixels, num_points - 1)]
    hypotenuse = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    if hypotenuse == 0: return 0.0
    ratio = np.clip((end[1] - start[1]) / hypotenuse, -1.0, 1.0)
    return (np.arccos(ratio) * 180 / np.pi)

def lenRoot(points, pixel_size=0.04):
    points_array = np.array(points)
    if len(points_array) < 2: return 0.0
    diffs = np.diff(points_array, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(segment_lengths) * pixel_size)

def export_rsml_and_json(out_dir, base_name, plate_meta, measurements_dict):
    rsml_tree = ET.Element('rsml')
    metadata = ET.SubElement(rsml_tree, 'metadata')
    ET.SubElement(metadata, 'version').text = '1.0'
    scene = ET.SubElement(rsml_tree, 'scene')

    export_data = plate_meta.copy()
    export_data["plants"] = []

    for uid, data in measurements_dict.items():
        # Strip all heavy UI drawing objects before JSON dump
        json_safe_data = {k: v for k, v in data.items() if k not in [
            "rsml_xml", "main_pts", "lateral_pts_list", "hull_pts", 
            "colored_skel_crop", "crop_offset", "main_root_colors", "graph_edges"
        ]}
        export_data["plants"].append(json_safe_data)
        scene.append(data["rsml_xml"]) 

    json_path = os.path.join(out_dir, f"{base_name}_Metrics.json")
    with open(json_path, 'w') as f: json.dump(export_data, f, indent=4)
    
    rsml_path = os.path.join(out_dir, f"{base_name}_Topology.rsml")
    tree = ET.ElementTree(rsml_tree)
    tree.write(rsml_path, encoding='utf-8', xml_declaration=True)