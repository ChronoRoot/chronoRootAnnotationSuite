import os
import json
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from .root_graph_builder import extract_skeleton, createGraph

# --- LEGACY ANGLE MATH ---
def tipAngle(points):
    start, end = points[0], points[-1]
    hypotenuse = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    if hypotenuse == 0: return 0.0
    vertical_distance = end[1] - start[1]
    return (np.arccos(vertical_distance / hypotenuse) * 180 / np.pi)

def emergenceAngle(points, distance_cm=1, pixel_size=0.04):
    distance_pixels = int(distance_cm / pixel_size)
    start = points[0]
    end = points[min(distance_pixels, len(points)) - 1]
    hypotenuse = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    if hypotenuse == 0: return 0.0
    vertical_distance = end[1] - start[1]
    return (np.arccos(vertical_distance / hypotenuse) * 180 / np.pi)

def calculate_length(points, cm_per_px):
    points_array = np.array(points)
    if len(points_array) < 2: return 0.0
    diffs = np.diff(points_array, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)) * cm_per_px)

# --- LEGACY RSML TRAVERSAL ---
def vecinos(ske, seed): 
    x, y = int(seed[0]), int(seed[1])
    h, w = ske.shape
    neighbors = []
    y_min, y_max = max(0, y-1), min(h, y+2)
    x_min, x_max = max(0, x-1), min(w, x+2)
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if i == y and j == x: continue
            if ske[i, j] != 0: neighbors.append([j, i])
    return neighbors

def add_point(polyline, point):
    ET.SubElement(polyline, 'point', {'x': str(point[0]), 'y': str(point[1])})

def get_next_node_rsml(ske, start, previous, polyline):
    current, prev = start, previous
    ske[current[1], current[0]] = 0
    add_point(polyline, current)
    pts_added = [current]
    
    while True:
        all_nbs = vecinos(ske, current)
        valid = [n for n in all_nbs if not np.array_equal(n, prev)]
        if len(valid) != 1: return ske, current, pts_added
        
        nxt = valid[0]
        add_point(polyline, nxt)
        pts_added.append(nxt)
        ske[nxt[1], nxt[0]] = 0
        prev, current = current, nxt

def traverse_rsml_legacy(ske2, seed, parent_xml, mainRootColors):
    raiz = ET.SubElement(parent_xml, 'root', {'id': 'mainRoot', 'label': 'mainRoot'})
    geo = ET.SubElement(raiz, 'geometry')
    polyline = ET.SubElement(geo, 'polyline')

    main_pts = [seed]
    add_point(polyline, seed)
    all_neighbors = vecinos(ske2, seed)
    ske2[seed[1], seed[0]] = 0
    
    if not all_neighbors: return 0, main_pts, []

    main_candidates = [n for n in all_neighbors if ske2[n[1], n[0]] in mainRootColors]
    if main_candidates:
        main_start = main_candidates[0]
        lateral_starts = [n for n in all_neighbors if n != main_start]
    else:
        main_start = all_neighbors[0]
        lateral_starts = all_neighbors[1:]

    lateral_queue = [{'start': ls, 'parent': seed, 'order': 1, 'elem': raiz} for ls in lateral_starts]

    if main_start:
        current, prev = main_start, seed
        while True:
            ske2, stop_node, pts_added = get_next_node_rsml(ske2, current, prev, polyline)
            main_pts.extend(pts_added)
            nbs = vecinos(ske2, stop_node)
            if not nbs:
                ske2[stop_node[1], stop_node[0]] = 0
                break
            
            next_main = None
            for n in nbs:
                if ske2[n[1], n[0]] == 0: continue
                if ske2[n[1], n[0]] in mainRootColors: next_main = n
                else: lateral_queue.append({'start': n, 'parent': stop_node, 'order': 1, 'elem': raiz})
            
            ske2[stop_node[1], stop_node[0]] = 0
            if next_main:
                prev, current = stop_node, next_main
            else: break

    lateral_pts_list = []
    counters = {}
    
    while lateral_queue:
        task = lateral_queue.pop(0)
        start, parent, order, p_elem = task['start'], task['parent'], task['order'], task['elem']
        if ske2[start[1], start[0]] == 0: continue
        
        idx = counters.get(order, 0)
        counters[order] = idx + 1
        lat_id = f"lat_o{order}_{idx}"
        
        lr = ET.SubElement(p_elem, 'root', {'id': lat_id, 'label': lat_id})
        geo = ET.SubElement(lr, 'geometry')
        poly = ET.SubElement(geo, 'polyline')
        add_point(poly, parent)
        lat_pts = [parent]
        
        ske2, stop_node, pts_added = get_next_node_rsml(ske2, start, parent, poly)
        lat_pts.extend(pts_added)
        lateral_pts_list.append(lat_pts)
        
        nbs = vecinos(ske2, stop_node)
        for n in nbs:
            if ske2[n[1], n[0]] != 0:
                lateral_queue.append({'start': n, 'parent': stop_node, 'order': order + 1, 'elem': lr})
        ske2[stop_node[1], stop_node[0]] = 0

    return counters.get(1, 0), main_pts, lateral_pts_list

# --- MAIN EXTRACTION ---
def extract_plate_metrics(model, plants_meta, cm_per_px):
    results = {}

    for p_meta in plants_meta:
        uid, genotype, plant_num = p_meta['uid'], p_meta['genotype'], p_meta['plant_num']
        
        patch_data = model._get_class_patch(uid)
        if not patch_data: continue
        patch, x_off, y_off = patch_data
        bin_crop = model.masks[uid][y_off:y_off+patch.shape[0], x_off:x_off+patch.shape[1]]
        
        # 1. SPATIAL METRICS
        contours, _ = cv2.findContours(bin_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        biggest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(biggest_contour)
        
        area_hull_cm2 = float(cv2.contourArea(hull) * (cm_per_px ** 2))
        x, y, w, h = cv2.boundingRect(hull)
        width_cm, height_cm = float(w * cm_per_px), float(h * cm_per_px)
        
        # 2. SKELETONIZATION & GRAPH
        root_bin_canvas = np.zeros(model.raw_image.shape[:2], dtype=np.uint8)
        for cid in [1, 2]: 
            active = (patch == cid) & (bin_crop > 0)
            p_y, p_x = np.where(active)
            root_bin_canvas[p_y + y_off, p_x + x_off] = 1
            
        full_skel, branches, endpoints, is_valid = extract_skeleton(root_bin_canvas, prune_iters=0)
        if not is_valid: continue

        mc_skel_bin = np.zeros_like(root_bin_canvas)
        for cid in [1, 2]: 
            active = (patch == cid) & (bin_crop > 0)
            p_y, p_x = np.where(active)
            mc_skel_bin[p_y + y_off, p_x + x_off] = cid
        mc_skel_bin[full_skel == 0] = 0

        skel_ys, skel_xs = np.where(full_skel > 0)
        root_base = [skel_xs[np.argmin(skel_ys)], skel_ys[np.argmin(skel_ys)]]
        
        graph, actual_base, colored_skel = createGraph(full_skel, mc_skel_bin, root_base, endpoints, branches)
        if not graph: continue
        
        # 3. RSML & ANGLES TRAVERSAL
        main_root_colors = [d.get('color', 0) for u, v, d in graph.edges(data=True) if d.get('root_type') == 1]
        
        plant_xml = ET.Element('plant', {'id': f'Plant_{uid}_{genotype}_{plant_num}'})
        
        lr_count, main_pts, lateral_pts_list = traverse_rsml_legacy(
            colored_skel.copy(), actual_base, plant_xml, main_root_colors
        )
        
        mr_len_cm = calculate_length(main_pts, cm_per_px)
        lr_len_cm = sum(calculate_length(pts, cm_per_px) for pts in lateral_pts_list)
        
        tip_angles = [tipAngle(pts) for pts in lateral_pts_list if len(pts) > 2]
        emergence_angles = [emergenceAngle(pts, 1.0, cm_per_px) for pts in lateral_pts_list if len(pts) > 2]

        results[uid] = {
            "uid": int(uid),
            "genotype": str(genotype),
            "plant_num": str(plant_num),
            "mr_length_cm": mr_len_cm,
            "lr_length_cm": lr_len_cm,
            "lr_count": lr_count,
            "hull_area_cm2": area_hull_cm2,
            "width_cm": width_cm,
            "height_cm": height_cm,
            "tip_angle_deg": float(np.mean(tip_angles)) if tip_angles else 0.0,
            "emergence_angle_deg": float(np.mean(emergence_angles)) if emergence_angles else 0.0,
            "rsml_xml": plant_xml, # Pre-built XML for export
            "main_pts": main_pts,  # Coordinates for the UI painter
            "lateral_pts_list": lateral_pts_list # Coordinates & Angles for UI
        }

    return results

def export_rsml_and_json(out_dir, base_name, plate_meta, measurements_dict):
    rsml_tree = ET.Element('rsml')
    metadata = ET.SubElement(rsml_tree, 'metadata')
    ET.SubElement(metadata, 'version').text = '1.0'
    scene = ET.SubElement(rsml_tree, 'scene')

    export_data = plate_meta.copy()
    export_data["plants"] = []

    for uid, data in measurements_dict.items():
        # Clean dictionary for JSON
        json_safe_data = {k: v for k, v in data.items() if k not in ["rsml_xml", "main_pts", "lateral_pts_list"]}
        export_data["plants"].append(json_safe_data)
        scene.append(data["rsml_xml"]) # Attach the pre-built XML tree

    json_path = os.path.join(out_dir, f"{base_name}_Metrics.json")
    with open(json_path, 'w') as f: json.dump(export_data, f, indent=4)
    
    rsml_path = os.path.join(out_dir, f"{base_name}_Topology.rsml")
    tree = ET.ElementTree(rsml_tree)
    tree.write(rsml_path, encoding='utf-8', xml_declaration=True)