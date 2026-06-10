import numpy as np
import cv2

def compute_metrics(graph, bin_crop, cm_per_px, lateral_pts_list, lr_count):
    # Unit conversion: cm to mm
    mm_per_px = cm_per_px * 10.0
    
    # =================================================================
    # 1. BASIC ARCHITECTURE
    # Using Euclidean edge weights directly from the graph
    # =================================================================
    mr_len_px = sum(d.get('weight', 0.0) for u, v, d in graph.edges(data=True) if d.get('root_type') == 10)
    lr_len_px = sum(d.get('weight', 0.0) for u, v, d in graph.edges(data=True) if d.get('root_type') != 10)
    
    mr_len_mm = float(mr_len_px * mm_per_px)
    lr_len_mm = float(lr_len_px * mm_per_px)
    tr_len_mm = mr_len_mm + lr_len_mm
    
    # Discrete LR Density: 10 * Number of LRs / MR Length (mm) -> equivalent to LRs/cm
    lr_density = (10.0 * lr_count / mr_len_mm) if mr_len_mm > 0 else 0.0
    
    # Main Over Total Root Ratio
    mr_over_tr = (mr_len_mm / tr_len_mm) if tr_len_mm > 0 else 0.0

    # =================================================================
    # 2. SPATIAL DISTRIBUTION (Convex Hull)
    # =================================================================
    area_hull_mm2 = 0.0
    w_mm = 0.0
    h_mm = 0.0
    root_density = 0.0
    aspect_ratio = 0.0
    hull_pts_local = []
    
    contours, _ = cv2.findContours(bin_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(biggest_contour)
        hull_pts_local = hull.copy()
        
        # Calculate Dimensions
        area_hull_mm2 = float(cv2.contourArea(hull) * (mm_per_px ** 2))
        x, y, w, h = cv2.boundingRect(hull)
        w_mm = float(w * mm_per_px)
        h_mm = float(h * mm_per_px)
        
        # Calculate Derived Spatial Metrics
        root_density = (tr_len_mm / area_hull_mm2) if area_hull_mm2 > 0 else 0.0
        aspect_ratio = (h_mm / w_mm) if w_mm > 0 else 0.0

    # =================================================================
    # 3. ANGLES (Preserved from original logic)
    # =================================================================
    tip_angles = []
    emergence_angles = []
    for pts in lateral_pts_list:
        t_ang = tipAngle(pts)
        e_ang = emergenceAngle(pts, distance_cm=0.2, pixel_size=cm_per_px)
        if not np.isnan(t_ang): tip_angles.append(t_ang)
        if not np.isnan(e_ang): emergence_angles.append(e_ang)

    mean_tip = float(np.mean(tip_angles)) if tip_angles else 0.0
    mean_emergence = float(np.mean(emergence_angles)) if emergence_angles else 0.0

    # Return the clean dictionary matching the manuscript parameters
    return {
        # Basic Architecture
        "mr_length_mm": mr_len_mm,
        "lr_length_mm": lr_len_mm,
        "tr_length_mm": tr_len_mm,
        "lr_count": int(lr_count),
        "lr_density_cm": lr_density,
        "mr_over_tr_ratio": mr_over_tr,
        # Spatial Distribution
        "hull_area_mm2": area_hull_mm2,
        "hull_width_mm": w_mm,
        "hull_height_mm": h_mm,
        "root_density_mm_mm2": root_density,
        "aspect_ratio": aspect_ratio,
        # Angles
        "tip_angle_deg": mean_tip,
        "emergence_angle_deg": mean_emergence,
        # Geometry for UI Rendering (Will be shifted to global later)
        "_hull_pts_local": hull_pts_local
    }

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