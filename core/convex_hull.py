import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def calculate_optimal_canvas(df, buffer=50):
    """
    Scans the dataframe to determine the minimal canvas size and optimal seed placement.
    Rotates all hulls around (0,0) to find the absolute maximum geometric extents.
    """
    print("Scanning dataset to determine minimal Atlas size...")
    
    global_min_x, global_max_x = 0, 0
    global_min_y, global_max_y = 0, 0
    
    for _, row in df.iterrows():
        ini = row.get('ini_pos')
        ftip = row.get('ftip_pos')
        hulls = row.get('hull_pts')

        # Safety checks
        if not isinstance(hulls, list) or not isinstance(ini, list) or not isinstance(ftip, list):
            continue
        if len(hulls) == 0 or len(ini) < 2 or len(ftip) < 2:
            continue

        dx = ftip[0] - ini[0]
        dy = ftip[1] - ini[1]
        if dx == 0 and dy == 0:
            continue
            
        # Fixed Angle Calculation (from our previous fix)
        v_root = np.array([dx, dy], dtype=np.float64)
        norm = np.linalg.norm(v_root)
        if norm == 0: continue
        v_root = v_root / norm
        
        v_vertical = np.array([0, 1], dtype=np.float64)
        dot_product = np.clip(np.dot(v_vertical, v_root), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = -np.degrees(angle_rad) if v_root[0] > 0 else np.degrees(angle_rad)

        # Rotate around `ini`, but translate `ini` to exactly (0,0)
        rot_mat = cv2.getRotationMatrix2D((float(ini[0]), float(ini[1])), angle_deg, 1.0)
        rot_mat[0, 2] += (0 - ini[0])
        rot_mat[1, 2] += (0 - ini[1])

        # Warp the points mathematically
        pts_array = np.array(hulls, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.transform(pts_array, rot_mat).reshape(-1, 2)
        
        # Find local bounds of this specific root
        min_x, min_y = np.min(warped_pts, axis=0)
        max_x, max_y = np.max(warped_pts, axis=0)
        
        # Update global extremes
        global_min_x = min(global_min_x, min_x)
        global_max_x = max(global_max_x, max_x)
        global_min_y = min(global_min_y, min_y) # Usually slightly negative if seed grows upward a bit
        global_max_y = max(global_max_y, max_y)

    # Calculate final canvas dimensions based on the extremes
    # Seed X needs to be shifted right by the maximum leftward drift
    dest_ini_x = int(abs(global_min_x) + buffer)
    # Seed Y needs to be shifted down by the maximum upward drift
    dest_ini_y = int(abs(global_min_y) + buffer) 
    
    canvas_w = int(dest_ini_x + global_max_x + buffer)
    canvas_h = int(dest_ini_y + global_max_y + buffer)
    
    print(f"Optimal Geometry: Size[{canvas_h}, {canvas_w}], Seed Center[{dest_ini_y}, {dest_ini_x}]")
    return (canvas_w, canvas_h), (dest_ini_x, dest_ini_y)

def get_rotated_hulls(row, dest_ini):
    ini = row.get('ini_pos')
    ftip = row.get('ftip_pos')
    hulls = row.get('hull_pts')

    if not isinstance(hulls, list) or not isinstance(ini, list) or not isinstance(ftip, list):
        return None
        
    if len(hulls) == 0 or len(ini) < 2 or len(ftip) < 2:
        return None

    dx = ftip[0] - ini[0]
    dy = ftip[1] - ini[1]
    
    if dx == 0 and dy == 0:
        return None
        
    v_root = np.array([dx, dy], dtype=np.float64)
    norm = np.linalg.norm(v_root)
    if norm == 0:
        return None
    v_root = v_root / norm
    
    v_vertical = np.array([0, 1], dtype=np.float64)
    
    # Calculate angle using dot product (clipped to prevent floating point errors with arccos)
    dot_product = np.clip(np.dot(v_vertical, v_root), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    
    # Apply the exact sign logic from Code 2
    angle_deg = -np.degrees(angle_rad) if v_root[0] > 0 else np.degrees(angle_rad)

    rot_mat = cv2.getRotationMatrix2D((float(ini[0]), float(ini[1])), angle_deg, 1.0)
    rot_mat[0, 2] += (dest_ini[0] - ini[0])
    rot_mat[1, 2] += (dest_ini[1] - ini[1])

    pts_array = np.array(hulls, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.transform(pts_array, rot_mat)
    
    return np.int32(warped.reshape(-1, 2))

def draw_atlas_grid_on_figure(df, fig, canvas_dims, dest_ini):
    """Draws the NxM grid of Accumulated Convex Hull heatmaps."""
    canvas_w, canvas_h = canvas_dims  # Unpack the dynamic dimensions
    
    # Drop NAs to prevent weird "nan" headers
    timepoints = sorted(df['timepoint'].dropna().unique(), key=natural_sort_key)
    
    df_copy = df.copy()
    df_copy['group'] = df_copy['condition'].astype(str) + " | " + df_copy['genotype'].astype(str)
    groups = sorted(df_copy['group'].dropna().unique())

    axes = fig.subplots(len(timepoints), len(groups), squeeze=False)
    
    for i, tp in enumerate(timepoints):
        for j, grp in enumerate(groups):
            ax = axes[i, j]
            subset = df_copy[(df_copy['timepoint'] == tp) & (df_copy['group'] == grp)]
            
            heatmap = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            valid_plants = 0
            
            # Aggregate hulls for this cell in the grid
            for _, row in subset.iterrows():
                warped_hulls = get_rotated_hulls(row, dest_ini)
                if warped_hulls is not None:
                    temp_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
                    cv2.fillPoly(temp_mask, [warped_hulls], 1.0)
                    heatmap += temp_mask
                    valid_plants += 1
                    
            # Render the Heatmap
            if valid_plants > 0:
                heatmap = heatmap / valid_plants # Normalize intensity (0.0 to 1.0)
                im = ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1.0)
            else:
                # Provide a visual warning instead of a silent white box
                ax.imshow(np.zeros((canvas_h, canvas_w)), cmap='binary')
                ax.text(canvas_w//2, canvas_h//2, "Missing Topology\nData in JSON", 
                        color='red', ha='center', va='center', fontsize=12, fontweight='bold')

            # Formatting
            ax.axis('off')
            if i == 0: 
                ax.set_title(grp, fontweight='bold', fontsize=12, pad=10)
            if j == 0: 
                ax.text(-0.15, 0.5, tp, transform=ax.transAxes, rotation=90, 
                        va='center', ha='center', fontweight='bold', fontsize=12)

    fig.tight_layout()

def generate_qualitative_grid(df, out_dir):
    """Wrapper used by the Full Report generator to save the atlas to disk."""
    print("Generating Qualitative Atlas Grid...")
    
    # 1. Calculate dynamic bounds first
    canvas_dims, dest_ini = calculate_optimal_canvas(df)
    
    timepoints = df['timepoint'].dropna().unique()
    groups = (df['condition'].astype(str) + " | " + df['genotype'].astype(str)).unique()
    
    # Make the figure large enough for high-res saving
    fig = plt.figure(figsize=(5 * len(groups), 6 * len(timepoints)))
    
    # 2. Pass dynamic variables to the drawing function
    draw_atlas_grid_on_figure(df, fig, canvas_dims, dest_ini)
    
    grid_path = os.path.join(out_dir, "Qualitative_Atlas_Grid.png")
    fig.savefig(grid_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)