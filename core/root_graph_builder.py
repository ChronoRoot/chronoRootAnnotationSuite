import numpy as np
import cv2
import networkx as nx
from skimage.morphology import skeletonize

# --- PRE-ALLOCATED KERNELS ---
T_KERNELS = [np.array([[-1, 1, -1], [1, 1, 1], [0, 0, 0]]), np.array([[-1, 1, 0], [1, 1, 0], [-1, 1, 0]]), np.array([[0, 0, 0], [1, 1, 1], [-1, 1, -1]]), np.array([[0, 1, -1], [0, 1, 1], [0, 1, -1]]), np.array([[1, -1, -1], [1, 1, -1], [-1, 1, -1]]), np.array([[-1, 1, -1], [1, 1, -1], [1, -1, -1]]), np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, 1]]), np.array([[-1, -1, -1], [-1, 1, 1], [1, 1, -1]]), np.array([[-1, 1, 1], [1, 1, -1], [-1, -1, -1]]), np.array([[1, 1, -1], [-1, 1, 1], [-1, -1, -1]]), np.array([[-1, -1, 1], [-1, 1, 1], [-1, 1, -1]]), np.array([[-1, 1, -1], [-1, 1, 1], [-1, -1, 1]]), np.array([[-1, 1, -1], [-1, 1, 1], [-1, -1, -1]]), np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, -1]]), np.array([[-1, 1, -1], [1, 1, -1], [-1, -1, -1]]), np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, -1]])]
EP_KERNELS = [np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]]), np.array([[-1, 1, -1], [-1, 1, -1], [-1, -1, -1]]), np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]]), np.array([[-1, -1, -1], [1, 1, -1], [-1, -1, -1]]), np.array([[-1, -1, -1], [-1, 1, 1], [-1, -1, -1]]), np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]]), np.array([[-1, -1, -1], [-1, 1, -1], [-1, 1, -1]]), np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]])]
X_KERNELS = [np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])]
T_BRANCH_KERNELS = [np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]]), np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]]), np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]]), np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]]), np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]]), np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]]), np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]]), np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])]
Y_KERNELS = [np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]]), np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]]), np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]]), np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]]), np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])]
_Y3 = np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]])
_Y4 = np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
Y_KERNELS.extend([np.rot90(_Y3), np.rot90(_Y4), np.rot90(np.rot90(_Y3))])

def get_roi_bounding_box(mask, padding=5):
    points = cv2.findNonZero(mask)
    if points is None: return None, None
    x, y, w, h = cv2.boundingRect(points)
    h_img, w_img = mask.shape
    x_start, y_start = max(0, x - padding), max(0, y - padding)
    x_end, y_end = min(w_img, x + w + padding), min(h_img, y + h + padding)
    return (slice(y_start, y_end), slice(x_start, x_end)), (x_start, y_start)

def extract_skeleton(binary_mask, prune_iters=3):
    """Generates skeleton and topological points directly from a numpy array."""
    roi_slice, offset = get_roi_bounding_box(binary_mask)
    if roi_slice is None:
        return np.zeros_like(binary_mask), np.array([]), np.array([]), False

    cropped_mask = binary_mask[roi_slice]
    skeleton_crop = np.array(skeletonize(cropped_mask > 0), dtype='uint8')

    # Apply customizable pruning pipeline
    skeleton_crop = trim(prune(skeleton_crop, prune_iters))
    
    branch_points, end_points = skeleton_nodes(skeleton_crop)
    
    if len(branch_points) > 0:
        branch_points[:, 0] += offset[0]
        branch_points[:, 1] += offset[1]
    if len(end_points) > 0:
        end_points[:, 0] += offset[0]
        end_points[:, 1] += offset[1]

    full_skeleton = np.zeros_like(binary_mask)
    full_skeleton[roi_slice] = skeleton_crop
    
    is_valid = len(end_points) >= 2
    return full_skeleton, branch_points, end_points, is_valid

def trim(ske):
    bp = np.zeros_like(ske)
    for t in T_KERNELS:
        bp = cv2.morphologyEx(ske, cv2.MORPH_HITMISS, t)
        ske = cv2.subtract(ske, bp)
    return ske

def prune(skel, num_it):
    if num_it == 0: return skel
    orig = skel.copy()
    for _ in range(num_it):
        current_skel = skel
        for kernel in EP_KERNELS:
            hit = cv2.morphologyEx(current_skel, cv2.MORPH_HITMISS, kernel)
            current_skel = cv2.subtract(current_skel, hit)
        skel = current_skel
        
    end = endPoints(skel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for _ in range(num_it):
        end = cv2.dilate(end, kernel)
        end = cv2.bitwise_and(end, orig)
    return cv2.bitwise_or(end, skel)

def endPoints(skel):
    ep = np.zeros_like(skel)
    for kernel in EP_KERNELS:
        ep = cv2.add(ep, cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel))
    return ep

def skeleton_nodes(ske):
    branch = branchedPoints(ske)
    end = endPoints(ske)
    bp = np.where(branch == 1)
    bnodes = [[bp[1][i], bp[0][i]] for i in range(len(bp[0]))]
    ep = np.where(end == 1)
    enodes = [[ep[1][i], ep[0][i]] for i in range(len(ep[0]))]
    return np.array(bnodes), np.array(enodes)

def branchedPoints(skel):
    bp = np.zeros(skel.shape, dtype=int)
    for x in X_KERNELS: bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, x)
    for y in Y_KERNELS: bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, y)
    for t in T_BRANCH_KERNELS: bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, t)
    return bp

# --- GRAPH BUILDER (Optimized & State-Safe) ---
def createGraph(skeleton_image, multiclass_skeleton, root_base_position, end_points, branch_points):
    skeleton_image = skeleton_image.astype(np.int32)
    graph = nx.Graph()
    edge_state = {'counter': 3}
    
    # 1. Find the topological root base
    actual_root_base, _, _ = find_nearest(root_base_position, np.array(end_points))
    if actual_root_base is None: 
        return graph, None, skeleton_image
    actual_root_base = tuple(actual_root_base)
    
    # Initialize the starting node
    graph.add_node(actual_root_base, pos=actual_root_base, type='base', age=0)
    skeleton_image[actual_root_base[1], actual_root_base[0]] = edge_state['counter']
    
    # 2. Kick off the recursive depth-first trace
    _trace_graph(graph, skeleton_image, multiclass_skeleton, actual_root_base, edge_state)
    
    return graph, actual_root_base, skeleton_image

def _trace_graph(graph, skeleton_image, multiclass_skeleton, current_node, edge_state):
    """Recursively walks the skeleton using dynamic 3x3 neighborhood checks."""
    # Find all UNTRAVERSED paths originating from this node (value == 1)
    neighbors = find_neighbors(skeleton_image, current_node, search_value=1)
    
    for start_pixel in neighbors:
        # Double check it wasn't already traversed by a previous branch in this loop
        if skeleton_image[start_pixel[1], start_pixel[0]] != 1:
            continue
            
        edge_color = edge_state['counter']
        edge_state['counter'] += 1
        
        bio_class = int(multiclass_skeleton[start_pixel[1], start_pixel[0]])
        acc_dist = np.linalg.norm(np.array(current_node) - np.array(start_pixel))
        
        # Mark as traversed
        skeleton_image[start_pixel[1], start_pixel[0]] = edge_color
        
        curr_pix = start_pixel
        next_node = None
        
        while True:
            # Look at the 3x3 grid for untraversed pixels
            next_neighbors = find_neighbors(skeleton_image, curr_pix, search_value=1)
            
            # Check if semantic class changes mid-edge
            color_changed = (multiclass_skeleton[curr_pix[1], curr_pix[0]] != bio_class)

            # If neighbors != 1, we hit a branch (>1), endpoint (0), or semantic split
            if len(next_neighbors) != 1 or color_changed:
                next_node = tuple(curr_pix)
                break
                
            # Move forward along the continuous edge
            next_pix = next_neighbors[0]
            skeleton_image[next_pix[1], next_pix[0]] = edge_color
            acc_dist += np.linalg.norm(np.array(curr_pix) - np.array(next_pix))
            curr_pix = next_pix

        # Create the new node and connect the edge
        if next_node not in graph.nodes:
            graph.add_node(next_node, pos=next_node, type='node', age=0)
            
        if not graph.has_edge(current_node, next_node):
            graph.add_edge(current_node, next_node, weight=acc_dist, color=edge_color, root_type=bio_class)

        # Recursively trace outward from this new node
        _trace_graph(graph, skeleton_image, multiclass_skeleton, next_node, edge_state)

def find_neighbors(skeleton_image, pixel, search_value=1):
    """Vectorized C-memory neighborhood search."""
    x, y = pixel[0], pixel[1]
    h, w = skeleton_image.shape
    
    # Define safe slice boundaries
    y_min, y_max = max(0, y-1), min(h, y+2)
    x_min, x_max = max(0, x-1), min(w, x+2)
    
    # Instantly extract the 3x3 neighborhood
    neighborhood = skeleton_image[y_min:y_max, x_min:x_max]
    
    # Vectorized search
    local_ys, local_xs = np.where(neighborhood == search_value)
    
    neighbors = []
    for ly, lx in zip(local_ys, local_xs):
        global_y = y_min + ly
        global_x = x_min + lx
        # Ignore the center pixel
        if global_x != x or global_y != y:
            neighbors.append([global_x, global_y])
            
    return neighbors

def find_nearest(target_pos, point_list):
    if len(point_list) == 0: return None, point_list, None
    dists = np.linalg.norm(target_pos - point_list, axis=1)
    idx = np.argmin(dists)
    return point_list[idx, :], np.delete(point_list, idx, axis=0), dists