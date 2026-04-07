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

# --- GRAPH BUILDER (State-Safe) ---
def createGraph(skeleton_image, multiclass_skeleton, root_base_position, end_points, branch_points):
    graph = nx.Graph()
    edge_state = {'counter': 3}
    
    actual_root_base, remaining_endpoints, _ = find_nearest(root_base_position, np.array(end_points))
    if actual_root_base is None: return graph, None, skeleton_image
    
    actual_root_base = tuple(actual_root_base)
    graph.add_node(actual_root_base, pos=actual_root_base, type='base', age=0)
    skeleton_image[actual_root_base[1], actual_root_base[0]] = edge_state['counter']
    
    neighbor_pixels = find_neighbors(skeleton_image, actual_root_base)
    if not neighbor_pixels: return graph, actual_root_base, skeleton_image
    
    skeleton_image, first_node, first_edge_length, bio_class = get_next_node(
        skeleton_image, multiclass_skeleton, neighbor_pixels[0], actual_root_base, [], 0, actual_root_base, edge_state
    )
    first_node = tuple(first_node)
    
    graph.add_node(first_node, pos=first_node, type='node', age=0)
    graph.add_edge(actual_root_base, first_node, weight=first_edge_length, color=edge_state['counter'], root_type=bio_class)
    edge_state['counter'] += 1
    
    rem_endpoints_list = [tuple(ep) for ep in remaining_endpoints]
    if first_node not in rem_endpoints_list:
        graph = continue_graph(graph, skeleton_image, multiclass_skeleton, first_node, actual_root_base, rem_endpoints_list, branch_points, edge_state)
    
    return graph, actual_root_base, skeleton_image

def continue_graph(graph, skeleton_image, multiclass_skeleton, current_pos, parent_pos, end_points_list, branch_points, edge_state):
    neighbor_pixels = find_neighbors(skeleton_image, current_pos)
    for neighbor_start in neighbor_pixels:
        if skeleton_image[neighbor_start[1], neighbor_start[0]] != 1:
            n_tuple = tuple(neighbor_start)
            if n_tuple in graph.nodes and not graph.has_edge(current_pos, n_tuple):
                dist = np.linalg.norm(np.array(current_pos) - np.array(neighbor_start))
                b_class = int(multiclass_skeleton[neighbor_start[1], neighbor_start[0]])
                graph.add_edge(current_pos, n_tuple, weight=dist, color=edge_state['counter'], root_type=b_class)
                edge_state['counter'] += 1
            continue
        
        skeleton_image, next_node, edge_length, bio_class = get_next_node(
            skeleton_image, multiclass_skeleton, neighbor_start, current_pos, neighbor_pixels, 0, current_pos, edge_state
        )
        next_node = tuple(next_node)
        if next_node == parent_pos: continue
        
        if next_node not in graph.nodes:
            graph.add_node(next_node, pos=next_node, type='node', age=0)
        if not graph.has_edge(current_pos, next_node):
            graph.add_edge(current_pos, next_node, weight=edge_length, color=edge_state['counter'], root_type=bio_class)
            edge_state['counter'] += 1
            
        if skeleton_image[next_node[1], next_node[0]] == 1:
            skeleton_image[next_node[1], next_node[0]] = edge_state['counter']
            
        if next_node not in end_points_list:
            graph = continue_graph(graph, skeleton_image, multiclass_skeleton, next_node, current_pos, end_points_list, branch_points, edge_state)
    return graph

def get_next_node(skeleton_image, multiclass_skeleton, current_pixel, parent_pixel, sibling_pixels, acc_dist, initial_pos, edge_state):
    if acc_dist == 0: acc_dist = np.linalg.norm(np.array(current_pixel) - np.array(initial_pos))
    edge_bio_class = int(multiclass_skeleton[current_pixel[1], current_pixel[0]])
    
    while True:
        neighbors = find_neighbors(skeleton_image, current_pixel)
        valid_children = [n for n in neighbors if not np.array_equal(n, parent_pixel) and n not in sibling_pixels]
        
        color_changed = False
        if not np.array_equal(current_pixel, initial_pos):
            if multiclass_skeleton[current_pixel[1], current_pixel[0]] != multiclass_skeleton[parent_pixel[1], parent_pixel[0]]:
                color_changed = True
                if len(valid_children) == 1:
                    next_c = valid_children[0]
                    if len([n for n in find_neighbors(skeleton_image, next_c) if not np.array_equal(n, current_pixel)]) != 1:
                        color_changed = False

        if not np.array_equal(current_pixel, initial_pos):
            skeleton_image[current_pixel[1], current_pixel[0]] = edge_state['counter']
            
        if len(valid_children) != 1 or color_changed:
            return skeleton_image, current_pixel, acc_dist, edge_bio_class
            
        skeleton_image[current_pixel[1], current_pixel[0]] = edge_state['counter']
        next_pixel = valid_children[0]
        acc_dist += np.linalg.norm(np.array(current_pixel) - np.array(next_pixel))
        parent_pixel, current_pixel, sibling_pixels = current_pixel, next_pixel, []

def find_neighbors(skeleton_image, pixel, search_value=1):
    x, y, h, w = pixel[0], pixel[1], skeleton_image.shape[0], skeleton_image.shape[1]
    return [[j, i] for i in range(max(0, y-1), min(h, y+2)) for j in range(max(0, x-1), min(w, x+2)) 
            if skeleton_image[i, j] == search_value and not (x == j and y == i)]

def find_nearest(target_pos, point_list):
    if len(point_list) == 0: return None, point_list, None
    dists = np.linalg.norm(target_pos - point_list, axis=1)
    idx = np.argmin(dists)
    return point_list[idx, :], np.delete(point_list, idx, axis=0), dists