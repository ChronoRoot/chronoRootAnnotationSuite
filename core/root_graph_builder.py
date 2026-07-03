import numpy as np
import cv2
import networkx as nx
from skimage.morphology import skeletonize


# Kernels for trim()
T_KERNELS = [
    np.array([[-1, 1, -1], [1, 1, 1], [0, 0, 0]]),
    np.array([[-1, 1, 0], [1, 1, 0], [-1, 1, 0]]),
    np.array([[0, 0, 0], [1, 1, 1], [-1, 1, -1]]),
    np.array([[0, 1, -1], [0, 1, 1], [0, 1, -1]]),
    np.array([[1, -1, -1], [1, 1, -1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [1, 1, -1], [1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, 1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [1, 1, -1]]),
    np.array([[-1, 1, 1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[1, 1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, 1], [-1, 1, 1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [-1, 1, 1], [-1, -1, 1]]),
    np.array([[-1, 1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, -1]])
]

# Kernels for endPoints() - Strict background
EP_KERNELS = [
    np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, 1, -1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [-1, 1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
]

# Perfectly symmetrical relaxed kernels for pruning
PRUNE_KERNELS = [
    # Straight directions (Relaxed corners using 0)
    np.array([[-1, -1, -1], [-1,  1, -1], [ 0,  1,  0]]), # UP
    np.array([[ 0,  1,  0], [-1,  1, -1], [-1, -1, -1]]), # DOWN
    np.array([[-1, -1,  0], [-1,  1,  1], [-1, -1,  0]]), # LEFT
    np.array([[ 0, -1, -1], [ 1,  1, -1], [ 0, -1, -1]]), # RIGHT
    
    # Diagonal directions (Strict corners)
    np.array([[-1, -1, -1], [-1,  1, -1], [ 1, -1, -1]]), # NE tip (Branch SW)
    np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1,  1]]), # NW tip (Branch SE) 
    np.array([[ 1, -1, -1], [-1,  1, -1], [-1, -1, -1]]), # SE tip (Branch NW)
    np.array([[-1, -1,  1], [-1,  1, -1], [-1, -1, -1]])  # SW tip (Branch NE)
]

# --- STRICT BRANCH POINT KERNELS ---

# 1. Crosses (Strict corners to avoid matching solid blocks)
X_KERNELS = [
    np.array([[-1,  1, -1], 
              [ 1,  1,  1], 
              [-1,  1, -1]]), # + cross
              
    np.array([[ 1, -1,  1], 
              [-1,  1, -1], 
              [ 1, -1,  1]])  # x cross
]

# 2. T-Branches (Strictly block the 4th side so they never match a Cross)
T_BASE_ORTHO = np.array([[-1,  1, -1], 
                         [ 1,  1,  1], 
                         [-1, -1, -1]]) # -1 at the bottom ensures this is ONLY a 'T'

T_BASE_DIAG = np.array([[ 1, -1,  1], 
                        [-1,  1, -1], 
                        [ 1, -1, -1]]) # -1 at bottom-right ensures this is ONLY a 3-way diagonal

T_BRANCH_KERNELS = []
for i in range(4):
    T_BRANCH_KERNELS.append(np.rot90(T_BASE_ORTHO, i))
    T_BRANCH_KERNELS.append(np.rot90(T_BASE_DIAG, i))


# 3. Y-Branches (Strictly block the negative space between forks)
Y_BASE_1 = np.array([[ 1, -1,  1], 
                     [-1,  1, -1], 
                     [-1,  1, -1]]) 

Y_BASE_2 = np.array([[-1,  1, -1], 
                     [ 1,  1, -1], 
                     [-1, -1,  1]])

Y_KERNELS = []
for i in range(4):
    Y_KERNELS.append(np.rot90(Y_BASE_1, i))
    Y_KERNELS.append(np.rot90(Y_BASE_2, i))

def get_roi_bounding_box(mask, padding=5):
    points = cv2.findNonZero(mask)
    if points is None: return None, None
    x, y, w, h = cv2.boundingRect(points)
    h_img, w_img = mask.shape
    x_start, y_start = max(0, x - padding), max(0, y - padding)
    x_end, y_end = min(w_img, x + w + padding), min(h_img, y + h + padding)
    return (slice(y_start, y_end), slice(x_start, x_end)), (x_start, y_start)

def extract_skeleton(binary_mask, prune_primary=5, prune_cleanup_1=3, prune_cleanup_2=3, *, prune_iters=None):
    """Generates skeleton and topological points directly from a numpy array."""
    if prune_iters is not None:
        prune_primary = prune_iters

    # Clean up mask with morphological operations 
    morph_kernel_size = 3
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    
    morph_kernel_size = 3
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    
    roi_slice, offset = get_roi_bounding_box(binary_mask)
    if roi_slice is None:
        return np.zeros_like(binary_mask), np.array([]), np.array([]), False

    cropped_mask = binary_mask[roi_slice]
    skeleton_crop = np.array(skeletonize(cropped_mask > 0), dtype='uint8')

    # Apply customizable pruning pipeline
    skeleton_crop = trim(prune(skeleton_crop, prune_primary))
    skeleton_crop = trim(prune(skeleton_crop, prune_cleanup_1))
    skeleton_crop = trim(prune(skeleton_crop, prune_cleanup_2))
    
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

def trim(ske): ## Removes unwanted pixels from the skeleton
    # Using pre-allocated global kernels
    bp = np.zeros_like(ske)
    for t in T_KERNELS:
        bp = cv2.morphologyEx(ske, cv2.MORPH_HITMISS, t)
        ske = cv2.subtract(ske, bp)
    return ske


def prune(skel, num_it): 
    ## Removes branches with length lower than num_it
    orig = skel
    
    # 1. Pruning loop using the relaxed PRUNE_KERNELS
    for i in range(0, num_it):
        current_skel = skel
        for kernel in PRUNE_KERNELS:
            hit = cv2.morphologyEx(current_skel, cv2.MORPH_HITMISS, kernel)
            current_skel = cv2.subtract(current_skel, hit)
        skel = current_skel
        
    # 2. Re-grow endpoints
    end = endPoints(skel) # Still correctly uses EP_KERNELS under the hood
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    for i in range(0, num_it):
        end = cv2.dilate(end, kernel)
        end = cv2.bitwise_and(end, orig)
        
    return cv2.bitwise_or(end, skel)


def endPoints(skel):
    ep = np.zeros_like(skel)
    # Use global EP_KERNELS
    for kernel in EP_KERNELS:
        ep = cv2.add(ep, cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel))
    return ep


def skeleton_nodes(ske):
    branch = branchedPoints(ske)
    end = endPoints(ske)
    
    bp = np.where(branch == 1)
    bnodes = []
    for i in range(len(bp[0])):
        bnodes.append([bp[1][i],bp[0][i]])
    
    ep = np.where(end == 1)
    enodes = []
    for i in range(len(ep[0])):
        enodes.append([ep[1][i],ep[0][i]])
    
    return np.array(bnodes), np.array(enodes)


def branchedPoints(skel):
    bp = np.zeros(skel.shape, dtype=int)
    
    for x in X_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, x)
    for y in Y_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, y)
    for t in T_BRANCH_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, t)
        
    return bp

# --- GRAPH BUILDER (State-Safe) ---
def createGraph(skeleton_image, multiclass_skeleton, root_base_position, end_points, branch_points):
    skeleton_image = skeleton_image.astype(np.int32)
    
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

def graphInit(graph, main_root_class=1):
    """
    Initializes a static plant graph by evaluating candidates (endpoints + extremities) 
    to find the path that maximizes the physical main root length.
    """
    if len(graph.nodes) == 0:
        raise Exception("Cannot initialize empty graph")

    # 1. Identify Main Root Sub-components & Set Traversal Costs EARLY
    c1_edges = [(u, v) for u, v, data in graph.edges(data=True) 
                if data.get('orig_root_type', data.get('root_type')) == main_root_class]
    
    c1_nodes_set = set()
    for u, v in c1_edges:
        c1_nodes_set.add(u)
        c1_nodes_set.add(v)
        
    for u, v, data in graph.edges(data=True):
        phys_len = data.get('weight', 1.0)
        edge_type = data.get('orig_root_type', data.get('root_type'))
        
        if edge_type == main_root_class:
            data['traversal_cost'] = phys_len
        else:
            if phys_len < 5.0 and (u in c1_nodes_set or v in c1_nodes_set):
                data['traversal_cost'] = phys_len * 2.0 
            else:
                data['traversal_cost'] = phys_len * 100.0

    c1_nodes = list(c1_nodes_set)
    ini_node, ftip_node = None, None

    # 2. Candidate Evaluation & Length Maximization
    if c1_nodes:
        # Get the absolute highest and lowest nodes (captures T-junction seeds/tips)
        abs_top_node = min(c1_nodes, key=lambda n: n[1])
        abs_bottom_node = max(c1_nodes, key=lambda n: n[1])
        
        # Find isolated nodes (degree 1)
        c1_endpoints = [n for n in c1_nodes if graph.degree(n) == 1]
        
        # Expand candidates: Endpoints UNION Absolute Extremities
        candidates = list(set(c1_endpoints + [abs_top_node, abs_bottom_node]))
        
        if len(candidates) >= 2:
            max_phys_length = -1
            best_pair = (None, None)
            
            # Compare all candidates to find the longest continuous main root path
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    n1, n2 = candidates[i], candidates[j]
                    try:
                        path = nx.shortest_path(graph, source=n1, target=n2, weight='traversal_cost')
                        phys_len = sum(graph.edges[path[k], path[k+1]].get('weight', 1.0) for k in range(len(path) - 1))
                        
                        if phys_len > max_phys_length:
                            max_phys_length = phys_len
                            best_pair = (n1, n2)
                            
                    except nx.NetworkXNoPath:
                        continue
                        
            if best_pair[0] is not None:
                # Sort the winning pair by Y-coordinate
                if best_pair[0][1] < best_pair[1][1]:
                    ini_node, ftip_node = best_pair[0], best_pair[1]
                else:
                    ini_node, ftip_node = best_pair[1], best_pair[0]
            else:
                ini_node, ftip_node = abs_top_node, abs_bottom_node
        else:
            ini_node, ftip_node = abs_top_node, abs_bottom_node
            
        if ini_node == ftip_node and len(c1_nodes) > 1:
            ini_node, ftip_node = abs_top_node, abs_bottom_node
            
    else:
        # Fallback if no main root edges exist
        all_nodes = list(graph.nodes())
        ini_node = min(all_nodes, key=lambda n: n[1])
        ftip_node = max(all_nodes, key=lambda n: n[1])

    # 3. Assign topological types
    for node in graph.nodes():
        if node == ini_node:
            graph.nodes[node]['type'] = "Ini"
            graph.nodes[node]['age'] = 1
        elif node == ftip_node:
            graph.nodes[node]['type'] = "FTip"
            graph.nodes[node]['age'] = 1
        else:
            degree = graph.degree(node)
            if degree > 2:
                graph.nodes[node]['type'] = "Bif"
            elif degree == 1:
                graph.nodes[node]['type'] = "LTip"
            else:
                graph.nodes[node]['type'] = "null"

    # 4. Main Path Extraction
    if ini_node != ftip_node:
        try:
            main_path = nx.shortest_path(
                graph, 
                source=ini_node, 
                target=ftip_node, 
                weight='traversal_cost'
            )
            for i in range(len(main_path) - 1):
                u, v = main_path[i], main_path[i + 1]
                graph.edges[u, v]['root_type'] = 10
                
        except nx.NetworkXNoPath:
            print(f"Warning: Disconnected graph structure between {ini_node} and {ftip_node}.")

    return graph