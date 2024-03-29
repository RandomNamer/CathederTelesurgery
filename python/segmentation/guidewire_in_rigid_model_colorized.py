import cv2
import numpy as np
import os
from os.path import join
import re
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from fil_finder import FilFinder2D
from astropy import units as u
import json
import numpyencoder


from common_configs import *
from utils.filesystem import recursive_mkdir


image_folder = f"{PROJECT_ROOT}/Data/Datasets/RigidModelVideo-2-20/2-20-3-clip0"
output_folder = f"{PROJECT_ROOT}/Data/Datasets/RigidModelVideo-2-20/2-20-3-clip0/outputs"
matching_regex = "frame\d{4}\.jpg"
fps = 30


def find_endpoint(skeleton):
    connectivity = find_connectivity(skeleton)
    # print(np.unique(connectivity, return_counts=True))
    return np.argwhere(connectivity == 2)

def find_connectivity(binary_image):
    result = np.zeros_like(binary_image, dtype=np.uint8)
    for [i, j] in np.argwhere(binary_image):
        conn_count = 0
        for ii in range(i-1, i+2):
            for jj in range(j-1, j+2):
                if binary_image[ii, jj]:
                    conn_count += 1
        result[i, j] = conn_count
    return result

def adaptive_threshold_wrapper(image, maxValue, block_size, C, max_block_mean=120):
    new_image = np.copy(image)
    new_image[new_image > max_block_mean] = max_block_mean
    return cv2.adaptiveThreshold(new_image, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=block_size, C=C)

def suggest_c_pct(image, pct=5):
    t = np.percentile(image, pct)
    return np.mean(image[image > t]) - np.mean(image[image < t])

def filament_pruning(skeleton, allow_fragmentation=False, endpoint_boundary=None, intermediate_outputs={}):
    if allow_fragmentation:
            main_skeleton = skeleton.astype(np.uint8)
    else:
        _, labeled, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), connectivity=8)
        # TODO: add bbox extraction
        main_skeleton = np.isin(labeled, np.argsort(stats[:, -1])[-2]).astype(np.uint8)
        #Filament analysis:
    fil = FilFinder2D(main_skeleton, distance=250 * u.pc, mask=main_skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
    final_mask = fil.skeleton_longpath.astype(np.uint8)
    if allow_fragmentation:
            # Remove outlier connectivities:
            _, labeled, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
            endpoints = find_endpoint(final_mask)
            endpoints_labels = [labeled[p[0], p[1]] for p in endpoints]
            eligiblity = [endpoint_boundary[0] < p[0] < endpoint_boundary[2] and endpoint_boundary[1] < p[1] < endpoint_boundary[3] for p in endpoints]
            
            intermediate_outputs['endpoints_labels'] = endpoints_labels
            intermediate_outputs['endpoints_fractured'] = endpoints
            intermediate_outputs['endpoints_eligible'] = eligiblity
            
            for label in np.unique(endpoints_labels):
                eligible_count = np.sum(np.logical_and(eligiblity, endpoints_labels == label))
                if eligible_count == 0:
                    final_mask[labeled == label] = 0
                    #Mark for removal:
                    endpoints[endpoints_labels == label] = (0,0) 
    else: 
            endpoints = fil.end_pts[0]
    return final_mask, endpoints

# New simplified param and workflow for colorized model:
def segment_guidewire(image: np.ndarray, connnected_components_thresholding='otsu', dialation_ksize=5, endpoint_bound_frac=0.05, skeleton_only=False, allow_fragmentation=False, ignore_fil_process=True):
    endpoint_boundary = get_boundary_by_frac(image, endpoint_bound_frac)
    intermediate_outputs = {}
    image_r = image[:, :, 2]
    mask = adaptive_threshold_wrapper(image_r, 255, block_size=33, C=suggest_c_pct(image_r, 5), max_block_mean=np.percentile(image_r, 60))
    mask = np.invert(mask.astype(bool))
    intermediate_outputs['initial_mask_size'] = mask.sum()
    # Morphological op pass 1:
    tmp = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (dialation_ksize, dialation_ksize)))
    num, labeled, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
    sizes = stats[1:, -1]
    labels = np.argsort(stats[:, -1])[-3:-1]
    if connnected_components_thresholding == 'otsu':
        t = threshold_otsu(sizes)
        labels = np.where(sizes >= t)[0] + 1
        intermediate_outputs['otsu_threshold'] = t
    elif connnected_components_thresholding == "manual":
        t = 50
        labels = np.where(sizes >= t)[0] + 1
        intermediate_outputs['manual_threshold'] = t
    elif connnected_components_thresholding == "take_one":
        labels = [np.argmax(sizes) + 1]
    intermediate_outputs['selected_labels'] = labels
    mask = np.isin(labeled, labels).astype(np.uint8)
    # Morphological op pass 2: try to close the gap between the fragmented components
    # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (dialation_ksize_skeleton, dialation_ksize_skeleton)))
    skeleton = skeletonize(mask)
    # temporarily disable fragmentation processing
    if not ignore_fil_process:
        final_mask, endpoints = filament_pruning(skeleton, allow_fragmentation, endpoint_boundary, intermediate_outputs)
    else:
        final_mask = skeleton.astype(np.uint8)
        endpoints = find_endpoint(skeleton)
    endpoints = [p for p in endpoints if endpoint_boundary[0] < p[0] < endpoint_boundary[2] and endpoint_boundary[1] < p[1] < endpoint_boundary[3]]
    return final_mask, endpoints, intermediate_outputs

def get_boundary_by_frac(image, endpoint_bound_frac=0.05):
    h, w = image.shape[:2]
    endpoint_boundary = (h * endpoint_bound_frac, w * endpoint_bound_frac, h * (1-endpoint_bound_frac), w * (1-endpoint_bound_frac))
    return endpoint_boundary

def remove_endpoint_outliers(endpoints, endpoint_boundary):
   endpoints = [p for p in endpoints if endpoint_boundary[0] < p[0] < endpoint_boundary[2] and endpoint_boundary[1] < p[1] < endpoint_boundary[3]]
   return endpoints
    

def find_search_area(skeleton, endpoints, connectpoint_padding=10, endpoint_padding=40):
    paired = []
    bboxes = []
    if endpoints.shape[0] == 0:
        return []
    for i, ep in enumerate(endpoints):
        if i in paired:
            continue
        distances = [cv2.norm(ep, e) for e in endpoints]
        for idx in np.argsort(distances):
            if idx == i:
                continue
            candidate = endpoints[idx]
            x_min = min(ep[0], candidate[0])
            x_max = max(ep[0], candidate[0])
            y_min = min(ep[1], candidate[1])
            y_max = max(ep[1], candidate[1])
            roi = skeleton[x_min:x_max, y_min:y_max]
            if np.sum(roi) > 1:
                continue    
            else :
                paired.append(i)
                paired.append(idx)
                bboxes.append([
                    x_min - connectpoint_padding, 
                    x_max + connectpoint_padding, 
                    y_min - connectpoint_padding, 
                    y_max + connectpoint_padding
                ])
                break
    left_most_point = endpoints[np.argmin(endpoints[:, 1])]
    bottom_most_point = endpoints[np.argmax(endpoints[:, 0])]
    bboxes.append([
        left_most_point[0] - endpoint_padding, 
        left_most_point[0] + endpoint_padding, 
        left_most_point[1] - endpoint_padding, 
        left_most_point[1]
    ])
    bboxes.append([
        bottom_most_point[0] - endpoint_padding,
        skeleton.shape[0] - 1, 
        bottom_most_point[1] - endpoint_padding, 
        bottom_most_point[1] + endpoint_padding
    ])
    return bboxes

def recursive_fill(skeleton, ref_frame, search_area_padding, endpoint_area_padding, update_thresh):
    endpoints = find_endpoint(skeleton)
    search_areas = find_search_area(skeleton, endpoints, search_area_padding, endpoint_area_padding)
    for bbox in search_areas:
        if len(bbox) == 0:
            continue
        roi_ref = ref_frame[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        roi_cur = skeleton[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        roi_diff = roi_ref.astype(int) - roi_cur

        diff_add = np.count_nonzero(roi_diff > 0)
        diff_sub = np.count_nonzero(roi_diff < 0)
        if (diff_add - diff_sub)  > update_thresh and diff_sub < 500 and diff_add < 1000: # have find something, append it
            search_result = np.bitwise_or.reduce((roi_ref, roi_cur))
            skeleton[bbox[0]:bbox[1], bbox[2]:bbox[3]] = search_result
            print(f"Frame updated at bbox {bbox}")
            return recursive_fill(skeleton, ref_frame, search_area_padding, endpoint_area_padding, update_thresh)
    return skeleton

def temporal_fill_process_4(skeleton, ref_stack, ref_count=5, search_area_padding = 10, endpoint_area_padding=40, update_thresh = 20, length_diff_limit=100):
    d = []
    current_frame = skeleton
    need_recompute_endpoints = False
    for r in range(ref_count):
        ref_frame = ref_stack[:, :, -r]
        d.append([np.count_nonzero(ref_frame), np.count_nonzero(current_frame)])
        if np.count_nonzero(ref_frame) - np.count_nonzero(current_frame) > length_diff_limit:
            print(f"begin patching from ref {r} of image")
            need_recompute_endpoints = True
            skeleton_filled = recursive_fill(current_frame, ref_frame, search_area_padding, endpoint_area_padding, update_thresh)
            skeleton_filled_d = cv2.morphologyEx(current_frame.astype(np.uint8), cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            new_skeleton = skeletonize(skeleton_filled_d)
            current_frame = new_skeleton
    return current_frame, need_recompute_endpoints
 
# Use contrasty color on red:    
def composite(image: np.ndarray, mask: np.ndarray, endpoints: tuple, alpha=0.7, speed=None):
    '''
    Note that endpoint coordinates are in (y, x) format
    '''
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr = np.where(mask_bgr == (255, 255, 255), (0, 255, 0), mask_bgr).astype(np.uint8)
    overlay = cv2.addWeighted(image, 1, mask_bgr, alpha, 0)
    title = f"Estimated tip position: {endpoints[0] if len(endpoints) == 1 else 'Multi'}, Speed: " + ("None" if speed is None else f"{speed:.2f}") + " Pixel/s"
    cv2.putText(overlay, title , (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (219, 59, 209), 2)
    for endpoint in endpoints:
        cv2.circle(overlay, (endpoint[1], endpoint[0]), 7, (221, 72, 254), -1)
        
    return overlay
    
if __name__ == "__main__":
    #configs:
    ref_count = 5
    start_from = 0

    
    recursive_mkdir(output_folder)
    debug_folder = join(output_folder, "debug")
    segementation_folder = join(output_folder, "segmentation")
    recursive_mkdir(debug_folder)
    recursive_mkdir(segementation_folder)
    prev_endpoint = None
    ref_stack = np.zeros((1080, 1920, ref_count), dtype=np.uint8)
    processed_count = 0
    for i, f in enumerate(os.listdir(image_folder)):
        if i < start_from:
            continue
        if re.match(matching_regex, f):
            print(f"Processing {f}")
            img = cv2.imread(join(image_folder, f))
            filename = f.split(".")[0]
            
            # skeleton, endpoints, inspection, dilated = segment_guidewire(img, allow_fragmentation=True, connnected_components_thresholding='take_one', ignore_fil_process=True)
            skeleton, endpoints, inspection = segment_guidewire(img, dialation_ksize=7, allow_fragmentation=True, connnected_components_thresholding='otsu', ignore_fil_process=False)
            
            if processed_count > ref_count:
                skeleton, need_recompute_endpoints = temporal_fill_process_4(skeleton, ref_stack, ref_count=ref_count, search_area_padding = 10, endpoint_area_padding=50, update_thresh = 20, length_diff_limit=40)
                if need_recompute_endpoints:
                    print("Recomputing endpoints and pruning")
                    boundary = get_boundary_by_frac(img, 0.05)
                    skeleton, endpoints = filament_pruning(skeleton, False, boundary)
                    endpoints = remove_endpoint_outliers(endpoints, boundary)
            
            ref_stack = np.roll(ref_stack, shift=-1, axis=2)
            ref_stack[:, :, -1] = skeleton
            dilated =  cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1) * 255
            processed_count += 104
            
            cv2.imwrite(join(segementation_folder, f), skeleton.astype(np.uint8) * 255)
            inspection['endpoints'] = endpoints
            json.dump(inspection, open(join(debug_folder, f"{filename}.json"), "w"), indent=2, cls=numpyencoder.NumpyEncoder)
            
            # Speed estimation:
            speed = None
            if len(endpoints) == 1:
                if prev_endpoint is not None:
                    speed = np.linalg.norm(np.array(endpoints[0]) - np.array(prev_endpoint)) * fps
                prev_endpoint = endpoints[0]
            if speed is not None and (speed < 0 or speed > 1000):
                speed = None
            cv2.imwrite(join(output_folder, f), composite(img, dilated, endpoints, alpha=0.5, speed=speed))
        else:
            print(f"Skipping {f}")
    os.system(f"ffmpeg -framerate 15 -i {output_folder}/frame%04d.jpg -r 15 -b:v 20M {output_folder}/composite.mp4")
        
