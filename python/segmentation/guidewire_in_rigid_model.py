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


image_folder = f"{PROJECT_ROOT}/Data/Datasets/RigidModelVideo-11-21/11-21-1-clip2"
output_folder = f"{PROJECT_ROOT}/Data/Datasets/RigidModelVideo-11-21/11-21-1-clip2/outputs_f_t_100"
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

def segment_guidewire(image: np.ndarray, connnected_components_thresholding='otsu', dialation_ksize_skeleton=7, endpoint_bound_frac=0.05, skeleton_only=False, allow_fragmentation=False):
    h, w = image.shape[:2]
    endpoint_boundary = (h * endpoint_bound_frac, w * endpoint_bound_frac, h * (1-endpoint_bound_frac), w * (1-endpoint_bound_frac))
    intermediate_outputs = {}
    image_r = image[:, :, 2]
    mask = adaptive_threshold_wrapper(image_r, 80, block_size=33, C=30, max_block_mean=110)
    mask = np.invert(mask.astype(bool))
    intermediate_outputs['initial_mask_size'] = mask.sum()
    # Morphological op pass 1:
    tmp = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    num, labeled, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
    sizes = stats[1:, -1]
    labels = np.argsort(stats[:, -1])[-3:-1]
    if (connnected_components_thresholding == 'otsu'):
        t = threshold_otsu(sizes)
        labels = np.where(sizes >= t)[0] + 1
        intermediate_outputs['otsu_threshold'] = t
    if (connnected_components_thresholding == "manual"):
        t = 50
        labels = np.where(sizes >= t)[0] + 1
        intermediate_outputs['manual_threshold'] = t
    intermediate_outputs['selected_labels'] = labels
    mask = np.isin(labeled, labels).astype(np.uint8)
    # Morphological op pass 2: try to close the gap between the fragmented components
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (dialation_ksize_skeleton, dialation_ksize_skeleton)))
    skeleton = skeletonize(mask)
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
    endpoints = [p for p in endpoints if endpoint_boundary[0] < p[0] < endpoint_boundary[2] and endpoint_boundary[1] < p[1] < endpoint_boundary[3]]
    if not skeleton_only:
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    return final_mask * 255, endpoints, intermediate_outputs
    
def composite(image: np.ndarray, mask: np.ndarray, endpoints: tuple, alpha=0.7, speed=None):
    '''
    Note that endpoint coordinates are in (y, x) format
    '''
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr = np.where(mask_bgr == (255, 255, 255), (0, 255, 0), mask_bgr).astype(np.uint8)
    overlay = cv2.addWeighted(image, 1, mask_bgr, alpha, 0)
    title = f"Estimated tip position: {endpoints if len(endpoints) < 2 else 'Multi'}, Speed: " + ("None" if speed is None else f"{speed:.2f}") + " Pixel/s"
    cv2.putText(overlay, title , (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 205), 2)
    for endpoint in endpoints:
        cv2.circle(overlay, (endpoint[1], endpoint[0]), 5, (128, 0, 245), -1)
        
    return overlay
    
if __name__ == "__main__":
    recursive_mkdir(output_folder)
    debug_folder = join(output_folder, "debug")
    segementation_folder = join(output_folder, "segmentation")
    recursive_mkdir(debug_folder)
    recursive_mkdir(segementation_folder)
    prev_endpoint = None
    for f in os.listdir(image_folder):
        if re.match(matching_regex, f):
            print(f"Processing {f}")
            img = cv2.imread(join(image_folder, f))
            filename = f.split(".")[0]
            segemented, endpoints, inspection = segment_guidewire(img, allow_fragmentation=True, connnected_components_thresholding='manual')
            cv2.imwrite(join(segementation_folder, f), segemented)
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
            cv2.imwrite(join(output_folder, f), composite(img, segemented, endpoints, alpha=0.5, speed=speed))
        else:
            print(f"Skipping {f}")
    os.system(f"ffmpeg -framerate 15 -i {output_folder}/frame%04d.jpg -r 15 -b:v 20M {output_folder}/composite.mp4")
        
