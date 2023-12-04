import os
import torch
from utils.filesystem import recursive_mkdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import SamModel, SamProcessor
from common_configs import PROJECT_ROOT
from segmentation.common import otsu

source_root = f"{PROJECT_ROOT}/Data/Datasets/RigidModelVideo-11-21/11-21-3-clip0"
source_images = [f"{source_root}/frame0001.jpg"]
# source_images = [f"{source_root}/{file}" for file in os.listdir(source_root) if file.endswith(".jpg")]

# target_path = f"{project_root}/Data/Datasets/RigidModelVideo-11-21/11-21-2-clip0/"
target_path = source_root

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamModel.from_pretrained('facebook/sam-vit-huge').to(device)
processor = SamProcessor.from_pretrained('facebook/sam-vit-huge')


def generate_points(image, bounding_box=[(20, 20), (1900, 1060)], step=100):
    points = [[x, y] for x in range(bounding_box[0][0], bounding_box[1][0], step) for y in
              range(bounding_box[0][1], bounding_box[1][1], step)]
    return [[point] for point in points]

def optimal_mask(masks, scores, strategy="minimize_size", min_score=0.7):
    if strategy == "minimize_size":
        mask_sizes = torch.stack([torch.count_nonzero(m) for m in masks])
        idx = torch.argmin(mask_sizes)
        if scores[idx] > min_score:
            return masks[idx]
        else: 
            return optimal_mask(masks, scores, strategy="maximize_score")
    elif strategy == "maximize_score":
        return masks[torch.argmax(scores)]
    else:
        raise ValueError("Invalid strategy. Must be one of 'minimize_size' or 'maximize_score'")


for image_path in source_images:
    print(f"Doing SAM on {image_path}")
    image = Image.open(image_path)
    image_name = image_path.split("/")[-1].split(".")[0]
    inputs = processor(image, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs['pixel_values'])
    
    generated_points = generate_points(image, bounding_box=[(20,20), (1900, 1060)], step=80)
    inputs = processor(image, input_points=[generated_points], return_tensors="pt").to(device)
    inputs.pop("pixel_values", None)
    inputs.update({'image_embeddings': image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores
    optimal_masks = torch.stack([ optimal_mask(masks[0][i], scores[0][i], strategy="minimize_size", min_score=0.88) for i in range(scores.shape[1]) ])
    mask_sizes = [torch.count_nonzero(m).cpu().detach().numpy() for m in optimal_masks]
    
    # decide threshold for sizes. We will not use CUDA since histogram and otsu is not supported on CUDA.
    hist, bins = np.histogram(mask_sizes, bins=1000)
    threshold_idx = 500
    if hist[400:600].sum() > 0 or True:
        threshold_idx = otsu(hist)
    mask_size_threshold = bins[threshold_idx]
    
    final_masks_stack = optimal_masks[mask_sizes < mask_size_threshold]
    final_mask = final_masks_stack.any(dim=0).cpu().detach().numpy()
    final_mask = np.invert(final_mask)
    
    cv2.imwrite(f"{target_path}/{image_name}_mask.png", final_mask * 255)
    
    
