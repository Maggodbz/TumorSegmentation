import fiftyone as fo
import os
from pathlib import Path
import numpy as np
import cv2
from data import load_brain_mri_scans

# Set FiftyOne environment variables
os.environ["FIFTYONE_DATABASE_DIR"] = "~/.fiftyone/var/lib/mongo"

def mask_to_bbox(mask_path):
    """
    Convert a binary mask to bounding boxes and their corresponding instance masks.
    
    Args:
        mask_path: Path to the mask image
        
    Returns:
        Tuple of (bboxes, instance_masks) where:
        - bboxes: List of bounding boxes in format [x, y, w, h] normalized to [0,1]
        - instance_masks: List of binary masks for each detection, cropped to their bounding box
    """
    # Read the mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Threshold to get binary mask (255 for tumor, 0 for background)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to bounding boxes and extract instance masks
    height, width = mask.shape
    bboxes = []
    instance_masks = []
    
    for contour in contours:
        # Get bounding box in (x,y,w,h) format
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create instance mask by cropping the binary mask to the bounding box
        instance_mask = binary[y:y+h, x:x+w]
        
        # Normalize coordinates to [0,1] but keep width/height format
        bbox = [x/width, y/height, w/width, h/height]  # [x,y,w,h] format
        
        bboxes.append(bbox)
        instance_masks.append(instance_mask)
    
    return bboxes, instance_masks

def add_brain_mri_scan_dataset_to_fiftyone(dataset_name: str, dataset_path: Path, class_name: str) -> fo.Dataset:
    """
    Add a brain MRI scan dataset to FiftyOne.

    Args:
        dataset_name (str): The name of the dataset.
        dataset_path (Path): The path to the dataset.
    """
    # Delete existing dataset if it exists to avoid conflicts
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    # Create a new dataset
    dataset = fo.Dataset(dataset_name)
    
    # Load the brain MRI scans
    brain_mri_scans = load_brain_mri_scans(dataset_path, class_name)
    
    # Add samples to the dataset
    for brain_mri_scan in brain_mri_scans:
        sample = fo.Sample(filepath=str(brain_mri_scan.flair))
        
        # Convert mask to bounding boxes and instance masks
        bboxes, instance_masks = mask_to_bbox(brain_mri_scan.mask)
        
        # Create detections from bounding boxes
        detections = []
        for bbox, instance_mask in zip(bboxes, instance_masks):
            # Convert mask to boolean (required by FiftyOne)
            instance_mask = instance_mask > 0
            
            detection = fo.Detection(
                label=class_name,
                bounding_box=bbox,  # [x,y,w,h] format
                mask=instance_mask,  # Binary mask cropped to bounding box
                confidence=1.0  # Since these are ground truth
            )
            detections.append(detection)
        
        # Add detections to the sample
        sample["ground_truth"] = fo.Detections(detections=detections)
        
        dataset.add_sample(sample)
    
    # Compute metadata and save the dataset
    dataset.compute_metadata()
    dataset.save()
    
    return dataset
    
if __name__ == "__main__":
    dataset_name = "brain-tumor-segmentation"
    base_path = Path.home() / ".cache/kagglehub/datasets/tobiasgrass/unet-data-png"
    class_name = "tumor"
    dataset = add_brain_mri_scan_dataset_to_fiftyone(dataset_name, base_path, class_name)
    app = fo.launch_app(dataset)
    app.wait()
    