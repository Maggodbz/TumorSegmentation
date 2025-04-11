import fiftyone as fo
import os
from pathlib import Path
from data import load_brain_mri_scans

# Set FiftyOne environment variables
os.environ["FIFTYONE_DATABASE_DIR"] = "~/.fiftyone/var/lib/mongo"

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
    
    # Set the mask targets for visualization
    dataset.default_mask_targets = {255: class_name}
    
    # Load the brain MRI scans
    brain_mri_scans = load_brain_mri_scans(dataset_path, class_name)
    
    # Add samples to the dataset
    for brain_mri_scan in brain_mri_scans:
        sample = fo.Sample(filepath=str(brain_mri_scan.flair))
        sample["mask"] = fo.Segmentation(mask_path=str(brain_mri_scan.mask))
        dataset.add_sample(sample)
    
    # Compute metadata and save the dataset
    dataset.compute_metadata()
    dataset.save()
    
    return dataset
    
    
if __name__ == "__main__":
    dataset_name = "brain-tumor-segmentation"
    base_path = Path.home() / ".cache/kagglehub/datasets/tobiasgrass/unet-data-png"
    class_name = "tumor"
    add_brain_mri_scan_dataset_to_fiftyone(dataset_name, base_path, class_name)
    