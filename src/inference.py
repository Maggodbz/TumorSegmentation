import fiftyone as fo
from pathlib import Path
from fiftyone_dataset import add_brain_mri_scan_dataset_to_fiftyone





if __name__ == "__main__":
    dataset_name = "brain-tumor-segmentation"
    base_path = Path.home() / ".cache/kagglehub/datasets/tobiasgrass/unet-data-png"
    class_name = "tumor"
    dataset = add_brain_mri_scan_dataset_to_fiftyone(dataset_name, base_path, class_name)
    