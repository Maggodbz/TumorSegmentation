import fiftyone as fo
from pathlib import Path
from fiftyone_dataset import add_brain_mri_scan_dataset_to_fiftyone
import ultralytics
from ultralytics import SAM


if __name__ == "__main__":
    dataset_name = "brain-tumor-segmentation"
    base_path = Path.home() / ".cache/kagglehub/datasets/tobiasgrass/unet-data-png"
    class_name = "tumor"
    dataset = add_brain_mri_scan_dataset_to_fiftyone(dataset_name, base_path, class_name)
    print(dataset)

    model = ultralytics.YOLO("yolo11n.pt")
    dataset.apply_model(
        model=model,
        label_field="predictions",
    )
    dataset.save()
    app = fo.launch_app(dataset)
    app.wait()