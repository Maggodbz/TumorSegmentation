from dataclasses import dataclass
from pathlib import Path

@dataclass
class BrainMRIScan:
    t1: Path | None = None
    t2: Path | None = None
    flair: Path | None = None
    mask: Path | None = None
    class_name: str | None = None



def load_brain_mri_scans(dataset_path: Path, class_name: str) -> list[BrainMRIScan]:
    """
    Load a brain MRI scan dataset from a directory.
    Directory structure:
    dataset_path/
        flair/
            image_1.png
            image_2.png
            ...
        t2/
            image_1.png
            image_2.png
            ...
        mask/
            image_1.png
            image_2.png
            ...
        class_name (str): The name of the class.
    """
    dataset = []
    flair_files = sorted(list((dataset_path / "flair").glob("*.png")))
    t2_files = sorted(list((dataset_path / "t2").glob("*.png")))
    mask_files = sorted(list((dataset_path / "mask").glob("*.png")))

    for flair_file, t2_file, mask_file in zip(flair_files, t2_files, mask_files):
        brain_mri_scan = BrainMRIScan(
            flair=flair_file,
            t2=t2_file,
            mask=mask_file,
            class_name=class_name,
        )
        dataset.append(brain_mri_scan)
    return dataset
