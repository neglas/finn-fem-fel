import json
import logging
from datetime import datetime
from pathlib import Path

# ---------- Error logger ----------
def setup_error_log(relative_path:Path):
    logging.basicConfig(
        
        filename=relative_path,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def log_error(message: str):
    logging.error(message)


# ---------- JSON handler ----------
def save_detections_to_json(image_name, box_dict, file_name):
    """
    Save or update detection results for a single image in a JSON file.

    If the image already exists, its boxes are replaced. Otherwise, a new entry is appended.
    """

    base_path = Path(__file__).parent
    folder = base_path / "output/results"
    folder.mkdir(parents=True, exist_ok=True)

    out_path = folder / file_name

    # Load existing data
    if out_path.exists():
        with open(out_path, "r") as f:
            all_data = json.load(f)
            if isinstance(all_data, dict):
                # wrap old single-image dict into a list
                all_data = [all_data]
    else:
        all_data = []

    # Create new data for this image
    new_entry = {
        "image_name": image_name,
        "boxes": [{"label": label, "bbox": box} for label, box in box_dict.items()]
    }

    # Check if image already exists
    updated = False
    for idx, entry in enumerate(all_data):
        if entry.get("image_name") == image_name:
            all_data[idx] = new_entry  
            updated = True
            break

    if not updated:
        all_data.append(new_entry)  

    # Save back
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=4)

    if updated:
        print(f"Updated boxes for image '{image_name}' in {out_path}")
    else:
        print(f"Saved {len(box_dict)} boxes for image '{image_name}' to {out_path}")

        