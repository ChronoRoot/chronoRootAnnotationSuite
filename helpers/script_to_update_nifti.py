import os
import nibabel as nib
import numpy as np
from ..model import PlantImageModel 

def update_nifti_files(root_folder):
    """
    Walks through all subfolders, loads the plant model for each JSON found,
    and updates/overwrites the .nii.gz file with the flattened multiclass map.
    """
    model = PlantImageModel()
    updated_count = 0

    print(f"Starting NIfTI update walk in: {root_folder}")

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json"):
                base_name = file.replace(".json", "")
                png_path = os.path.join(root, f"{base_name}.png")
                nii_path = os.path.join(root, f"{base_name}.nii.gz")

                # Only process if the corresponding PNG and NIfTI exist
                if os.path.exists(png_path) and os.path.exists(nii_path):
                    try:
                        # 1. Load the task into the model (decodes RLE and populates masks)
                        model.load_task(root, base_name)
                        
                        # 2. Generate the flattened multiclass label map
                        final_labels = model._flatten_multiclass()
                        
                        if final_labels is not None:
                            # 3. Save the updated NIfTI (matching your model's orientation logic)
                            new_nifti = nib.Nifti1Image(final_labels.astype(np.uint8).T, np.eye(4))
                            nib.save(new_nifti, nii_path)
                            
                            print(f"Updated: {nii_path}")
                            updated_count += 1
                        else:
                            print(f"Skipped (No masks found): {base_name}")
                            
                    except Exception as e:
                        print(f"Error processing {base_name}: {e}")

    print(f"\nFinished! Updated {updated_count} files.")

if __name__ == "__main__":
    # Change this to your target directory
    target_dir = "../ChronoRoot2"
    update_nifti_files(target_dir)