# ChronoRoot Annotation Suite: Official User Guide

Manual annotation of complex root systems is a demanding task. This custom tool was built to replace generic software (like ITK-SNAP) with a streamlined, biology-first workflow. 


## 1. Getting Started: The File Browser

The application automatically tracks your progress, ensuring you never lose track of which frames still need review.

1. **Select Database:** Click the **`...`** button in the top left to select your root database folder (e.g., `ArabidopsisDataset`).
2. **Navigate:** Double-click folders to navigate. You will see progress bars indicating how many frames are `Completed` versus `In Progress`.
3. **Load a Task:** Click any file marked `[Pending]` or `[In Progress]`. The tool will automatically load the raw `.png` image and any existing `.nii.gz` or `.json` predictions.

![Main interface of ChronoRoot Annotation Suite showing the file browser.](tutorialImages/1_main_interface.png) 
![Selecting a specific task folder from the database.](tutorialImages/2_selecting_a_task.png) 
![A loaded task displaying the raw root image and plant list.](tutorialImages/3_task_loaded.png)


## 2. Canvas Navigation & Controls

Efficient movement around the canvas is critical for reducing annotation fatigue.

* **Pan:** Hold **Ctrl + Left-Click** and drag to move around the high-resolution image.
* **Zoom:** Use the **Mouse Wheel** to zoom in and out.
* **Brush Size:** Use the **Brush Size slider** in the left panel to precisely adjust your stroke thickness.
* **Opacity:** Use the **Overlay Opacity slider** to see the raw root texture beneath your painted masks. Adjust this frequently to verify edges.
* **Bounding Box:** Click the **Show Bounding Boxes** button to toggle boxes that show the exact pixel dimensions of all plant instances.

![Overlay opacity set to high, obscuring the raw root.](tutorialImages/4_opacity_slider_1.png) 
![Overlay opacity lowered to clearly see the raw root beneath the mask.](tutorialImages/5_opacity_slider_2.png) 
![Canvas view with plant bounding boxes toggled off.](tutorialImages/6_bounding_boxes_off.png)


## 3. The Plant-by-Plant Workflow: Examples

The automatic model provides a great starting point, but it will make mistakes. For every plant in the image, you will cycle through three core steps:
1. **Fix Instance:** Does it need to be merged with another piece, split from a neighbor, or created from scratch?
2. **Fix Binary Shape:** Use **Shape Paint** to define the exact outer limits of the plant. (Left-click to paint, right-click to erase background noise).
3. **Multi-Class Segmentation:** Use **Multi-Class Paint** to assign biological labels (Primary Root, Lateral Root, Hypocotyl, etc.) inside the area you just defined.

Here is how this exact workflow applies to the three most common scenarios you will face.

### Scenario A: The Fused or Overlapping Plant (Splitting)
When plants grow densely, the model often fuses two overlapping roots into a single ID. You must separate them manually and reconstruct the overlap.

1. **Split:** Select the fused plant in the list. Click the **Split (Knife)** tool, draw a red line completely across the connection point, and **double-click** to execute the cut. The system will separate the pieces into two IDs.
2. **Fix Overlaps (Shape Paint):** Because the split tool divides pixels exclusively, you must recreate the physical overlap. Select one of the separated plants, click **Shape Paint**, and paint the shared crossing area back in. *(Pixels can belong to multiple plants as long as they have different Plant IDs).*
3. **Multi-Class:** Finally, select the **Multi-Class Paint** tool and assign the correct biological labels to the newly separated root structures.

![Two distinct root systems merged into one ID.](tutorialImages/7_example_fused_plants.png) 
![Drawing the red cut line across the fusion.](tutorialImages/8_example_splitting.png) 
![Plants successfully separated into two distinct IDs.](tutorialImages/9_a_example_splitted.png) 
![Showing the structural overlap of the two plants.](tutorialImages/9c_overlapped_plants_example.png)
![Repainting the shared crossing pixels using Shape Paint.](tutorialImages/9_b_fixing_with_shape.png) 
![Applying the correct biological labels to the separated roots.](tutorialImages/9_d_fixing_multiclass.png) 

### Scenario B: The Broken Plant (Merging)
If a single plant is fractured into multiple IDs due to faint or thin roots, you need to stitch it back together.

1. **Merge:** Hold **Shift** and click the broken pieces in the "Plant Instances" list to select them all. Click **Merge Selected**. They will combine into a brand-new, unified Plant ID.
2. **Shape Paint:** Select the newly merged plant, click the **Shape Paint** tool, and draw connecting lines to fill the empty gaps between the fragments.
3. **Multi-Class:** Apply the appropriate biological labels over the entire continuous structure.

![A single root split into multiple fragments by the model.](tutorialImages/10_broken_plant.png)
![Selecting all fragment IDs in the left-hand list.](tutorialImages/11_dual_selection.png)
![The fragments successfully merged into a single Plant ID.](tutorialImages/12_a_merged.png)
![Paiting](tutorialImages/12_b_fixing_with_painter.png) 
![Painting 2](tutorialImages/12_c_painter_green_adds.png) 
![Painting multiclass](tutorialImages/12_d_reviewing_multiclass.png)

### Scenario C: The Missing Plant (Creating)
If the model completely missed a seed, newly germinated root, or obscured plant:

1. **Create:** Click **+ Create New Plant**.
2. **Shape Paint:** Draw your first stroke over the unmapped plant. It will instantly be assigned a new ID and a unique color. Continue painting until the binary boundary is accurate.
3. **Multi-Class:** Switch to the **Multi-Class Paint** tool and label the seed, root, or hypocotyl accordingly.

![An unannotated plant visible in the raw image.](tutorialImages/13_missing_plant_noticed.png)
![Clicking the Create New Plant button.](tutorialImages/14_creation_button.png) 
![The newly painted binary mask for the plant.](tutorialImages/15_binary_creation.png) 
![Applying semantic classes to the newly created plant.](tutorialImages/16_multiclass_labels.png)


## 4. Final Review & Export

Before baking the data for the AI trainer, you must verify the entire image context.

1. Click **Global Multi-Class View**. This renders every plant and every biological label simultaneously. 
2. Visually verify that:
    * Every plant has a tight, accurate bounding box.
    * No labels are bleeding into the background.
    * Crossings and overlaps are logically accounted for.
3. Click **Save Progress** (Yellow) if you need to take a break. This saves your work safely to a `.json` file.
4. Click **Finish Annotation** (Green) when the image is perfect. 

![Global Multi-Class View showing all annotated plants with accurate bounding boxes and class labels.](tutorialImages/17_final_validation.png)

### Automated Validation
When you click Finish, the suite runs a background check. If it detects a single Plant ID broken into disconnected floating pieces (forgotten fragments), it will block the save and warn you. You must use the Shape Paint tool to connect the pieces, or the Split tool to separate them into distinct IDs, ensuring mathematically clean training data.