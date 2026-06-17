# ChronoRoot Annotation Suite: Official User Guide

Manual annotation of complex root systems is a demanding task. This tool provides a streamlined, biology-first workflow for correcting model segmentations, validating root tracing, and exporting phenotypic measurements.


## 0. Input Files per Plate

Each plate frame uses the **same base name** in one folder:

| File | Description |
|------|-------------|
| `PlateName.png` | Scanner photo of the petri dish |
| `PlateName.nii.gz` | **Automatic segmentation** from your model — the starting draft, not ground truth |
| `PlateName.json` | Your saved edits (created when you click **Save Progress**) |

After measurement and export:

| File | Description |
|------|-------------|
| `PlateName_Metrics.json` | Per-plant morphometrics, plate metadata |
| `PlateName_Topology.rsml` | Root architecture (RSML format) |

The model may miss plants or fuse neighbors. You correct those in **Annotation**. If you paint a **wide** mask to capture faint roots, use **Centerline Cleanup (optional)** to thin it before the tracing check.

> Finish **Finish Annotation** when mask and tracing work is done. Sections 7–8 cover trait measurement and batch comparison.


## 1. Getting Started: The File Browser

The application tracks progress so you can see which frames still need work.

1. **Dataset folder:** Click the **`...`** button to select your experiment folder (e.g. `ArabidopsisDataset`).
2. **Navigate:** Double-click folders. Progress bars show **To annotate**, **In progress**, and **Done** counts.
3. **Load a task:** Click any image marked **Missing** or **In Progress**. The tool loads the `.png` photo and model `.nii.gz` (plus `.json` if you saved before).

![Main interface of ChronoRoot Annotation Suite showing the file browser.](tutorialImages/1_main_interface.png) 
![Selecting a specific task folder from the database.](tutorialImages/2_selecting_a_task.png) 
![A loaded task displaying the raw root image and plant list.](tutorialImages/3_task_loaded.png)


## 2. Canvas Navigation & Controls

* **Pan:** Hold **Ctrl + Left-Click** and drag.
* **Zoom:** Use the **Mouse Wheel**.
* **Brush Size:** Adjust stroke thickness in the left panel.
* **Overlay Opacity:** See the raw root beneath painted masks.
* **Bounding Boxes:** Toggle boxes showing plant extents.

![Overlay opacity set to high, obscuring the raw root.](tutorialImages/4_opacity_slider_1.png) 
![Overlay opacity lowered to clearly see the raw root beneath the mask.](tutorialImages/5_opacity_slider_2.png) 
![Canvas view with plant bounding boxes toggled off.](tutorialImages/6_bounding_boxes_off.png)


## 3. The Plant-by-Plant Workflow: Examples

The model provides a starting point but will make mistakes. For every plant, cycle through:

1. **+ New / Merge / Split** — create, join, or separate plants.
2. **Shape Paint** — left = add, right = erase.
3. **Root Part Labels** — **Multi-Class Paint** (Main Root, Lateral Root, Hypocotyl, etc.).

### Scenario A: The Fused or Overlapping Plant (Splitting)

1. **Split:** Select the fused plant. **Split (Knife)** → draw a red line across the connection → **double-click** to cut.
2. **Fix Overlaps (Shape Paint):** Repaint shared crossing areas on each plant.
3. **Multi-Class Paint:** Assign biological labels to separated structures.

![Two distinct root systems merged into one ID.](tutorialImages/7_example_fused_plants.png) 
![Drawing the red cut line across the fusion.](tutorialImages/8_example_splitting.png) 
![Plants successfully separated into two distinct IDs.](tutorialImages/9_a_example_splitted.png) 
![Showing the structural overlap of the two plants.](tutorialImages/9c_overlapped_plants_example.png)
![Repainting the shared crossing pixels using Shape Paint.](tutorialImages/9_b_fixing_with_shape.png) 
![Applying the correct biological labels to the separated roots.](tutorialImages/9_d_fixing_multiclass.png) 

### Scenario B: The Broken Plant (Merging)

1. **Merge:** Shift+click fragments in the plant list → **Merge**.
2. **Shape Paint:** Connect gaps between fragments.
3. **Multi-Class Paint:** Label the full structure.

![A single root split into multiple fragments by the model.](tutorialImages/10_broken_plant.png)
![Selecting all fragment IDs in the left-hand list.](tutorialImages/11_dual_selection.png)
![The fragments successfully merged into a single Plant ID.](tutorialImages/12_a_merged.png)
![Painting](tutorialImages/12_b_fixing_with_painter.png) 
![Painting 2](tutorialImages/12_c_painter_green_adds.png) 
![Painting multiclass](tutorialImages/12_d_reviewing_multiclass.png)

### Scenario C: The Missing Plant (Creating)

1. **+ New** — create a plant the model missed.
2. **Shape Paint** — outline the plant (a wide stroke is fine for faint roots).
3. **Multi-Class Paint** — label seed, root, hypocotyl.

![An unannotated plant visible in the raw image.](tutorialImages/13_missing_plant_noticed.png)
![Clicking the Create New Plant button.](tutorialImages/14_creation_button.png) 
![The newly painted binary mask for the plant.](tutorialImages/15_binary_creation.png) 
![Applying semantic classes to the newly created plant.](tutorialImages/16_multiclass_labels.png)


## 4. Final Annotation Review

Before finishing, verify the whole plate:

1. Click **Review all labels** — all plants and root part labels at once.
2. Check: tight bounding boxes, no label bleed, crossings accounted for.
3. **Save Progress** (yellow) — pause and resume later (writes `.json`).
4. **Finish Annotation** (green) — marks the plate completed.

![Review all labels — whole-plate validation before Finish.](tutorialImages/17_final_validation.png)

### Automated Validation on Finish

If one Plant ID has **disconnected floating pieces**, Finish is blocked. Connect pieces with Shape Paint, or use **Split disconnected fragments** / Split tools to separate into distinct IDs.


## 5. Centerline Cleanup (optional)

Use the **Centerline Cleanup** tab when:

* You painted a **wide** mask in Annotation to capture faint roots.
* Masks look thick or uneven and you want them centered on the root.

**Skip this tab** if Annotation masks already look thin and accurate.

1. Select one plant.
2. Compare the four panels (grayscale, current mask, root-strength map, proposed refinement).
3. Adjust parameters if needed.
4. **Apply to this plant** — then **Save Progress** or **Finish Annotation** to persist.


## 6. Root Tracing Check (before Finish)

**Check every plant** on the **Root Tracing Check** tab before **Finish Annotation**.

### Hard problem: disconnected mask

If the plant mask is not **one connected piece**, tracing cannot proceed. Return to **Annotation**:

* **Merge** fragments, or **Shape Paint** to connect pieces.
* **Split disconnected fragments** if one ID should be several plants.

### Soft problem: wrong main-root path

The white **skeleton** may look fine, but the colored **main root path** may be wrong. The software could not place the automatic start/end correctly.

**Fix options:**

1. On the interactive graph: set **Start** (green) and **End** (blue) nodes; use yellow waypoints if needed → **Save root tracing for this plant**.
2. Return to Annotation and adjust Main Root / Lateral Root labels or outline.

When skeleton and main-root path look correct for every plant, proceed to **Finish Annotation**. You are done here unless you continue to trait measurement.


## 7. Measurement & Export (optional)

Skip this section unless you need phenotypic traits or batch comparisons.

Open the middle panel sub-tab **Plant Metadata** (beside Annotation Controls).

### Plate information & scale

* **Plate ID**, **Condition**, **Timepoint** — stored in exports and batch reports.
* **Calibration** — Scanner DPI, known plate height/width, or **Set via Measurement** on a known distance.
* **Check scale on image** — test the ruler without changing calibration.

### Per-plant identification

* **Plant** — internal ID (spatial order on plate).
* **Genotype** and **Plant #** — your experiment labels.
* Bulk-assign: select table rows → choose genotype → **Apply**.

### Run analysis

1. **Measure** — computes traits; opens **Phenomics Inspector**. Requires **Finish Annotation** first.
2. **Export** — writes trait outputs for this plate.


## 8. Phenomics Inspector & Batch Reports (optional)

### Phenomics Inspector

Read-only review after Measure.

### Batch Reports

Lists exported plates from your dataset. Check plates → **Load Selected Data** → configure plots and statistics → **Add to Report** → **Generate Report**.

Use the in-app **Help** button on the Batch Reports tab for statistical test details.

---

For a quick reference, see [README.md](../README.md) or [chronoroot.github.io](https://chronoroot.github.io/).
