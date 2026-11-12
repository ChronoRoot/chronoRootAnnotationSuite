# ChronoRoot Annotation Suite: Official User Guide

Manual annotation of complex root systems is a demanding task. This tool provides a biology-first workflow for correcting model segmentations, validating root tracing, and exporting phenotypic measurements.

The plant-by-plant annotation scenarios (split, merge, create) use screenshots from the Etiolation plate `2024-08-03_20-03-24_1`. The side panel in those figures may look slightly older; button names in the text match the **current** app. All other figures are current-UI screenshots.


## 0. Input Files per Plate

Each plate frame uses the **same base name** in one folder:

| File | Description |
|------|-------------|
| `PlateName.png` | Scanner photo of the petri dish |
| `PlateName.nii.gz` | Automatic segmentation from your model — the starting draft, not ground truth |
| `PlateName.json` | Your saved edits (created when you click **Save Progress** or **Finish Annotation**) |

After measurement and export:

| File | Description |
|------|-------------|
| `PlateName_Metrics.json` | Per-plant morphometrics and plate metadata |
| `PlateName_Topology.rsml` | Root architecture (RSML format) |

The model may miss plants or fuse neighbors. You correct those in **Annotation**. If you paint a **wide** mask to capture faint roots, use **Centerline Cleanup** (optional) to thin it before the tracing check.

> Click **Finish Annotation** when mask and tracing work are done. Later sections cover trait measurement and batch comparison.


## 1. Getting Started: The File Browser

1. **Dataset folder:** Click the **`...`** button to select your experiment folder.
2. **Navigate:** Double-click folders. Folder summaries show counts by status.
3. **Statuses:** **Without Annotation**, **Pending**, **In Progress**, **Completed**.
4. **Hide non-annotated images** — hides photo-only plates.
5. **Load a task:** Click an image.

![File browser showing plate statuses.](tutorialImages/01_browser_folder.png)
![Annotation workspace with a loaded plate, plant list, and tools.](tutorialImages/02_annotation_overview.png)


## 2. Canvas Navigation & Controls

* **Pan:** Hold **Ctrl + Left-Click** and drag.
* **Zoom:** Use the **Mouse Wheel**.
* **Brush Size** / **Overlay Opacity** — paint width and mask transparency.
* **Bounding Boxes** — plant extents.
* **Canvas Label Settings** — show **Plant #** / **Genotype**, choose Text or Index, and set **Font size** (try **18**) so labels stay readable when zoomed in.

![Overlay opacity set high.](tutorialImages/03_opacity_high.png)
![Overlay opacity lowered to see the raw root beneath the mask.](tutorialImages/04_opacity_low.png)
![Canvas with plant bounding boxes turned off.](tutorialImages/05_bboxes_off.png)


## 3. The Plant-by-Plant Workflow: Examples

For every plant, cycle through: **+ New / Merge / Split** → **Shape Paint** → **Multi-Class Paint**.

Plant list: **+ New**, **Merge**, **Delete**, **Reset Plant IDs**.

Tools: **Clear Selection**, **Bounding Boxes**, **Review multiclass**, **Shape Paint**, **Split (Knife)**, **Split disconnected fragments**, **Undo Last Action**, **Multi-Class Paint**.

### Scenario A: The Fused or Overlapping Plant (Splitting)

1. **Split (Knife)** — draw a red line across the fusion → **double-click** to cut.
2. **Shape Paint** — fix shared crossing pixels.
3. **Multi-Class Paint** — assign biological labels.

![Two distinct root systems merged into one ID.](tutorialImages/scenario_a_01_fused.png)
![Drawing the red cut line across the fusion with Split (Knife).](tutorialImages/scenario_a_02_knife.png)
![Plants successfully separated into two distinct IDs.](tutorialImages/scenario_a_03_split.png)
![Structural overlap remaining after the split.](tutorialImages/scenario_a_04_overlap.png)
![Repainting shared crossing pixels with Shape Paint.](tutorialImages/scenario_a_05_shape_paint.png)
![Applying biological labels with Multi-Class Paint.](tutorialImages/scenario_a_06_multiclass.png)

### Scenario B: The Broken Plant (Merging)

1. Shift+click fragments → **Merge**.
2. **Shape Paint** to connect gaps.
3. **Multi-Class Paint**.

![A single root split into multiple fragments by the model.](tutorialImages/scenario_b_01_broken.png)
![Selecting fragment IDs in the plant list.](tutorialImages/scenario_b_02_select.png)
![Fragments merged into a single Plant ID.](tutorialImages/scenario_b_03_merged.png)
![Connecting gaps with Shape Paint.](tutorialImages/scenario_b_04_shape_paint.png)
![Left-click adds mask (green stroke).](tutorialImages/scenario_b_05_paint_add.png)
![Reviewing multiclass labels on the merged plant.](tutorialImages/scenario_b_06_multiclass.png)

### Scenario C: The Missing Plant (Creating)

1. **+ New** → **Shape Paint** → **Multi-Class Paint**.

![An unannotated plant visible in the raw image.](tutorialImages/scenario_c_01_missing.png)
![Creating a new plant from the plant list.](tutorialImages/scenario_c_02_new.png)
![The newly painted binary mask.](tutorialImages/scenario_c_03_shape.png)
![Applying semantic classes to the new plant.](tutorialImages/scenario_c_04_multiclass.png)


## 4. Reset Plant IDs & Final Review

**Reset Plant IDs** re-numbers plants **left-to-right in rows**. Genotypes stay on the same physical plant. Use it before exporting genotypes for other timepoints of the same plate.

![Plant numbers intentionally out of order before reset.](tutorialImages/06_before_reset_plant_ids.png)
![After Reset Plant IDs: numbers follow left-to-right row order.](tutorialImages/07_after_reset_plant_ids.png)

1. **Review multiclass** — whole-plate labels.
2. **Save Progress** or **Finish Annotation**.

![Review multiclass — whole-plate label check.](tutorialImages/08_review_multiclass.png)
![Whole-plate validation view before Finish.](tutorialImages/scenario_review_all.png)

If one Plant ID has disconnected pieces, Finish is blocked — connect with **Shape Paint** or split into separate IDs.


## 5. Centerline Cleanup (optional) — both image types

Use when masks are thick/uneven. **Skip** if Annotation masks already look thin.

| Image source | Typical look | **Dark Roots** | Parameter tip |
|--------------|--------------|----------------|---------------|
| ChronoRoot / ManyRoots | Dark roots on light background | **On** | **Min/Max Root Radius** often **1–10**; set **Contrast Enhancement** and **Smooth Filter** |
| ScanPlates / scanner agar | Light roots on dark agar | **Off** | Smaller radii often suffice; still check contrast/smoothing |

1. Select a plant → set **Dark Roots** correctly.
2. Tune **Min/Max Root Radius**, then **Contrast Enhancement** and **Smooth Filter** until the root-strength map and proposed refinement look right.
3. **Apply to this plant** → **Save Progress** / **Finish Annotation**.

### ChronoRoot / ManyRoots (Dark Roots On, Max Radius 10)

Tune **Image Adjustments** carefully: **Contrast Enhancement** (e.g. Before Smoothing) and **Smooth Filter** (e.g. Medium) matter as much as the radius range for recovering faint ChronoRoot roots. Example settings also use a larger **Radius Step**, **Final Width**, and **Bridge Gaps** when roots are thin or broken.

![Centerline Cleanup on ManyRoots with Dark Roots on; contrast enhancement and smoothing enabled so the proposed centerline follows the root.](tutorialImages/09_manyroots_centerline_dark_on.png)

### ScanPlates-style plate (Dark Roots Off)

![Centerline Cleanup on a scanner plate with Dark Roots off.](tutorialImages/11_scanplates_centerline_dark_off.png)


## 6. Root Tracing Check — review every plant

**Root Tracing Check** builds the skeleton and main-root path from your masks. It is also a place to **clean up the segmentation so the main root is one continuous path**: wrong colors / broken main roots can be corrected by setting Start/End nodes (and waypoints) and clicking **Save root tracing for this plant**, which writes the traced colors back into the plant mask.

**Review every plant** on the plate. If something looks wrong, fix it here (or return to Annotation) and **Save root tracing for this plant** before moving on.

Disconnected masks cannot be traced — fix connectivity in Annotation first.

### ChronoRoot / ManyRoots

![Root Tracing Check on a ManyRoots plant.](tutorialImages/10_manyroots_root_tracing.png)

### ScanPlates-style plate

![Root Tracing Check on the scanner plate used for RSML fixes.](tutorialImages/12_scanplates_root_tracing.png)


## 7. RSML Lateral Root Fixes (optional)

Skip when the automatic tracing is already right. Use it when a lateral is broken into a tertiary past a cross, or when parallel laterals were fused.

### Example: merge Lateral 3 with its tertiary

On plant 1, automatic tracing sometimes splits a continuous lateral into **Lateral 3** plus **Lateral 3 · tertiary 1**. Merge them with **Connect**:

1. Open **RSML Lateral Root Fixes**, select the plant, enable **Each lateral root** (and **Show direction** / **Show nodes** while debugging).
2. Click **Lateral 3** in the list (or on the canvas). Zoom in on the break.
3. **Shift-click** or **Ctrl-click** the tertiary segment so the legend shows **1 Target: Lateral 3** and **2 Source: … tertiary …**.
4. Click **Connect** (pieces must share an endpoint).
5. Click **Keep fixes for this plant** (memory only). Measurement / **Save Results** write `Topology.rsml`.

**Reset** rebuilds the automatic tracing if you want to start over.

![Lateral 3 selected (zoomed).](tutorialImages/13_rsml_lateral3_selected.png)
![Lateral 3 + its tertiary selected — click Connect.](tutorialImages/14_rsml_lateral3_and_tertiary_selected.png)
![After Connect: one continuous Lateral 3.](tutorialImages/15_rsml_after_connect_lateral3.png)
![Keep fixes for this plant stores the curated topology in memory.](tutorialImages/15b_rsml_keep_fixes.png)


## 8. Plant Metadata, Genotypes & Measurement

Open the middle panel sub-tab **Plant Metadata**. Keep **Canvas Label Settings → Font size** at **18** so plant numbers and genotypes stay readable when zoomed in on a few plants (not the full plate).

### Plate information & scale

**Plate ID**, **Condition**, **Timepoint**, and **Calibration** (Scanner DPI, known size, or **Set via Measurement**). Use **Check scale on image** to verify the ruler.

### Fill plate metadata for every plate

**Plate ID**, **Condition**, **Timepoint**, and scale must be filled **per plate** (they travel into `_Metrics.json` and Batch Reports).

![Empty Plate ID / Condition / Timepoint fields and blank Genotype column — fill plate fields here; canvas shows plant boxes (Nº 1, Nº 2 ignored, …) without genotype names yet.](tutorialImages/17b_metadata_plate_fields_empty.png)

### Genotype List

**Genotype List…** manages saved genotype names for the dropdowns (examples: `Genotype_A`, `Genotype_B`, `Genotype_C`).

![Manage Saved Genotypes dialog listing Genotype_A, Genotype_B, Genotype_C — use + Add Row / Save_Close to keep the list.](tutorialImages/17_genotype_list_dialog.png)

### Assigning genotypes on the plate

Select table rows → **Bulk genotype** → **Apply** (or edit cells one by one). Canvas labels update immediately when **Show → Genotype** is on.

![Before assign: Genotype column blank; canvas labels are plant numbers only (Nº 1, Nº 2 ignored, Nº 3) at font size 18, zoomed on the top plants.](tutorialImages/18_metadata_genotypes_before.png)
![After Bulk Apply Genotype_A: every row shows Genotype_A; canvas reads “Nº 1 Genotype_A”, “Nº 2 Genotype_A (ignored)”, “Nº 3 Genotype_A”.](tutorialImages/19_metadata_genotypes_assigned.png)
![After Bulk Apply Genotype_B on plants 2–4: table rows 2–4 show Genotype_B (plant 1 still Genotype_A); canvas shows “Nº 1 Genotype_A” next to “Nº 2 Genotype_B (ignored)”.](tutorialImages/19b_metadata_bulk_apply_genotype_b.png)

### Import / Export Genotypes across timepoints

For the **same physical plate** at several timepoints:

1. **Reset Plant IDs** so **Plant #** matches left-to-right order on every timepoint.
2. **Export Genotypes** → `{ "1": "Genotype_A", … }`.
3. On another timepoint (or after clearing), **Import Genotypes**.

**Plant # must match.** Unmatched numbers are reported — add missing seeds with **+ Non-Germinated** (or fix Plant #), then import again.

![Genotypes cleared again: Genotype column blank and canvas back to plant numbers only — ready to click Import Genotypes.](tutorialImages/19c_metadata_cleared_before_import.png)
![After Import Genotypes: Plant # → genotype map restored (Genotype_A / Genotype_B as exported); canvas labels match the table again.](tutorialImages/20_metadata_after_import.png)

Also: **+ Non-Germinated** (traits = 0) and **Ignore** (excluded from results).

### Run analysis

1. **Run Measurements** — computes traits (plate should be finished first). The app then switches to the **Phenomics Inspector** tab (covered in the next section).
2. **Save Results** — writes `_Metrics.json` / topology for this plate.

After that, analyzed plates show **Analysis: Analyzed** in the left file browser (Annotation still open — this is not the Inspector yet):

![Left file browser on the phenomics folder: green checkmarks with Annotation: Completed and Analysis: Analyzed; Annotation tab still active with plants 1–3 zoomed.](tutorialImages/16_phenomics_browser.png)


## 9. Phenomics Inspector

Review traits and overlays per plant after **Run Measurements** (this is the Inspector tab, not the Annotation workspace).

Toggle **Show Convex Hull** and angle modes (**Hide Angles**, **Lateral emergence angle (2 mm)**, **Overall Tip Angles**). Use **Export Editable Vector (.SVG)** to save qualitative figures for papers and slides.

![Inspector without convex hull.](tutorialImages/21_inspector_no_hull.png)
![Inspector with convex hull.](tutorialImages/22_inspector_with_hull.png)
![Inspector with lateral emergence angles.](tutorialImages/23_inspector_with_angles.png)
![Another plant on the same plate.](tutorialImages/24_inspector_other_plant.png)


## 10. Batch Reports

1. Check plates → **Load Selected Data**.
2. Choose **Plot Type**, metric, axes, hue, and **Statistics** (compare factor, test, alpha).
3. **Add to Report** for each figure.
4. **Generate Report** writes queued figures + stats files.
5. **Export Table (.CSV)** for your own analysis.

![Batch Reports plate picker.](tutorialImages/25_batch_reports_list.png)

### Different plot types and metrics

![Box plot — Main Root Length by genotype.](tutorialImages/26_batch_box_main_root.png)
![Violin plot — Lateral Root Length by genotype.](tutorialImages/27_batch_violin_lateral.png)
![Swarm plot — Convex Hull Area, with statistics panel.](tutorialImages/28_batch_swarm_hull_stats.png)
![Qualitative Atlas — average root-system shapes by genotype (architecture overview).](tutorialImages/29_batch_qualitative_atlas.png)

### What gets exported

**Generate Report** creates a folder with SVG figures, aggregated data, and statistical summaries:

![Contents of the exported report folder.](tutorialImages/30_batch_exported_folder.png)

**Export Table (.CSV)** / master aggregated CSV columns (plant traits + plate metadata) for external analysis:

![Preview of exported table columns.](tutorialImages/31_batch_exported_table_columns.png)

Use the in-app **Help** on Batch Reports for statistical test details.

---

For a quick reference, see [README.md](../README.md) or [chronoroot.github.io](https://chronoroot.github.io/).
