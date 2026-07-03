"""Centralized in-app help text and dialogs for ChronoRoot Annotation Suite."""

from PyQt5.QtWidgets import QMessageBox

HELP_WORKFLOW = """<b>How to use this app</b><br><br>
<b>File browser</b><br>
• Double-click folders to browse; click an image to open it.<br>
• Plate status: <b>Without Annotation</b> (photo only), <b>Pending</b> (.nii.gz only),
<b>In Progress</b>, or <b>Completed</b> (.json present).<br>
• <b>Hide non-annotated images</b> — hides photo-only plates (no .json and no .nii.gz).<br>
• <b>Save Progress</b> — write edits and resume later.<br>
• <b>Finish Annotation</b> — mark the plate completed.<br><br>
<b>Annotation path</b><br>
• <b>Annotation</b> — select a plant, fix masks and root-part labels.<br>
• <b>Centerline Cleanup</b> — compare panels; <b>Apply to this plant</b> if refinement looks better.<br>
• <b>Root Tracing Check</b> — confirm skeleton and main-root path per plant.<br><br>
<b>Traits path</b><br>
• <b>Plant Metadata</b> — plate fields, scale, genotypes.<br>
• <b>Measure</b>, then <b>Export</b>.<br>
• <b>Phenomics Inspector</b> — review per plant.<br>
• <b>Batch Reports</b> — compare plates across an experiment."""

HELP_ANNOTATION = """<b>Annotation</b><br><br>
<b>Plant list</b><br>
• <b>+ New</b> — paint in a missed plant.<br>
• <b>Merge</b> — join fragments (select 2+ plants first).<br>
• <b>Delete</b> — remove a plant.<br><br>
<b>Tools</b><br>
• <b>Clear Selection</b> — deselect all plants.<br>
• <b>Bounding Boxes</b> — show or hide plant boxes.<br>
• <b>Review all labels</b> — see every label on the plate at once.<br>
• <b>Shape Paint</b> — left adds outline, right erases.<br>
• <b>Split (Knife)</b> — cut fused plants; double-click to finish the line.<br>
• <b>Split disconnected fragments</b> — one ID split into separate plants.<br>
• <b>Undo Last Action</b> — undo the last paint, split, or merge.<br>
• <b>Multi-Class Paint</b> — Main Root, Lateral Root, Hypocotyl, and other root-part labels.<br><br>
<b>Brush Size</b> and <b>Overlay Opacity</b> — adjust paint brush and mask transparency.<br><br>
<b>Navigation:</b> Ctrl+drag to pan, wheel to zoom."""

HELP_FRANGI = """<b>Root Cleanup Tools</b><br><br>
Select a plant in the list. Compare <b>Current Segmentation</b> with <b>Proposed Refinement</b>.
Tune parameters, then click <b>Apply to this plant</b>.<br><br>

<b>Modes</b><br>
• <b>Apply Correction</b> — <b>On:</b> uses the root-strength map (Frangi) to redraw roots.
<b>Off:</b> reshapes the existing mask to uniform width only.<br>
• <b>Dark Roots</b> — <b>On:</b> infrared/backlight (dark roots on light background).
<b>Off:</b> standard scans (light roots on dark agar).<br>
• <b>Keep Disconnected Pieces</b> — <b>On:</b> keep all floating fragments.
<b>Off:</b> keep only the main connected root system.<br><br>

<b>Parameters</b><br>
• <b>Min Root Radius</b> / <b>Max Root Radius</b> — root width range (pixels) for the heatmap.
Raise on high-resolution images.<br>
• <b>Radius Step</b> — step between scales in the Frangi filter; increase to speed up processing.<br>
• <b>Final Width</b> — output mask width in pixels (use 1 for standard resolution, 2–3 for high-res).<br>
• <b>Search Range</b> — pixels outside the current mask to search for missing root tips.<br>
• <b>Bridge Gaps</b> — maximum gap (pixels) to connect broken heatmap segments.<br>
• <b>Core Thresh</b> — strict threshold (0–1) for solid root tissue; raise to drop noise.<br>
• <b>Faint Thresh</b> — relaxed threshold (0–1); faint pixels kept only if connected to core (hysteresis).<br><br>

<b>Image Adjustments</b><br>
• <b>Image Channel</b> — grayscale source; Red-Blue Avg suits most scanned plates.<br>
• <b>Contrast Enhancement</b> — local CLAHE for uneven lighting.<br>
• <b>Smooth Filter</b> — blur before heatmap generation (None / Light / Medium / High).<br><br>

<b>Classes to Refine</b> — checked classes are refined; unchecked classes are copied unchanged
(Main Root, Lateral Root, Seed, Hypocotyl, Leaves/Aerial, Petiole).<br><br>

Use <b>Save Progress</b> or <b>Finish Annotation</b> to write applied changes to disk."""

HELP_GRAPH = """<b>Root tracing check</b><br><br>
Select each plant and confirm skeleton and main-root path look correct.<br><br>

<b>Graph Settings</b><br>
• <b>Prune Iterations</b> — remove short skeleton spurs (higher = more pruning).<br>
• <b>Dilation Base Thick</b> — thickness when applying graph colors to the mask.<br><br>

<b>Node Click Interaction</b><br>
• <b>Set Start Node (Green)</b> — click to place the root start.<br>
• <b>Set End Node (Blue)</b> — click to place the root end.<br>
• <b>Toggle Waypoint (Yellow)</b> — click to add or remove waypoints.<br>
• <b>Reset Nodes to Auto</b> — clear manual nodes and use automatic detection.<br><br>

<b>Classes to Graph</b> — which root classes participate in tracing.<br><br>

<b>Legend:</b> Green = start, Blue = end, Yellow = waypoint.<br><br>

Disconnected masks — use <b>Split disconnected fragments</b> or fix in Annotation.<br>
Click <b>Save root tracing for this plant</b> to commit colors to the mask."""

HELP_METADATA = """<b>Plate information &amp; scale</b><br><br>
Fill plate ID, condition, timepoint, and image scale.<br>
Enter genotype and plant number in the table.<br><br>

<b>Calibration</b> — Scanner DPI, Known Image Height (cm), Known Image Width (cm),
or Custom Ratio (px/cm).<br>
• <b>Set via Measurement</b> — draw a line on the image to set scale.<br>
• <b>Check scale on image</b> — verify the ruler overlay.<br><br>

<b>Genotype List…</b> — manage saved genotypes.<br>
<b>Auto Renumber</b> — reorder plant numbers left-to-right.<br><br>

Click <b>Measure</b> to compute traits (plate should be finished first).<br>
Click <b>Export</b> to write trait outputs for this plate."""

HELP_INSPECTOR = """<b>Phenomics review</b><br><br>

<b>Root Representation</b><br>
• <b>Colored root parts (annotation)</b> — semantic labels from Annotation.<br>
• <b>Colored skeleton</b> — mask pixels validated by Root Tracing Check.<br>
• <b>RSML architecture preview</b> — topology overlay.<br>
• <b>Raw image only</b> — scanner image without overlays.<br><br>

<b>Biological Features</b><br>
• <b>Hide Angles</b>, <b>Lateral emergence angle (2 mm)</b>, or <b>Overall Tip Angles</b>.<br>
• <b>Show Convex Hull</b> — draw the convex hull around the root system.<br><br>

<b>Render Settings</b><br>
• <b>Line/Mask Width</b> — stroke width for overlays.<br><br>

Click <b>Export Editable Vector (.SVG)</b> to save the current view."""

HELP_BATCH = """<b>Batch Reports</b><br><br>
<b>Workflow:</b> check plates → <b>Load Selected Data</b> → tune the plot →
<b>Add to Report</b> → <b>Generate Report</b>.<br><br>

<b>Plate picker</b><br>
• <b>Open Task</b> — jump to a plate in Annotation.<br>
• <b>Load Selected Data</b> — load checked metrics files into the report.<br><br>

<b>Plot</b><br>
• <b>Plot Type</b>, <b>Line Error Bars</b>, <b>Y-Axis (Metric)</b>,
<b>X-Axis (Group)</b>, <b>Hue (Color)</b>.<br>
• Qualitative Atlas uses <b>Atlas Columns</b> (Condition, Genotype, or Condition | Genotype).<br><br>

<b>Statistics</b><br>
• <b>Compare groups by:</b> main factor (e.g. genotype).<br>
• <b>Within each:</b> optional stratum (e.g. timepoint).<br>
• <b>And within:</b> optional nested stratum (e.g. condition).<br>
• <b>Test:</b> Automatic, Mann-Whitney U, Kruskal-Wallis, or One-way ANOVA.<br>
• <b>Alpha:</b> significance level.<br><br>

<b>Report queue:</b> <b>Add to Report</b>, <b>Remove</b>, then <b>Generate Report</b>.<br>
<b>Export Table (.CSV)</b> — export the aggregated dataset.<br><br>

<b>Empty results?</b> Usually only one genotype/condition in a stratum, too few plants, or missing values."""

# Backward-compatible alias for report_tab imports
REPORT_HELP_TEXT = HELP_BATCH


def show_help(parent, title, html_text):
    QMessageBox.information(parent, title, html_text)
