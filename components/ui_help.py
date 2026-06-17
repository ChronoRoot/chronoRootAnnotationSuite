"""Centralized in-app help text and dialogs for ChronoRoot Annotation Suite."""

from PyQt5.QtWidgets import QMessageBox

HELP_WORKFLOW = """<b>How to use this app</b><br><br>
<b>Annotation path</b><br>
• Load a plate from the file browser.<br>
• <b>Annotation</b> — select a plant, fix masks and root-part labels.<br>
• <b>Centerline Cleanup</b> — compare panels; apply if refinement looks better.<br>
• <b>Root Tracing Check</b> — confirm skeleton and main-root path per plant.<br>
• <b>Finish Annotation</b> when every plant is done.<br><br>
<b>Traits path</b><br>
• <b>Plant Metadata</b> — plate fields, scale, genotypes.<br>
• <b>Measure</b>, then <b>Export</b>.<br>
• <b>Phenomics Inspector</b> — review per plant.<br>
• <b>Batch Reports</b> — compare plates across an experiment."""

HELP_ANNOTATION = """<b>Annotation</b><br><br>
<b>Pick a plant</b> in the plant list (Shift+click to select several).<br><br>
<b>Adjust the plant mask</b><br>
• <b>+ New</b> — paint in a missed plant.<br>
• <b>Merge</b> — join fragments (select 2+ plants first).<br>
• <b>Split (Knife)</b> — cut fused plants; double-click to finish the line.<br>
• <b>Split disconnected fragments</b> — one ID split into separate plants.<br>
• <b>Shape Paint</b> — left adds root, right erases.<br><br>
<b>Paint root-part labels</b><br>
• <b>Multi-Class Paint</b> — Main Root, Lateral Root, Hypocotyl, etc.<br><br>
<b>Check the whole plate</b><br>
• <b>Review all labels</b> — see every label at once.<br><br>
<b>Navigation:</b> Ctrl+drag to pan, wheel to zoom."""

HELP_FRANGI = """<b>Root Refinement & Cleanup</b><br><br>
Select a plant in the list. Compare the <b>Current Segmentation</b> with the <b>Proposed Refinement</b>.<br>
Tune parameters until the refinement panel matches the true root structure, then click <b>Apply to this plant</b>.<br><br>

<b>Core Processing Modes</b><br>
• <b>Centerline correction</b> — <b>On:</b> Uses Frangi vesselness to find and redraw roots based on the image heatmap. <b>Off:</b> Skips the heatmap and simply reshapes your <i>existing</i> mask to a uniform width.<br>
• <b>Dark roots</b> — <b>On:</b> Use for infrared or backlight pictures (dark roots on a light background). <b>Off:</b> Use for standard scanned images (light roots on dark agar).<br>
• <b>Allow disconnected</b> — <b>On:</b> Keeps every separate root fragment. <b>Off:</b> Discards disconnected floating pieces and keeps only the main connected root system.<br><br>

<b>Thresholds & Structure</b><br>
• <b>Thick</b> — The final uniform width of the root mask. Use <b>1</b> for standard/low-resolution images. Increase to <b>2 or 3</b> for high-resolution images.<br>
• <b>Core (Strong Conf.)</b> — The strict heatmap threshold (0–1). Only very clear, high-confidence root pixels pass. Raise this to drop background noise; lower it if the main root core is disappearing.<br>
• <b>Faint (Weak Conf.)</b> — The relaxed threshold (0–1). Weak pixels are kept <i>only</i> if they physically connect to a 'Core' pixel (Hysteresis). Lower this to recover faint lateral tips that branch off the main root.<br>
• <b>Search</b> — How many pixels outside the current mask to look for missing roots. Raise this when root tips are cut off too early.<br>
• <b>Bridge</b> — Maximum gap (in pixels) to jump across broken heatmap segments. Raise to connect "dotted" roots; lower if neighboring separate roots are merging together.<br>
• <b>Ignore</b> — Discards heatmap blobs smaller than this pixel count. Raise to remove salt-and-pepper noise; lower to keep tiny lateral roots.<br><br>

<b>Image Adjustments</b><br>
• <b>Image</b> — Channel used for the grayscale panel. "Red-Blue Avg" works best for typical plates. Switch if contrast is poor.<br>
• <b>Contrast</b> — Local contrast boost (CLAHE). Use if the image has uneven lighting.<br>
• <b>Smooth</b> — Blurs the grayscale panel to reduce noise. Use Light/Medium/High if the resulting heatmap looks too speckled.<br>
• <b>Class Checkboxes</b> — Only checked root classes are refined. Unchecked classes are copied perfectly from the current mask.<br><br>

Use <b>Save Progress</b> or <b>Finish Annotation</b> to write applied changes to disk."""

HELP_GRAPH = """<b>Root Tracing Check</b><br><br>
Select each plant and confirm skeleton and main-root path look correct.<br><br>
<b>Disconnected mask</b> — merge or connect in Annotation, or use Split disconnected fragments.<br><br>
<b>Wrong main-root path</b> — set Start (green) and End (blue) on the graph, or fix labels in Annotation.<br><br>
<b>Legend:</b> Green = start, Blue = end, Yellow = waypoint."""

HELP_METADATA = """<b>Plant Metadata</b><br><br>
Set plate ID, condition, timepoint, and image scale.<br>
Fill genotype and plant number in the table.<br><br>
Click <b>Measure</b> to compute traits (plate must be finished first).<br>
Click <b>Export</b> to write trait outputs for this plate."""

HELP_INSPECTOR = """<b>Phenomics Inspector</b><br><br>
Use the overlay options to compare labels, tracing, skeleton, or architecture.<br>
Adjust lateral emergence angle if needed.<br>
Click <b>Export SVG</b> to save the current view."""

HELP_BATCH = """<b>Batch Reports</b><br><br>
<b>Workflow:</b> check plates → Load Selected Data → tune the plot → Add to Report → Generate Report.<br><br>
<b>Compare / Within / And within:</b><br>
Example: Compare <i>genotype</i>, within each <i>timepoint</i>, and within each <i>condition</i>
tests whether genotypes differ at every timepoint, separately for each condition.<br><br>
<b>Statistical tests:</b><br>
• <b>Automatic</b> — Mann-Whitney U for 2 groups; Kruskal-Wallis + pairwise Mann-Whitney for 3+.<br>
• <b>Mann-Whitney U</b> — non-parametric; best for 2 independent groups.<br>
• <b>Kruskal-Wallis</b> — non-parametric; needs 3+ groups (2 groups use Mann-Whitney).<br>
• <b>One-way ANOVA</b> — parametric; needs 3+ groups and roughly normal data.<br><br>
<b>Empty results?</b> Usually only one genotype/condition in a stratum, too few plants, or missing values."""

# Backward-compatible alias for report_tab imports
REPORT_HELP_TEXT = HELP_BATCH


def show_help(parent, title, html_text):
    QMessageBox.information(parent, title, html_text)
