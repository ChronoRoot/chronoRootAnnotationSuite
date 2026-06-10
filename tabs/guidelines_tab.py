from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextBrowser

class GuidelinesTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setStyleSheet(
            "font-size: 14px; background-color: #fcfcfc; border: 1px solid #ccc; padding: 15px; line-height: 1.5;"
        )

        about_html = """
        <h2 style="color: #333; margin-bottom: 5px;">ChronoRoot Annotation Suite: User Guide</h2>

        <hr style="border: 0; border-top: 1px solid #ddd; margin: 15px 0;">

        <h3 style="color: #333;">Canvas Navigation</h3>
        <ul>
            <li><b>Pan:</b> Hold <b>Ctrl + Left-Click and Drag</b> (or use Middle-Click and Drag) to move around the image.</li>
            <li><b>Zoom:</b> Use the <b>Mouse Wheel</b> to zoom in and out.</li>
            <li><b>Brush Size:</b> Use the <b>Brush Size slider</b> in the left panel to adjust the thickness of the paint brush.</li>
        </ul>

        <h3 style="color: #333;">Stage 1: Fix Instances (Split & Merge)</h3>
        <p>The automatic model groups touching plants into a single instance. You must separate them manually.</p>
        <ul>
            <li><b>Merge:</b> If one plant is broken into pieces, hold <b>Shift + Click</b> in the list to select all its parts, then click <b>Merge Selected</b>. After merging, use the <b>Shape Paint</b> tool to fill in the gaps.</li>
            <li><b>Split:</b> If two plants are fused together, select the ID, click the <b>Split (Knife)</b> tool, draw a line across the connection, and double-click to cut. </li>
            <li><b>Creating Overlaps:</b> Plants often overlap physically (roots, hypocotyls, leaves). Because the Split tool divides pixels exclusively, you must recreate the overlap manually. After splitting, select one of the separated plants, click <b>Shape Paint</b>, and paint the shared area back in. Pixels can belong to multiple plants as long as they have different Plant IDs.</li>
        </ul>
        
        <h3 style="color: #333;">Stage 2: Fix Binary Shape</h3>
        <p>Define the exact outer limits of each individual plant.</p>
        <ul>
            <li>Select a plant and click <b>Shape Paint</b>.</li>
            <li><b>Left-Click</b> to paint missing parts.</li>
            <li><b>Right-Click</b> to erase background noise or incorrect areas.</li>
        </ul>

        <h3 style="color: #333;">Stage 3: Multi-Class Segmentation</h3>
        <p>Assign specific biological labels to the plant parts.</p>
        <ul>
            <li>Select a plant and click <b>Multi-Class Paint</b>.</li>
            <li>Select a biological class from the list (e.g., Main Root, Hypocotyl).</li>
            <li><b>Important:</b> You can only paint labels inside the area you defined in Stage 2. Always fix the binary shape first before painting biological parts.</li>
        </ul>
        
        <h3 style="color: #333;">Stage 4: Final Review</h3>
        <p>Verify your annotations across the entire image.</p>
        <ul>
            <li>Once all instances are corrected and labeled, click <b>Global Multi-Class View</b>.</li>
            <li>Use this view to perform a final visual check of all plants and their biological parts together before clicking <b>Finish Annotation</b>.</li>
        </ul>
        """
        
        self.text_browser.setHtml(about_html)
        layout.addWidget(self.text_browser)