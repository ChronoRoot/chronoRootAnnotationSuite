from PyQt5.QtWidgets import QTabWidget
from PyQt5.QtCore import pyqtSignal

# Import your existing tabs
from tabs.review_tab import ReviewCanvasTab
from tabs.frangi_tab import FrangiCanvasTab
from tabs.graph_tab import GraphCanvasTab
from tabs.inspector_tab import PhenomicsInspectorTab
from tabs.report_tab import ComparisonReportTab
from tabs.guidelines_tab import GuidelinesTab
from tabs.about_tab import AboutTab

class WorkspaceManager(QTabWidget):
    workspace_changed = pyqtSignal(int)
    
    # NEW: Added main_window argument
    def __init__(self, main_window, shared_model):
        super().__init__()
        self.main_window = main_window
        self.model = shared_model
        self.init_workspaces()
        
    def init_workspaces(self):
        # 1. Pixel Annotation
        self.canvas_review = ReviewCanvasTab(self.model)
        self.addTab(self.canvas_review, "1. Annotation & Review")
        
        # 2. Skeleton Refinement
        self.canvas_frangi = FrangiCanvasTab(self.model)
        self.addTab(self.canvas_frangi, "2. Centerline Refinement")
        
        # 3. Topology Graphing
        self.canvas_graph = GraphCanvasTab(self.model)
        self.addTab(self.canvas_graph, "3. Topology Graph")
        
        # 4. Phenomics Inspector
        self.inspector_tab = PhenomicsInspectorTab()
        self.addTab(self.inspector_tab, "4. Phenomics Inspector")
        
        # 5. Batch Reports (NEW: Passing the main_window reference)
        self.report_tab = ComparisonReportTab(self.main_window)
        self.addTab(self.report_tab, "5. Batch Reports")

        self.guidelines_tab = GuidelinesTab()
        self.addTab(self.guidelines_tab, "Guidelines")

        self.about_tab = AboutTab()
        self.addTab(self.about_tab, "About")
        
        # Connect tab change to internal handler
        self.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        self.workspace_changed.emit(index)