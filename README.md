# ChronoRoot Annotation Suite

An open-source desktop tool for correcting automatic root segmentations on petri-dish scanner images, and optionally measuring and comparing root traits across experiments.

**Website:** [chronoroot.github.io](https://chronoroot.github.io/) — full user guide and documentation.

## Annotation workflow

1. **Annotation** — fix plants, shapes, and root-part labels.
2. **Centerline Cleanup** — compare panels; apply if refinement looks better.
3. **Root Tracing Check** — confirm each plant traces correctly.
4. **Finish Annotation** — done.

## Traits workflow

1. **Plant Metadata** — plate info, image scale, genotypes.
2. **Measure** then **Export**.
3. **Phenomics Inspector** — review per plant.
4. **Batch Reports** — compare plates.

Measure requires the plate to be finished first.

## Installation

You can install the ChronoRoot Annotation Suite using either an Apptainer/Singularity container (recommended for Linux systems to avoid local python dependency conflicts) or directly inside a Conda environment.

### Option 1: Apptainer/Singularity (Recommended)
This installer builds an isolated container image (~800 MB) from the `image.def` file, creates a wrapper script to handle desktop display bindings, and registers a desktop shortcut.

Run the following command:
```bash
wget https://raw.githubusercontent.com/ChronoRoot/chronoRootAnnotationSuite/main/install_apptainer.sh && chmod +x install_apptainer.sh && ./install_apptainer.sh
```

### Option 2: Conda Environment
This installer sets up a local Conda environment named `chronoRootAnnotation` with all package dependencies (PyQt5, OpenCV, NumPy, NetworkX, SciPy, etc.), builds a launcher script, and registers a desktop shortcut.

Run the following command:
```bash
wget https://raw.githubusercontent.com/ChronoRoot/chronoRootAnnotationSuite/main/install_conda.sh && chmod +x install_conda.sh && ./install_conda.sh
```

## Launch

After installation, you can launch the application in three ways:

1. **System Menu**: Search for **ChronoRoot Annotation Suite** in your desktop application launcher/applications menu.
2. **Wrapper Script**: Run the generated execution script (installed by default to `~/.local/chronoroot-annotation-suite/ChronoRootAnnotationSuite.sh`).
3. **Manual Launch**:
   - For Container: Run `./main.sh` from the repository root.
   - For Conda: Activate the conda environment and run python:
     ```bash
     conda activate chronoRootAnnotation
     python main.py
     ```

**Config Location:** `~/.config/chronoRootAnnotationSuite/config.json`

## Input and output files

See [Tutorial/tutorial.md](Tutorial/tutorial.md) for file formats (scanner image, model segmentation, saved annotations, and analysis exports).

## License and updates

See [chronoroot.github.io](https://chronoroot.github.io/). In-app **About → Check for Updates** runs `git pull` on git checkouts.
