# FFAST - Force Field Analysis and Screening Tool

A Python GUI application for analyzing and visualizing Machine Learning Force Field (MLFF) models. FFAST provides interactive tools for comparing predictions from different models against ground truth data, with rich visualization capabilities including error analysis plots and a 3D molecular viewer.

**Key Features:**
- Comprehensive error analysis tools (basic, atomic, total force, subsystem, cluster, scatter, gyration)
- Interactive 3D molecular visualization ("Loupe" viewer) with geometry measurement
- Full support for variable-sized molecular datasets
- Energy shift correction (subtract mean energy offset) across all energy error plots
- Scriptable headless mode for running predictions on remote machines without a GUI
- Dynamic sub-dataset creation from plot zoom/selection
- Automatic caching of expensive computations via MD5 fingerprinting

**Please cite:** Fonseca G, Poltavsky I, Tkatchenko A. *J Chem Theory Comput.* 2023;19(23):8706-8717. [DOI: 10.1021/acs.jctc.3c00985](https://doi.org/10.1021/acs.jctc.3c00985)

---

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Using uv (Recommended)](#using-uv-recommended)
  - [Using pip](#using-pip)
  - [Install Model Support (Optional)](#install-model-support-optional)
  - [Verify Installation](#verify-installation)
- [Quick Start](#quick-start)
- [Features](#features)
  - [Model Support](#model-support)
  - [Dataset Support](#dataset-support)
  - [3D Molecular Viewer (Loupe)](#3d-molecular-viewer-loupe)
  - [Error Analysis Tools](#error-analysis-tools)
  - [Advanced Features](#advanced-features)
  - [Keyboard Shortcuts](#keyboard-shortcuts)
- [Usage Guide](#usage-guide)
  - [Working with Datasets](#working-with-datasets)
  - [Working with Models](#working-with-models)
  - [Using the Loupe 3D Viewer](#using-the-loupe-3d-viewer)
  - [Error Analysis Workflows](#error-analysis-workflows)
  - [Headless Batch Processing](#headless-batch-processing)
- [Example Workflow](#example-workflow)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Installation

### Prerequisites

- **Python 3.9-3.11** (Python 3.12+ is not compatible with PySide6 6.4.2 and will cause a segmentation fault)
- **Supported OS**: Linux, macOS, Windows

### Using uv (Recommended)

```bash
# Create a virtual environment and install all dependencies
uv venv --python 3.11
uv sync
```

### Using pip

```bash
# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install FFAST and all core dependencies
pip install -e .
```

<!-- ### Install Model Support (Optional)

Install packages for the ML models you plan to use:

```bash
pip install sgdml        # sGDML
pip install schnetpack   # SchNet
pip install mace-torch   # MACE
pip install nequip       # Nequip
pip install spookynet    # SpookyNet
``` -->

### Verify Installation

```bash
python main.py
```

If the GUI opens successfully, installation is complete.

---

## Quick Start

### 1. Launch FFAST

```bash
python main.py
```

Optionally specify a working directory for file dialogs:
```bash
python main.py --workdir /path/to/your/data
```

### 2. Load a Dataset

- Menu: File > Load Dataset (or `Ctrl+d`)
- Supported formats:
  - sGDML `.npz` files (with `R`, `E`, `F`, `z` keys) for fixed-size datasets (same system per configuration)
  - ASE-compatible formats (`.db`, `.extxyz`, `.traj`, `.xyz`, and others)
  - Variable-sized datasets (different atom counts per configuration) are automatically detected

The dataset appears in the left sidebar.

### 3. Load a Pre-computed Predictions

<!-- **Method A: Load a trained model file**
- Menu: File > Load Model (or `Ctrl+m`)
- Select your model file (`.model`, `.pth`, `.npz`, etc.)
- Model type is auto-detected -->

<!-- **Method B: Load pre-computed predictions** -->
- Menu: File > Load Prediction (or `Ctrl+p`)
- Select an `.npz` file with `E` (energies) and `F` (forces) keys
- Select the corresponding dataset from the dropdown

### 4. Explore Error Analysis

Once a model and dataset are loaded:
- Click the **Basic Errors** tab to see energy and force MAE/RMSE timelines, distributions, and metric tables
- Explore other tabs: Atomic Errors, Scatter Errors, Total Force Errors, Subsystem Errors, Cluster Error, Gyration

### 5. Open the 3D Viewer

- Menu: Loupe > New (or `Ctrl+n`)
- Select your dataset from the dropdown
- Use the slider to navigate through configurations
- Left-drag to rotate, right-drag to pan, scroll to zoom

---

## Features

### Model Support

<!-- - **Supported models**: sGDML, MACE, Nequip, SchNet, SpookyNet -->
<!-- - **Custom predictions**: Load pre-computed energies/forces from `.npz` files -->
<!-- - **Ghost models**: When loading a saved session, models are reconstructed from cached predictions if the original model file is unavailable -->
<!-- - **Model comparison**: Load multiple models and compare side-by-side with automatic color coding -->
- **Zero model**: Load a reference model that predicts zero for all outputs (File > Load Zero Model, or `Ctrl+0`)
   - Used for quick check of suspicious energy/force ranges in the dastaset.

### Dataset Support

- **sGDML format**: `.npz` files with `R`, `E`, `F`, `z` keys
- **ASE formats**: `.db`, `.extxyz`, `.traj`, `.xyz`, and all other ASE-supported formats
- **Variable-sized molecules**: Full support for datasets with different atom counts per configuration, automatically detected on load
- **Unit cells**: Periodic boundary conditions are supported when present in the data

### 3D Molecular Viewer (Loupe)

Interactive 3D visualization with:
- **Atom rendering**: Customizable colors (by element, force error, mean force error, or displacement) and adjustable sizes
- **Bond visualization**: Dynamic bond detection with adjustable distance cutoff, or fixed bonds
- **Force vectors**: Display force arrows with adjustable length, normalization, and temporal averaging
- **Unit cells**: Visualize periodic boundary conditions
- **Geometry measurement**: Measure distances (2 atoms), angles (3 atoms), and dihedral angles (4 atoms) interactively
- **Atom alignment**: Align structures using a 3-atom reference frame
- **Atom indices**: Overlay atom index labels in the 3D view
- **XYZ axes**: Display orientation axes in the viewport corner
- **Camera controls**: Manual positioning, field of view adjustment, center-of-mass tracking, save/load camera positions
- **Atom filtering**: Select specific atoms to focus analysis on a subset alows you to isolate and analyze specific regions of a molecule, such as an active site or functional group
- **Selection**: Click atoms to select, or rectangle-select with Ctrl+drag
- **Export**: Save screenshots as PNG (with optional transparent background)
- **Animation**: Navigate trajectory with the frame slider

**Loupe menu controls** (apply to all open Loupe windows):
- Bond Width: Thin, Normal, Thick, Extra Thick
- Atom Size: 50%, 75%, 100%, 150%, 200%
- Bond Color and Background Color pickers

### Error Analysis Tools

Multiple tabs for comprehensive error analysis:

- **Basic Errors**: MAE/RMSE timelines and KDE distributions for both energies and forces, plus MAE and RMSE summary tables. Includes a "Subtract mean energy offset" checkbox that removes the constant energy bias from all energy error plots and tables.
- **Atomic Errors**: Per-atom and per-element force error heatmaps and distributions to identify problematic atoms
- **Scatter Errors**: Predicted vs. actual correlation plots for both energies and forces, with interactive point selection for sub-dataset creation
- **Total Force Errors**: System-level total force magnitude error distributions and MAE/RMSE tables
- **Subsystem Errors**: Error analysis on molecular subsystems with aggregated per-geometry force errors
- **Cluster Error**: Dataset clustering (agglomerative or K-Means) and per-cluster error bar charts. Note: not yet supported for variable-sized datasets.
- **Gyration**: Radius of gyration analysis (weighted by atomic number) with timeline and distribution plots

### Advanced Features

- **Sub-datasets**: Create filtered datasets from plot zoom/selection
  - Click the "Sub" button on any compatible plot
  - Sub-dataset updates dynamically as you zoom/pan
  - Can be opened in a separate Loupe window for 3D inspection
- **Energy shift correction**: A global checkbox in the Basic Errors tab subtracts the mean energy offset (mean of predicted minus true energies) from all energy error calculations, affecting distributions, timelines, scatter plots, cluster errors, and MAE/RMSE tables across all tabs
- **Atom filtering**: Focus analysis on specific atoms via the Loupe atom filter panel
- **Headless mode**: Batch processing without the GUI for large-scale computations on remote machines
- **Data caching**: Automatic caching of expensive computations, keyed by MD5 fingerprints of models and datasets
- **Save/Load sessions**: Save the entire working state (datasets, models, all cached computations) to a directory for later restoration

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+d` | Load dataset |
| `Ctrl+m` | Load model |
| `Ctrl+p` | Load prediction |
| `Ctrl+l` | Load saved session |
| `Ctrl+s` | Save session |
| `Ctrl+n` | New Loupe window |
| `Ctrl+0` | Load zero model |

---

## Usage Guide

### Working with Datasets

#### Loading Datasets

FFAST supports two main dataset formats:

**1. sGDML .npz format:**
- Required keys: `R` (positions), `E` (energies), `F` (forces), `z` (atomic numbers)
- Shapes: R: `(N, n_atoms, 3)`, E: `(N,)`, F: `(N, n_atoms, 3)`, z: `(n_atoms,)`
- Optional: lattice vectors for periodic systems

**2. ASE-compatible formats:**
- `.db`, `.extxyz`, `.traj`, `.xyz`, and others
- Energies read from `.info['energy']` or `.get_potential_energy()`
- Forces read from `.arrays['forces']` or `.get_forces()`
- Automatically detects whether atom counts are uniform or variable across configurations

#### Dataset Information

After loading, view dataset details in the left sidebar:
- Number of configurations
- Atom count (range shown for variable-sized datasets)
- Chemical formula
- Dataset fingerprint (MD5 hash used for cache matching)

<!-- ### Pre-computed Predictions -->

<!-- #### Method 1: Load Trained Model Files

1. Menu: File > Load Model (or `Ctrl+m`)
2. Select your model file:
   - MACE: `.model`
   - Nequip: `.pth`
   - sGDML: `.npz`
   - SchNet: `.pth`
   - SpookyNet: `.pth`
3. Model type is automatically detected
4. Model appears in the sidebar -->

### Load Pre-computed Predictions

<!-- Useful for sharing results without sharing trained models: -->
#### For .npz predictions:
1. Create an `.npz` file with:
   - `E`: energies array, shape `(N,)`
   - `F`: forces array, shape `(N, n_atoms, 3)`
2. Menu: File > Load Prediction (or `Ctrl+p`)
3. Select your `.npz` file
4. Select the corresponding dataset from the dropdown
5. The prediction appears as a model in the sidebar

#### For ASe-compatible predictions:
1. Create an ASE-readable file (e.g., `.xyz`, `.db`) with:
   - Energies in `.info['energy']` or via `.get_potential_energy()`
   - Forces in `.arrays['forces']` or via `.get_forces()`
2. Load the file as a dataset (File > Load Dataset)
3. The energies and forces are automatically treated as predictions for error analysis

<!-- #### Generating Predictions

Predictions are generated automatically when needed (e.g., when opening error plots). Progress is shown in the sidebar task list. For large datasets, consider using [headless mode](#headless-batch-processing) to pre-compute predictions. -->

<!-- #### Model Fingerprints

Models are identified by MD5 fingerprints based on their parameters. This enables automatic matching of cached predictions to models and datasets across sessions. -->

### Using the Loupe 3D Viewer

#### Opening Loupe

Menu: Loupe > New (or `Ctrl+n`). A window opens with a dataset selection dropdown.

#### Basic Controls

| Action | Control |
|--------|---------|
| Rotate view | Left-click and drag |
| Pan | Right-click and drag |
| Zoom | Mouse scroll wheel |
| Select atom | Left-click on atom |
| Rectangle select | Ctrl + drag |

#### Navigating the Trajectory

- **Frame slider**: Drag to change configuration
- **Frame number**: Displays the current frame index

#### Sidebar Panels

**ATOMS:**
- Show/Hide atoms
- Size: Adjust atom sphere radius
- Coloring modes: Elements (default), Force Error, Mean Force Error, Total Displacement, Mean Displacement

**BONDS:**
- Show/Hide bonds
- Width: Adjust bond line thickness
- Type: Dynamic (distance-based detection) or Fixed
- Cutoff lenience: Multiplier for bond detection distance threshold

**FORCE VECTORS:**
- Enable/Disable force arrow display
- Length: Scale arrow length
- Normalized: Set all arrows to equal length per frame
- Avg. window: Smooth forces over N frames

**UNIT CELL:**
- Show/Hide periodic cell boundary edges (available when lattice data is present)

**CAMERA:**
- Manual camera positioning (coordinates and target)
- Field of view adjustment
- Center of mass tracking (auto-center on molecular COM)
- Save/Load camera positions

**INFO / MEASUREMENT:**
- Select 1 atom: View position, element, and index
- Select 2 atoms: Measure distance
- Select 3 atoms: Measure bond angle
- Select 4 atoms: Measure dihedral angle

**ALIGNMENT:**
- Select 3 reference atoms to align the molecular structure
- Provides translation and rotation alignment

**INDICES:**
- Toggle atom index label overlays in the 3D view
- Adjustable font size

**AXES:**
- Toggle XYZ orientation axes display in the viewport corner

**ATOM FILTER:**
- Select specific atoms by clicking or rectangle-selecting
- Apply filter to focus analysis on atom subsets

**EXPORT:**
- Save current view as PNG
- Optional transparent background

### Error Analysis Workflows

#### Basic Error Analysis

1. Load a dataset and model (or pre-computed predictions)
2. Click the **Basic Errors** tab
3. View plots:
   - Energy MAE timeline: Identifies configurations with high energy errors
   - Force MAE timeline: Tracks force prediction quality across the trajectory
   - Energy/Force error distributions: KDE-smoothed histograms of error magnitudes
   - MAE and RMSE summary tables: Per-model, per-dataset metrics

**Energy shift correction:** Check "Subtract mean energy offset" to remove the constant energy bias (mean of E_predicted - E_true) from all energy error calculations. When active, all affected plots and tables update their titles to show "(shifted)".

#### Identifying Problematic Configurations

**Using timeline plots:**
1. In Basic Errors, look for peaks in the MAE timeline
2. Note the frame index of high-error configurations

**Creating sub-datasets:**
1. Open any error timeline plot
2. Zoom/pan to a region of interest (e.g., a high-error region)
3. Click the **Sub** toggle button in the plot toolbar
4. A new sub-dataset appears in the sidebar: "Sub: [dataset_name]"
5. Open the sub-dataset in Loupe to visualize these configurations in 3D

Sub-datasets update dynamically as you zoom.

#### Atomic-Level Error Analysis

1. Click the **Atomic Errors** tab
2. View the per-element force error distributions
3. Identify which elements or specific atoms have consistently high errors

#### Correlation Analysis

1. Click the **Scatter Errors** tab
2. View predicted vs. actual scatter plots for energies and forces
3. Points close to the diagonal indicate good predictions; outliers indicate problematic configurations
4. Click or box-select points to create a sub-dataset from outliers

#### Cluster Analysis

1. Click the **Cluster Error** tab
2. The dataset is automatically clustered using configured schemes (agglomerative with Coulomb distance, K-Means with energy)
3. View per-cluster error bar charts
4. Click on bars to select clusters for further analysis

Note: Cluster analysis is not currently supported for variable-sized datasets.

### Headless Batch Processing

For expensive computations on large datasets, use headless mode to run without a GUI, including on remote compute nodes.

#### Example Script

```python
import os
import sys
from pathlib import Path

# Set working directory and Python path to the FFAST project root
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from client.environment import startHeadlessEnvironment

# Initialize headless environment
env = startHeadlessEnvironment()

# Load dataset (use "sGDML" for .npz or "ase (auto)" for ASE formats)
env.taskLoadDataset("examples/data/dataset.xyz", "ase (auto)")
env.waitForTasks(verbose=True)

# Get the loaded dataset and its fingerprint
dataset = env.getDatasetFromPath("examples/data/dataset.xyz")

# Load pre-computed predictions (ASE file with energies and forces)
# The second argument is the dataset fingerprint to match against
env.loadPrepredictedDataset("examples/data/prediction.xyz", dataset.fingerprint)

# Get the model created from the predictions (ghost model)
model = env.getAllModels()[0]

# Queue error computations
env.addToGenerationQueue("energyError", model=model, dataset=dataset)
env.addToGenerationQueue("forcesError", model=model, dataset=dataset)
env.addToGenerationQueue("energyErrorMetrics", model=model, dataset=dataset)
env.addToGenerationQueue("forcesErrorMetrics", model=model, dataset=dataset)
env.addToGenerationQueue("energyErrorDist", model=model, dataset=dataset)
env.addToGenerationQueue("forcesErrorDist", model=model, dataset=dataset)
env.waitForTasks(verbose=True)

# Retrieve computed metrics
eMetrics = env.getData("energyErrorMetrics", model=model, dataset=dataset)
fMetrics = env.getData("forcesErrorMetrics", model=model, dataset=dataset)

print(f"Energy MAE: {eMetrics.get('mae'):.4f}")
print(f"Energy RMSE: {eMetrics.get('rmse'):.4f}")
print(f"Force MAE: {fMetrics.get('mae'):.4f}")
print(f"Force RMSE: {fMetrics.get('rmse'):.4f}")

# Save session for later use in the GUI
# Creates a directory at the given path containing:
#   info.json      - dataset/model metadata
#   cache/*.npz    - all computed data (errors, distributions, metrics)
# Load it in the GUI via File > Load (Ctrl+l).
savePath = os.path.join(PROJECT_ROOT, "results")
env.save(savePath)
print(f"\nSession saved to: {savePath}")

# Clean up
env.headlessQuit()
```

Run:
```bash
python examples/headless/headless.py
```

#### Loading Pre-computed Results in the GUI

1. Launch the GUI: `python main.py`
2. Menu: File > Load (or `Ctrl+l`)
3. Navigate to the saved directory (e.g., `results/`)
4. Datasets, models, and all cached computations are restored automatically

Note: Original dataset files must still be accessible at their saved paths. If model files are unavailable, ghost models are created from the cached predictions.

---

## Example Workflow

This tutorial uses the pre-computed predictions in `examples/` for two MD22 datasets.

### Prerequisites

Download MD22 datasets from http://www.sgdml.org/#datasets:
- MD22 Docosahexaenoic acid (DHA)
- MD22 Stachyose

Save as `dha.npz` and `stachyose.npz` in a directory of your choice.

### Step-by-Step Tutorial

#### 1. Launch FFAST

```bash
python main.py --workdir /path/to/downloaded/datasets
```

#### 2. Load Datasets

- Menu: File > Load Dataset (`Ctrl+d`)
- Select `dha.npz`, then repeat for `stachyose.npz`
- Both datasets appear in the left sidebar

#### 3. Load Pre-computed Predictions

The `examples/` directory contains pre-computed MACE and Nequip predictions.

- Menu: File > Load (`Ctrl+l`)
- Navigate to `examples/MACE/` and open it
- Repeat for `examples/Nequip/`

If the dataset fingerprints match, you will see models appear in the sidebar. These may appear as ghost models (reconstructed from cached predictions).

You should now have 2 datasets and 2 models.

#### 4. View Basic Errors

Click the **Basic Errors** tab. You will see:
- Energy and force MAE timelines
- Error distribution histograms (KDE-smoothed)
- MAE and RMSE summary tables

Exploration tips:
- Hover over points to see values
- Zoom with the scroll wheel
- Pan by dragging
- Right-click for view options

Both models are shown in different colors for comparison.

#### 5. Explore Atomic Errors

Click the **Atomic Errors** tab to see per-element force error distributions. Look for elements with consistently higher errors.

#### 6. Create a Sub-dataset

1. Go to the **Basic Errors** tab
2. In the force MAE timeline, zoom into a region with high errors
3. Click the **Sub** toggle button in the plot toolbar
4. A new "Sub: dha" dataset appears in the sidebar, containing only the configurations visible in the zoomed view

#### 7. Visualize in Loupe

1. Menu: Loupe > New (`Ctrl+n`)
2. Select "Sub: dha" from the dropdown
3. Enable force vectors in the FORCE VECTORS panel
4. Drag the frame slider to browse configurations with high errors
5. Use the Info/Measurement panel to measure distances or angles of interest

#### 8. Compare Model Performance

In the **Scatter Errors** tab, compare predicted vs. actual scatter plots for both models. Check how tightly the points cluster around the diagonal.

Explore all other tabs (Cluster Error, Gyration, Total Force Errors, Subsystem Errors) for additional insights.

---

## Configuration

### Command-line Options

```bash
python main.py [--workdir PATH]
```

- `--workdir PATH`: Set default directory for file dialogs

Debug logging is automatically saved to `debug.log` in the FFAST directory.

### Alternative Entry Point

If you encounter Qt plugin path issues, try:
```bash
python run_ffast.py
```

This wrapper script explicitly sets `QT_PLUGIN_PATH` before launching the application.

### Configuration Files

- `config/default.json`: Default settings (plot parameters, clustering schemes, Loupe defaults, colors)
- `config/userConfig.py`: User configuration overrides
- `config/atoms.py`: Atomic element data (colors, covalent radii, element names)

### Key Configuration Options (default.json)

| Option | Default | Description |
|--------|---------|-------------|
| `plotDistNum` | 500 | Number of points in KDE distributions |
| `scatterPlotNPoints` | 50000 | Maximum points in scatter plots |
| `plotPenWidth` | 3 | Line width in plots |
| `energyUnit` | null | Energy unit label (auto-detected if null) |
| `forceUnit` | null | Force unit label (auto-detected if null) |
| `loupeBondsWidth` | 25 | Default bond line width |
| `loupeAtomSizeScale` | 1.0 | Default atom size multiplier |
| `loupeBondsLenience` | 1.1 | Bond detection distance multiplier |
| `loupeBGColor` | "#000000" | Loupe background color |
| `loupeBondsColor` | "#404040" | Default bond color |
| `loupeForceErrorPercentile` | 0.995 | Percentile for force error color scaling |
| `clusterScheme` | (see file) | Clustering methods and parameters |

---

## Troubleshooting

### Installation Issues

**Segmentation fault on startup:**
- Ensure you are using Python 3.9-3.11. Python 3.12+ causes a segmentation fault with PySide6 6.4.2.
- Recreate the virtual environment: `rm -rf .venv && uv venv --python 3.11 && uv sync`
- Test PySide6: `python -c "from PySide6.QtWidgets import QApplication"`

**ImportError: No module named 'PySide6':**
- Install with: `pip install pyside6==6.4.2`

**OpenGL errors on startup:**
- Update graphics drivers
- On Linux: `sudo apt install libgl1-mesa-glx`

### Qt Platform Plugin Issues

**"Could not find the Qt platform plugin 'cocoa'" (macOS):**

This can occur when PySide6 is installed in a directory synced by iCloud Drive.

Solutions:
1. **Move the virtual environment outside of iCloud Drive:**
   ```bash
   python -m venv ~/venvs/ffast
   source ~/venvs/ffast/bin/activate
   pip install -e .
   ```

2. **Use the alternative entry point** which explicitly sets Qt plugin paths:
   ```bash
   python run_ffast.py
   ```

3. **Recreate the environment:**
   ```bash
   rm -rf .venv && uv venv --python 3.11 && uv sync
   ```

**UI elements not rendering correctly:**
- Ensure PySide6 version is exactly 6.4.2: `pip install pyside6==6.4.2`

### Model Loading Issues

**"Model type not recognized":**
- Install the corresponding model package (see [Install Model Support](#install-model-support-optional))

**"Fingerprint mismatch" when loading predictions:**
- The dataset has changed since predictions were computed. Regenerate predictions with the current dataset, or use the exact same dataset file.

**Model loads but predictions fail:**
- Verify the model was trained for the correct dataset format
- Check that the dataset has all required fields (`R`, `E`, `F`, `z`)
- Check `debug.log` for detailed error messages

### Dataset Issues

**"Cannot load dataset" error:**
- Verify the file format is supported (sGDML `.npz` or ASE-compatible)
- For `.npz`: Check it contains `R`, `E`, `F`, `z` keys
- For ASE formats: Test with `python -c "import ase.io; ase.io.read('file')"`

### Performance Issues

**Slow predictions on large datasets:**
- Use headless mode for batch processing
- Predictions are cached automatically for reuse

**Loupe viewer is laggy:**
- Hide bonds or force vectors when not needed
- Reduce atom size
- Create a smaller sub-dataset
- Update graphics drivers

**Loupe window is blank:**
- Check OpenGL support: `glxinfo | grep OpenGL` (Linux)
- Try software rendering: `export LIBGL_ALWAYS_SOFTWARE=1` before running
- Update graphics drivers

### Data Issues

**"No data available" in plots:**
- Ensure both a dataset and model are loaded
- Wait for predictions to finish (check progress in the sidebar)
- Verify the model and dataset are compatible

**Ghost models appearing (models with hash names):**
- These are created from cached predictions when the original model file is unavailable. They function normally for viewing pre-computed results. Delete them from the sidebar if not needed.

### Getting Help

If you encounter issues not covered here:
1. Check `debug.log` in the FFAST directory for detailed error messages
2. Report issues at the project's GitHub repository

---

## Development

### For Developers

If you want to contribute to the development of FFAST, here are some guidelines:

#### Code Structure

- `main.py`: Entry point and main event loop
- `UI/`: User interface components (MainWindow, SideBar, Plots, Loupe, Templates)
- `modules/`: Auto-discovered pluggable modules (error analysis, model loaders, Loupe features)
- `client/`: Core logic (Environment, DataType/DataEntity, TaskManager)
- `loaders/`: Dataset and model loader base classes
- `config/`: Configuration files

#### Adding a New Module

1. Create `modules/my_module.py`
2. Define `DEPENDENCIES = ["other_module"]` if needed
3. Implement one or more hooks:
   - `loadData(env)`: Register data types
   - `loadUI(UIHandler, env)`: Add UI components (plots, panels, tabs)
   - `loadLoupe(loupeViewer, env, dataset)`: Add 3D viewer features
4. The module is automatically discovered and loaded in dependency order

### Contributing

Contributions are welcome:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Follow code style and add tests if applicable
4. Test your changes by running `python main.py` and checking `debug.log`
5. Submit a pull request

For major changes, open an issue first to discuss the approach.

---

## Citation

If you use FFAST in your research, please cite:

```bibtex
@article{fonseca2023ffast,
  title={Force Field Analysis Software and Tools (FFAST): Assessing Machine Learning Force Fields under the Microscope},
  author={Fonseca, Gregory and Poltavsky, Igor and Tkatchenko, Alexandre},
  journal={Journal of Chemical Theory and Computation},
  volume={19},
  number={23},
  pages={8706--8717},
  year={2023},
  publisher={American Chemical Society},
  doi={10.1021/acs.jctc.3c00985},
  pmid={38011895},
  pmcid={PMC10720330}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **MD22 datasets**: Stefan Chmiela et al., sgdml.org
- **ASE**: Atomic Simulation Environment developers
- **Vispy**: High-performance interactive 2D/3D data visualization library
- **PyQtGraph**: Scientific graphics and GUI library
- **Model frameworks**: MACE, Nequip, sGDML, SchNet, SpookyNet developers

---