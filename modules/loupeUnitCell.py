import numpy as np
from config.userConfig import getConfig
from functools import partial
import logging
from UI.loupeProperties import VisualElement, CanvasProperty

logger = logging.getLogger("FFAST")
DEPENDENCIES = ["loupeCamera"]


class UnitCellElement(VisualElement):
    """Visual element for drawing unit cell edges as lines."""

    def __init__(self, *args, parent=None, width=2, **kwargs):
        from vispy import scene

        self.lines = scene.visuals.Line(
            pos=None,
            parent=parent,
            color=(0.5, 0.5, 0.5, 0.8),  # Gray color for unit cell
            width=width,
            connect="segments",
            antialias=True,
        )
        super().__init__(*args, **kwargs, singleElement=self.lines)
        self.width = width

    def onNewGeometry(self):
        self.queueVisualRefresh()

    def onCameraChange(self):
        dist = self.canvas.props["camera"].get("distance")
        if dist is None:
            dist = 1
        self.lines.set_data(width=self.width / dist)

    def _draw(self, picking=False, pickingColors=None):
        """Draw the unit cell edges."""
        unitCellEdges = self.canvas.props["unitCellEdges"].get("edges")
        width = self.canvas.props["camera"].get("distance")

        if width is None:
            width = 1

        if unitCellEdges is None or not self.canvas.settings.get("showUnitCell"):
            self.hide()
        else:
            self.show()
            self.lines.set_data(pos=unitCellEdges, width=self.width / width)


class UnitCellProperty(CanvasProperty):
    """Property that computes unit cell edge positions."""

    key = "unitCellEdges"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def onNewGeometry(self):
        self.clear()

    def generate(self):
        """Generate the 12 edges of the unit cell box."""
        dataset = self.canvas.dataset
        current_index = self.canvas.index

        # Get lattice vectors from dataset for the current frame
        try:
            lattice = dataset.getLattice(current_index)
        except Exception as e:
            logger.debug(f"Could not get lattice: {e}")
            self.set(edges=None)
            return

        if lattice is None:
            self.set(edges=None)
            return

        # Get current atomic positions to determine cell origin
        R = self.canvas.getCurrentR()
        if R is None or len(R) == 0:
            origin = np.array([0.0, 0.0, 0.0])
        else:
            # Use center of atoms as reference, or you could use min corner
            origin = np.mean(R, axis=0) - np.sum(lattice, axis=0) / 2

        # Extract lattice vectors
        # lattice can be a Cell object (ASE) or array
        if hasattr(lattice, 'array'):
            # ASE Cell object
            lattice_array = np.array(lattice.array)
        else:
            lattice_array = np.array(lattice)

        a, b, c = lattice_array[0], lattice_array[1], lattice_array[2]

        # Define the 8 corners of the unit cell
        corners = np.array([
            origin,              # 0: origin
            origin + a,          # 1: along a
            origin + b,          # 2: along b
            origin + a + b,      # 3: a + b
            origin + c,          # 4: along c
            origin + a + c,      # 5: a + c
            origin + b + c,      # 6: b + c
            origin + a + b + c,  # 7: a + b + c
        ])

        # Define the 12 edges (pairs of corner indices)
        edges = [
            # Bottom face (z = 0)
            (0, 1), (1, 3), (3, 2), (2, 0),
            # Top face (z = c)
            (4, 5), (5, 7), (7, 6), (6, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        # Convert to line segments for Vispy
        edge_positions = []
        for i, j in edges:
            edge_positions.append(corners[i])
            edge_positions.append(corners[j])

        edge_positions = np.array(edge_positions)
        self.set(edges=edge_positions)


def addSettings(UIHandler, loupe):
    """Add unit cell settings to loupe."""
    settings = loupe.settings
    settings.addParameters(**{
        "showUnitCell": [False, "updateGeometry"],
    })

    # Add the canvas property
    loupe.addCanvasProperty(UnitCellProperty)


def addUnitCellObject(UIHandler, loupe):
    """Add the unit cell visual element."""
    loupe.addVisualElement(UnitCellElement, "UnitCellElement")


def addSettingsPane(UIHandler, loupe):
    """Add unit cell settings to the sidebar."""
    from UI.Templates import SettingsPane

    settings = loupe.settings
    pane = SettingsPane(UIHandler, loupe.settings, parent=loupe)

    pane.addSetting(
        "CheckBox",
        "Show Unit Cell",
        settingsKey="showUnitCell",
        toolTip="Display unit cell edges",
    )

    loupe.addSidebarPane("UNIT CELL", pane)


def loadLoupe(UIHandler, loupe):
    """Main entry point for loading the unit cell module."""
    addSettings(UIHandler, loupe)
    addUnitCellObject(UIHandler, loupe)
    addSettingsPane(UIHandler, loupe)
