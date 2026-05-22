import numpy as np
import logging
from UI.loupeProperties import VisualElement

logger = logging.getLogger("FFAST")
DEPENDENCIES = ["loupeAtoms"]

_LINE_WIDTH = 3
_LINE_OUTLINE_WIDTH_FACTOR = 1.5
_ARROW_OUTLINE_SIZE = 3.0
_ARROW_INNER_SIZE = 1.5

class ForceVectorsElement(VisualElement):
    pos = None

    def __init__(self, *args, parent=None, width=_LINE_WIDTH, **kwargs):
        from vispy import scene

        self.outline = scene.visuals.Arrow(
            pos=None,
            parent=parent,
            color="black",
            width=width * _LINE_OUTLINE_WIDTH_FACTOR,
            connect="segments",
            arrow_size=_ARROW_OUTLINE_SIZE,
            arrows=None,
            arrow_color="black",
            method="gl",
        )
        self.lines = scene.visuals.Arrow(
            pos=None,
            parent=parent,
            color="white",
            width=width,
            connect="segments",
            arrow_size=_ARROW_INNER_SIZE,
            arrows=None,
            arrow_color="white",
            method="gl",
        )
        self.lines.set_gl_state(depth_test=False)
        super().__init__(*args, **kwargs, singleElement=None)
        self.width = width

    def show(self):
        self.hidden = False
        self.outline.visible = True
        self.lines.visible = True

    def hide(self):
        self.hidden = True
        self.outline.visible = False
        self.lines.visible = False

    def onNewGeometry(self):
        self.update()

    def _get_forces(self):
        """Return (N_atoms, 3) forces for current frame, or None if unavailable."""
        settings = self.canvas.settings
        model_key = settings.get("forceVectorsModelKey")
        dataset = self.canvas.dataset
        window = settings.get("forceVectorsAvgWindow")
        index = self.canvas.index

        if model_key is not None:
            env = self.canvas.loupe.env
            model = env.getModel(model_key)
            if model is None:
                return None, "no_prediction"
            data = env.getData("forces", model=model, dataset=dataset)
            if data is None:
                return None, "no_prediction"
            forces_all = data.get("forces")
            if forces_all is None:
                return None, "no_prediction"
            if window > 0:
                n = dataset.getN()
                indices = np.arange(-window, window + 1) + index
                indices = indices[(indices >= 0) & (indices < n)]
                if forces_all.ndim == 3:
                    F = np.mean(forces_all[indices], axis=0)
                else:
                    F = forces_all
            else:
                if forces_all.ndim == 3:
                    F = forces_all[index]
                else:
                    F = forces_all
            return F, None

        # Ground truth
        if window > 0:
            n = dataset.getN()
            indices = np.arange(-window, window + 1) + index
            indices = indices[(indices >= 0) & (indices < n)]
            F = dataset.getForces(indices=indices)
            F = np.mean(F, axis=0)
        else:
            F = dataset.getForces(indices=index)
        return F, None

    def update(self):
        settings = self.canvas.settings
        show = settings.get("showForceVectors")
        status_label = getattr(self.canvas.loupe, "_forceVectorsStatusLabel", None)

        if not show:
            self.hide()
            if status_label:
                status_label.setVisible(False)
            return

        self.show()

        F, err = self._get_forces()

        if err == "no_prediction":
            self.pos = None
            if status_label:
                status_label.setText("No predictions computed for this model")
                status_label.setVisible(True)
            self.queueVisualRefresh()
            return

        if status_label:
            status_label.setVisible(False)

        lengthFactor = settings.get("forceVectorsLength")
        normalised = settings.get("forceVectorsNormalised")
        R = self.canvas.getCurrentR()

        for vOrM in self.canvas.currentTransformations:
            if vOrM.ndim == 2:
                F = F @ vOrM

        if normalised:
            normF = F / np.max(np.linalg.norm(F, axis=1)) * lengthFactor / 5
        else:
            normF = F * lengthFactor / 500

        pos = np.empty((R.shape[0] * 2, 3))
        pos[0::2, :] = R
        pos[1::2, :] = R + normF
        self.pos = pos

        self.queueVisualRefresh()

    def _draw(self, **kwargs):
        show = self.canvas.settings.get("showForceVectors")

        if self.pos is None or not show:
            self.hide()
            return

        self.show()
        arrows = self.pos.reshape(-1, 6)

        self.outline.set_data(
            pos=self.pos,
            color="black",
            width=self.width * _LINE_OUTLINE_WIDTH_FACTOR,
            arrows=arrows,
        )
        self.lines.set_data(
            pos=self.pos,
            color="white",
            width=self.width,
            arrows=arrows,
        )


def loadLoupe(UIHandler, loupe):
    from UI.Templates import SettingsPane, ObjectComboBox
    from PySide6.QtWidgets import QLabel

    loupe.addVisualElement(ForceVectorsElement, "ForceVectorsElement")

    settings = loupe.settings
    settings.addParameters(
        **{
            "showForceVectors": [False, "updateGeometry"],
            "forceVectorsModelKey": [None, "updateGeometry"],
            "forceVectorsLength": [5, "updateGeometry"],
            "forceVectorsAvgWindow": [0, "updateGeometry"],
            "forceVectorsNormalised": [False, "updateGeometry"],
        }
    )
    settings.markAsPerDataset("forceVectorsModelKey")

    # SETTINGS PANE
    pane = SettingsPane(UIHandler, loupe.settings, parent=loupe)
    loupe.addSidebarPane("FORCE VECTORS", pane)

    pane.addSetting(
        "CheckBox",
        "Enable",
        settingsKey="showForceVectors",
        toolTip="Show a vector field corresponding to the forces",
    )

    # SOURCE SELECTOR
    class _ForceSourceComboBox(ObjectComboBox):
        def updateList(self, *args):
            self.currentlyUpdatingList = True
            model_keys = self.env.getAllModelKeys()
            self.currentKeyList = [None] + model_keys
            self.clear()
            self.addItems(
                ["Ground Truth"]
                + [self.env.getModelOrDataset(k).getDisplayName() for k in model_keys]
            )
            if self.selectedKey in self.currentKeyList:
                self.setCurrentIndex(self.currentKeyList.index(self.selectedKey))
                self.currentlyUpdatingList = False
            elif self.currentKeyList:
                self.setCurrentIndex(0)
                self.currentlyUpdatingList = False
                self.forceUpdate()
            else:
                self.currentlyUpdatingList = False

    sourceCombo = _ForceSourceComboBox(UIHandler, hasDatasets=False)
    sourceCombo.setOnIndexChanged(
        lambda key: settings.setParameter("forceVectorsModelKey", key)
    )
    pane.layout.addWidget(sourceCombo)

    # STATUS LABEL (shown when no predictions available)
    statusLabel = QLabel("No predictions computed for this model")
    statusLabel.setWordWrap(True)
    statusLabel.setStyleSheet("color: orange;")
    statusLabel.setVisible(False)
    pane.layout.addWidget(statusLabel)
    loupe._forceVectorsStatusLabel = statusLabel

    pane.addSetting(
        "Slider",
        "Length",
        settingsKey="forceVectorsLength",
        toolTip="Change the length of the force vectors",
        nMin=1,
        nMax=50,
    )
    pane.addSetting(
        "Slider",
        "Avg. window",
        settingsKey="forceVectorsAvgWindow",
        toolTip="Set the number of points to average around for a smoother result.",
        nMin=0,
        nMax=10000,
    )
    pane.addSetting(
        "CheckBox",
        "Normalised",
        settingsKey="forceVectorsNormalised",
        toolTip="If enabled, set the longest vector for every frame to the same length",
    )
