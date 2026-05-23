import logging

import numpy as np

from UI.loupeProperties import CanvasProperty
from client.mathUtils import kabschTransforms

logger = logging.getLogger("FFAST")

DEPENDENCIES = ["loupeAtoms", "loupeAtomAlign", "loupeCamera"]


class KabschAlignProperty(CanvasProperty):

    key = "kabschAlign"
    changesR = True

    def onNewGeometry(self):
        canvas = self.canvas
        settings = canvas.loupe.settings

        if not settings.get("kabschAlign"):
            return

        r0 = canvas.getR(0)

        heavy_only = settings.get("kabschAlignHeavyOnly")
        indices = None
        if heavy_only:
            z = canvas.dataset.getElements()
            heavy = np.where(z > 1)[0]
            if len(heavy) > 0:
                indices = heavy

        # Keep camera on the aligned molecule's resting position (ref_centroid).
        # Must be set unconditionally: originCenterOfMass is disabled while
        # Kabsch is active, so nothing else moves the camera. Without this,
        # enabling Kabsch on a non-zero frame (where COM tracking had moved the
        # camera to that frame's centroid) leaves the camera pointing at the
        # wrong position after the molecule is snapped back to frame-0 centroid.
        r0_sel = r0[indices] if indices is not None else r0
        ref_centroid = r0_sel.mean(axis=0)
        canvas.camera.center = ref_centroid

        if canvas.index == 0:
            return

        r = canvas.getCurrentR()

        if r.shape != r0.shape:
            logger.warning(
                "Kabsch alignment skipped: frame shapes differ "
                f"({r.shape} vs {r0.shape})"
            )
            return

        transforms = kabschTransforms(r, r0, indices=indices)
        canvas.currentTransformations = canvas.currentTransformations + transforms


def addSettings(UIHandler, loupe):
    settings = loupe.settings
    settings.addParameters(
        **{
            "kabschAlign": [False, "updateGeometry"],
            "kabschAlignHeavyOnly": [True, "updateGeometry"],
        }
    )
    settings.markAsPerDataset("kabschAlign")
    settings.markAsPerDataset("kabschAlignHeavyOnly")

    ## MUTUAL EXCLUSIVITY
    def _disable_others_for_kabsch():
        if settings.get("kabschAlign"):
            settings.setParameter("originCenterOfMass", False)
            settings.setParameter("alignAtoms", False)

    settings.addParameterActions("kabschAlign", _disable_others_for_kabsch)
    settings.addParameterActions(
        "originCenterOfMass",
        lambda: settings.setParameter("kabschAlign", False)
        if settings.get("originCenterOfMass")
        else None,
    )
    settings.addParameterActions(
        "alignAtoms",
        lambda: settings.setParameter("kabschAlign", False)
        if settings.get("alignAtoms")
        else None,
    )

    ## SETTINGS PANE
    pane = loupe.getSettingsPane("ATOMS")
    pane.addSetting(
        "CheckBox",
        "Align (Kabsch)",
        settingsKey="kabschAlign",
        toolTip="Align all frames to frame 0 using Kabsch rigid rotation (minimises RMSD)",
    )

    heavyOnlyBox = pane.addSetting(
        "CheckBox",
        "Heavy atoms only",
        settingsKey="kabschAlignHeavyOnly",
        toolTip="Use only heavy atoms (z > 1) to compute the alignment rotation",
    )
    heavyOnlyBox.setHideCondition(lambda: not settings.get("kabschAlign"))


def loadLoupe(UIHandler, loupe):
    addSettings(UIHandler, loupe)
    loupe.addCanvasProperty(KabschAlignProperty)
