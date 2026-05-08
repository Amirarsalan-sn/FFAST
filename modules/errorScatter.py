"""import numpy as np
from config.userConfig import getConfig
import logging

logger = logging.getLogger("FFAST")
DEPENDENCIES = ["basicErrors"]


def loadUI(UIHandler, env):
    from UI.ContentTab import ContentTab
    from UI.Plots import BasicPlotWidget

    ct = ContentTab(UIHandler)  # adding a new one manually
    UIHandler.addContentTab(ct, "Scatter Errors")

    class EnergyScatterPlot(BasicPlotWidget):
        def __init__(self, handler, **kwargs):
            super().__init__(
                handler,
                title="Energy Scatter",
                name="Energy Scatter",
                **kwargs,
            )
            self.setDataDependencies("energy")
            self.setXLabel("True Energy", getConfig("energyUnit"))
            self.setYLabel("Predicted Energy", getConfig("energyUnit"))

            # this list will save which indices are currently selected for each plot
            # since there's a maximum number of indices on a scatter plot
            # see "scatterPlotNPoints" in the config file
            self.indices = {}
            self.eventSubscribe(
                "ENERGY_SHIFT_CHANGED", self.onEnergyShiftChanged
            )

        def onEnergyShiftChanged(self):
            shifted = self.handler.energyShiftEnabled
            self.titleLabel.setText(
                "Energy Scatter (shifted)"
                if shifted
                else "Energy Scatter"
            )
            self.visualRefresh(force=True)

        def addPlots(self):
            self.indices.clear()
            shifted = self.handler.energyShiftEnabled
            for data in self.getWatchedData():

                predE = data["dataEntry"].get("energy")
                trueE = data["dataset"].getEnergies()

                if shifted:
                    shift = np.mean(predE - trueE)
                    predE = predE - shift

                # this is a unique key for the model/dataset combination
                # perfect for saving the indices below uniquely
                key = self.getKey(data["dataset"], data["model"])

                n = getConfig("scatterPlotNPoints")
                if len(predE) > n:
                    idx = np.round(np.linspace(0, len(predE) - 1, n)).astype(
                        int
                    )
                    predE = predE[idx]
                    trueE = trueE[idx]
                    self.indices[key] = idx
                else:
                    self.indices[key] = None

                self.plot(
                    trueE, predE, autoColor=data, scatter=True, autoLabel=data
                )

        def getKey(self, dataset, model):
            # Creates a unique key based on the dataset/model combination
            # perfect for saving the indices for each plot item, needed when
            # subbing (see getDatasetSubIndices)
            return f"{dataset.fingerprint}__{model.fingerprint}"

        def getDatasetSubIndices(self, dataset, model):
            (xRange, yRange) = self.getRanges()
            key = self.getKey(dataset, model)
            idxs = self.indices[key]

            x0, x1 = xRange
            y0, y1 = yRange

            de = self.env.getData("energy", dataset=dataset, model=model)
            predE = de.get("energy")
            trueE = dataset.getEnergies()

            xTruth = (predE > x0) & (predE < x1)
            yTruth = (trueE > y0) & (trueE < y1)
            args = np.argwhere(xTruth & yTruth).flatten()

            if idxs is None:
                return args
            else:
                return idxs[args]

    plt = EnergyScatterPlot(UIHandler, parent=ct)
    ct.addWidget(plt, 0, 0)
    ct.addDataSelectionCallback(plt.setModelDatasetDependencies)

    class ForcesScatterPlot(BasicPlotWidget):
        def __init__(self, handler, **kwargs):
            super().__init__(
                handler,
                title="Forces Scatter",
                name="Forces Scatter",
                **kwargs,
            )
            self.setDataDependencies("forces")
            self.setXLabel("True Forces", getConfig("forcesUnit"))
            self.setYLabel("Predicted Forces", getConfig("forcesUnit"))

            self.indices = {}

        def getKey(self, dataset, model):
            # Creates a unique key based on the dataset/model combination
            # perfect for saving the indices for each plot item, needed when
            # subbing (see getDatasetSubIndices)
            return f"{dataset.fingerprint}__{model.fingerprint}"

        def addPlots(self):

            for data in self.getWatchedData():
                dataset = data["dataset"]
                pred_forces = data["dataEntry"].get("forces")
                true_forces = dataset.getForces()

                # Handle variable vs uniform datasets
                if isinstance(pred_forces, list):
                    # Variable dataset: concatenate and flatten
                    predE = np.concatenate([f.flatten() for f in pred_forces])
                    trueE = np.concatenate([f.flatten() for f in true_forces])
                else:
                    # Uniform dataset: flatten directly
                    predE = pred_forces.flatten()
                    trueE = true_forces.flatten()

                key = self.getKey(dataset, data["model"])

                n = getConfig("scatterPlotNPoints")
                if len(predE) > n:
                    idx = np.round(np.linspace(0, len(predE) - 1, n)).astype(
                        int
                    )
                    predE = predE[idx]
                    trueE = trueE[idx]
                    self.indices[key] = idx
                else:
                    self.indices[key] = None

                self.plot(
                    trueE, predE, autoColor=data, scatter=True, autoLabel=data
                )

        def getDatasetSubIndices(self, dataset, model):
            (xRange, yRange) = self.getRanges()
            key = self.getKey(dataset, model)
            idxs = self.indices[key]

            x0, x1 = xRange
            y0, y1 = yRange

            de = self.env.getData("forces", dataset=dataset, model=model)
            pred_forces = de.get("forces")
            true_forces = dataset.getForces()

            # Handle variable vs uniform datasets
            if isinstance(pred_forces, list):
                # Variable dataset: concatenate and flatten
                predF = np.concatenate([f.flatten() for f in pred_forces])
                trueF = np.concatenate([f.flatten() for f in true_forces])
            else:
                # Uniform dataset: flatten directly
                predF = pred_forces.flatten()
                trueF = true_forces.flatten()

            xTruth = (predF > x0) & (predF < x1)
            yTruth = (trueF > y0) & (trueF < y1)
            args = np.argwhere(xTruth & yTruth).flatten()

            if idxs is None:
                idxs = args
            else:
                idxs = idxs[args]

            # this is indices of the flattened forces by component
            # but we need the index of the geometry
            if hasattr(dataset, 'isVariable') and dataset.isVariable:
                # Variable dataset: use molecule_offsets to map flat indices to molecules
                # Each flat index corresponds to a force component (atom * 3)
                # We need to find which molecule each flat index belongs to
                atom_offsets = dataset.molecule_offsets  # [0, n1, n1+n2, ...]
                force_offsets = atom_offsets * 3  # Convert to force component offsets

                # For each flat index, find which molecule it belongs to
                mol_indices = []
                for idx in idxs:
                    # Find which molecule this flat index belongs to
                    mol_idx = np.searchsorted(force_offsets[1:], idx, side='right')
                    mol_indices.append(mol_idx)

                idxs = np.unique(mol_indices).astype(int)
            else:
                # Uniform dataset: all molecules have same number of atoms
                nEntriesPerConf = dataset.getNAtoms() * 3
                idxs = np.unique(np.floor(idxs / nEntriesPerConf)).astype(int)

            return idxs

    plt = ForcesScatterPlot(UIHandler, parent=ct)
    ct.addWidget(plt, 0, 1)
    ct.addDataSelectionCallback(plt.setModelDatasetDependencies)
"""