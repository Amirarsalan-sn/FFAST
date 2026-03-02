import numpy as np
from client.dataType import DataType
import logging
from scipy.stats import gaussian_kde
from config.userConfig import getConfig

logger = logging.getLogger("FFAST")

DEPENDENCIES = []


def loadData(env):
    # Helper function for variable dataset indexing
    def flatIndexToConfigIndex(flat_indices, dataset):
        """
        Convert flat force component indices to configuration indices.

        For uniform datasets: uses simple division
        For variable datasets: uses molecule_offsets with searchsorted
        """
        if hasattr(dataset, 'isVariable') and dataset.isVariable:
            # Variable: use offsets
            offsets = dataset.molecule_offsets * 3  # 3 components per atom
            config_indices = np.searchsorted(offsets[1:], flat_indices, side='right')
            return np.unique(config_indices)
        else:
            # Uniform: simple division
            nAtoms = dataset.getNAtoms()
            return np.unique(flat_indices // (nAtoms * 3))

    class EnergyPredictionError(DataType):
        modelDependent = True
        datasetDependent = True
        key = "energyError"
        dependencies = ["energy"]
        iterable = True
        atomConstant = True

        def __init__(self, *args):
            super().__init__(*args)

        def data(self, dataset=None, model=None, taskID=None):
            env = self.env

            ePred = env.getData("energy", model=model, dataset=dataset)
            eData = dataset.getEnergies()
            print ("eData:", eData)
            print ("ePred:", ePred.get("energy"))

            diff = ePred.get("energy") - eData
            shift = np.mean(diff)
            de = self.newDataEntity(diff=diff, shift=shift)
            env.setData(de, self.key, model=model, dataset=dataset)
            return True

    class ForcesPredictionError(DataType):
        modelDependent = True
        datasetDependent = True
        key = "forcesError"
        dependencies = ["forces"]
        iterable = True
        atomFilterable = True

        def __init__(self, *args):
            super().__init__(*args)

        def data(self, dataset=None, model=None, taskID=None):
            env = self.env

            fPred = env.getData("forces", model=model, dataset=dataset)
            fData = dataset.getForces()

            if hasattr(dataset, 'isVariable') and dataset.isVariable:
                # Variable dataset: fPred and fData are lists
                fPred_forces = fPred.get("forces")
                diff_list = []
                for i in range(len(fData)):
                    diff_list.append(fPred_forces[i] - fData[i])
                diff = diff_list
            else:
                # Uniform dataset: numpy arrays
                diff = fPred.get("forces") - fData

            de = self.newDataEntity(
                diff=diff
            )
            env.setData(de, self.key, model=model, dataset=dataset)
            return True

    class EnergyErrorDist(DataType):
        modelDependent = True
        datasetDependent = True
        key = "energyErrorDist"
        dependencies = ["energyError"]
        iterable = False

        def __init__(self, *args):
            super().__init__(*args)

        def _computeKDE(self, absDiff):
            mirrored = np.concatenate([absDiff, -absDiff])
            nPts = getConfig("plotDistNum")

            if np.std(mirrored) < 1e-10:
                distX = np.linspace(
                    0,
                    max(np.max(mirrored), 1e-10),
                    nPts,
                )
                distY = np.zeros_like(distX)
                closest_idx = np.argmin(
                    np.abs(distX - np.mean(mirrored))
                )
                distY[closest_idx] = 1.0
            else:
                kde = gaussian_kde(mirrored)
                delta = np.max(mirrored)
                distX = np.linspace(
                    0,
                    np.max(mirrored) + 0.05 * delta,
                    nPts,
                )
                distY = kde(distX)
            return distX, distY

        def data(self, dataset=None, model=None, taskID=None):
            env = self.env

            eErr = env.getData("energyError", model=model, dataset=dataset)
            rawDiff = eErr.get("diff")
            shift = eErr.get("shift")

            # Unshifted KDE
            distX, distY = self._computeKDE(np.abs(rawDiff))

            # Shifted KDE
            shiftedDiff = rawDiff - shift
            sDistX, sDistY = self._computeKDE(np.abs(shiftedDiff))

            de = self.newDataEntity(
                distY=distY,
                distX=distX,
                shiftedDistX=sDistX,
                shiftedDistY=sDistY,
            )
            env.setData(de, self.key, model=model, dataset=dataset)
            return True

    class ForcesErrorDist(DataType):
        modelDependent = True
        datasetDependent = True
        key = "forcesErrorDist"
        dependencies = ["forcesError"]
        iterable = False

        def __init__(self, *args):
            super().__init__(*args)

        def data(self, dataset=None, model=None, taskID=None):
            env = self.env

            err = env.getData("forcesError", model=model, dataset=dataset)
            diff = err.get("diff")

            if hasattr(dataset, 'isVariable') and dataset.isVariable:
                # Variable dataset: diff is list of arrays
                # Flatten each molecule's diff and compute MAE per molecule
                mae_list = []
                for diff_mol in diff:
                    diff_flat = np.abs(diff_mol).reshape(-1)
                    mae_list.append(np.mean(diff_flat))
                mae = np.array(mae_list)
            else:
                # Uniform dataset: diff is (N, M, 3) array
                diff = np.abs(diff)
                diff = diff.reshape(diff.shape[0], -1)
                mae = np.mean(diff, axis=1)

            # Mirror for symmetric distribution
            mae = np.concatenate([-np.abs(mae), np.abs(mae)])

            # Check if data has sufficient variance for KDE
            if np.std(mae) < 1e-10:
                # Zero or near-zero variance: create simple distribution
                distX = np.linspace(
                    0,
                    max(np.max(mae), 1e-10),
                    getConfig("plotDistNum"),
                )
                # Delta function approximation at mean value
                distY = np.zeros_like(distX)
                closest_idx = np.argmin(np.abs(distX - np.mean(mae)))
                distY[closest_idx] = 1.0
            else:
                # Normal KDE calculation
                kde = gaussian_kde(mae)
                delta = np.max(mae) - 0

                distX = np.linspace(
                    0,
                    np.max(mae) + delta * 0.05,
                    getConfig("plotDistNum"),
                )
                distY = kde(distX)

            de = self.newDataEntity(distY=distY, distX=distX)
            env.setData(de, self.key, model=model, dataset=dataset)
            return True

    class EnergyErrorMetrics(DataType):
        modelDependent = True
        datasetDependent = True
        key = "energyErrorMetrics"
        dependencies = ["energyError"]
        iterable = False

        def __init__(self, *args):
            super().__init__(*args)

        def data(self, dataset=None, model=None, taskID=None):
            env = self.env

            eErr = env.getData("energyError", model=model, dataset=dataset)

            diff = eErr.get("diff")
            shift = eErr.get("shift")

            # Unshifted metrics
            absDiff = np.abs(diff)
            mae = np.mean(absDiff)
            rmse = np.sqrt(np.mean(diff ** 2))

            # Shifted metrics
            shiftedDiff = diff - shift
            shiftedMae = np.mean(np.abs(shiftedDiff))
            shiftedRmse = np.sqrt(np.mean(shiftedDiff ** 2))

            de = self.newDataEntity(
                mae=mae,
                rmse=rmse,
                shiftedMae=shiftedMae,
                shiftedRmse=shiftedRmse,
            )
            env.setData(de, self.key, model=model, dataset=dataset)
            return True

    class ForcesErrorMetrics(DataType):
        modelDependent = True
        datasetDependent = True
        key = "forcesErrorMetrics"
        dependencies = ["forcesError"]
        iterable = False

        def __init__(self, *args):
            super().__init__(*args)

        def data(self, dataset=None, model=None, taskID=None):
            env = self.env

            err = env.getData("forcesError", model=model, dataset=dataset)
            diff = err.get("diff")

            if hasattr(dataset, 'isVariable') and dataset.isVariable:
                # Variable dataset: diff is list of arrays
                atomicMAE_list = []
                for diff_mol in diff:
                    # Per-atom MAE for this molecule: (n_atoms_i,)
                    atomicMAE_list.append(np.mean(np.abs(diff_mol), axis=1))

                # Global metrics: concatenate all
                diff_flat = np.vstack(diff)  # (total_atoms, 3)
                mae = np.mean(np.abs(diff_flat))
                rmse = np.sqrt(np.mean(diff_flat ** 2))

                de = self.newDataEntity(atomicMAE=atomicMAE_list, mae=mae, rmse=rmse)
            else:
                # Uniform dataset: diff is (N, M, 3) array
                diff = np.abs(diff)
                atomicMAE = np.mean(diff, axis=2)  # (N, M)
                mae = np.mean(diff)
                rmse = np.sqrt(np.mean(diff ** 2))

                de = self.newDataEntity(atomicMAE=atomicMAE, mae=mae, rmse=rmse)

            env.setData(de, self.key, model=model, dataset=dataset)
            return True

    env.registerDataType(EnergyPredictionError)
    env.registerDataType(ForcesPredictionError)
    env.registerDataType(EnergyErrorDist)
    env.registerDataType(ForcesErrorDist)
    env.registerDataType(EnergyErrorMetrics)
    env.registerDataType(ForcesErrorMetrics)


def loadUI(UIHandler, env):
    from UI.ContentTab import ContentTab
    from UI.Plots import BasicPlotWidget, Table
    from UI.Templates import Slider, HorizontalContainerScrollArea
    from PySide6.QtWidgets import QCheckBox

    ct = ContentTab(UIHandler)
    UIHandler.addContentTab(ct, "Basic Errors")

    # Energy shift checkbox (global toggle)
    shiftCheckBox = QCheckBox("Subtract mean energy offset", parent=ct)
    shiftCheckBox.setToolTip(
        "Remove constant energy offset by subtracting "
        "mean(E_predicted - E_true) from all energy errors"
    )

    def onShiftToggled(state):
        UIHandler.energyShiftEnabled = bool(state)
        UIHandler.eventPush("ENERGY_SHIFT_CHANGED")

    shiftCheckBox.stateChanged.connect(onShiftToggled)
    ct.topLayout.addWidget(shiftCheckBox)

    # PLOTS
    class EnergyErrorDistPlot(BasicPlotWidget):
        def __init__(self, handler, **kwargs):
            super().__init__(
                handler,
                title="Energy MAE distribution",
                isSubbable=True,
                name="Energy Error Distribution",
                **kwargs,
            )
            self.setDataDependencies("energyErrorDist")
            self.setXLabel("Energy MAE", getConfig("energyUnit"))
            self.setYLabel("Density")
            self.eventSubscribe(
                "ENERGY_SHIFT_CHANGED", self.onEnergyShiftChanged
            )

        def onEnergyShiftChanged(self):
            shifted = self.handler.energyShiftEnabled
            self.titleLabel.setText(
                "Energy MAE distribution (shifted)"
                if shifted
                else "Energy MAE distribution"
            )
            self.visualRefresh(force=True)

        def addPlots(self):
            shifted = self.handler.energyShiftEnabled
            for data in self.getWatchedData():
                de = data["dataEntry"]
                if shifted:
                    x = de.get("shiftedDistX")
                    y = de.get("shiftedDistY")
                else:
                    x = de.get("distX")
                    y = de.get("distY")
                self.plot(x, y, autoColor=data, autoLabel=data)

        def getDatasetSubIndices(self, dataset, model):
            (xRange, yRange) = self.getRanges()
            N = dataset.getN()
            x0, x1 = xRange

            eErr = env.getData("energyError", model=model, dataset=dataset)
            diff = eErr.get("diff")
            if self.handler.energyShiftEnabled:
                diff = diff - eErr.get("shift")
            diff = np.abs(diff)

            idxs = np.argwhere((diff >= x0) & (diff <= x1))
            idxs = np.unique(idxs)

            return idxs

    class ForcesErrorDistPlot(BasicPlotWidget):
        def __init__(self, handler, **kwargs):
            super().__init__(
                handler,
                title="Forces MAE distribution",
                isSubbable=False,  # not finding a good way to do it
                # If we say: at least one force component f with x0 < f < x1
                # then for large molecules, even small windows have a shitload
                # of indices
                # If we average the force error by geometry
                # then outliers can be hidden within otherwise okay geometries
                # as tested on DHA
                # Perhaps error scatters are better for the outliers,
                # but there only one/two points can realistically be selected
                # at the same time
                name="Force Error Distribution",
                **kwargs,
            )
            self.setDataDependencies("forcesErrorDist")
            self.setXLabel("Forces MAE", getConfig("forceUnit"))
            self.setYLabel("Density")

        def addPlots(self):
            for data in self.getWatchedData():
                de = data["dataEntry"]
                x, y = de.get("distX"), de.get("distY")
                self.plot(x, y, autoColor=data, autoLabel=data)

        def getDatasetSubIndices(self, dataset, model):
            (xRange, yRange) = self.getRanges()
            N = dataset.getN()
            x0, x1 = xRange

            err = env.getData("forcesError", model=model, dataset=dataset)
            diff = err.get("diff")

            if hasattr(dataset, 'isVariable') and dataset.isVariable:
                # Variable dataset: diff is list of arrays
                diff_flat = np.concatenate([np.abs(d).reshape(-1) for d in diff])
                idxs_flat = np.argwhere((diff_flat >= x0) & (diff_flat <= x1)).flatten()

                # Convert flat indices to config indices using helper
                idxs = flatIndexToConfigIndex(idxs_flat, dataset)
            else:
                # Uniform dataset: diff is numpy array
                diff = np.abs(diff)
                nConf = diff.shape[0]
                diff = diff.reshape(-1)

                idxs = np.argwhere((diff >= x0) & (diff <= x1))
                idxs = flatIndexToConfigIndex(idxs, dataset)

            return idxs

    class EnergyErrorPlot(BasicPlotWidget):
        smoothing = 1

        def __init__(self, handler, **kwargs):
            super().__init__(
                handler,
                title="Energy MAE timeline",
                name="Energy Error TimelineP",
                **kwargs,
            )
            self.setDataDependencies("energyError")
            self.setXLabel("Configuration index")
            self.setYLabel("Energy MAE", getConfig("energyUnit"))

            self.slider = Slider(
                parent=self,
                hasEditBox=True,
                label="Smoothing",
                nMin=1,
                nMax=10000,
            )
            self.slider.setToolTip("Number of points in sliding average")
            self.addOption(self.slider)
            self.slider.setCallbackFunc(self.updateSmoothing)
            self.eventSubscribe(
                "ENERGY_SHIFT_CHANGED", self.onEnergyShiftChanged
            )

        def onEnergyShiftChanged(self):
            shifted = self.handler.energyShiftEnabled
            self.titleLabel.setText(
                "Energy MAE timeline (shifted)"
                if shifted
                else "Energy MAE timeline"
            )
            self.visualRefresh(force=True)

        def updateSmoothing(self, value):
            self.smoothing = value
            self.visualRefresh(force=True, noAutoRange=True)

        def addPlots(self):
            #self.slider.setMinMax(1, self.env.getMaxSize()//2+1)  # requires more thinking (what happens to current smoothings when deleting datasets)

            smoothing = self.smoothing
            shifted = self.handler.energyShiftEnabled
            for data in self.getWatchedData():
                de = data["dataEntry"]
                err = de.get("diff")
                if shifted:
                    err = err - de.get("shift")
                err = np.convolve(
                    err, np.ones(smoothing) / smoothing, mode="valid"
                )
                self.plot(
                    np.arange(err.shape[0]),
                    np.abs(err),
                    autoColor=data,
                    autoLabel=data,
                )

        def getDatasetSubIndices(self, dataset, model):
            (xRange, yRange) = self.getRanges()
            N = dataset.getN()
            x0, x1 = xRange
            return np.arange(
                max(0, int(x0 + self.smoothing)),
                min(N, int(x1 + self.smoothing)),
            )

    class ForcesErrorPlot(BasicPlotWidget):

        smoothing = 1

        def __init__(self, handler, **kwargs):
            super().__init__(
                handler,
                title="Forces MAE timeline",
                name="Force Error Timeline",
                **kwargs,
            )
            self.setDataDependencies("forcesError")
            self.setXLabel("Configuration index")
            self.setYLabel("Forces MAE", getConfig("forceUnit"))

            self.slider = Slider(
                parent=self,
                hasEditBox=True,
                label="Smoothing",
                nMin=1,
                nMax=10000,
            )
            self.addOption(self.slider)
            self.slider.setCallbackFunc(self.updateSmoothing)

        def updateSmoothing(self, value):
            self.smoothing = value
            self.visualRefresh(force=True, noAutoRange=True)

        def addPlots(self):
            #self.slider.setMinMax(1, self.env.getMaxSize()//2+1)  # requires more thinking (what happens to current smoothings when deleting datasets)
            # also what happens if the smoothing value is larger than the new added dataset's size
            smoothing = self.smoothing
            for data in self.getWatchedData():
                err = data["dataEntry"].get("diff")

                # Handle variable vs uniform datasets
                if isinstance(err, list):
                    # Variable dataset: err is list of arrays
                    mae = np.array([np.mean(np.abs(e)) for e in err])
                else:
                    # Uniform dataset: err is (N, M, 3) array
                    mae = err.reshape(err.shape[0], -1)
                    mae = np.mean(np.abs(mae), axis=1)

                # Apply smoothing
                mae = np.convolve(
                    mae, np.ones(smoothing) / smoothing, mode="valid"
                )
                self.plot(
                    np.arange(mae.shape[0]),
                    mae,
                    autoColor=data,
                    autoLabel=data,
                )

        def getDatasetSubIndices(self, dataset, model):
            (xRange, yRange) = self.getRanges()
            N = dataset.getN()
            x0, x1 = xRange
            return np.arange(
                max(0, int(x0 + self.smoothing)),
                min(N, int(x1 + self.smoothing)),
            )

    plt = EnergyErrorDistPlot(UIHandler, parent=ct)
    ct.addWidget(plt, 0, 0)
    ct.addDataSelectionCallback(plt.setModelDatasetDependencies)

    plt = ForcesErrorDistPlot(UIHandler, parent=ct)
    ct.addWidget(plt, 0, 1)
    ct.addDataSelectionCallback(plt.setModelDatasetDependencies)

    plt = EnergyErrorPlot(UIHandler, parent=ct)
    ct.addWidget(plt, 1, 0)
    ct.addDataSelectionCallback(plt.setModelDatasetDependencies)

    plt = ForcesErrorPlot(UIHandler, parent=ct)
    ct.addWidget(plt, 1, 1)
    ct.addDataSelectionCallback(plt.setModelDatasetDependencies)

    # TABLES
    scrollContainer = HorizontalContainerScrollArea(parent=ct)
    scrollContainer.content.layout.setSpacing(32)

    class BaseTable(Table):
        def __init__(self, **kwargs):
            super().__init__(UIHandler, parent=ct, **kwargs)
            ct.addDataSelectionCallback(self.setModelDatasetDependencies)

        def getSize(self):
            nCols = len(self.getDatasetDependencies())
            nRows = len(self.getModelDependencies())
            return (nRows, nCols)

        def getLeftHeader(self, i):
            models = self.getModelDependencies()
            model = self.handler.env.getModel(models[i])
            return f"{model.getDisplayName()}"

        def getTopHeader(self, i):
            datasets = self.getDatasetDependencies()
            dataset = self.handler.env.getDataset(datasets[i])
            return f"{dataset.getDisplayName()}"

    class EnergyMAETable(BaseTable):
        def __init__(self):
            super().__init__(title="Energy MAE")
            self.setDataDependencies("energyErrorMetrics")
            self.eventSubscribe(
                "ENERGY_SHIFT_CHANGED", self.onEnergyShiftChanged
            )

        def onEnergyShiftChanged(self):
            shifted = self.handler.energyShiftEnabled
            self.titleLabel.setText(
                "Energy MAE (shifted)" if shifted else "Energy MAE"
            )
            self.visualRefresh()

        def getValue(self, i, j):
            env = self.handler.env
            model = self.getModelDependencies()[i]
            dataset = self.getDatasetDependencies()[j]
            de = env.getData(
                "energyErrorMetrics",
                model=env.getModel(model),
                dataset=env.getDataset(dataset),
            )

            if de is None:
                return ""
            key = "shiftedMae" if self.handler.energyShiftEnabled else "mae"
            return f"{de.get(key):.2f}"

    class EnergyRMSETable(BaseTable):
        def __init__(self):
            super().__init__(title="Energy RMSE")
            self.setDataDependencies("energyErrorMetrics")
            self.eventSubscribe(
                "ENERGY_SHIFT_CHANGED", self.onEnergyShiftChanged
            )

        def onEnergyShiftChanged(self):
            shifted = self.handler.energyShiftEnabled
            self.titleLabel.setText(
                "Energy RMSE (shifted)" if shifted else "Energy RMSE"
            )
            self.visualRefresh()

        def getValue(self, i, j):
            env = self.handler.env
            model = self.getModelDependencies()[i]
            dataset = self.getDatasetDependencies()[j]
            de = env.getData(
                "energyErrorMetrics",
                model=env.getModel(model),
                dataset=env.getDataset(dataset),
            )

            if de is None:
                return ""
            key = (
                "shiftedRmse" if self.handler.energyShiftEnabled else "rmse"
            )
            return f"{de.get(key):.2f}"

    class ForcesMAETable(BaseTable):
        def __init__(self):
            super().__init__(title="Forces MAE")
            self.setDataDependencies("forcesErrorMetrics")

        def getValue(self, i, j):
            env = self.handler.env
            model = self.getModelDependencies()[i]
            dataset = self.getDatasetDependencies()[j]
            de = env.getData(
                "forcesErrorMetrics",
                model=env.getModel(model),
                dataset=env.getDataset(dataset),
            )

            if de is None:
                return ""
            else:
                return f"{de.get('mae'):.2f}"

    class ForcesRMSERable(BaseTable):
        def __init__(self):
            super().__init__(title="Forces RMSE")
            self.setDataDependencies("forcesErrorMetrics")

        def getValue(self, i, j):
            env = self.handler.env
            model = self.getModelDependencies()[i]
            dataset = self.getDatasetDependencies()[j]
            de = env.getData(
                "forcesErrorMetrics",
                model=env.getModel(model),
                dataset=env.getDataset(dataset),
            )

            if de is None:
                return ""
            else:
                return f"{de.get('rmse'):.2f}"

    scrollContainer.addContent(EnergyMAETable())
    scrollContainer.addContent(EnergyRMSETable())
    scrollContainer.addContent(ForcesMAETable())
    scrollContainer.addContent(ForcesRMSERable())
    scrollContainer.addStretch()

    # argument are (row, col, rowSpan, colSpan)
    ct.addWidget(scrollContainer, 2, 0, 1, 2)
