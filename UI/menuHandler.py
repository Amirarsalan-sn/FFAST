from events import EventClass
from PySide6.QtWidgets import QFileDialog
from UI.Templates import customFileDialog
import os


class MenuHandler(EventClass):
    def __init__(self, window):
        self.handler = window.handler
        self.window = window
        self.connectActions()

    def connectActions(self):
        handler, window = (self.handler, self.window)
        mb = window.menuBar()

        # FILE
        File = mb.addMenu("&File")
        File.addAction("Save", self.onSave, "Ctrl+s")
        File.addAction("Load", self.onLoad, "Ctrl+l")

        File.addAction("Load Dataset", self.onDatasetLoad, "Ctrl+d")
        File.addAction("Load Model", self.onModelLoad, "Ctrl+m")

        File.addAction("Load Zero Model", self.loadZeroModel, "Ctrl+0")
        File.addAction("Load Prediction", self.loadPrepredictedModel, "Ctrl+p")

        # File.addAction("Preferences", self.onPreferences)
        # File.addAction("Exit", self.onExit)

        # LOUPE
        Loupe = mb.addMenu("&Loupe")
        Loupe.addAction("New", self.newLoupe, "Ctrl+n")

    def onSave(self):
        workdir = self.handler.workdir
        (path, _) = QFileDialog.getSaveFileName(self.handler.window, "Save File", workdir)
        if path is None or path.strip() == "":
            return

        self.handler.env.newTask(
            self.handler.env.save,
            args=(path,),
            visual=True,
            name=f"Saving at {os.path.basename(path)}",
            threaded=True,
        )

    def onLoad(self):
        workdir = self.handler.workdir
        path = QFileDialog.getExistingDirectory(self.handler.window, "Select Directory", workdir)
        if path is None or path.strip() == "":
            return

        self.handler.env.newTask(
            self.handler.env.load,
            args=(path,),
            visual=True,
            name=f"Loading {os.path.basename(path)}",
            threaded=True,
        )

    def onPreferences(self):
        pass

    def onExit(self):
        self.eventPush("QUIT_EVENT")

    def onDatasetLoad(self):
        env = self.handler.env
        workdir = self.handler.workdir
        fileTypes = sorted(list(env.datasetTypes.keys()))
        extensions = [
            env.datasetTypes[x].datasetFileExtension for x in fileTypes
        ]
        path, typ = customFileDialog(
            self.handler.window, fileTypes=fileTypes, extensions=extensions, directory=workdir
        )

        env.taskLoadDataset(path, typ)

    def onModelLoad(self):
        env = self.handler.env
        workdir = self.handler.workdir
        fileTypes = list(env.modelTypes.keys())
        extensions = [env.modelTypes[x].modelFileExtension for x in fileTypes]
        path, typ = customFileDialog(
            self.handler.window, fileTypes=fileTypes, extensions=extensions, directory=workdir
        )

        env.taskLoadModel(path, typ)

    def loadPrepredictedModel(self):
        env = self.handler.env
        workdir = self.handler.workdir
        names = [x.getName() for x in env.getAllDatasets(excludeSubs=True)]
        keys = [x.fingerprint for x in env.getAllDatasets(excludeSubs=True)]
        extensions = ["*"] * len(names)
        extensions += ["*.npz"] * len(names)
        names += names

        path, typ = customFileDialog(
            self.handler.window, fileTypes=names, extensions=extensions, directory=workdir
        )
        idx = names.index(typ)
        env.loadPrepredictedDataset(path, keys[idx])

    def newLoupe(self):
        self.handler.newLoupe()

    def loadZeroModel(self):
        env = self.handler.env
        env.loadZeroModel()
