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

        File.addAction("Load Zero Model", self.onZeroModelLoad, "Ctrl+0")
        File.addAction("Load Prediction", self.onPrepredictedModelLoad, "Ctrl+p")

        # File.addAction("Preferences", self.onPreferences)
        # File.addAction("Exit", self.onExit)

        # LOUPE
        Loupe = mb.addMenu("&Loupe")
        Loupe.addAction("New", self.newLoupe, "Ctrl+n")
        Loupe.addSeparator()

        # Bond Width submenu
        bondMenu = Loupe.addMenu("Bond Width")
        bondMenu.addAction("Thin (10)", lambda: self.setBondWidth(10))
        bondMenu.addAction("Normal (25)", lambda: self.setBondWidth(25))
        bondMenu.addAction("Thick (50)", lambda: self.setBondWidth(50))
        bondMenu.addAction("Extra Thick (100)", lambda: self.setBondWidth(100))
        # TODO: add custom bond width dialog
        # bondMenu.addSeparator()
        # bondMenu.addAction("Custom...", self.showBondWidthDialog)

        # Atom Size submenu
        atomMenu = Loupe.addMenu("Atom Size")
        atomMenu.addAction("50%", lambda: self.setAtomSize(0.5))
        atomMenu.addAction("75%", lambda: self.setAtomSize(0.75))
        atomMenu.addAction("100%", lambda: self.setAtomSize(1.0))
        atomMenu.addAction("150%", lambda: self.setAtomSize(1.5))
        atomMenu.addAction("200%", lambda: self.setAtomSize(2.0))
        # TODO: add custom atom size dialog
        # atomMenu.addSeparator()
        # atomMenu.addAction("Custom...", self.showAtomSizeDialog) 

        # Colors submenu
        colorMenu = Loupe.addMenu("Colors")
        colorMenu.addAction("Bond Color...", self.showBondColorPicker)
        colorMenu.addAction("Background Color...", self.showBackgroundColorPicker)

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

    def onPrepredictedModelLoad(self):
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
        env.taskLoadPrepredictedDataset(path, keys[idx])

    def newLoupe(self):
        self.handler.newLoupe()

    def onZeroModelLoad(self):
        env = self.handler.env
        env.taskLoadZeroModel()

    def setBondWidth(self, width):
        """Set bond width for the active Loupe."""
        loupe = self.handler.getActiveLoupe()
        if not loupe:
            return
        loupe.settings.setParameter("bondWidth", width, refresh=True)

    def setAtomSize(self, scale):
        """Set atom size scale for the active Loupe."""
        loupe = self.handler.getActiveLoupe()
        if loupe and hasattr(loupe, 'settings'):
            loupe.settings.setParameter("atomSizeScale", scale, refresh=True)

    def showBondWidthDialog(self):
        """Show custom bond width input dialog."""
        loupe = self.handler.getActiveLoupe()
        if not loupe:
            return

        from PySide6.QtWidgets import QInputDialog
        current = loupe.settings.get("bondWidth", 200)
        value, ok = QInputDialog.getInt(
            self.window,
            "Bond Width",
            "Enter bond width (pixels):",
            value=current,
            min=10,
            max=1000,
            step=10
        )
        if ok:
            self.setBondWidth(value)

    def showAtomSizeDialog(self):
        """Show custom atom size input dialog."""
        loupe = self.handler.getActiveLoupe()
        if not loupe:
            return

        from PySide6.QtWidgets import QInputDialog
        current = loupe.settings.get("atomSizeScale", 1.0)
        value, ok = QInputDialog.getDouble(
            self.window,
            "Atom Size",
            "Enter atom size scale:",
            value=current,
            min=0.1,
            max=10.0,
            decimals=2
        )
        if ok:
            self.setAtomSize(value)

    def showBondColorPicker(self):
        """Show bond color picker dialog."""
        loupe = self.handler.getActiveLoupe()
        if not loupe:
            return

        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        from config.userConfig import getConfig

        current_hex = loupe.settings.get("bondColor", getConfig("loupeBondsColor", "#404040"))
        current_color = QColor(current_hex)

        color = QColorDialog.getColor(
            current_color,
            self.window,
            "Select Bond Color"
        )

        if color.isValid():
            hex_color = color.name()
            loupe.settings.setParameter("bondColor", hex_color, refresh=True)

    def showBackgroundColorPicker(self):
        """Show background color picker dialog."""
        loupe = self.handler.getActiveLoupe()
        if not loupe:
            return

        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        from config.userConfig import getConfig

        current_hex = getConfig("loupeBGColor", "#000000")
        current_color = QColor(current_hex)

        color = QColorDialog.getColor(
            current_color,
            self.window,
            "Select Background Color"
        )

        if color.isValid():
            # Update canvas background directly
            loupe.canvas.canvas.bgcolor = color.getRgbF()[:3]
            loupe.canvas.canvas.update()
