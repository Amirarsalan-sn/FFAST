import os
import logging
import numpy as np
from datasetLoaders.loader import DatasetLoader, VariableDatasetLoader
import ase.io
from collections.abc import Iterable

logger = logging.getLogger("FFAST")

class aseDatasetLoader(DatasetLoader):
    datasetName = "ase"
    datasetFileExtension = "*"
    saveFormats = ["db", "xyz", "extxyz", "traj", "vasp", "dftb"]

    def __init__(self, path, *args, **kwargs):
        super().__init__(path)
        self.atomsList = ase.io.read(path, index=":")
        _, self.file_extension = os.path.splitext(path)
        self.N = len(self.atomsList)

        exAtoms = self.atomsList[0]  # assumes all the same molecule!!

        self.nAtoms = len(exAtoms)
        self.z = exAtoms.get_atomic_numbers()

        if hasattr(exAtoms, "cell"):
            self.lattice = exAtoms.cell
        else:
            self.lattice = None

        self.chem = self.zToChemicalFormula(self.z)


    def ForceKeys(self):
        exAtoms = self.atomsList[0]
        num_key = 0
        forcekeys = []
        for key in exAtoms.arrays.keys():
            if "force" in key.lower():
                logger.debug(f"Found forces in array '{key}' for index 0.")
                num_key += 1
                forcekeys.append(key)

        return forcekeys

    def EneregyKeys(self):
        exAtoms = self.atomsList[0]
        num_key = 0
        energykeys = []
        for key in exAtoms.info.keys():
            if "energy" in key.lower():
                logger.debug(f"Found energy in array '{key}' for index 0.")
                num_key += 1
                energykeys.append(key)

        return energykeys

    def getN(self):
        return self.N

    def getNAtoms(self):
        return self.nAtoms

    def getChemicalFormula(self):
        return self.chem

    def getCoordinates(self, indices=None):
        # probably should just do it once at the start and save it as np arrays?
        if indices is None:
            indices = np.arange(self.N)
        elif not isinstance(indices, Iterable):
            return self.atomsList[indices].get_positions()

        R = []
        for idx in indices:
            R.append(self.atomsList[idx].get_positions())

        return np.array(R)

    def getEnergies(self, indices=None):
        # probably should just do it once at the start and save it as np arrays?
        keys = self.EneregyKeys()
        if len(keys) == 0:
            if indices is None:
                indices = np.arange(self.N)
            elif not isinstance(indices, Iterable):
                return self.atomsList[indices].get_potential_energy()

            R = []
            for idx in indices:
                R.append(self.atomsList[idx].get_potential_energy())
        else:
            key = keys[0]
            logger.info(f"Using energies from array(s) {key} out of {keys}.")
            if indices is None:
                indices = np.arange(self.N)
            elif not isinstance(indices, Iterable):
                return self.atomsList[indices].info[key]

            R = []
            for idx in indices:
                R.append(self.atomsList[idx].info[key])
        return np.array(R)

    def getForces(self, indices=None):
        # probably should just do it once at the start and save it as np arrays?
        keys = self.ForceKeys()

        if len(keys) == 0:
            if indices is None:
                indices = np.arange(self.N)
            elif not isinstance(indices, Iterable):
                return self.atomsList[indices].get_forces()

            R = []
            for idx in indices:
                R.append(self.atomsList[idx].get_forces())
        else:
            key = keys[0]
            logger.info(f"Using forces from array(s) {key} out of {keys}.")
            if indices is None:
                indices = np.arange(self.N)
            elif not isinstance(indices, Iterable):
                return self.atomsList[indices].arrays[key]

            R = []
            for idx in indices:
                R.append(self.atomsList[idx].arrays[key])

        return np.array(R)

    def getElements(self):
        return self.z

    def getLattice(self, indices=None):
        """Return the unit cell/lattice for specified frame(s)."""
        if indices is None:
            indices = np.arange(self.N)
        elif not isinstance(indices, Iterable):
            return self.atomsList[indices].get_cell()

        R = []
        for idx in indices:
            R.append(self.atomsList[idx].get_cell())
        return np.array(R)

    @staticmethod
    def saveDataset(dataset, path, format=None, taskID=None):
        from ase import Atoms
        from ase.calculators.calculator import Calculator

        R, F = dataset.getCoordinates(), dataset.getForces()
        E, zStr = dataset.getEnergies(), dataset.getElementsName()

        atoms = []

        class FakeCalc(Calculator):
            def __init__(self):
                pass

        for i in range(R.shape[0]):
            atom = Atoms(positions=R[i], symbols=zStr)
            atom.calc = FakeCalc()
            atom.calc.results = {"forces": F[i], "energy": E[i]}
            atoms.append(atom)

        ase.io.write(path, atoms, format=format)


class VariableASEDatasetLoader(VariableDatasetLoader):
    """Loader for ASE datasets with variable-sized molecules."""

    datasetName = "ase (variable)"
    datasetFileExtension = "*"
    saveFormats = ["db", "xyz", "extxyz", "traj", "vasp", "dftb"]

    def __init__(self, path, *args, **kwargs):
        super().__init__(path)
        self.atomsList = ase.io.read(path, index=":")
        _, self.file_extension = os.path.splitext(path)
        self.N = len(self.atomsList)

        # Build flat arrays
        R_list, F_list, E_list, z_list = [], [], [], []
        offsets = [0]

        # Detect force and energy keys from first frame
        force_keys = self._detectForceKeys()
        energy_keys = self._detectEnergyKeys()

        for atoms in self.atomsList:
            n_atoms = len(atoms)
            R_list.append(atoms.get_positions())

            # Handle forces
            if force_keys:
                F_list.append(atoms.arrays[force_keys[0]])
            else:
                try:
                    F_list.append(atoms.get_forces())
                except:
                    # If forces not available, use zeros
                    F_list.append(np.zeros((n_atoms, 3)))

            # Handle energies
            if energy_keys:
                E_list.append(atoms.info[energy_keys[0]])
            else:
                try:
                    E_list.append(atoms.get_potential_energy())
                except:
                    # If energy not available, use zero
                    E_list.append(0.0)

            z_list.append(atoms.get_atomic_numbers())
            offsets.append(offsets[-1] + n_atoms)

        # Convert to flat arrays
        self.R_flat = np.vstack(R_list)
        self.F_flat = np.vstack(F_list)
        self.E = np.array(E_list)
        self.z_flat = np.concatenate(z_list)
        self.molecule_offsets = np.array(offsets)

        # Store lattice info if present
        if hasattr(self.atomsList[0], "cell"):
            self.lattice = [atoms.get_cell() for atoms in self.atomsList]
        else:
            self.lattice = None

        # Chemical formula: show range
        atom_counts = self.getNAtoms()
        self.chem = f"Variable ({atom_counts.min()}-{atom_counts.max()} atoms)"

    def _detectForceKeys(self):
        """Detect force keys in the first atoms object."""
        if not self.atomsList:
            return []
        exAtoms = self.atomsList[0]
        force_keys = []
        for key in exAtoms.arrays.keys():
            if "force" in key.lower():
                logger.debug(f"Found forces in array '{key}' for index 0.")
                force_keys.append(key)
        return force_keys

    def _detectEnergyKeys(self):
        """Detect energy keys in the first atoms object."""
        if not self.atomsList:
            return []
        exAtoms = self.atomsList[0]
        energy_keys = []
        for key in exAtoms.info.keys():
            if "energy" in key.lower():
                logger.debug(f"Found energy in array '{key}' for index 0.")
                energy_keys.append(key)
        return energy_keys

    def getLattice(self, indices=None):
        """Return the unit cell/lattice for specified frame(s)."""
        if not hasattr(self, 'lattice') or self.lattice is None:
            return None

        if indices is None:
            return np.array(self.lattice)
        elif not isinstance(indices, Iterable):
            return self.lattice[indices]

        return np.array([self.lattice[i] for i in indices])

    @staticmethod
    def saveDataset(dataset, path, format=None, taskID=None):
        """Save variable dataset to ASE format."""
        from ase import Atoms
        from ase.calculators.calculator import Calculator

        if not dataset.isVariable:
            # Fall back to uniform saver
            return aseDatasetLoader.saveDataset(dataset, path, format, taskID)

        atoms = []

        class FakeCalc(Calculator):
            def __init__(self):
                pass

        for i in range(dataset.getN()):
            r = dataset.getCoordinates(i)  # (n_atoms_i, 3)
            f = dataset.getForces(i)       # (n_atoms_i, 3)
            e = dataset.getEnergies(i)     # scalar
            z = dataset.getElements(i)     # (n_atoms_i,)
            zStr = [dataset.zIntToZStr[x] for x in z]

            atom = Atoms(positions=r, symbols=zStr)
            atom.calc = FakeCalc()
            atom.calc.results = {"forces": f, "energy": e}
            atoms.append(atom)

        ase.io.write(path, atoms, format=format)


def loadData(env):
    """Smart loader that auto-detects uniform vs variable datasets."""

    class SmartASELoader:
        datasetName = "ase (auto)"
        datasetFileExtension = "*"
        saveFormats = ["db", "xyz", "extxyz", "traj", "vasp", "dftb"]

        def __call__(self, path):
            # Read all frames to detect if uniform or variable
            atomsList = ase.io.read(path, index=":")
            atom_counts = [len(atoms) for atoms in atomsList]

            if len(set(atom_counts)) == 1:
                # Uniform dataset
                logger.info(f"Loading uniform ASE dataset: {len(atomsList)} molecules, {atom_counts[0]} atoms each")
                return aseDatasetLoader(path)
            else:
                # Variable dataset
                logger.info(f"Loading variable ASE dataset: {len(atomsList)} molecules, {min(atom_counts)}-{max(atom_counts)} atoms")
                return VariableASEDatasetLoader(path)

    env.initialiseDatasetType(SmartASELoader())
