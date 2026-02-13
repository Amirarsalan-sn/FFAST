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
