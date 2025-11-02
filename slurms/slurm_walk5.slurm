#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="zbot-train-walk5"
#SBATCH --time=18:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --constraint="(a100|h100|h200)&(gpu40g|gpu80g)"
#SBATCH --output=slurm_outputs/walk5_output.txt
#SBATCH --error=slurm_outputs/walk5_error.txt
#SBATCH --mail-user=sebasrb@ntnu.no
#SBATCH --mail-type=ALL

# Get the project root directory (parent of slurms folder)
# SLURM_SUBMIT_DIR is the directory where sbatch was called from
cd ${SLURM_SUBMIT_DIR}/..
WORKDIR=$(pwd)
echo "Running from directory: $WORKDIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"

# Create output directory if it doesn't exist
mkdir -p slurm_outputs

# Path to your conda env (change if needed)
ENV_PATH="/cluster/work/$USER/zeroth-trainings/ksim-env/"

module purge
module load Anaconda3/2024.02-1

# Remove old environment if it exists (to ensure Python 3.11)
if [ -d "$ENV_PATH" ]; then
  echo "Removing old environment..."
  rm -rf ${ENV_PATH}
fi

# Create fresh conda environment with Python 3.11
echo "Creating Python 3.11 environment..."
conda create -y --prefix ${ENV_PATH} python=3.11

# Use Python directly from the environment (don't rely on activation)
PYTHON_BIN="${ENV_PATH}/bin/python"
PIP_BIN="${ENV_PATH}/bin/pip"

# Verify we're using the right Python
echo "Python version: $(${PYTHON_BIN} --version)"
echo "Python path: ${PYTHON_BIN}"

# Install dependencies
${PIP_BIN} install --force-reinstall -r requirements.txt
${PIP_BIN} install 'jax[cuda12]'  # If using GPU machine, install JAX CUDA libraries

echo "Installed dependencies:"
pip freeze

# Verify GPU backend
echo "Checking JAX backend:"
${PYTHON_BIN} -c "import jax; print(jax.default_backend())"  # Should print "gpu"

# Set MuJoCo to use headless rendering backend (osmesa)
export MUJOCO_GL=osmesa
echo "Set MUJOCO_GL=osmesa for headless rendering"

# Run training (VARIANT 5: Aggressive Lateral - Strong lateral control)
${PYTHON_BIN} -m training_variants.train_walk5


# Example: to load pretrained checkpoint and run in view mode, uncomment:
# python -m training_variants.train_walk5ing_variants.train_walk5 load_from_ckpt_path=zbot_walking_task/run_9/checkpoints/ckpt.bin # run_mode=view

# Find the run directory that was created/used
RUN_DIR=$(ls -td zbot_walking_task/run_* 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ]; then
  RUN_NUM=$(basename "$RUN_DIR")
  echo "Training used directory: $RUN_DIR ($RUN_NUM)"

  # Rename SLURM output files to use run number instead of job ID
  

  
else
  echo "Warning: Could not find run directory"
fi

conda deactivate

echo "Job finished"
