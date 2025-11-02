#!/bin/bash
#SBATCH --job-name=clean_conda
#SBATCH --output=slurms/slurm_outputs/clean_conda_output.txt
#SBATCH --error=slurms/slurm_outputs/clean_conda_error.txt
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1

echo "Cleaning corrupted conda cache..."
echo "This will remove all cached packages and force re-download on next install"

# Clean all conda caches
conda clean --all --yes

echo ""
echo "Conda cache cleaned successfully!"
echo "The next environment creation will download fresh packages."
