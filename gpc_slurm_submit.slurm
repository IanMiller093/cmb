#!/bin/bash
#SBATCH -J pysm_job
#SBATCH -p mathm_compute
#SBATCH --qos=mathm
#SBATCH -o test.%j.out
#SBATCH -e test.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miller42@upenn.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home3/miller42/projects/cmb_component_separation/.venv/bin/activate

"$@"