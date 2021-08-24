#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=3:15:00

BASE=$HOME/mixing
RUN=${1}
OUTDIR=${SCRATCH}/mixing/$RUN

#Prepare SCRATCH and move there
#rm -r ${OUTDIR}; mkdir -p ${OUTDIR}
cp ${BASE}/experiments/${RUN}/* ${OUTDIR}/.
ls -l

cd ${OUTDIR}

echo "Starting"

module --force purge
module load CCEnv
module load StdEnv/2020 fftw-mpi mpi4py hdf5-mpi python

export HDF5_MPI=ON
export LDSHARED="icc -shared"
export FFTW_PATH="$SCINET_FFTW_MPI_ROOT"
export MPI_PATH="$I_MPI_ROOT"
export MPLBACKEND=pdf

source ${HOME}/python_env/bin/activate

mpirun python3.8 nl_strat_simulation.py

mpirun python3.8 merge_file.py ./data-mixingsim-00/ --cleanup=True

deactivate

exit
