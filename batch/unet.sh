source $SCRATCH/virtualenvs/env-tensorflow-0.9.0/bin/activate
export LD_LIBRARY_PATH=/apps/daint/UES/5.2.UP04/sandbox-ds/tensorflow/cudadnn/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/daint/UES/5.2.UP04/sandbox-ds/tensorflow/cudadnn/lib64/:$LD_LIBRARY_PATH
srun time python ../scripts/rfi_launcher.py