#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=jupyterNB
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
##SBATCH --cpus-per-task=2
#SBATCH --job-name=viz-cap

module purge
module load python/intel/2.7.12

#source /beegfs/sb3923-share/pytorch-cpu/py3.6.3/bin/activate
module load fftw/intel/3.3.6-pl2
python -m pip install pyFFTW --user
python -c "import pyfftw; print(pyfftw.__file__); print(pyfftw.__version__)"
module load h5py/intel/2.7.0rc2

pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp27-cp27mu-linux_x86_64.whl
pip install torchvision 
#python3 -m pip install numpy cython mpi4py --user
#python3 -m pip install nbodykit[extras] --user

port=$(shuf -i 6000-9999 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1
cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to prince with command

ssh $USER@prince

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on prince compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)




