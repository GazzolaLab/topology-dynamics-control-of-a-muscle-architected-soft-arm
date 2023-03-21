# Topology, dynamics and control of dynamics, and control of an octopus-analog muscular hydrostat

This repository corresponds to the software and case files for running Topology, 
dynamics and control of dynamics, and control of an octopus-analog muscular hydrostat paper. At the backend we 
are using `pyelastica` software for simulations. This repository won't be updated and maintained, but we will keep
developing modules and examples used here in this **REPO**.

## Installation

Our software runs on `python` and some other dependencies which can be installed easily using `pip`. 
Below are steps of how to install . 

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Virtual python workspace: `conda`.

We recommend using python version above 3.9.

```bash
conda create --name topology-dynamics-control-of-octopus-env
conda activate topology-dynamics-control-of-octopus-env
conda install python=3.9
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

## How to run cases

After successfully completing the installation, you can run the case files for the simulations in [Cases](./Cases) directory.
In order to run the simulation files, please change the bash directory to the directory of the simulation you would like to run and type following command.

```bash
python run_script.py
```

Following the successful completion of the simulation corresponding videos and data will be saved inside the same directory.