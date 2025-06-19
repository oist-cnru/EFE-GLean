# EFE-GLean experimental demonstration

This is a demonstration of EFE-GLean, a goal-directed active inference planning scheme as described in Active Inference with Dynamic Planning and Information Gain in Continuous Space by Inferring Low-Dimensional Latent States.

This repository contains the necessary datasets, programs, and configuration files to run the experiments and visualize the results as seen in the aforementioned paper.

## Requirements
* Linux on a recent x86-64 CPU (Tested on Ubuntu Linux 20.04, 22.04 and 24.04)
* Python 3.8, 3.10 or 3.12

## Install LibPvrnn
This program uses a pre-release build of LibPvrnn as its backbone. For this paper, LibPvrnn is provided as a wheel file for Python 3.8, 3.10 and 3.12.

To run the experiments, it is first necessary to install and configure LibPvrnn as follows.
```bash
python -m venv .venv
pip install LibPvrnn/dist/libpvrnn-2.1a2-cp312-cp312-linux_x86_64.whl 
export PVRNN_SAVE_DIR="/path/to/EFE-GLean/LibPvrnn"
```
Set the ```PVRNN_SAVE_DIR``` environmental variable to the path to where the LibPvrnn directory is placed.

**This build of LibPvrnn is not optimized for performance. It is only intended to preview the capabilities of EFE-GLean. Functionality outside of the provided scripts is not guaranteed.**

For any inquiries regarding LibPvrnn, please contact the authors.

## Run experiments
To run Experiments 1, 2 and 3 from the paper, first install LibPvrnn as shown above, then run the following commands.
```bash
pip install matplotlib
cd Tmaze
python inference_discrete_aifg.py ../LibPvrnn/configs/2d_pftdsg.toml --input_goal_reached
python inference_aifg_rgbdc.py ../LibPvrnn/configs/2d_pft_msgs_w2.toml --er_samples 100 --trials 100 --input_goal_sense --max_step_size 0.5 --max_steps 25
python inference_aifg_rgbdc.py ../LibPvrnn/configs/2d_pft_rgbdc4s.toml --er_samples 100 --trials 100 --input_goal_sense --input_corner_sense --rgb_color
```

To generate the figures, please use the included Jupyter notebooks in the ```Tmaze``` directory.
