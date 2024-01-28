# tree-compression
This project is part of the [2024 Research Project](https://github.com/TU-Delft-CSE/Research-Project) at [TU Delft](https://github.com/TU-Delft-CSE).

# Abstract

Trees are essential components of both real and digital environments. Therefore, it is important to have 3D models of trees that are of high quality and computationally efficient. One way to achieve this is by compressing a high-quality model using billboard rendering, which involves partitioning the tree into multiple planes to produce a similar result to the original.

Our study explores the compression of 3D models using an optimization loop and adapting billboard rendering techniques. We use computer vision primitives to render basic models, which we then optimize by adjusting the texture to resemble the original tree. The models consist of multiple upright planes that are rotated around the central vertical axis of the original tree. We use different optimization functions, such as L1 and L2 losses, to determine the best approach.

We can improve the initial models by bounding the billboards and limiting their heights and widths to that of the trees. Additionally, we can use double-sided textures for the billboards to allow more flexibility for optimizing different species of trees. However, optimizing multiple tree types performs differently for each species, leading to improvements that only benefit certain trees in specific scenarios.

Using quantitative metrics, we determined which models perform best and how similar they are to the original after training. We found that our compressed models generally resemble the original while having only a fraction of the original size.

# Installation

Requires Python 3.9+, VS2019+, Cuda 11.3+ and PyTorch 1.10+

Tested in Anaconda3 with Python 3.9 and PyTorch 1.10

## One time setup (Windows)
Install the Cuda toolkit (required to build the PyTorch extensions). We have used [Cuda 12.3](https://developer.nvidia.com/cuda-downloads) for the duration of the project, however, newer versions should still work. Below is an example for Cuda 12.3:

```
conda create -n env python=3.9 
conda activate env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install matplotlib image-similarity-measures pyfftw openpyxl ninja
```
## Every new command prompt

```
activate env
```

# Example

The normal dataset of quaking aspen can be run by the following comand: 

```
python train.py
```

Running the training algorithm will create a new folder `training_out\` into which the results of a training session are categorized by renders such as the one below, textures, the `model.obj` and the average evaluation results of the test set in `avg_results`. 

![10000](https://github.com/SebastianManda/tree-compression/assets/99266062/b2854bc4-3b7f-4215-bb2a-a671f71dfb28)


# Dataset

Three datasets are provided in the `data\` directory, one for each tree species chosen for testing. New datasets can be created by running the `generate_gt.py` file, however, it is not possible to execute it in the environment provided and must be ran externally.

Requires Python 3.7 and the latest BPY (Blender Python Api) which can be installed as follows:

```
pip install bpy
```

# Licence

This work is made available under the Nvidia Source Code License.

For business inquiries, please visit our website and submit the form: NVIDIA Research Licensing

Models under `models\willow.obj`, `\models\quaking_aspen.obj` and `\models\acer.obj` are derived from the blender plugin https://github.com/friggog/tree-gen under GPL-3.0 License.
