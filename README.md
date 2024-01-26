# tree-compression

Runs on Cuda 12.3

```
conda create -n env python=3.9 
conda activate env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install matplotlib image-similarity-measures pyfftw openpyxl ninja
```
