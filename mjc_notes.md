
install env

```sh
export PROJ=glounts10.0
conda create -n $PROJ python=3.8 -y
conda activate $PROJ
mamba install -y ipykernel pip ipywidgets
# pip install torch==1.10.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
mamba install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install gluonts==0.10.0
pip install -e .
```
```sh
export PROJ=gluonts10.0
conda create -n $PROJ python=3.9 -y
conda activate $PROJ
mamba install -y ipykernel pip ipywidgets
# torch
mamba install  -y ipykernel pip ipywidgets pytorch==1.13.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install -y xformers -c xformers/label/dev
mamba install -y conda-lock
# gluonts
pip install gluonts==0.10.0
pip install -e .
# pip install "gluonts[torch,pro]" pytorchts vectorbt
python -m ipykernel install --user --name $PROJ --display-name $PROJ
```
