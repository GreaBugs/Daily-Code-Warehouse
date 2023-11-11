#!/bin/bash

cd /home/baoyue.shen/fastinst
source .venv/bin/activate
echo "已激活环境"
# CUDA 11.0
# pip uninstall torch torchvision torchaudio
# pip install -U torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/cu110/torch_stable.html
pip install -U torch==1.10.0+cu114 torchvision==0.11.1+cu114 -f https://download.pytorch.org/whl/cu114/torch_stable.html --no-cache-dir


# pip install -U torch==1.9.0 torchvision==0.10.0
# pip install -U torch==1.10.0 torchvision==0.11.0
pip install -U opencv-python
cd /home/baoyue.shen/fastinst/detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
cd ..
# pip install -U git+https://github.com/mcordts/cityscapesScripts.git
pip install -r /home/baoyue.shen/fastinst/FastInst-dis/requirements.txt
python $1
