# AuESINet
The official code repository of "Pseudodata-guided Invariant Representation Learning Boosts the Out-of-Distribution Generalization in Enzymatic Kinetic Parameter Prediction"

# Installation
Create a new environment for OmniESI:
```shell
conda create -n AuESI python=3.8
conda activate AuESI
```
Installation for pytorch 1.12.1:
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Installation for dgl:
```
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
```
Installation for other dependencies:
```shell
pip install -r requirements.txt
```

# Data preparation

Please refer to the data preparation section of OmniESI. Ref:https://github.com/Hong-yu-Zhang/OmniESI


# Quick reproduction

```shell
python test_ood.py --model configs/model/MESI.yaml --data configs/data/[Data].yaml --weight results/[Data] --task regression
```
