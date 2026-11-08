# SCRF

This repository provides the official implementation of **SCRF**, a structure-aware distillation framework for multimodal knowledge graph completion.
(一种结构感知蒸馏的多模态知识图谱补全方法).

## Code Structure

```text
SCRF
├── configs
│   ├── DB15K.json
│   ├── MKG-W.json
│   └── MKG-Y.json
├── datasets
│   ├── DB15K
│   ├── MKG-W
│   └── MKG-Y
├── layers
│   ├── __init__.py
│   └── tucker.py
├── models
│   ├── __init__.py
│   └── scrf.py
├── utils
│   ├── __init__.py
│   ├── data.py
│   └── training.py
├── requirements.txt
├── README.md
└── train.py
```

## Data

Please refer to the [Data preparation section of MoMoK](https://github.com/zjukg/MoMoK#-data-preparation) for the dataset resources and multimodal feature downloads. Download and extract the data, then place the required files in the corresponding directories below.

| Dataset | Download instructions | Local directory |
|---|---|---|
| DB15K | [MoMoK: Data preparation](https://github.com/zjukg/MoMoK#-data-preparation) | `datasets/DB15K/` |
| MKG-W | [MoMoK: Data preparation](https://github.com/zjukg/MoMoK#-data-preparation) | `datasets/MKG-W/` |
| MKG-Y | [MoMoK: Data preparation](https://github.com/zjukg/MoMoK#-data-preparation) | `datasets/MKG-Y/` |

Place the following seven files in each dataset directory:

```text
datasets/<dataset>/
├── entity2id.txt
├── relation2id.txt
├── train.txt
├── valid.txt
├── test.txt
├── img_features.pth
└── text_features.pth
```

The provided configurations use the original multimodal features (`img_features.pth` and `text_features.pth`), rather than the CLIP variants. Each feature tensor must have one row per entity, ordered according to `entity2id.txt`. Keep the entity mappings, data splits, and feature files from the same dataset release together.

The loader applies L2 normalization to image and text features. It constructs inverse relations at runtime; no additional preprocessing script is required for files in this format. Use `--data-root /path/to/datasets` if the data is stored elsewhere.

## Dependencies

The implementation has been run with Python 3.9 and PyTorch 2.5. Install the dependencies from the project directory:

```bash
conda create -n scrf python=3.9 -y
conda activate scrf
python -m pip install -r requirements.txt
```

The commands below use `cuda:0`. Select another GPU with `--device cuda:1`, or use `--device cpu`.

## Train

Run the following commands from the project directory. Selecting a dataset automatically loads its configuration from `configs/`.

### DB15K

```bash
python -u train.py --dataset DB15K --device cuda:0 --output-dir runs/DB15K
```

### MKG-W

```bash
python -u train.py --dataset MKG-W --device cuda:0 --output-dir runs/MKG-W
```

### MKG-Y

```bash
python -u train.py --dataset MKG-Y --device cuda:0 --output-dir runs/MKG-Y
```

## Acknowledgements

Dataset download instructions are provided by [MoMoK](https://github.com/zjukg/MoMoK#-data-preparation). We thank the authors for making their code and dataset resources available.
