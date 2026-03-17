# HW1 — MedViTV2: Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention

[![Paper](https://img.shields.io/badge/arXiv-2502.13693-b31b1b.svg)](https://arxiv.org/abs/2502.13693)
[![Published](https://img.shields.io/badge/Elsevier-Applied_Soft_Computing-blue)](https://doi.org/10.1016/j.asoc.2025.114045)
[![Original Repo](https://img.shields.io/badge/GitHub-MedViTV2-181717?logo=github)](https://github.com/Omid-Nejati/MedViTV2)

This homework is based on the MedViTV2 paper. The goal is to understand the architecture, replicate the original results, and extend the model to custom medical datasets.

---

## Paper Overview

MedViTV2 is the first architecture to integrate **Kolmogorov–Arnold Network (KAN)** layers into a Vision Transformer for generalized medical image classification. It introduces three key improvements over MedViTV1:

### 1. KAN-Integrated Transformer Block
KAN (Kolmogorov–Arnold Network) replaces the standard MLP in the transformer block. This reduces computational complexity (FLOPs) by **44%** while improving classification accuracy.

### 2. Dilated Neighborhood Attention (DiNA)
MedViTV1 used standard Neighborhood Attention (NA). MedViTV2 replaces it with **Dilated Neighborhood Attention**, which expands the receptive field and mitigates feature collapse, improving robustness to corrupted inputs.

| Version | Attention |
|---------|-----------|
| MedViTV1 | Neighborhood Attention |
| MedViTV2 | **Dilated** Neighborhood Attention |

### 3. Hierarchical Hybrid Strategy
A hierarchical architecture that alternates between **Local Feature Perception (LFP)** and **Global Feature Perception (GFP)** blocks, balancing fine-grained local detail and high-level global context.

---

## Modifications from Original

The following changes were made to the original MedViTV2 codebase:

### `MedViT.py`
- Uncommented `attn_drop=drop` and `**extra_args` parameters in the attention block (lines 306–310), enabling full attention dropout and extra argument passing.

### `datasets.py`
- Added two custom dataset loaders using `torchvision.datasets.ImageFolder`:
  - **`Bone_Multiclass`** — multi-class bone fracture classification (10 categories)
  - **`Bone_Binary`** — binary fracture detection (fractured vs. normal)
- Both datasets auto-detect number of classes from the folder structure.

### `main.py`
- Added `print(f'lr: ...')` logging in both MedMNIST and general training loops to monitor learning rate per epoch.
- Changed default `model_name` from `MedViT_tiny` → **`MedViT_small`**.
- Changed default `dataset` from `PAD` → **`Bone_Binary`**.
- Changed default `checkpoint_path` to match `MedViT_small`.
- Fixed `torch.load(..., weights_only=False)` to suppress deprecation warning in PyTorch ≥ 2.x.
- Added example run commands at the bottom of the file.

### `Tutorials/Evaluation.ipynb`
- Updated natten installation command:
  - Before: `pip3 install natten==0.17.3+torch250cu124 -f https://shi-labs.com/natten/wheels/`
  - After: `pip install natten==0.17.5+torch250cu124 -f https://whl.natten.org`

---

## Datasets

### Experiment 1 — CPN (Covid-Pneumonia-Normal)
- **Task**: 3-class X-ray classification (COVID-19 / Pneumonia / Normal)
- **Source**: Kumar, Sachin (2022). *Covid19-Pneumonia-Normal Chest X-Ray Images*. Mendeley Data, V1.
  - DOI: [10.17632/dvntn9yhd2.1](https://doi.org/10.17632/dvntn9yhd2.1)
- **Training**: Replicated using default configuration with MedViT_small pretrained weights.

### Experiment 2 — Bone Fracture Multi-Region (Multiclass)
- **Task**: 10-class bone fracture classification across multiple body regions
- **Source**: [Kaggle — Fracture Multi-Region X-ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data/data)
- **Preprocessing**: 18 damaged/corrupted images removed. Image rotation augmentation applied to train, validation, and test sets.
- **Training**: Replicated using default configuration; also tested with tuned hyperparameters.

### Experiment 3 — Bone Break Classification (Binary)
- **Task**: Binary fracture detection (fractured vs. normal)
- **Source**: [Kaggle — Bone Break Classification Image Dataset](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset?resource=download)
- **Note**: For each fracture category, images are pre-split into `train/` and `test/` directories.

---

## Setup

Clone and install dependencies:

```bash
git clone https://github.com/<your-username>/Deep-Learning-Paper-Implementation-Practice.git
cd Deep-Learning-Paper-Implementation-Practice/MedViT2
```

Install PyTorch 2.5:
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

Install natten:
```bash
pip install natten==0.17.5+torch250cu124 -f https://whl.natten.org
```

Install remaining requirements:
```bash
pip install -r requirements.txt
```

---

## Training

Train on MedMNIST datasets:
```bash
python main.py --model_name 'MedViT_small' --dataset 'breastmnist' --pretrained False
```

Train on CPN (pretrained):
```bash
python main.py --model_name 'MedViT_small' --dataset 'CPN' --pretrained True
```

Train on Bone Binary (pretrained):
```bash
python main.py --model_name 'MedViT_small' --dataset 'Bone_Binary' --pretrained True
```

Train on Bone Multiclass (pretrained, various configs):
```bash
# Default config
python main.py --model_name 'MedViT_small' --dataset 'Bone_Multiclass' --pretrained True

# Tuned hyperparameters
python main.py --model_name 'MedViT_small' --dataset 'Bone_Multiclass' --pretrained True --lr 0.001 --epochs 200 --batch_size 64

# Tiny model
python main.py --model_name 'MedViT_tiny' --dataset 'Bone_Multiclass' --pretrained True --lr 0.0001 --epochs 100 --batch_size 64
python main.py --model_name 'MedViT_tiny' --dataset 'Bone_Multiclass' --pretrained True --lr 0.0001 --epochs 300 --batch_size 64
```

---

## Results

### CPN — MedViT_small (default config, pretrained)
Results replicated using the default configuration from the original repository.
See slide 9 of `DeepLearning_HW1.pptx` for the full performance table.

### Bone Fracture Multiclass — MedViT_small / MedViT_tiny (pretrained)
Rotation augmentation was applied. 18 damaged images were removed from the dataset.
See slide 11 of `DeepLearning_HW1.pptx` for the full performance table.

### Bone Break Binary — MedViT_small (pretrained)
See slide 13 of `DeepLearning_HW1.pptx` for the full performance table.

---

## File Structure

```
MedViT2/
├── MedViT.py                  # Model architecture (modified)
├── datasets.py                # Dataset loaders incl. custom datasets (modified)
├── main.py                    # Training script (modified)
├── fasterkan.py               # FasterKAN implementation
├── requirements.txt
├── DeepLearning_HW1.pptx      # Homework presentation slides
└── Tutorials/
    ├── Evaluation.ipynb       # Training & evaluation tutorial (modified)
    └── Visualization.ipynb    # Grad-CAM visualization tutorial
```

---

## References

- [MedViTV2 (Original Repo)](https://github.com/Omid-Nejati/MedViTV2)
- [MedViTV1](https://github.com/Omid-Nejati/MedViT)
- [FasterKAN](https://github.com/AthanasiosDelis/faster-kan)
- [NATTEN](https://github.com/SHI-Labs/NATTEN)

## Citation

```bibtex
@article{manzari2025medical,
  title={Medical image classification with KAN-integrated transformers and dilated neighborhood attention},
  author={Manzari, Omid Nejati and Asgariandehkordi, Hojat and Koleilat, Taha and Xiao, Yiming and Rivaz, Hassan},
  journal={Applied Soft Computing},
  pages={114045},
  year={2025},
  publisher={Elsevier}
}

@article{manzari2023medvit,
  title={MedViT: a robust vision transformer for generalized medical image classification},
  author={Manzari, Omid Nejati and Ahmadabadi, Hamid and Kashiani, Hossein and Shokouhi, Shahriar B and Ayatollahi, Ahmad},
  journal={Computers in Biology and Medicine},
  volume={157},
  pages={106791},
  year={2023},
  publisher={Elsevier}
}
```
