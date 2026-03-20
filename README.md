# Deep Learning Paper Implementation Practice

Weekly SOTA paper-to-code practice — each week re-implements a recent paper, adapts it to a new dataset, and evaluates against published baselines or ablation studies.

> Course: Deep Learning Practice — Institute of Biomedical Informatics

---

## Homework Overview

| # | Folder | Paper | Task | New Dataset |
|---|--------|-------|------|-------------|
| HW1 | [MedViT2](./MedViT2) | MedViTV2 (Applied Soft Computing 2025) | Medical Image Classification | Bone Fracture (binary & multi-class) |
| HW2 | [PDFNet](./PDFNet) | PDFNet (CVPR 2026) | Dichotomous Image Segmentation | Kvasir-SEG (medical polyp) |
| HW3 | [cascade-detr](./cascade-detr) | Cascade-DETR (ICCV 2023) | Object Detection | Aquarium Dataset |
| HW4 | [moonshine](./moonshine) | Moonshine ASR (arXiv 2024) | Speech Recognition Fine-tuning | Fluent Speech Commands |
| HW5 | [PokeGemma](./PokeGemma) | — | LLM SFT (LoRA) | Pokémon GO Battle (self-built, 16K samples) |

---

## HW1 — MedViTV2: Medical Image Classification

[![Paper](https://img.shields.io/badge/arXiv-2502.13693-b31b1b.svg)](https://arxiv.org/abs/2502.13693)
[![Published](https://img.shields.io/badge/Elsevier-Applied_Soft_Computing-blue)](https://doi.org/10.1016/j.asoc.2025.114045)

Replicated MedViTV2, which integrates **KAN (Kolmogorov–Arnold Networks)** into a Vision Transformer and replaces standard Neighborhood Attention with **Dilated Neighborhood Attention** to prevent feature collapse. Extended to two bone fracture datasets beyond the original MedMNIST benchmarks.

**Key results:** KAN reduces FLOPs by 44% vs. standard MLP while maintaining or improving classification accuracy.

---

## HW2 — PDFNet: Dichotomous Image Segmentation

[![Paper](https://img.shields.io/badge/arXiv-2503.06100-b31b1b.svg)](https://arxiv.org/abs/2503.06100)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](https://arxiv.org/abs/2503.06100)

Reproduced PDFNet, which fuses **RGB + pseudo-depth maps** and fine-grained patch features via shared Swin-B encoder. Transferred to **Kvasir-SEG** (colonoscopy polyp segmentation) — a domain shift from natural DIS images. Investigated the Depth Integrity-Prior loss in a depth-ambiguous medical setting.

---

## HW3 — Cascade-DETR: Universal Object Detection

[![Paper](https://img.shields.io/badge/arXiv-2307.11035-b31b1b.svg)](https://arxiv.org/abs/2307.11035)
[![ICCV 2023](https://img.shields.io/badge/ICCV-2023-blue)](https://arxiv.org/abs/2307.11035)

Fine-tuned Cascade-DETR on the **Aquarium Dataset**, applying the paper's cascade IoU-based matching and attention injection mechanism to an out-of-domain underwater detection scenario.

---

## HW4 — Moonshine: ASR Fine-tuning

[![Paper](https://img.shields.io/badge/arXiv-2410.15608-b31b1b.svg)](https://arxiv.org/abs/2410.15608)

Explored fine-tuning Moonshine Tiny (27M params) — a variable-length ASR model using **RoPE** and raw Conv1D audio frontend — on the **Fluent Speech Commands** dataset. Compared four strategies: full fine-tuning, frozen encoder, no-padding (gradient accumulation), and varying learning rates. Baseline Moonshine Tiny achieves near-zero WER on FSC out-of-the-box; fine-tuning degraded performance, suggesting strong pre-trained generalization.

---

## HW5 — PokeGemma: LLM Domain Fine-tuning

[![Model](https://img.shields.io/badge/HuggingFace-google%2Fgemma--2--2b--it-yellow)](https://huggingface.co/google/gemma-2-2b-it)

Fine-tuned **Gemma-2 2B** via **LoRA SFT** using LLaMA Factory on a self-built Pokémon GO battle dataset (16,323 instruction-response pairs generated from pvpoke rankings data). Goal: lift the model from fact retrieval to tactical type-matchup reasoning. LoRA reduces trainable parameters to 10.4M (0.39% of 2.6B total).
