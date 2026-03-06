# Attention Smoothing Is All You Need For Unlearning

Official implementation of the paper:

**Attention Smoothing Is All You Need For Unlearning**  
*International Conference on Learning Representations (ICLR) 2026*

**Saleh Zare Zade, Xiangyu Zhou, Sijia Liu, Dongxiao Zhu**

📄 **Paper Links**

[![arXiv](https://img.shields.io/badge/arXiv-2603.01285-b31b1b.svg)](https://arxiv.org/abs/2603.01285)
[![ICLR](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=sX9HbELwLO)

---

# Overview

Large Language Models (LLMs) may memorize sensitive, copyrighted, or unsafe information during training. Removing this knowledge after training is challenging because retraining from scratch is computationally expensive and many existing unlearning methods degrade model utility.

This repository provides the implementation of **Attention Smoothing Unlearning (ASU)**. The method removes memorized knowledge by modifying the attention behavior of transformer models. Specifically, ASU increases the **softmax temperature inside attention**, which smooths the attention distribution and weakens token associations responsible for reconstructing memorized content.

ASU formulates unlearning as **self-distillation from a forget-teacher derived from the model’s own attention**. The approach suppresses lexical and semantic associations while preserving general language capability.

Experiments show that ASU achieves strong forgetting performance while maintaining model utility across several benchmarks.

---

# Method

The key mechanism in ASU is **attention smoothing**.

Standard transformer attention is defined as:

$$
\alpha_{ij} = \text{softmax}\left(\frac{q_i k_j^T}{\sqrt{d}}\right)
$$

ASU introduces a temperature parameter $\tau > 1$:

$$
\alpha_{ij}^{(\tau)} =
\text{softmax}\left(
\frac{q_i k_j^T}{\tau \sqrt{d}}
\right)
$$

Increasing the temperature $\tau$:

- flattens attention distributions
- weakens lexical and semantic token associations
- suppresses reconstruction of memorized knowledge

The smoothed attention is used to create a **forget-teacher**, and the model is trained through **self-distillation** to match the teacher’s behavior while preserving useful capabilities.

---

# Features

- Attention-based LLM unlearning framework
- Minimal modification to transformer architectures
- Compatible with common open-source LLMs
- Supports multiple unlearning benchmarks
- Preserves model utility while removing memorized knowledge

---

# Benchmarks

This implementation supports evaluation on several unlearning benchmarks:

- **TOFU** — benchmark for LLM unlearning evaluation  
- **MUSE** — benchmark for measuring unlearning behavior  
- **WMDP** — safety benchmark for knowledge removal

---

# Installation

Clone the repository and create a conda environment.

```bash
git clone https://github.com/<username>/attention-smoothing-unlearning
cd attention-smoothing-unlearning

conda create -n asu python=3.10
conda activate asu

pip install -r requirements.txt
```

---

# Dataset Setup

Download the datasets used in the experiments.

```bash
bash scripts/download_tofu.sh
bash scripts/download_muse.sh
bash scripts/download_wmdp.sh
```

If you already have the datasets, update the dataset paths in the configuration files.

---

# Training

Example training command:

```bash
python train_unlearning.py \
  --model_name meta-llama/Llama-2-7b \
  --dataset tofu \
  --temperature 2.0 \
  --learning_rate 2e-5 \
  --batch_size 8
```

Key parameters:

| Parameter | Description |
|---|---|
| `temperature` | attention smoothing temperature |
| `forget_set` | dataset to remove knowledge from |
| `retain_set` | dataset to preserve knowledge |
| `distill_weight` | weight for self-distillation loss |

---

# Evaluation

Run evaluation on the supported benchmarks.

```bash
python evaluate.py \
  --model_path checkpoints/asu \
  --benchmark tofu
```

Evaluation metrics include:

- forgetting effectiveness
- retained model utility
- task performance on QA and generation tasks

---

# Repository Structure

```
.
├── configs
│   └── training configurations
├── datasets
│   └── dataset loaders and preprocessing
├── models
│   └── transformer modifications for ASU
├── training
│   └── unlearning training pipeline
├── evaluation
│   └── benchmark evaluation scripts
├── scripts
│   └── dataset download scripts
└── README.md
```

---

# Acknowledgements

This repository builds upon several open-source projects:

- https://github.com/sail-sg/closer-look-LLM-unlearning  
- https://github.com/swj0419/muse_bench  
- https://github.com/centerforaisafety/wmdp  
- https://github.com/locuslab/open-unlearning  

We thank the authors of these repositories for making their work publicly available.

---

# Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zarezade2026attention,
  title={Attention Smoothing Is All You Need For Unlearning},
  author={Zare Zade, Saleh and Zhou, Xiangyu and Liu, Sijia and Zhu, Dongxiao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

---

# License

This project is released under the MIT License.
