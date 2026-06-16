# CCG: Uncertainty-aware Cycle Diffusion Model for Fair Glaucoma Diagnosis

Official PyTorch implementation of the MIDL 2026 paper

> **Uncertainty-aware Cycle Diffusion Model for Fair Glaucoma Diagnosis**
> *Ziheng Wang, Shuran Yang, Yan Lin, Wenrui Zang, Yanda Meng*
> Medical Imaging with Deep Learning (MIDL), 2026

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0-orange.svg)](https://pytorch.org/)

---

## 🛠️ Installation

```bash
git clone https://github.com/WANG-ZIHENG/CCG.git
cd CCG
conda create -n ccg python=3.10 -y
conda activate ccg
pip install -r requirements.txt
```

The code is tested with PyTorch 2.0 (CUDA 11.8) on Linux.

---

## 📁 Data Preparation

1. Download the **Harvard-FairVLMed** dataset following the instructions of [FairCLIP](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP).
2. Organise the dataset as follows (paths are configurable via CLI flags):

   ```
   <DATA_ROOT>/fairvlmed10k/
     ├── All/                        # raw .npz files (one per case)
     ├── mask/                       # cup/disc segmentation masks: data_XXXXX_predict.png
     ├── training_slo_fundus/        # real fundus PNGs used as FID reference
     └── data_summary.csv            # demographic + clinical summary
   ```

3. The repository already ships the **paper-aligned 7,363-sample subset** under `data/`, following the subset released in our previous MICCAI 2025 work, [Fairness-Aware vCDR-Controlled Generation for Glaucoma Diagnosis](https://link.springer.com/chapter/10.1007/978-3-032-05114-1_25).

---

## 📦 Pre-trained Checkpoints

| Checkpoint | Size | Purpose | Download |
|---|---|---|---|
| `control_sd21_ini.ckpt` | 6.7 GB | Stable Diffusion 2.1 + empty ControlNet branch (training start) | [HuggingFace](https://huggingface.co/lllyasviel/ControlNet) |
| `ccg_controlnet_sd21.ckpt` | 13.7 GB | Trained CCG ControlNet (Lightning format) | [Google Drive (TBD)](https://drive.google.com/) |

Place both files under `checkpoints/`:

```
checkpoints/
  ├── control_sd21_ini.ckpt
  └── ccg_controlnet_sd21.ckpt
```

Integrity (SHA-256):

```
ccg_controlnet_sd21.ckpt  96dbd1633a66b0c10e14815c53820359169dc3a330b87fd5bfd0efd371373ac2
```

---

## 🚀 Inference

CCG has **two** inference paths.

### 1. In-loop inference during training (automatic)

When you launch `cycle_train.py`, the generator automatically synthesises the hard-case set $D_g$ at every cycle (the top-$m\%$ samples ranked by *Overconfident Error*; see Section 4.3 of the paper). For each generated case the script writes three separate files into `<result_dir>/sd_gen<global_epoch>/`:

| File | Content |
|---|---|
| `data_XXXXX_<uuid>_mask.png` | the cup/disc segmentation mask used as the ControlNet condition |
| `data_XXXXX_<uuid>_generate.png` | the synthesised SLO fundus image |
| `data_XXXXX_<uuid>_prompt.txt` | the short demographic prompt (e.g. *"Male, White, Non-Hispanic, with Glaucoma"*) |

These files are then consumed by the classifier in the next cycle. You **do not** need to invoke anything separately — it runs as part of training (see the [Training](#-training) section below).

### 2. Stand-alone example (`run_inference.py`)

For quickly inspecting how the generator behaves on a few real test-set masks, use the demo script:

```bash
python run_inference.py
```

By default it loads `checkpoints/ccg_controlnet_sd21.ckpt`, picks 3 cases from the test set, and writes one `[real fundus | cup/disc mask | generated]` triptych per case to `gen_out/` — purely for human review, **not** consumed by training.

Override the defaults via positional arguments:

```bash
python run_inference.py <ckpt_path> <n_samples> <out_dir>
```

---

## 🔄 Training

```bash
python cycle_train.py \
  --max_cycles=5 \
  --num_epochs=30 \
  --batch_size=16 \
  --lr=1e-5 \
  --wd=6e-5 \
  --mode=both \
  --use_gen_data=True \
  --model_arch=efficientnet_b0 \
  --dataset_dir=<DATA_ROOT>/fairvlmed10k \
  --summarized_note_file=<DATA_ROOT>/fairvlmed10k/data_summary.csv \
  --real_train_png_dir=<DATA_ROOT>/fairvlmed10k/training_slo_fundus \
  --result_dir=./results/ccg_run
```

---

## 🗂️ Repository Layout

```
CCG/
├── cycle_train.py            # main CCG cycle training entry
├── train_pretrain_model.py   # standalone classifier baseline
├── run_inference.py          # generate fundus images from masks
├── cldm/                     # ControlNet model + DDIM sampler
├── ldm/                      # latent diffusion model
├── cycle_resnet_src/         # classifier + Sorter
├── pytorch_fid/              # FID metric
├── data/                     # paper-subset filter file + summary CSV
├── models/cldm_v21.yaml      # ControlNet-SD2.1 architecture
├── checkpoints/              # download SD2.1 init + trained CCG weight here
├── requirements.txt          # python dependencies
└── LICENSE                   # Apache-2.0
```

---

## 📑 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wang2026uncertainty,
  title={Uncertainty-aware Cycle Diffusion Model for Fair Glaucoma Diagnosis},
  author={Wang, Ziheng and Yang, Shuran and Lin, Yan and Meng, Yanda and others},
  booktitle={Medical Imaging with Deep Learning},
  year={2026}
}
```

---

## 🙏 Acknowledgments

- Built upon [ControlNet](https://github.com/lllyasviel/ControlNet) (Zhang et al., 2023) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).
- Dataset: [Harvard-FairVLMed](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP) (Luo et al., 2024).
- Optic disc segmentation prior provided by [TransUNet](https://github.com/Beckschen/TransUNet) (Chen et al., 2021) fine-tuned on [Harvard FairSeg](https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg) (Tian et al., 2023).
