
# Uncertainty-aware Cycle Diffusion Model for Fair Glaucoma Diagnosis

[![GitHub license](https://img.shields.io/github/license/用户名/仓库名)](https://github.com/用户名/仓库名/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## 🛠️ Install dependencies

```python
conda create -n GlaucoDiff python=3.11.7
conda activate GlaucoDiff
pip install -r requirements.txt
```

## 📁 Data preparation

The Harvard-FairVLMed dataset (named as **fairvlmed10k**) can be accessed via this [link](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP).

We provide the `data_summary.csv` and `filter_file.txt` files for both datasets, which include the filenames used in our experiments, along with information on whether each file is used for the training, validation, or test set, as well as the demographic information and medical records from the source data.

**The complete code will be made publicly available after the paper is accepted. Coming Soon.**
