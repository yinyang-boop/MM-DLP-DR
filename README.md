# DDLS 2025 Final Project: Multi‑modal Machine Learning for Drug Repurposing

## Overview
This repository contains the implementation of my DDLS 2025 final project. The project explores **drug repurposing** using three public datasets:
- **ChEMBL** (bioactivity, pChEMBL labels)
- **DrugBank** (mechanisms → proxy supervised labels)
- **GDSC** (drug sensitivity, LN_IC50 → sensitive labels)

The work follows the course milestones:
- **Phase I**: Dataset exploration, splits, visualizations
- **Phase II**: Baseline model training, evaluation, improvement
- **Phase III**: Minimal toolset for accessibility (MCP‑style)

## Repository Structure
- **notebooks/** Jupyter/Colab notebooks for Phase I–III 
- **src/** Modular Python code (data loaders, models, utils) 
- **scripts/** Data download/preprocessing scripts 
- **results/** Figures, metrics, predictions (generated after running notebooks) 
- **models/** Trained model artifacts (optional, or via Releases/Zenodo)

## Data
- **ChEMBL**: Download via chembl_webresource_client
- **DrugBank**: Requires license; proxy labels from mechanisms
- **GDSC**: Public CSV download from cancerrxgene.org
Reminder: Raw/processed data files are not included in this repo. Use the scripts in scripts/ to download and preprocess.

## Reproducibility
- Fixed random seeds for deterministic splits.
- Dependencies pinned in requirements.txt.
- All figures and metrics saved under results/.
- Trained model weights can be uploaded to GitHub Releases or Zenodo.

## License
MIT License (see LICENSE file).

## Citation
If you use this code, please cite the original datasets: ChEMBL, DrugBank, GDSC.

## Getting Started
### Run in Colab
[![Open In Colab](https://colab.research.google.com/drive/1wxda2X0pbDMfC0L910kBJY177beaeTuD#scrollTo=Zz3ME3BZSHq2)]

### Local Setup
```bash
git clone https://github.com/yinyang-boop/MM-DLP-DR.git
cd <repo-name>
pip install -r requirements.txt
