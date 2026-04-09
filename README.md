# ENEN-645-Group-4-Final
Final project for ENEN-645: Data Mining and Machine Learning, focused on **plant disease classification under a lab-to-field domain shift**.

## Project Overview

Deep learning models for plant disease classification often perform extremely well on clean, laboratory-style datasets, but their performance can degrade significantly when applied to real-world field images. This project investigates that generalization gap by comparing multiple model training strategies on a unified benchmark dataset.

The work studies transfer from:
- **Lab-style images** from datasets such as **PlantVillage**
to
- **Field-style images** from datasets such as **PlantDoc** and related sources

Three primary modeling approaches were evaluated:
1. **Method 01:** ResNet-18 trained from scratch
2. **Method 02:** Transfer learning with ResNet-50 and test-time augmentation (TTA)
3. **Method 03:** EfficientNet-B0 with CutMix augmentation

Each of these methods is presented as a Jupyter Notebook in the "models" folder

The final report discusses performance on both in-distribution and out-of-distribution data, along with qualitative Grad-CAM analysis.

---

## Dataset

The unified dataset created for this project is available on Kaggle:

**PlantLab2RealGeneralization**  
https://www.kaggle.com/datasets/maciekpopik/plantlab2realgeneralization
For references, please refer to the final report or the Citations section on the above Kaggle page.

This dataset was created using the scripts in the `dataset_creation/` folder.

