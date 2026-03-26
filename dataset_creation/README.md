# Dataset Creation

This folder contains scripts used to prepare the dataset for the project.

## Includes
- Merging PlantVillage and PlantDoc datasets
- Filtering overlapping classes
- Creating train/validation/test splits
- Generating in-distribution and out-of-distribution test sets

## Note
These scripts are provided for reproducibility and were used to generate the dataset used in training.
To successfully run this script, the entire Plant Village dataset, Plant Doc dataset and the datasets mentioned in the "Alternative" comun of the excel mapping must present in the same folder.

## Dataset Source and Description
The dataset used in this project was constructed and documented on Kaggle:

https://www.kaggle.com/datasets/maciekpopik/plantlab2realgeneralization

The Kaggle page includes:
- Full description of dataset construction
- Source datasets (PlantVillage, PlantDoc, etc.)
- Class mapping details
- Dataset structure and splits

This repository contains the scripts used to generate the dataset from those sources.