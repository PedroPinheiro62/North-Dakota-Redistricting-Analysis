# North-Dakota-Redistricting-Analysis

This repository contains the code and part of the data used for performing Random Walks and Short Burst runs on the state of North Dakota at census blocks level.

## Data
The raw data used, as well as the cleaned data used by the code in this repo, is available on Google Drive:
[ND-DATA](https://drive.google.com/drive/folders/1ij4kOO3iKgNRNZtMJNzjVf2X2NHvRIb7?usp=sharing)

## Structure
The repository is structured as follows:

### Data Cleaning
- `data_cleaning/`: Contains Jupyter notebooks and data for cleaning and preparing the data for the Markov Chain runs.

### Ensemble Analysis
- `Random_Walk.py`: Contains the code for performing a random walk on the cleaned data.
- `Ensembles Analysis.ipynb`: A Jupyter notebook for the ensemble analysis (Histograms and Box Plots).

### Short Bursts
- `Short_Burst.py`: Contains the code for performing short burst runs on the cleaned data.
- `data_sb`: Contains the Short Burst generated data (Generated by Short_Burst.py).
- `Short Burst Analysis.ipynb`: A Jupyter notebook for analysing the short burst runs data.
