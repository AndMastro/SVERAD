![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# SVERAD: Shapley Value-Expressed Radial Basis Function
This is the official repository for the work **Calculation of exact Shapley values for explaining support vector machine models using the radial basis function kernel**.

## Before you start

If you want to use the code, we suggest to use a ```conda``` environmnet with one of the provided environment files (tested of Ubuntu 20.04) and clone the repository (a pip package for SVERAD will be released in the near future).

## SVERAD facilities

The class ```ExplainingSVC``` contains all the facilities needed to train a Support Vector Classifier and explain its predictions in terms of exact SVERAD Shapley values.

If you want to indepentently compute SVERAD SV for RBF kernel in your code, use the function ```compute_sverad_sv()``` available in the ```sverad_kernel.py``` module. 

## Reproducibility

The repo contains the source code and the notebooks usable to reproduce the experiments and results in the paper. It is possible to use the nooteboks provided to replicate the experiments:

* ```explanations_rbf_random_vectors.ipynb``` replicates the experiments for the computation of exact SV with small randomly geenrated vectors.
* ```rbf_50_compounds_SVERAD_vs_SHAP.ipynb``` compares SVERAD and SHAP for randomly drawn compounds.
* ```calculation_SVs_SVM_RF.ipynb``` is used to both train and optimize the SVM and RF models via Grid Search and to compute exact SV with SVERAD and TreeSHAP and SHAP values with KernelSHAP.
* ```analysis_SVs_SVM_RF.ipynb``` derives analyses and statistics on the computed SV and SHAP values.

## Running times

As a reference measurement for comparison, executing SVERAD on the dataset provided within the repo with a machine mounting an Intel Core i7-12700H with 4.70 GHz of maximum clock speed and 16 GB of RAM took around 22 seconds, analogously to TreeSHAP, while running KernelSHAP calculations took more than 5.5 hours.

### Contacts

For any queries or information, feel free to drop an [email](mailto:mastropietro@diag.uniroma1.it).

## Citations

If you like and use our work, please cite our publication on **Scientific Reports** 😄

Mastropietro, A., Feldmann, C. & Bajorath, J. [Calculation of exact Shapley values for explaining support vector machine models using the radial basis function kernel](https://www.nature.com/articles/s41598-023-46930-2). Sci Rep 13, 19561 (2023). https://doi.org/10.1038/s41598-023-46930-2
