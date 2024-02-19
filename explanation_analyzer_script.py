#!/usr/bin/env python
# coding: utf-8

# # Feature explanations analysis

# In[1]:


import sverad.feature_importance as feim
from sverad.utils import set_seeds
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import makedirs
from os.path import exists
from tqdm.auto import tqdm
import yaml
# import pickle
import dill
from rdkit import Chem
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score

import time

start = time.time()

with open("parameters.yml") as paramFile:  
    args = yaml.load(paramFile, Loader=yaml.FullLoader)

SEED  = args["explanation_analyzer"]["SEED"]
TARGET_UNIPROT_ID = args["trainer_explainer"]["TARGET_UNIPROT_ID"]
SAVE_DATA_PATH = args["explanation_analyzer"]["SAVE_DATA_PATH"]
SAVE_PLOTS = args["explanation_analyzer"]["SAVE_PLOTS"]
SAVE_MAPPINGS = args["explanation_analyzer"]["SAVE_MAPPINGS"]
FIGURE_FORMAT = args["explanation_analyzer"]["FIGURE_FORMAT"]
DATASET_PATH = args["explanation_analyzer"]["DATASET_PATH"]
DATASET_PICKLE_PATH = args["explanation_analyzer"]["DATASET_PICKLE_PATH"]
FINGERPRINTS_PICKLE_PATH = args["explanation_analyzer"]["FINGERPRINTS_PICKLE_PATH"]
MODEL_PATH = args["trainer_explainer"]["MODEL_PATH"]
EXPLANATION_PATH = args["trainer_explainer"]["EXPLANATION_PATH"]
PREDICTION_PATH = args["trainer_explainer"]["PREDICTION_PATH"]
COLOR_ABSENT_FEATURES = args["explanation_analyzer"]["COLOR_ABSENT_FEATURES"]
COLOR_PRESENT_FEATURES = args["explanation_analyzer"]["COLOR_PRESENT_FEATURES"]

set_seeds(SEED)

# Defining assessed target and test-training split
dataset_name = TARGET_UNIPROT_ID
trial = 0

SAVE_FIGURES_PATH = SAVE_DATA_PATH + dataset_name
SAVE_MAPPINGS_PATH = SAVE_DATA_PATH + dataset_name + "/shapley_values_mapping/"


mpl.rcParams.update({'font.size': 14})


# ## Loading the data



dataset_df = pd.read_csv(DATASET_PATH, sep="\t")





dataset_df.pivot_table(index="uniprot_id", columns="label", values="nonstereo_aromatic_smiles", aggfunc="nunique", fill_value=0)




dataset_dict = None
fingerprint_gen_dict = None
shap_dict = None
obtained_models = None
with open(DATASET_PICKLE_PATH, "rb") as infile:
    dataset_dict = dill.load(infile)
with open(FINGERPRINTS_PICKLE_PATH, "rb") as infile:
    fingerprint_gen_dict = dill.load(infile)
with open(EXPLANATION_PATH, "rb") as infile:
    shap_dict = dill.load(infile)
with open(MODEL_PATH, "rb") as infile:
    obtained_models = dill.load(infile)
prediction_df = pd.read_csv(PREDICTION_PATH, sep="\t")




print("Best params for SVM_RBF:", obtained_models[TARGET_UNIPROT_ID,0, 'SVC_RBF'].best_params_)
print("Best params for SVM_Tanimoto:", obtained_models[TARGET_UNIPROT_ID,0, 'SVC_Tanimoto'].best_params_)




svc_rbf_model = obtained_models[(dataset_name, trial, "SVC_RBF")]
support_vectors_rbf = svc_rbf_model.best_estimator_.explicit_support_vectors

svc_tanimoto_model = obtained_models[(dataset_name, trial, "SVC_Tanimoto")]
support_vectors_tanimoto = svc_tanimoto_model.best_estimator_.explicit_support_vectors

num_samples = dataset_dict[dataset_name].feature_matrix.shape[0]
avg_intesecting = []
avg_diff = []
avg_union = []

support_vectors = [support_vectors_rbf, support_vectors_tanimoto]
kernel_types = ["RBF", "Tanimoto"]
for support_vectors_ in support_vectors:
    for i in range(num_samples):
        vector = dataset_dict[dataset_name].feature_matrix[i, :]
        repeated_vector = vector[np.zeros(support_vectors_.shape[0]), :]
        intersecting_f = vector.multiply(support_vectors_)
        only_vector = repeated_vector - intersecting_f
        only_support = support_vectors_ - intersecting_f

        n_shared = intersecting_f.sum(axis=1)
        n_only_v = only_vector.sum(axis=1)
        n_only_sv = only_support.sum(axis=1)

        n_difference = n_only_v + n_only_sv
        n_union_features = n_shared + n_difference

        avg_intesecting.append(n_shared.mean())
        avg_diff.append(n_difference.mean())
        avg_union.append(n_union_features.mean())

    print(f"\nSVM with {kernel_types.pop(0)} kernel:")
    print(f"Average number of intersecting features: {np.mean(avg_intesecting)}")
    print(f"Average number of symmetric difference features: {np.mean(avg_diff)}")
    print(f"Average number of union features: {np.mean(avg_union)}")
    
print("=====================================")

# ## Model performances



for data_set_name, dataset_grpd_df in prediction_df.groupby(["dataset_name"]):
    for (algorithm, traininset), split_df in prediction_df.groupby(["algorithm", "trainingset"]):
        split_name = "training set" if traininset else "test set"
        print("{} - {}: {:0.2f}".format(algorithm, split_name, balanced_accuracy_score(split_df.label, split_df.prediction)))


# # Analysis

# ### Setting up variables for analysis




trial_df = prediction_df.query("dataset_name == @dataset_name & trial == @trial")
fingerprint = fingerprint_gen_dict[dataset_name]
dataset = dataset_dict[dataset_name]

svc_rbf = obtained_models[(dataset_name, trial, "SVC_RBF")].best_estimator_
svc_rbf_shapley = shap_dict[(dataset_name, trial, "SVC_RBF")]["sverad_values"]
svc_rbf_shapley_E = shap_dict[(dataset_name, trial, "SVC_RBF")]["expected_value"]

svc_tanimoto = obtained_models[(dataset_name, trial, "SVC_Tanimoto")].best_estimator_
svc_tanimoto_shapley = shap_dict[(dataset_name, trial, "SVC_Tanimoto")]["sveta_values"]
svc_tanimoto_shapley_E = shap_dict[(dataset_name, trial, "SVC_Tanimoto")]["expected_value"]



# ### Expected values

# In[11]:


print("SVM RBF expected value: ", svc_rbf.expected_value[0])
print("SVM Tanimoto expected value: ", svc_tanimoto.expected_value[0])
print("=====================================")


if not exists(SAVE_FIGURES_PATH):
    makedirs(SAVE_FIGURES_PATH)
if not exists(SAVE_MAPPINGS_PATH):
    makedirs(SAVE_MAPPINGS_PATH)


# ## Absence and presence of features

# ### Defining assessed compounds and transforming the dataframe for analysis



correct_predicted_df = trial_df.query("label == prediction & testset")



correct_predicted_df_molten = correct_predicted_df.melt(id_vars=["nonstereo_aromatic_smiles", "dataset_idx", "label", "algorithm"],
                                                        value_vars=["present_shap", "absent_shap"],
                                                        value_name="Shapley sum")




# Renaming the columns from 0 and 1 to random and active, respectively.
correct_predicted_df_molten["label_str"] = "Random"
correct_predicted_df_molten.loc[correct_predicted_df_molten["label"] == 1, "label_str"] = "Active"

correct_predicted_df_molten["variable_str"] = ""
correct_predicted_df_molten.loc[correct_predicted_df_molten.variable == "present_shap", "variable_str"] = "Present features"
correct_predicted_df_molten.loc[correct_predicted_df_molten.variable == "absent_shap", "variable_str"] = "Absent features"


# ### SVs for SVM using SVERAD


mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots() #fig, ax = plt.subplots() edited to save in larger plot
sns.boxplot(data=correct_predicted_df_molten.query("algorithm == 'SVC_RBF'"), x="label_str", hue="variable_str", y="Shapley sum", hue_order=["Present features", "Absent features"], palette=[COLOR_PRESENT_FEATURES, COLOR_ABSENT_FEATURES])
xlim = ax.get_xlim()
ax.hlines(0, *xlim, color="gray", ls="--")
ax.set_xlim(xlim)
ax.set_title("SVM with RBF kernel")

ax.set_xlabel("Class")
ax.set_ylabel(r"Sum of Shapley values for log odds")
ax.legend(*ax.get_legend_handles_labels())
plt.tight_layout() #added to save in larger plot. Remove if not needed
if SAVE_PLOTS:
    plt.savefig(SAVE_FIGURES_PATH + "_boxplot_present_absent_sverad_svm." + FIGURE_FORMAT, dpi=300, bbox_inches='tight') #bbox_inches='tight'used with plt.tight_layout() to save in larger plot

# ### SVs for SVM using SVETA


mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots() #fig, ax = plt.subplots() edited to save in larger plot
sns.boxplot(data=correct_predicted_df_molten.query("algorithm == 'SVC_Tanimoto'"), x="label_str", hue="variable_str", y="Shapley sum", hue_order=["Present features", "Absent features"], palette=[COLOR_PRESENT_FEATURES, COLOR_ABSENT_FEATURES])
xlim = ax.get_xlim()
ax.hlines(0, *xlim, color="gray", ls="--")
ax.set_xlim(xlim)
ax.set_title("SVM with Tanimoto kernel")

ax.set_xlabel("Class")
ax.set_ylabel(r"Sum of Shapley values for log odds")
ax.legend(*ax.get_legend_handles_labels())
plt.tight_layout() #added to save in larger plot. Remove if not needed
if SAVE_PLOTS:
    plt.savefig(SAVE_FIGURES_PATH + "_boxplot_present_absent_sveta_svm." + FIGURE_FORMAT, dpi=300, bbox_inches='tight') #bbox_inches='tight'used with plt.tight_layout() to save in larger plot

# #### Contributions of all features of active compounds.

# In[18]:

print("\nContribution of all features in active compounds:")
print("Avg sum of SVs for SVM RBF: ", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 1")[["present_shap", "absent_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 1")[["present_shap", "absent_shap"]].sum(axis=1).std())

print("Avg sum of SVs for SVM Tanimoto: ", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 1")[["present_shap", "absent_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 1")[["present_shap", "absent_shap"]].sum(axis=1).std())


# ##### Contribution of present features

# In[19]:

print("\nContributions of present features in active compounds:")
print("Avg sum of SVs for SVM RBF: ", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 1")[["present_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 1")[["present_shap"]].sum(axis=1).std())

print("Avg sum of SVs for SVC_Tanimoto: ", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 1")[["present_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 1")[["present_shap"]].sum(axis=1).std())


# ##### Contribution of absent features

# In[20]:

print("\nContributions of absent features in active compounds:")
print("Avg sum of SVs for SVM RBF: ", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 1")[["absent_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 1")[["absent_shap"]].sum(axis=1).std())

print("Avg sum of SVs for SVM Tanimoto: ", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 1")[["absent_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 1")[["absent_shap"]].sum(axis=1).std())


# #### Contributions of all features of random compounds.

# In[21]:

print("\nContribution of all features in random compounds:")

print("Avg sum of SVs for SVM RBF: ", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 0")[["present_shap", "absent_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 0")[["present_shap", "absent_shap"]].sum(axis=1).std())

print("Avg sum of SVs for SVM Tanimoto: ", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 0")[["present_shap", "absent_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 0")[["present_shap", "absent_shap"]].sum(axis=1).std())


# ##### Contribution of present features

# In[22]:

print("\nContributions of present features in random compounds.\n")

print("Avg sum of SVs for SVM RBF: ", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 0")[["present_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 0")[["present_shap"]].sum(axis=1).std())

print("Avg sum of SVs for SVM Tanimoto: ", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 0")[["present_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 0")[["present_shap"]].sum(axis=1).std())


# ##### Contribution of absent features

# In[23]:

print("\nContributions of absent features in random compounds.\n")

print("Avg sum of SVs for SVM RBF: ", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 0")[["absent_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_RBF' & label == 0")[["absent_shap"]].sum(axis=1).std())

print("Avg sum of SVs for SVM Tanimoto: ", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 0")[["absent_shap"]].sum(axis=1).mean())
print("Std dev", correct_predicted_df.query("algorithm == 'SVC_Tanimoto' & label == 0")[["absent_shap"]].sum(axis=1).std())


# ### Generating mapping for all the correctly predicted test cpds

# In[36]:


if SAVE_MAPPINGS:

    print("\n=====================================\n")
    print("Generating mappings for all the correctly predicted test compounds...\n")
    
    if not exists(SAVE_MAPPINGS_PATH + "active/SVERAD"):
        makedirs(SAVE_MAPPINGS_PATH + "active/SVERAD")
    if not exists(SAVE_MAPPINGS_PATH + "active/SVETA"):
        makedirs(SAVE_MAPPINGS_PATH + "active/SVETA")

    #for SVERAD SVM
    print("Generating mappings for SVERAD SVM on active compunds...\n")
    test_active_df_SVM_RBF = trial_df.query("testset & label == 1 & prediction == 1 & algorithm == 'SVC_RBF'")
    for i, row in tqdm(test_active_df_SVM_RBF.iterrows(), total=len(test_active_df_SVM_RBF)):
        vis_cpd_idx = row["dataset_idx"]
        vis_cpd_smiles = dataset.nonstereo_aromatic_smiles[vis_cpd_idx]
        vis_cpd_mol_obj = Chem.MolFromSmiles(vis_cpd_smiles)
        svr_present_shap = svc_rbf_shapley[vis_cpd_idx] * dataset.feature_matrix.toarray()[vis_cpd_idx]
        atomweight_svr = feim.shap2atomweight(vis_cpd_mol_obj, fingerprint, svr_present_shap)
        svg_image = feim.rdkit_gaussplot(vis_cpd_mol_obj, atomweight_svr, n_contourLines=5)
        fig = feim.show_png(svg_image.GetDrawingText())
        fig.save(SAVE_MAPPINGS_PATH + "active/SVERAD/SVM_SVERAD_"+ str(vis_cpd_idx) + "_" + vis_cpd_smiles + ".png", dpi=(300,300))
        
    # for SVETA SVM
    print("Generating mappings for SVETA SVM on active compunds...\n")
    test_active_df_SVM_Tanimoto = trial_df.query("testset & label == 1 & prediction == 1 & algorithm == 'SVC_Tanimoto'")
    for i, row in tqdm(test_active_df_SVM_Tanimoto.iterrows(), total=len(test_active_df_SVM_Tanimoto)):
        vis_cpd_idx = row["dataset_idx"]
        vis_cpd_smiles = dataset.nonstereo_aromatic_smiles[vis_cpd_idx]
        vis_cpd_mol_obj = Chem.MolFromSmiles(vis_cpd_smiles)
        svr_present_shap = svc_tanimoto_shapley[vis_cpd_idx] * dataset.feature_matrix.toarray()[vis_cpd_idx]
        atomweight_svr = feim.shap2atomweight(vis_cpd_mol_obj, fingerprint, svr_present_shap)
        svg_image = feim.rdkit_gaussplot(vis_cpd_mol_obj, atomweight_svr, n_contourLines=5)
        fig = feim.show_png(svg_image.GetDrawingText())
        fig.save(SAVE_MAPPINGS_PATH + "active/SVETA/SVM_SVETA_"+ str(vis_cpd_idx) + "_" + vis_cpd_smiles + ".png", dpi=(300,300))




    if not exists(SAVE_MAPPINGS_PATH + "random/SVETA"):
        makedirs(SAVE_MAPPINGS_PATH + "random/SVETA")
    if not exists(SAVE_MAPPINGS_PATH + "random/SVERAD"):
        makedirs(SAVE_MAPPINGS_PATH + "random/SVERAD")

    #for SVERAD SVM
    print("Generating mappings for SVERAD SVM on random compunds...\n")
    test_random_df_SVM_RBF = trial_df.query("testset & label == 0 & prediction == 0 & algorithm == 'SVC_RBF'")
    for i, row in tqdm(test_random_df_SVM_RBF.iterrows(), total=len(test_random_df_SVM_RBF)):
        vis_cpd_idx = row["dataset_idx"]
        vis_cpd_smiles = dataset.nonstereo_aromatic_smiles[vis_cpd_idx]
        vis_cpd_mol_obj = Chem.MolFromSmiles(vis_cpd_smiles)
        svr_present_shap = svc_rbf_shapley[vis_cpd_idx] * dataset.feature_matrix.toarray()[vis_cpd_idx]
        atomweight_svr = feim.shap2atomweight(vis_cpd_mol_obj, fingerprint, svr_present_shap)
        svg_image = feim.rdkit_gaussplot(vis_cpd_mol_obj, atomweight_svr, n_contourLines=5)
        fig = feim.show_png(svg_image.GetDrawingText())
        fig.save(SAVE_MAPPINGS_PATH + "random/SVERAD/SVM_SVERAD_"+ str(vis_cpd_idx) + "_" + vis_cpd_smiles + ".png", dpi=(300,300))
        
    # for SVETA SVM
    print("Generating mappings for SVETA SVM on random compunds...\n")
    test_random_df_SVM_Tanimoto = trial_df.query("testset & label == 0 & prediction == 0 & algorithm == 'SVC_Tanimoto'")
    for i, row in tqdm(test_random_df_SVM_Tanimoto.iterrows(), total=len(test_random_df_SVM_Tanimoto)):
        vis_cpd_idx = row["dataset_idx"]
        vis_cpd_smiles = dataset.nonstereo_aromatic_smiles[vis_cpd_idx]
        vis_cpd_mol_obj = Chem.MolFromSmiles(vis_cpd_smiles)
        svr_present_shap = svc_tanimoto_shapley[vis_cpd_idx] * dataset.feature_matrix.toarray()[vis_cpd_idx]
        atomweight_svr = feim.shap2atomweight(vis_cpd_mol_obj, fingerprint, svr_present_shap)
        svg_image = feim.rdkit_gaussplot(vis_cpd_mol_obj, atomweight_svr, n_contourLines=5)
        fig = feim.show_png(svg_image.GetDrawingText())
        fig.save(SAVE_MAPPINGS_PATH + "random/SVETA/SVM_SVETA_"+ str(vis_cpd_idx) + "_" + vis_cpd_smiles + ".png", dpi=(300,300))

end = time.time()
print(f"\nRunning the script took {end-start} seconds.")