from sverad.sverad_svm import ExplainingSVC as SVERADExplainingSVC
from sverad.utils import DataSet, UnfoldedMorganFingerprint, set_seeds
from sveta.svm import ExplainingSVC as SVETAExplainingSVC

import numpy as np
import pandas as pd
import yaml
# import pickle
import dill #using dill since it allows more flexibility in pickling (nested functions)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm

import time

args = None

with open("parameters.yml") as paramFile:  
    args = yaml.load(paramFile, Loader=yaml.FullLoader)

SEED  = args["trainer_explainer"]["SEED"]
SAVE_DATASET_PICKLE = args["trainer_explainer"]["SAVE_DATASET_PICKLE"]
LOAD_DATASET_PICKLE = args["trainer_explainer"]["LOAD_DATASET_PICKLE"]
EMPTY_SET_VALUE = args["trainer_explainer"]["EMPTY_SET_VALUE"]
SAVE_EXPLANATIONS = args["trainer_explainer"]["SAVE_EXPLANATIONS"]
SAVE_MODELS = args["trainer_explainer"]["SAVE_MODELS"]
DATASET_PATH = args["trainer_explainer"]["DATASET_PATH"]
DATASET_PICKLE_PATH = args["trainer_explainer"]["DATASET_PICKLE_PATH"]
FINGERPRINTS_PICKLE_PATH = args["trainer_explainer"]["FINGERPRINTS_PICKLE_PATH"]
MODEL_PATH = args["trainer_explainer"]["MODEL_PATH"]
EXPLANATION_PATH = args["trainer_explainer"]["EXPLANATION_PATH"]
PREDICTION_PATH = args["trainer_explainer"]["PREDICTION_PATH"]
TARGET_UNIPROT_ID = args["trainer_explainer"]["TARGET_UNIPROT_ID"]

set_seeds(SEED)


# ## Definition of models and searched hyperparameter space.


model_list = [{"name": "SVC_RBF",
               "algorithm": SVERADExplainingSVC(empty_set_value=EMPTY_SET_VALUE), #added empty_set_value need seed?
               "parameter": {"C": [0.1, 1, 10, 50, 100, 200, 400, 500, 750, 1000], 
                            "gamma_value": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            },
              },
              {"name": "SVC_Tanimoto",
               "algorithm": SVETAExplainingSVC(no_player_value=EMPTY_SET_VALUE),
               "parameter": {"C": [0.1, 1, 10, 50, 100, 200, 400, 500, 750, 1000],
                            },
              },
              
             ]


# ## Loading the pre-compiled compound set.



dataset_df = pd.read_csv(DATASET_PATH, sep="\t")
dataset_df.pivot_table(index="uniprot_id", columns="label", values="nonstereo_aromatic_smiles", aggfunc="nunique", fill_value=0)


# ## Creating the dataset



dataset_dict = dict()
fingerprint_gen_dict = dict()

if not LOAD_DATASET_PICKLE:
    for dataset_name, data_grpd_df in dataset_df.groupby("uniprot_id"):
        
        # label: 1: active, 0: random
        labels = np.array([1 if l == 'active' else 0 for l in data_grpd_df.label])
        # Creating Fingerprint
        morgan_radius2 = UnfoldedMorganFingerprint(radius=2)
        morgan_radius2.fit_smiles(data_grpd_df.nonstereo_aromatic_smiles.tolist())
        
        # Constructing Dataset
        fp_matrix = morgan_radius2.transform_smiles(data_grpd_df.nonstereo_aromatic_smiles.tolist())
        # Constructing Dataset
        dataset = DataSet(labels, fp_matrix)
        dataset.add_attribute("nonstereo_aromatic_smiles", data_grpd_df.nonstereo_aromatic_smiles.values)
        
        dataset_dict[dataset_name] = dataset
        fingerprint_gen_dict[dataset_name] = morgan_radius2
else:
    with open(DATASET_PICKLE_PATH, "rb") as infile:
        dataset_dict = dill.load(infile)
    with open(FINGERPRINTS_PICKLE_PATH, "rb") as infile:
        fingerprint_gen_dict = dill.load(infile)



if SAVE_DATASET_PICKLE:
    with open(DATASET_PICKLE_PATH, "wb") as outfile:
        dill.dump(dataset_dict, outfile)
    with open(FINGERPRINTS_PICKLE_PATH, "wb") as outfile:
        dill.dump(fingerprint_gen_dict, outfile)


ds = dataset_dict[TARGET_UNIPROT_ID]
print("Number of samples and features: " + str(ds.feature_matrix.todense().shape))

# ## Training the models and generation of feature contributions


n_splits = 1 # Number of test-training splits

prediction_df = []
failed = 0
hyperparamter_dict = dict()
obtained_models = dict()
shap_dict = dict()

# Loop over multiple data-sets. Here only one is assessed.
for dataset_name, dataset in tqdm(dataset_dict.items(), total=len(dataset_dict)): #it takes ~5:30h to run Kernel SHAP for SVC + ~7h for RF

    # Loop over test-training splits. Only one is assessed. `n_splits == 1`
    print("Splitting dataset.")
    data_splitter = StratifiedShuffleSplit(n_splits=n_splits, random_state=SEED, test_size=0.50)
    for trial_nr, (train_idx, test_idx) in tqdm(enumerate(data_splitter.split(dataset.feature_matrix, dataset.label)), leave=False, total=n_splits, disable=True):
        training_set = dataset[train_idx]
        test_set = dataset[test_idx]

        #Iterating over assessed models.
        for model_dict in model_list:
            # print(model_dict["parameter"])
            # Setting up hyperparameter search.
            param_grid = model_dict["parameter"]
            model = GridSearchCV(estimator = model_dict["algorithm"],
                                 param_grid = param_grid,
                                 n_jobs=-1,
                                 scoring= "neg_mean_squared_error", #"neg_mean_squared_error", "accuracy"
                                 cv=StratifiedShuffleSplit(n_splits=10, random_state=SEED, test_size=0.5),
                                verbose=0,
                                )
            # Determining optimal hyperparameters and fitting the model to the entire training set with these hyperparamters
            print("Model fitting and tuning to obtain optimal hyperparameters via Grid Search...")
            model.fit(training_set.feature_matrix, training_set.label)
            preds = model.predict(test_set.feature_matrix)
            print("Model accuracy: ", np.mean(preds == test_set.label))
            obtained_models[(dataset_name, trial_nr, model_dict["name"])] = model
            print("Model trained and optimized.")
            
            # Saving hyperparameters
            if model_dict["name"] not in hyperparamter_dict:
                hyperparamter_dict[model_dict["name"]] = []
            best_param = dict(model.best_params_)
            best_param["dataset_name"] = dataset_name
            best_param["trial"] = trial_nr
            hyperparamter_dict[model_dict["name"]].append(best_param)

            # break
            # SVs.
            shap_values = None
            print("Model name: ", model_dict["name"])
            
            if model_dict["name"] == "SVC_RBF":
                print("Explaining model using SVERAD...")
                start = time.time()
                shap_values = model.best_estimator_.feature_weights(dataset.feature_matrix)
                end = time.time()
                expected_value = model.best_estimator_.expected_value
                print(f"SVERAD took {end-start} seconds.")
                print("SVERAD values computation done.")
            elif model_dict["name"] == "SVC_Tanimoto":
                print("Explaining model using SVETA...")
                start = time.time()
                shap_values = model.best_estimator_.feature_weights(dataset.feature_matrix)
                end = time.time()
                expected_value = model.best_estimator_.expected_value
                print(f"SVETA took {end-start} seconds.")
                print("SVETA values computation done.") 
            else:
                raise(ValueError("Model not implemented."))
            
            
            # Creating a DataFrame with all relevant data.
            trial_df = pd.DataFrame()
            trial_df["nonstereo_aromatic_smiles"] = dataset.nonstereo_aromatic_smiles
            trial_df["dataset_idx"] = range(len(dataset.label))
            trial_df["label"] = dataset.label
            trial_df["prediction"] = model.best_estimator_.predict(dataset.feature_matrix)
            if model_dict["name"] == "SVC_RBF" or model_dict["name"] == "SVC_Tanimoto":
                trial_df["log_odds"] = model.best_estimator_.predict_log_odds(dataset.feature_matrix)[:, 1]
            trial_df["proba"] = model.best_estimator_.predict_proba(dataset.feature_matrix)[:, 1]
            trial_df["trainingset"] = trial_df.nonstereo_aromatic_smiles.isin(training_set.nonstereo_aromatic_smiles)
            trial_df["testset"] = trial_df.nonstereo_aromatic_smiles.isin(test_set.nonstereo_aromatic_smiles)
            trial_df["trial"] = trial_nr
            trial_df["dataset_name"] = dataset_name
            trial_df["algorithm"] = model_dict["name"]
            
            # Creating a data set storing all SVs and SHAP values.
            shap_dict[(dataset_name, trial_nr, model_dict["name"])] = dict()
            if shap_values is not None:
                trial_df["present_shap"] = (shap_values * dataset.feature_matrix.toarray()).sum(axis=1)
                trial_df["absent_shap"] = (shap_values * (1-dataset.feature_matrix.toarray())).sum(axis=1) 
                if model_dict["name"] == "SVC_RBF":
                    shap_dict[(dataset_name, trial_nr, model_dict["name"])]["sverad_values"] = shap_values
                if model_dict["name"] == "SVC_Tanimoto":
                    shap_dict[(dataset_name, trial_nr, model_dict["name"])]["sveta_values"] = shap_values 
                
                shap_dict[(dataset_name, trial_nr, model_dict["name"])]["expected_value"] = expected_value
            
            
            prediction_df.append(trial_df)
prediction_df = pd.concat(prediction_df)
print(f"Number of failed datasets: {failed}")

print("Explanation done. Saving results...")


print(model.best_params_)
print(model.best_estimator_.get_params())
print(model.best_score_)
print(model_dict["parameter"])

# ## Storing all results.
# 
# We use dill since it allows more flexibility. Needed for the nested function used for the custom SVC with RBF kernel. All the pickle dumps/load can be safely substituted with dill. Will do that in the future.


if SAVE_MODELS:
    with open(MODEL_PATH, "wb") as outfile:
        # pickle.dump(obtained_models, outfile)
        dill.dump(obtained_models, outfile)



if SAVE_EXPLANATIONS:
    with open(EXPLANATION_PATH, "wb") as outfile:
        dill.dump(shap_dict, outfile)
    prediction_df.to_csv(PREDICTION_PATH, sep="\t", index=False)
    print("Results saved.")

