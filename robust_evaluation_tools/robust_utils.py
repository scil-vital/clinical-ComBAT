import re
import pandas as pd
import numpy as np
import os

import glob

from scripts import combat_info

from clinical_combat.harmonization.QuickCombat import QuickCombat
import matplotlib.pyplot as plt


# Compare if two DataFrames have the same values in the 'sid' column
def compare_sid(df1, df2):
    # Get the unique 'sid' values from both DataFrames
    sid_df1 = set(df1['sid'])
    sid_df2 = set(df2['sid'])
    
    # Check if they are equal
    return sid_df1 == sid_df2

def get_bundles(mov_data_file):
    return combat_info.get_bundles(mov_data_file)

def get_info(mov_data_file):
    [df,bundles] = combat_info.info(mov_data_file)
    nb_hc = int(re.findall('HC\(n=(\d+)',df["DetailInfos"]["Disease"])[0])
    nb_total = df["DetailInfos"]["Number of Subject"]
    nb_sick = nb_total - nb_hc
    return [nb_total,nb_hc,nb_sick]

def robust_text(x):
    return "NoRobust" if x == 'No' else x

def rwp_text(x):
    return "RWP" if x else "NoRWP"

def get_site(mov_data_file):
    mov_data = pd.read_csv(mov_data_file)
    return mov_data.site.unique()[0]

def add_nb_patients_and_diseased(df):
  df['num_patients'] = df['site'].str.extract(r'(\d+)_patients')[0].astype(int)
  df['disease_ratio'] = df['site'].str.extract(r'(\d+)_percent')[0].astype(int)
  df['num_diseased'] = (df['num_patients'] * df['disease_ratio']/100).astype(int)
  return df

def scatter(df1,df2, title, bundle='mni_MCP'):
    df1_bundle = df1[df1['bundle'] == bundle]
    df2_bundle = df2[df2['bundle'] == bundle]

    plt.figure(figsize=(10, 5))
    plt.scatter(df1_bundle['age'], df1_bundle['mean'], label='Train', alpha=0.5, color='green')
    plt.scatter(df2_bundle['age'], df2_bundle['mean'], label='Test', alpha=0.5, color='red')
    plt.xlabel('Age')
    plt.ylabel('Mean')
    plt.title(title)
    plt.legend()
    plt.show()

def get_complete_combination(folder_path, file_pattern='adni_compilation*.csv.gz'):
    # Define the folder path and the pattern to match files
    file_pattern_path = os.path.join(folder_path, file_pattern)

    # Get a list of all matching files
    file_list = glob.glob(file_pattern_path)

    # Initialize an empty list to store DataFrames
    df_list = []

    # Loop through the file list and read each file
    for file in file_list:
        df = pd.read_csv(file)
        df = remove_covariates_effects(df)
        df_list.append(df)

    # Concatenate all DataFrames
    df_combined = pd.concat(df_list, ignore_index=True)

    # Display the combined DataFrame
    return df_combined


def get_design_matrices(df):
    design = []
    Y = []
    for bundle in list(np.unique(df["bundle"])):
        data = df.query("bundle == @bundle")
        hstack_list = []
        hstack_list.append(np.ones(len(data["sid"])))  # intercept
        hstack_list.append(QuickCombat.to_category(data["sex"]))
        hstack_list.append(QuickCombat.to_category(data["handedness"]))
        ages = data["age"].to_numpy()
        hstack_list.append(ages)
        design.append(np.array(hstack_list))
        Y.append(data["mean"].to_numpy())
    return design, Y

def remove_covariates_effects(df):
    df = df.sort_values(by=["site", "sid", "bundle"])
    design, y = get_design_matrices(df)
    alpha, beta = QuickCombat.get_alpha_beta(design, y)

    df['mean_no_cov'] = df['mean']
    for i, bundle in enumerate(list(np.unique(df["bundle"]))):
        bundle_df = df[df['bundle'] == bundle]
        covariate_effect = np.dot(design[i][1:, :].transpose(), beta[i])
        df.loc[df['bundle'] == bundle, 'mean_no_cov'] = (y[i] - covariate_effect)
    return df


def transform_into_matrix(df):
    # Étape 1 : Vérifier si la colonne 'metric_bundle' existe, sinon l'ajouter
    if 'metric_bundle' not in df.columns:
        df['metric_bundle'] = df['bundle'] + '_' + df['metric']
    # Étape 2 : Réorganiser les données avec pivot_table
    df_pivot = df.pivot(index='sid', columns='metric_bundle', values='mean')

    # Étape 3 : Réinitialiser les index pour obtenir un DataFrame classique
    df_pivot = df_pivot.reset_index()

    # Étape 4 : Optionnel - Renommer les colonnes pour avoir un format clair
    df_pivot.columns.name = None  # Supprimer le nom des colonnes après pivot
    df_pivot = df_pivot.rename(columns=lambda x: x.replace(" ", "_") if isinstance(x, str) else x)

    # Étape 5 : Supprimer les colonnes AGE, HANDLINESS, etc., si elles ne sont pas nécessaires
    # Dans ce cas, elles sont déjà enlevées dans le pivot_table.

    # Résultat
    print(df_pivot)