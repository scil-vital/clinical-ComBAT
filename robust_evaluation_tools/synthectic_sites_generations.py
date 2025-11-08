import os
import numpy as np
import pandas as pd
import subprocess
import json

from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from clinical_combat.harmonization.QuickCombat import QuickCombat

import matplotlib.pyplot as plt

def augment_df(df, new_copies=2):
    
    augmented_dfs = [df]

    for copy_index in range(1, new_copies):
        temp_df = df.copy()
        sid_modifications = {}
        for sid_val in temp_df['sid'].unique():
            sid_modifications[sid_val] = np.random.choice([-1, 1])

        temp_df['age'] = temp_df.apply(
            lambda row: row['age'] + sid_modifications[row['sid']], axis=1
        )
        temp_df['sid'] = temp_df['sid'].astype(str) + f'_aug{copy_index}'
        temp_df['mean'] = temp_df['mean'] * (
            1 + np.random.choice([-0.02,-0.01, 0.01,0.02], size=len(temp_df))
        )

        augmented_dfs.append(temp_df)
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    return final_df

def save_scatter_plots_from_merged(merged, output_folder="TRUTH"):
    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Liste des combinaisons uniques (metric, bundle)
    unique_combinations = merged[['metric', 'bundle']].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        metric = row['metric']
        bundle = row['bundle']

        # Filtrer le merged selon metric et bundle
        filt = merged[(merged['metric'] == metric) & (merged['bundle'] == bundle)]

        # Créer la figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Scatter Plots for Metric: {metric} | Bundle: {bundle}", fontsize=16)

        # Rangée du haut (OLD)
        axes[0, 0].scatter(filt['age'], filt['mean_combined'])
        axes[0, 0].set_title("Combined - Mean")

        axes[0, 1].scatter(filt['age'], filt['whatever_biased_old'])
        axes[0, 1].set_title("Old - whatever")

        axes[0, 2].scatter(filt['age'], filt['mean_biased_old'])
        axes[0, 2].set_title("Old - Mean")

        # Rangée du bas (NEW)
        axes[1, 0].scatter(filt['age'], filt['mean_combined'])
        axes[1, 0].set_title("Combined - Mean (Again)")

        axes[1, 1].scatter(filt['age'], filt['whatever_biased_new'])
        axes[1, 1].set_title("New - whatever")

        axes[1, 2].scatter(filt['age'], filt['mean_biased_new'])
        axes[1, 2].set_title("New - Mean")

        # Ajout des labels pour chaque sous-graphique
        for ax in axes.flat:
            ax.set_xlabel("Age")
            ax.set_ylabel("Value")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Laisse de la place pour le titre principal

        # Nom du fichier
        filename = f"{metric}_{bundle}_scatter.png".replace("/", "-")
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath)
        plt.close(fig)

    return f"{len(unique_combinations)} figures saved in '{output_folder}'"

# Ce bloc est prêt à être utilisé dans un script Python ou un notebook
# Exemple : save_scatter_plots(combined, biased_df_old, biased_df)

def get_design_matrices(df, ignore_handedness=False, ignore_sex=False):
    design = []
    Y = []
    for bundle in list(np.unique(df["bundle"])):
        data = df.query("bundle == @bundle")
        hstack_list = []
        hstack_list.append(np.ones(len(data["sid"])))  # intercept
        if not ignore_sex:
            hstack_list.append(QuickCombat.to_category(data["sex"]))
        if not ignore_handedness:
            hstack_list.append(QuickCombat.to_category(data["handedness"]))
        ages = data["age"].to_numpy()
        hstack_list.append(ages)
        design.append(np.array(hstack_list))
        Y.append(data["mean"].to_numpy())
    return design, Y
    

def split_train_test(df, test_size=0.2, random_state=None):
    """
    Split the DataFrame into training and testing sets, ensuring the same proportion of HC and non-HC patients
    and that data from the same sid are in the same dataset.

    Parameters:
    file_path (str): The path to the CSV file to split.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    pd.DataFrame: Training set.
    pd.DataFrame: Testing set.
    """
    
    # Group by 'sid' and get unique sids
    unique_sids = df.groupby('sid').first().reset_index()
    
    # Split the unique sids into train and test sets
    train_sids, test_sids = train_test_split(unique_sids, test_size=test_size, random_state=random_state, stratify=unique_sids['disease'])
    
    # Create train and test DataFrames by filtering the original DataFrame
    train_df = df[df['sid'].isin(train_sids['sid'])]
    test_df = df[df['sid'].isin(test_sids['sid'])]
    
    return train_df, test_df

def sample_patients(df, num_patients, disease_ratio,index):
    # Lire le fichier CSV dans un DataFrame
    
    # Calculer le nombre de patients malades et sains
    num_diseased = int(num_patients * disease_ratio)
    num_healthy = num_patients - num_diseased
    
    # Filtrer les patients en santé (HC) et malades
    healthy_patients = df[df['disease'] == 'HC']
    diseased_patients = df[df['disease'] != 'HC']
    
    # S'assurer qu'il y a assez de patients pour chaque catégorie
    if len(healthy_patients['sid'].unique()) < num_healthy or len(diseased_patients['sid'].unique()) < num_diseased:
        raise ValueError("Nombre insuffisant de patients en santé ou malades pour l'échantillon demandé.")
    
    sampled_healthy_sid = healthy_patients['sid'].drop_duplicates().sample(n=num_healthy)
    sampled_diseased_sid = diseased_patients['sid'].drop_duplicates().sample(n=num_diseased)

    sampled_healthy = healthy_patients[healthy_patients['sid'].isin(sampled_healthy_sid)]
    sampled_diseased = diseased_patients[diseased_patients['sid'].isin(sampled_diseased_sid)]

    
    # Combiner les échantillons pour obtenir le DataFrame final
    sampled_df = pd.concat([sampled_healthy, sampled_diseased])
    # Modifier les valeurs de 'site' pour toutes les lignes
    sampled_df['site'] = f"{num_patients}_patients_{int(disease_ratio*100)}_percent_{index}"

    # Retourner le DataFrame final
    return sampled_df


def generate_biaised_data(df1, df2,ssv, fixed_bias=False,
                additive_uniform_low=-3, additive_uniform_high=3, 
                multiplicative_uniform_low=0.5, multiplicative_uniform_high=2, 
                additive_std_low=0.01, additive_std_high=0.1, 
                multiplicative_std_low=0.01, multiplicative_std_high=0.1):
    """
    Génère des biais additifs et multiplicatifs pour chaque bundle en fonction de df1, puis applique ces biais à df1 et df2
    de manière indépendante en tenant compte des covariables (âge, sexe, latéralité) et en centrant les résidus.

    Parameters:
    - df1, df2 (pd.DataFrame): Les DataFrames sur lesquels appliquer les biais.
    - additive_uniform_low, additive_uniform_high : paramètres pour le biais additif.
    - multiplicative_uniform_low, multiplicative_uniform_high : paramètres pour le biais multiplicatif.
    - additive_std_low, additive_std_high : paramètres pour l'écart-type du biais additif.
    - multiplicative_std_low, multiplicative_std_high : paramètres pour l'écart-type du biais multiplicatif.

    Returns:
    - tuple : Deux DataFrames avec les biais appliqués indépendamment.
    """
    
    # Dictionnaires pour stocker les biais par bundle
    additive_bias_per_bundle = {}
    multiplicative_bias_per_bundle = {}

    # # Tirer les moyennes de biais de distributions uniformes pour le bundle
    additive_mean = np.random.uniform(low=additive_uniform_low, high=additive_uniform_high)
    multiplicative_mean = np.random.uniform(low=multiplicative_uniform_low, high=multiplicative_uniform_high)
    
    # # Tirer les écarts-types de biais de distributions uniformes pour le bundle
    additive_std = np.random.uniform(low=additive_std_low, high=additive_std_high)
    multiplicative_std = np.random.uniform(low=multiplicative_std_low, high=multiplicative_std_high)

    # Calcul des biais pour chaque bundle unique dans df1

    # Déterminer la colonne à utiliser pour l'itération
    bundle_column = 'metric_bundle' if 'metric_bundle' in df1.columns else 'bundle'

    # Parcourir les valeurs uniques de la colonne sélectionnée
    for bundle in df1[bundle_column].unique():
        # Générer un biais additif et multiplicatif spécifique au metric_bundle
        if fixed_bias:
            additive_bias_per_bundle[bundle] = 2
            multiplicative_bias_per_bundle[bundle] = 1.25
        else:
            additive_bias_per_bundle[bundle] = np.random.normal(loc=additive_mean, scale=additive_std)
            multiplicative_bias_per_bundle[bundle] = np.random.normal(loc=multiplicative_mean, scale=multiplicative_std)

    
    # Appliquer les biais indépendamment à df1 et df2 en utilisant les mêmes biais générés
    biased_df = apply_bias_2(df1,df2, additive_bias_per_bundle, multiplicative_bias_per_bundle)
    
        
    
    # combined_renamed = combined.rename(columns={
    #     'mean': 'mean_combined'
    # })

    # biased_old_renamed = biased_df_old.rename(columns={
    #     'mean': 'mean_biased_old',
    #     'whatever': 'whatever_biased_old'
    # })

    # biased_new_renamed = biased_df.rename(columns={
    #     'mean': 'mean_biased_new',
    #     'whatever': 'whatever_biased_new'
    # })
    # merged = combined_renamed.merge(biased_old_renamed, on=['metric', 'bundle', 'sid'])
    # merged = merged.merge(biased_new_renamed, on=['metric', 'bundle', 'sid'])
    # merged_aa = merged[merged['sid'].isin(df1['sid'])]

    # save_scatter_plots_from_merged(merged_aa)

    
    biased_df1 = biased_df[biased_df['sid'].isin(df1['sid'])]
    biased_df2 = biased_df[biased_df['sid'].isin(df2['sid'])]
    bias_parameters = {
        'additive_mean': additive_mean,
        'multiplicative_mean': multiplicative_mean,
        'additive_std': additive_std,
        'multiplicative_std': multiplicative_std
    }
    
    return biased_df1, biased_df2, additive_bias_per_bundle, multiplicative_bias_per_bundle, bias_parameters

def apply_bias(dataframe, additive_bias_per_bundle, multiplicative_bias_per_bundle):
    biased_df = dataframe.copy()
    
    # Application de la régression et des biais pour chaque bundle unique
    bundle_column = 'metric_bundle' if 'metric_bundle' in biased_df.columns else 'bundle'

    for bundle in biased_df[bundle_column].unique():
        # Filtrer le DataFrame pour le bundle actuel
        bundle_df = biased_df[biased_df[bundle_column] == bundle]

        bundle_df_hc = bundle_df[bundle_df['disease'] == 'HC']

        X_hc = bundle_df_hc[['age', 'sex', 'handedness']]
        y_hc = bundle_df_hc['mean']

        # Préparer les covariables pour la régression
        X = bundle_df[['age', 'sex', 'handedness']]
        y = bundle_df['mean']

        # Ajuster le modèle de régression linéaire pour le bundle
        model = LinearRegression()
        model.fit(X_hc, y_hc)
        
        # Calculer les prédictions et les résidus pour le bundle
        predicted_mean = model.predict(X)
        residuals = y - predicted_mean

        # Récupérer les biais pour le bundle actuel
        additive_bias = additive_bias_per_bundle[bundle]
        multiplicative_bias = multiplicative_bias_per_bundle[bundle]
        
        # Appliquer les biais aux résidus centrés et réintégrer les effets des covariables
        biased_means_bundle = residuals * multiplicative_bias + additive_bias * np.std(residuals) + predicted_mean
        # biased_df.loc[biased_df[bundle_column] == bundle, 'whatever'] = residuals
        biased_df.loc[biased_df[bundle_column] == bundle, 'mean'] = biased_means_bundle
    
    # Assigner les valeurs biaisées calculées au DataFrame
    return biased_df

def apply_bias_2(df1,df2, additive_bias_per_bundle, multiplicative_bias_per_bundle):
    biased_df_all = pd.concat([df1, df2], ignore_index=True)
    new_biased_df = pd.DataFrame()
    for metric in biased_df_all['metric'].unique():
        # Filtrer le DataFrame pour le metric actuel
        biased_df = biased_df_all[biased_df_all['metric'] == metric]
        # Appliquer les biais pour le metric actuel
        biased_df = biased_df.sort_values(by=["site", "sid", "bundle"])
        ignore_handedness = True
        ignore_sex = False
        if biased_df['sex'].nunique() == 1:
            ignore_sex = True
        if biased_df['handedness'].nunique() == 1:
            ignore_handedness = True
        design, y = get_design_matrices(biased_df, ignore_handedness, ignore_sex)
        design_hc, y_hc = get_design_matrices(biased_df[biased_df['disease'] == 'HC'], ignore_handedness, ignore_sex)
        alpha, beta = QuickCombat.get_alpha_beta(design_hc, y_hc)

        for i, bundle in enumerate(list(np.unique(biased_df["bundle"]))):
                    # Récupérer les biais pour le bundle actuel
            additive_bias = additive_bias_per_bundle[metric+"_"+bundle]
            multiplicative_bias = multiplicative_bias_per_bundle[metric+"_"+bundle]

            bundle_df = biased_df[biased_df["bundle"] == bundle]
            covariate_effect = np.dot(design[i][1:, :].transpose(), beta[i])
            biased_df.loc[biased_df["bundle"] == bundle, 'mean'] = (y[i] - covariate_effect - alpha[i]) * multiplicative_bias + additive_bias * np.std(y[i]) + (covariate_effect + alpha[i])
        new_biased_df = pd.concat([new_biased_df, biased_df], ignore_index=True)
    # Assigner les valeurs biaisées calculées au DataFrame
    return new_biased_df

def process_test(sample_size, disease_ratio, i, train_df, test_df, directory, data_path, ssv= 'v1', fixed_biais=False):
    sizeDir = os.path.join(directory, f"{sample_size}_{int(disease_ratio*100)}")
    tempDir = os.path.join(sizeDir, f"{i}")
    os.makedirs(tempDir, exist_ok=True)
    if ssv == 'v2':
        sampled_df= sample_patients(train_df, sample_size, disease_ratio, i)
        sampled_df_biaied, test_df_biaised, gammas, deltas, parameters = generate_biaised_data(sampled_df, test_df, ssv, fixed_biais)
    else:
        train_df_biaised, test_df_biaised, gammas, deltas, parameters = generate_biaised_data(train_df, test_df, ssv, fixed_biais)
        sampled_df_biaied = sample_patients(train_df_biaised, sample_size, disease_ratio, i)

    train_sids = sampled_df_biaied['sid'].unique()

    ground_truth_train = train_df[train_df['sid'].isin(train_sids)]

    ground_truth_test = test_df
    

    if 'metric_bundle' in sampled_df_biaied.columns:
        temp_train_file = os.path.join(tempDir, f"train_{sample_size}_{int(disease_ratio*100)}_{i}_all.csv")
        sampled_df_biaied.to_csv(temp_train_file, index=False)

        temp_test_file = os.path.join(tempDir, f"test_{sample_size}_{int(disease_ratio*100)}_{i}_all.csv")
        test_df_biaised.to_csv(temp_test_file, index=False)

        ground_truth_train_file = os.path.join(tempDir, f"gt_train_{sample_size}_{int(disease_ratio*100)}_{i}.csv")
        ground_truth_train.to_csv(ground_truth_train_file, index=False)

        ground_truth_test_file = os.path.join(tempDir, f"gt_test_{sample_size}_{int(disease_ratio*100)}_{i}.csv")
        ground_truth_test.to_csv(ground_truth_test_file, index=False)

        for metric in sampled_df_biaied['metric'].unique():
            metric_df = sampled_df_biaied[sampled_df_biaied['metric'] == metric]
            metric_test_df = test_df_biaised[test_df_biaised['metric'] == metric]
            gt_metric_df = ground_truth_train[ground_truth_train['metric'] == metric]
            gt_metric_test_df = ground_truth_test[ground_truth_test['metric'] == metric]

            metric_train_file = os.path.join(tempDir, f"train_{sample_size}_{int(disease_ratio*100)}_{i}_{metric}.csv")
            metric_df.to_csv(metric_train_file, index=False)

            metric_test_file = os.path.join(tempDir, f"test_{sample_size}_{int(disease_ratio*100)}_{i}_{metric}.csv")
            metric_test_df.to_csv(metric_test_file, index=False)

            gt_metric_train_file = os.path.join(tempDir, f"gt_train_{sample_size}_{int(disease_ratio*100)}_{i}_{metric}.csv")
            gt_metric_df.to_csv(gt_metric_train_file, index=False)

            gt_metric_test_file = os.path.join(tempDir, f"gt_test_{sample_size}_{int(disease_ratio*100)}_{i}_{metric}.csv")
            gt_metric_test_df.to_csv(gt_metric_test_file, index=False)

            # subprocess.call(
            # f"scripts/combat_visualize_data.py {data_path} {temp_train_file} --out_dir {os.path.join(tempDir, 'VIZ')} -f",
            # shell=True)

    else:
        temp_train_file = os.path.join(tempDir, f"train_{sample_size}_{int(disease_ratio*100)}_{i}.csv")
        sampled_df_biaied.to_csv(temp_train_file, index=False)

        temp_test_file = os.path.join(tempDir, f"test_{sample_size}_{int(disease_ratio*100)}_{i}.csv")
        test_df_biaised.to_csv(temp_test_file, index=False)

        ground_truth_train_file = os.path.join(tempDir, f"gt_train_{sample_size}_{int(disease_ratio*100)}_{i}.csv")
        ground_truth_train.to_csv(ground_truth_train_file, index=False)
        
        ground_truth_test_file = os.path.join(tempDir, f"gt_test_{sample_size}_{int(disease_ratio*100)}_{i}.csv")
        ground_truth_test.to_csv(ground_truth_test_file, index=False)

        with open(os.path.join(tempDir, 'parameters.json'), 'w') as file:
            json.dump({'parameters': parameters, 'gammas': gammas, 'deltas': deltas}, file, indent=4)

        subprocess.call(
            f"scripts/combat_visualize_data.py {data_path} {temp_train_file} --out_dir {os.path.join(tempDir, 'VIZ')} -f",
            shell=True)
        subprocess.call(
            f"scripts/combat_visualize_data.py {data_path} {temp_test_file} --out_dir {os.path.join(tempDir, 'VIZ_TEST')} -f",
            shell=True)

def generate_sites(sample_sizes, disease_ratios, num_tests, directory, data_path, SYNTHETIC_SITES_VERSION='v1', disease=None, fixed_biais=False, n_jobs=-1):
    df = pd.read_csv(data_path)
    df = df[~df['bundle'].isin(['left_ventricle', 'right_ventricle'])]
    df = df[~((df['disease'] == 'HC') & (df['old_site'] != 'CamCAN'))]
    if disease == "ASTMIX":
        df = df[df['disease'].isin(['AD', 'SCHZ', 'TBI', 'HC'])]
    elif disease is not None and disease != "ALL":
        df = df[(df['disease'] == disease) | (df['disease'] == 'HC')]

    train_df, test_df = split_train_test(df, test_size=0.05, random_state=43)

    Parallel(n_jobs=n_jobs)(
    delayed(process_test)(
        sample_size, disease_ratio, i, train_df, test_df, directory, data_path, SYNTHETIC_SITES_VERSION, fixed_biais
    )
    for sample_size in sample_sizes
    for disease_ratio in disease_ratios
    for i in range(num_tests)
)
    

def generate_sites_no_file(sample_sizes, disease_ratios, num_tests, df,  disease=None, n_jobs=-1):
    df = df[~df['bundle'].isin(['left_ventricle', 'right_ventricle'])]
    df = df[~((df['disease'] == 'HC') & (df['old_site'] != 'CamCAN'))]
    if disease == "ASTMIX":
        df = df[df['disease'].isin(['AD', 'SCHZ', 'TBI', 'HC'])]
    elif disease is not None:
        df = df[(df['disease'] == disease) | (df['disease'] == 'HC')]
    dfs = Parallel(n_jobs=n_jobs)(
        delayed(sample_patients)(
            df, sample_size, disease_ratio, i
        )
        for sample_size in sample_sizes
        for disease_ratio in disease_ratios
        for i in range(num_tests)
    )
    return dfs 