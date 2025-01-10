import os
import numpy as np
import pandas as pd
import subprocess
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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


def generate_biaised_data(df1, df2, 
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
        additive_bias_per_bundle[bundle] = np.random.normal(loc=additive_mean, scale=additive_std)
        multiplicative_bias_per_bundle[bundle] = np.random.normal(loc=multiplicative_mean, scale=multiplicative_std)

    
    # Appliquer les biais indépendamment à df1 et df2 en utilisant les mêmes biais générés
    combined = pd.concat([df1, df2], ignore_index=True)
    biased_df = apply_bias(combined, additive_bias_per_bundle, multiplicative_bias_per_bundle)
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

        # Préparer les covariables pour la régression
        X = bundle_df[['age', 'sex', 'handedness']]
        y = bundle_df['mean']
        
        # Ajuster le modèle de régression linéaire pour le bundle
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculer les prédictions et les résidus pour le bundle
        predicted_mean = model.predict(X)
        residuals = y - predicted_mean

        # Récupérer les biais pour le bundle actuel
        additive_bias = additive_bias_per_bundle[bundle]
        multiplicative_bias = multiplicative_bias_per_bundle[bundle]
        
        # Appliquer les biais aux résidus centrés et réintégrer les effets des covariables
        biased_means_bundle = residuals * multiplicative_bias + additive_bias * np.std(residuals) + predicted_mean
        biased_df.loc[biased_df[bundle_column] == bundle, 'mean'] = biased_means_bundle
    
    # Assigner les valeurs biaisées calculées au DataFrame
    return biased_df

#GENERATE SITES
def generate_sites(sample_sizes, disease_ratios, num_tests, directory, data_path, disease=None):
    df = pd.read_csv(data_path)
    if disease is not None:
        df = df[(df['disease'] == disease) | (df['disease'] == 'HC')]

    train_df, test_df = split_train_test(df, test_size=0.2, random_state=42)
    # Initialize DataFrames to store the results
    for sample_size in sample_sizes:
        for disease_ratio in disease_ratios:  
            sizeDir = os.path.join(directory, f"{sample_size}_{int(disease_ratio*100)}")
            for i in range(num_tests):
                
                tempDir = os.path.join(sizeDir, f"{i}")
                os.makedirs(tempDir, exist_ok=True)

                train_df_biaised, test_df_biaised, gammas, deltas, parameters= generate_biaised_data(train_df, test_df)

                sampled_df_biaied =  sample_patients(train_df_biaised, sample_size, disease_ratio,i)

                # Iterate through each unique metric and save files
                if 'metric_bundle' in sampled_df_biaied.columns:
                    # Sauvegarder l'échantillon dans un fichier temporaire
                    temp_train_file = os.path.join(tempDir, f"train_{sample_size}_{int(disease_ratio*100)}_{i}_all.csv")
                    sampled_df_biaied.to_csv(temp_train_file, index=False)
                    
                    temp_test_file = os.path.join(tempDir, f"test_{sample_size}_{int(disease_ratio*100)}_{i}_all.csv")
                    test_df_biaised.to_csv(temp_test_file, index=False)
                    for metric in sampled_df_biaied['metric'].unique():
                        metric_df = sampled_df_biaied[sampled_df_biaied['metric'] == metric]
                        metric_test_df = test_df_biaised[test_df_biaised['metric'] == metric]

                        # Save the metric-specific train and test files
                        metric_train_file = os.path.join(tempDir, f"train_{sample_size}_{int(disease_ratio*100)}_{i}_{metric}.csv")
                        metric_df.to_csv(metric_train_file, index=False)
                        
                        metric_test_file = os.path.join(tempDir, f"test_{sample_size}_{int(disease_ratio*100)}_{i}_{metric}.csv")
                        metric_test_df.to_csv(metric_test_file, index=False)

                        # Apply visualization for each metric-specific file
                        
                        cmd = (
                            "scripts/combat_visualize_data.py"
                            + " "
                            + f"DONNES/adni_compilation.{metric}.csv.gz"
                            + " "
                            + metric_train_file
                            + " --out_dir "
                            + os.path.join(tempDir, f"VIZ/{metric}")
                            + " -f"
                            #+ " --bundles all"
                        )
                        #subprocess.call(cmd, shell=True)
                        cmd = (
                            "scripts/combat_visualize_data.py"
                            + " "
                            + f"DONNES/adni_compilation.{metric}.csv.gz"
                            + " "
                            + metric_test_file
                            + " --out_dir "
                            + os.path.join(tempDir, f"VIZ_TEST/{metric}")
                            + " -f"
                            #+ " --bundles all"
                        )
                        #subprocess.call(cmd, shell=True)
                else:
                    # Sauvegarder l'échantillon dans un fichier temporaire
                    temp_train_file = os.path.join(tempDir, f"train_{sample_size}_{int(disease_ratio*100)}_{i}.csv")
                    sampled_df_biaied.to_csv(temp_train_file, index=False)
                    
                    temp_test_file = os.path.join(tempDir, f"test_{sample_size}_{int(disease_ratio*100)}_{i}.csv")
                    test_df_biaised.to_csv(temp_test_file, index=False)

                    # Sauvegarde dans un fichier JSON
                    with open(os.path.join(tempDir,'parameters.json'), 'w') as file:
                        json.dump({'parameters': parameters, 'gammas': gammas, 'deltas': deltas}, file, indent=4)

                    cmd = (
                        "scripts/combat_visualize_data.py"
                        + " "
                        + data_path
                        + " "
                        + temp_train_file
                        + " --out_dir "
                        + os.path.join(tempDir, "VIZ")
                        + " -f"
                        #+ " --bundles all"
                    )
                    subprocess.call(cmd, shell=True)
                    cmd = (
                        "scripts/combat_visualize_data.py"
                        + " "
                        + data_path
                        + " "
                        + temp_test_file
                        + " --out_dir "
                        + os.path.join(tempDir, "VIZ_TEST")
                        + " -f"
                        #+ " --bundles all"
                    )
                    subprocess.call(cmd, shell=True)
