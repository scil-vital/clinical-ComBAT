import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

from clinical_combat.utils.robust import find_outliers_IQR, find_outliers_MAD, reject_outliers_until_mad_equals_mean, find_outliers_VS, find_outliers_VS2, remove_top_x_percent, cheat

import matplotlib.pyplot as plt

def find_outliers(df, robust_method, args = []):
    # Find outliers
    outliers_idx = []
    if robust_method in ['IQR', 'MAD', 'MAD_MEAN', 'VS', 'VS2','TOP5', 'TOP10', 'TOP20', 'TOP30', 'TOP40', 'TOP50', 'CHEAT']:
        for metric in df['metric'].unique():
            for bundle in df['bundle'].unique():
                data = df[(df['metric'] == metric) & (df['bundle'] == bundle)]
                outliers_idx += use_robust_method(data, robust_method)
    elif robust_method == 'kmeans':
        outliers_idx = use_robust_method(data, robust_method)
    return outliers_idx

def use_robust_method(data, robust_method, args = []):
    if robust_method == 'IQR':
        return find_outliers_IQR(data)
    elif robust_method == 'MAD':
        return find_outliers_MAD(data, args)
    elif robust_method == 'MAD_MEAN':
        return reject_outliers_until_mad_equals_mean(data, args)
    elif robust_method == 'VS':
        return find_outliers_VS(data)
    elif robust_method == 'VS2':
        return find_outliers_VS2(data)
    elif robust_method in ['TOP5', 'TOP10', 'TOP20', 'TOP30', 'TOP40', 'TOP50']:
        return remove_top_x_percent(data, x=int(robust_method
        .replace('TOP', '')))
    elif robust_method == 'CHEAT':
        return cheat(data)
    else:
        raise ValueError("Invalid robust method. Choose between 'iqr' and 'mad'.")

def analyze_detection_performance(outliers_idx, mov_data):
    
    metrics_list = []
    mov_data['is_malade'] = mov_data['disease'].apply(lambda x: 0 if x == 'HC' else 1)
    mov_data['is_outlier'] = 0
    mov_data.loc[outliers_idx, 'is_outlier'] = 1
    
    bundle_column = 'metric_bundle' if 'metric_bundle' in mov_data.columns else 'bundle'
    for bundle in mov_data[bundle_column].unique():
        bundle_data = mov_data[mov_data[bundle_column] == bundle]
        #plot_outliers_data(bundle_data)

        y_true = bundle_data['is_malade'].tolist()
        y_pred = bundle_data['is_outlier'].tolist()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        taux_faux_positifs = fp / (fp + tn) if (fp + tn) != 0 else 0
        f1 = f1_score(y_true, y_pred)

        metrics_list.append({
            'bundle': bundle,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'taux_faux_positifs': taux_faux_positifs,
            'f1_score': f1
        })

    # Overall metrics
    overall_outliers_sid = mov_data.loc[outliers_idx]['sid'].unique().tolist()
    mov_data['is_outlier'] = mov_data['sid'].apply(lambda x: 1 if x in overall_outliers_sid else 0)
    patients = mov_data.drop_duplicates(subset='sid')

    y_true = patients['is_malade'].tolist()
    y_pred = patients['is_outlier'].tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    taux_faux_positifs = fp / (fp + tn) if (fp + tn) != 0 else 0
    f1 = f1_score(y_true, y_pred)

    metrics_list.append({
        'bundle': 'overall',
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'taux_faux_positifs': taux_faux_positifs,
        'f1_score': f1
    })
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('bundle', inplace=True)
    metrics_df = metrics_df.sort_index(axis=1)
    metrics_df = metrics_df.T.reset_index()
    metrics_df.rename(columns={'index': 'metric'}, inplace=True)
    metrics_df['site'] = mov_data.site.unique()[0]
    return metrics_df



def get_matching_indexes(file_path, subset_path):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file_path)
    df2 = pd.read_csv(subset_path)

    # Find the matching indexes where entire rows are the same
    matching_indexes = df2["old_index"].tolist()

    # On vérifie ligne par ligne que DF2.iloc[i] == DF1.loc[DF2.iloc[i]["OldIndex"]]
    colonnes_a_verifier = ["sid", "bundle", "mean", "age"]

    # Boucle de vérification
    for i in range(len(df2)):
        index_dans_df1 = df2.iloc[i]["old_index"]
        ligne_df1 = df1.loc[index_dans_df1, colonnes_a_verifier]
        ligne_df2 = df2.iloc[i][colonnes_a_verifier]

        if not ligne_df1.equals(ligne_df2):
            print(f"Mismatch à la ligne {i} :")
            print("Dans df1 :\n", ligne_df1)
            print("Dans df2 :\n", ligne_df2)
            print("-" * 40)


    return matching_indexes


import pandas as pd
import numpy as np

def z_score_detection(df_file,
                      mean_col: str = "mean_no_cov",
                      threshold: float = 1.5) -> list[str]:
    """
    Repère les patients (sid) dont la moyenne du |z-score| dépasse `threshold`
    et renvoie la liste de ces sid.

    Paramètres
    ----------
    df : DataFrame
        Doit contenir les colonnes 'sid', 'metric_bundle' et `mean_col`.
    mean_col : str
        Colonne sur laquelle on calcule le z-score (défaut 'mean_no_cov').
    threshold : float
        Seuil de la moyenne du z-score (défaut 1.5).

    Retour
    ------
    outlier_sids : list[str]
        Liste des sid identifiés comme outliers.
    """
    # 1) stats par metric_bundle

    df = pd.read_csv(df_file)
    stats = (df.groupby("metric_bundle")[mean_col]
               .agg(['mean', 'std'])
               .rename(columns={'mean': 'global_mean', 'std': 'global_std'}))
    stats["global_std"].replace(0, 1e-6, inplace=True)  # éviter div/0

    # 2) merge pour récupérer les stats
    df_z = df.merge(stats, on="metric_bundle", how="left")

    # 3) |z-score|
    df_z["abs_zscore"] = ((df_z[mean_col] - df_z["global_mean"])
                          / df_z["global_std"]).abs()

    # 4) moyenne |z-score| par patient
    mean_z = (df_z.groupby("sid", as_index=False)
                    .agg(mean_abs_zscore=("abs_zscore", "mean")))

    # 5) liste des outliers
    outlier_sids = mean_z.loc[
        mean_z["mean_abs_zscore"] > threshold, "sid"
    ].tolist()

    return outlier_sids



def flag_sid(df, sids, method):
    df[method] = df['sid'].isin(sids).astype(int)
    return df