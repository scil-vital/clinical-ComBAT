from clinical_combat.harmonization import from_model_name
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import json
import os
from scipy.spatial.distance import euclidean



def remove_outliers(ref_data, mov_data, args):
    print("Removing outliers with method", args.robust)

    find_outlier = ROBUST_METHODS.get(args.robust)
    rwp = args.rwp

    # Initialize QC model
    QC = from_model_name(
        args.method.lower(),
        ignore_handedness_covariate=args.ignore_handedness,
        ignore_sex_covariate=args.ignore_sex,
        use_empirical_bayes=not args.no_empirical_bayes,
        limit_age_range=args.limit_age_range,
        degree=args.degree,
        regul_ref=args.regul_ref,
        regul_mov=args.regul_mov,
        nu=args.nu,
        tau=args.tau,
    )
    QC.fit(ref_data, mov_data, False)
    site = mov_data['site'].unique()[0]
    rwp_str = "RWP" if rwp else "NoRWP"

    mov_data = remove_covariates_effects2(mov_data)

    # Find outliers
    outliers_idx = []
    for bundle in QC.bundle_names:
        data = mov_data.query("bundle == @bundle")
        outliers_idx += find_outlier(data)
    if not rwp:
        # plot_distributions_by_bundle(mov_data, outliers_idx, args.out_dir)
        # scatter_plot_with_colors(mov_data, outliers_idx, 'mean_no_cov', os.path.join(args.out_dir, 'scatter'), site, title='scatter plot with outliers')
        plot_distributions_and_scatter(mov_data, outliers_idx, args.out_dir, y_column='mean_no_cov', robust_method=args.robust)

    mov_data = mov_data.drop(columns=['mean_no_cov'])

    

    # Save outliers
    outliers_filename = os.path.join(args.out_dir,f"outliers_{site}_{args.robust}_{rwp_str}.csv")
    outliers = mov_data.loc[outliers_idx]
    outliers.index.name = "old_index"
    outliers.to_csv(outliers_filename, index=True)

    # Remove outliers from movement data
    if rwp:
        print("RWP is applied")
        outlier_patients_ids = mov_data.loc[outliers_idx]['sid'].unique().tolist()
        if len(outlier_patients_ids) < (len(mov_data['sid'].unique().tolist())-1):
            mov_data = mov_data[~mov_data['sid'].isin(outlier_patients_ids)]
        else:
            print("All patients are outliers. RWP not applied.")
            mov_data = mov_data.drop(outliers_idx)
    else:
        mov_data = mov_data.drop(outliers_idx)
    return mov_data

def find_outliers_IQR(data):

    Q1 = data['mean_no_cov'].quantile(0.25)
    Q3 = data['mean_no_cov'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrer les valeurs aberrantes
    outliers = data[(data['mean_no_cov'] < lower_bound) | (data['mean_no_cov'] > upper_bound)]

    return outliers.index.to_list()

def find_outliers_VS(data, column='mean_no_cov'):
    """
    Équilibre les valeurs autour de la médiane en supprimant les valeurs les plus éloignées
    jusqu'à ce que la somme des écarts à droite et à gauche de la médiane soit équilibrée.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser pour équilibrer les valeurs.

    Retourne :
        list: Une liste des indices des valeurs supprimées.
    """
    # --- Classement des métriques (comme décidé plus tôt) -------------
    METRICS_HIGH = {'md', 'mdt', 'rd', 'rdt', 'fw', 'ad', 'adt'}  # patho ↑
    METRICS_LOW  = {'fa', 'fat', 'afd'}                           # patho ↓
    # ------------------------------------------------------------------

    metric_name = data['metric'].iloc[0]

    if metric_name in METRICS_HIGH:
        side = 'right'   # on enlève les plus grosses valeurs
    elif metric_name in METRICS_LOW:
        side = 'left'    # on enlève les plus petites valeurs
    else:
        return []        # métrique non classée → on ne fait rien

    outliers_idx = []
    median = data[column].median()

    while True:
        # Moyennes des écarts de chaque côté de la médiane
        left_mean  = abs((data[data[column] < median][column] - median).mean())
        right_mean = abs((data[data[column] > median][column] - median).mean())

        # Équilibre atteint (même critère qu’avant)
        if abs(left_mean - right_mean) <= 1e-6:
            break

        if side == 'right':
            # On continue seulement si la droite domine encore
            if right_mean <= left_mean:
                break
            target_idx = data[column].idxmax()   # plus grand écart à droite
        else:  # side == 'left'
            if left_mean <= right_mean:
                break
            target_idx = data[column].idxmin()   # plus grand écart à gauche

        outliers_idx.append(target_idx)
        data = data.drop(target_idx)

    return outliers_idx


def find_outliers_VS2(data, column='mean_no_cov'):
    """
    Équilibre les valeurs autour de la médiane en supprimant les valeurs les plus éloignées
    jusqu'à ce que la somme des écarts à droite et à gauche de la médiane soit équilibrée.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser pour équilibrer les valeurs.

    Retourne :
        list: Une liste des indices des valeurs supprimées.
    """
    # --- Classement des métriques (comme décidé plus tôt) -------------
    METRICS_HIGH = {'md', 'mdt', 'rd', 'rdt', 'fw', 'ad', 'adt'}  # patho ↑
    METRICS_LOW  = {'fa', 'fat', 'afd'}                           # patho ↓
    # ------------------------------------------------------------------

    metric_name = data['metric'].iloc[0]

    if metric_name in METRICS_HIGH:
        side = 'right'   # on enlève les plus grosses valeurs
    elif metric_name in METRICS_LOW:
        side = 'left'    # on enlève les plus petites valeurs
    else:
        return []        # métrique non classée → on ne fait rien

    outliers_idx = []
    median = data[column].median()

    while True:
        # Moyennes des écarts de chaque côté de la médiane
        median = data[column].median()
        
        left_mean  = abs((data[data[column] < median][column] - median).mean())
        right_mean = abs((data[data[column] > median][column] - median).mean())

        # Équilibre atteint (même critère qu’avant)
        if abs(left_mean - right_mean) <= 1e-6:
            break

        if side == 'right':
            # On continue seulement si la droite domine encore
            if right_mean <= left_mean:
                break
            target_idx = data[column].idxmax()   # plus grand écart à droite
        else:  # side == 'left'
            if left_mean <= right_mean:
                break
            target_idx = data[column].idxmin()   # plus grand écart à gauche

        outliers_idx.append(target_idx)
        data = data.drop(target_idx)

    return outliers_idx

def find_outliers_MAD(data, column='mean_no_cov', threshold=3.5):
    """
    Détecte les valeurs aberrantes dans un DataFrame à l'aide de la méthode MAD.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser pour détecter les outliers.
        threshold (float): Le seuil pour considérer une valeur comme un outlier. Par défaut : 3.5.

    Retourne :
        list: Une liste des indices des valeurs aberrantes.
    """
    # Calcul de la médiane de la colonne
    median = data[column].median()

    # Calcul du MAD
    mad = np.median(np.abs(data[column] - median))

    # Calcul des scores normalisés (distance modifiée)
    if mad == 0:
        print("MAD is zero, all values will appear as non-outliers.")
        return []

    modified_z_scores = 0.6745 * (data[column] - median) / mad

    # Identification des indices des outliers
    outliers = data[np.abs(modified_z_scores) > threshold]

    return outliers.index.to_list()


def reject_outliers_until_mad_equals_mean(data, threshold=0.001): 
    column = 'mean_no_cov'
    outliers_idx = []

    # --- Définis tes trois métriques ici ------------------------------
    METRICS_HIGH = {'md', 'mdt', 'rd', 'rdt', 'fw', 'ad', 'adt'}
    METRICS_LOW  = {'fa', 'fat', 'afd'}
    # ------------------------------------------------------------------

    metric_name = data['metric'].iloc[0]

    if metric_name in METRICS_HIGH:
        pick_idx = lambda s: s.idxmax()
        direction_valid = lambda mean, median: mean > median
    elif metric_name in METRICS_LOW:
        pick_idx = lambda s: s.idxmin()
        direction_valid = lambda mean, median: mean < median
    else:
        return outliers_idx

    while True:
        median = data[column].median()
        mean   = data[column].mean()

        # Si la direction n'est pas respectée, on arrête tout de suite
        if not direction_valid(mean, median):
            break

        if abs(median - mean) / median < threshold:
            break

        target_idx = pick_idx(data[column])
        outliers_idx.append(target_idx)
        data = data.drop(target_idx)

    return outliers_idx

def remove_top_x_percent(data, column='mean_no_cov', x=5):
    """
    Supprime les x % des valeurs les plus élevées dans une colonne donnée.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser.
        x (float): Le pourcentage des valeurs les plus élevées à supprimer.

    Retourne :
        list: Une liste des indices des valeurs supprimées.
    """
    if x <= 0 or x > 100:
        raise ValueError("x doit être un pourcentage entre 0 et 100.")

    # Calcul du nombre de valeurs à supprimer
    num_to_remove = int(len(data) * (x / 100.0))

    # Trouver les indices des x % des valeurs les plus élevées
    outliers_idx = data.nlargest(num_to_remove, column).index.to_list()
    # Afficher les 'sid' des outliers

    return outliers_idx

def top5(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=5)

def top10(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=10)

def top20(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=20)

def top30(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=30)

def top40(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=40)   

def top50(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=50)   

def cheat(data):
    return data[data['disease'] != 'HC'].index.to_list()

def zscore(data):
    return flagged(data, method='Z_SCORE')

def flagged(data, method):
    return data[data[method] == 1].index.to_list() 

def rien(data):
    return []

    
ROBUST_METHODS = {
    "IQR": find_outliers_IQR,
    "MAD": find_outliers_MAD,
    "MMS": reject_outliers_until_mad_equals_mean,
    "VS": find_outliers_VS,
    "VS2": find_outliers_VS2,
    "TOP5": top5,
    "TOP10": top10,
    "TOP20": top20,
    "TOP30": top30,
    "TOP40": top40,
    "TOP50": top50,
    "CHEAT": cheat,
    "FLIP": rien,
    "Z_SCORE": zscore
}
from clinical_combat.harmonization.QuickCombat import QuickCombat

def get_design_matrices(df, ignore_handedness=False):
    design = []
    Y = []
    for bundle in list(np.unique(df["bundle"])):
        data = df.query("bundle == @bundle")
        hstack_list = []
        hstack_list.append(np.ones(len(data["sid"])))  # intercept
        hstack_list.append(QuickCombat.to_category(data["sex"]))
        if not ignore_handedness:
            hstack_list.append(QuickCombat.to_category(data["handedness"]))
        ages = data["age"].to_numpy()
        hstack_list.append(ages)
        design.append(np.array(hstack_list))
        Y.append(data["mean"].to_numpy())
    return design, Y

def remove_covariates_effects2(df):
    df = df.sort_values(by=["site", "sid", "bundle"])
    ignore_handedness = False
    if df['handedness'].nunique() == 1:
        ignore_handedness = True
    design, y = get_design_matrices(df, ignore_handedness)
    alpha, beta = QuickCombat.get_alpha_beta(design, y)

    df['mean_no_cov'] = df['mean']
    for i, bundle in enumerate(list(np.unique(df["bundle"]))):
        bundle_df = df[df['bundle'] == bundle]
        covariate_effect = np.dot(design[i][1:, :].transpose(), beta[i])
        df.loc[df['bundle'] == bundle, 'mean_no_cov'] = (y[i] - covariate_effect)
    return df


def plot_distributions_and_scatter(df_before, outliers_idx, output_dir, y_column='mean_no_cov', robust_method=''):
    df_after = df_before.drop(outliers_idx)
    df_before = df_before.copy()
    df_before['is_malade'] = df_before['disease'].apply(lambda x: 0 if x == 'HC' else 1)
    df_before['is_outlier'] = 0
    df_before.loc[outliers_idx, 'is_outlier'] = 1

    disease = [d for d in df_before['disease'].unique() if d != 'HC'][0]
    site = df_before['site'].unique()[0]
    metric = df_before['metric'].unique()[0]
    parts = output_dir.split(os.sep)
    parts[parts.index(robust_method)] = "DISTRIBUTION"
    new_base = os.path.join(*parts)  # enlève /0/robust/


    bundle_column = 'metric_bundle' if 'metric_bundle' in df_before.columns else 'bundle'
    bundles = df_before[bundle_column].unique()

    for bundle in bundles:
        df_b_before = df_before[df_before[bundle_column] == bundle]
        df_b_after = df_after[df_after[bundle_column] == bundle]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

        # --- KDE Avant filtrage ---
        sns.kdeplot(df_b_before[df_b_before['disease'] == 'HC'][y_column], label='HC', ax=axes[0], linewidth=2)
        sns.kdeplot(df_b_before[df_b_before['disease'] != 'HC'][y_column], label='Malades', ax=axes[0], linewidth=2)
        sns.kdeplot(df_b_before[y_column], label='Tous', linestyle='--', ax=axes[0], linewidth=2)

        mean_hc_before = df_b_before[df_b_before['disease'] == 'HC'][y_column].mean()
        std_hc_before = df_b_before[df_b_before['disease'] == 'HC'][y_column].std()
        mean_all_before = df_b_before[y_column].mean()
        std_all_before = df_b_before[y_column].std()

        axes[0].axvline(mean_hc_before, color='blue', linestyle=':', linewidth=1.5, label='Moyenne HC')
        axes[0].axvline(mean_all_before, color='black', linestyle='-.', linewidth=1.5, label='Moyenne Tous')
        axes[0].axvspan(mean_hc_before - std_hc_before, mean_hc_before + std_hc_before,
                        color='blue', alpha=0.1, label='±1 STD HC')
        axes[0].axvspan(mean_all_before - std_all_before, mean_all_before + std_all_before,
                        color='black', alpha=0.1, label='±1 STD Tous')

        text_stats = (
            f"HC:\nμ={mean_hc_before:.6f}\nσ={std_hc_before:.6f}\n"
            f"Tous:\nμ={mean_all_before:.6f}\nσ={std_all_before:.6f}"
        )
        axes[0].text(0.02, 0.95, text_stats, transform=axes[0].transAxes,
                     verticalalignment='top', horizontalalignment='left', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        axes[0].set_title("Avant filtrage")
        axes[0].set_ylabel('Densité')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # --- KDE Après filtrage ---
        sns.kdeplot(df_b_after[df_b_after['disease'] == 'HC'][y_column], label='HC', ax=axes[1], linewidth=2)
        sns.kdeplot(df_b_after[df_b_after['disease'] != 'HC'][y_column], label='Malades', ax=axes[1], linewidth=2)
        sns.kdeplot(df_b_after[y_column], label='Tous', linestyle='--', ax=axes[1], linewidth=2)

        mean_hc_after = df_b_after[df_b_after['disease'] == 'HC'][y_column].mean()
        std_hc_after = df_b_after[df_b_after['disease'] == 'HC'][y_column].std()
        mean_all_after = df_b_after[y_column].mean()
        std_all_after = df_b_after[y_column].std()

        axes[1].axvline(mean_hc_after, color='blue', linestyle=':', linewidth=1.5, label='Moyenne HC')
        axes[1].axvline(mean_all_after, color='black', linestyle='-.', linewidth=1.5, label='Moyenne Tous')
        axes[1].axvspan(mean_hc_after - std_hc_after, mean_hc_after + std_hc_after,
                        color='blue', alpha=0.1, label='±1 STD HC')
        axes[1].axvspan(mean_all_after - std_all_after, mean_all_after + std_all_after,
                        color='black', alpha=0.1, label='±1 STD Tous')

        text_stats = (
            f"HC:\nμ={mean_hc_after:.6f}\nσ={std_hc_after:.6f}\n"
            f"Tous:\nμ={mean_all_after:.6f}\nσ={std_all_after:.6f}"
        )
        axes[1].text(0.02, 0.95, text_stats, transform=axes[1].transAxes,
                     verticalalignment='top', horizontalalignment='left', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        axes[1].set_title("Après filtrage")
        axes[1].legend(loc='upper right')
        axes[1].grid(True)

        # --- Scatterplot ---
        colors = df_b_before.apply(lambda row: (
            'blue' if row['is_malade'] == 0 and row['is_outlier'] == 0 else
            'green' if row['is_malade'] == 1 and row['is_outlier'] == 1 else
            'red' if row['is_malade'] == 0 and row['is_outlier'] == 1 else
            'orange'
        ), axis=1)

        axes[2].scatter(df_b_before['age'], df_b_before[y_column], c=colors)
        axes[2].set_title("Scatter MEAN_NO_COV")
        axes[2].set_xlabel('Âge')
        axes[2].set_ylabel(y_column)
        axes[2].grid(True)

        # Légende couleur pour le scatter plot (en haut à gauche)
        legend_elements = [
            Patch(facecolor='blue', label='Sain & pas outlier'),
            Patch(facecolor='green', label='Malade & outlier'),
            Patch(facecolor='red', label='Sain & outlier'),
            Patch(facecolor='orange', label='Malade & pas outlier')
        ]
        axes[2].legend(handles=legend_elements, loc='upper left', title='Légende couleurs')

        plt.suptitle(f"Analyse Maladie:{disease} - {site} ({metric}- {bundle})")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        dir_path = os.path.join(new_base, bundle)
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(os.path.join(dir_path, f"{bundle}_{site}_{robust_method}.png"))
        plt.close()