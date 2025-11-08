from clinical_combat.harmonization import from_model_name
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.neighbors import LocalOutlierFactor
import json
import os
import inspect
from scipy.spatial.distance import euclidean
from scipy.stats import shapiro


def remove_outliers(ref_data, mov_data, args):
    print("Removing outliers with method", args.robust)

    find_outlier = ROBUST_METHODS.get(args.robust)
    if find_outlier is None:
        raise ValueError(f"Robust method '{args.robust}' is not supported.")
    rwp = args.rwp
    threshold = getattr(args, "robust_threshold", None)

    # Initialize QC model
    covbat_pve = getattr(args, "covbat_pve", 0.95)
    covbat_max_components = getattr(args, "covbat_max_components", None)
    gam_n_knots = getattr(args, "gam_n_knots", 7)
    gmm_components = getattr(args, "gmm_components", 2)
    gmm_tol = getattr(args, "gmm_tol", 1e-4)
    gmm_max_iter = getattr(args, "gmm_max_iter", 200)

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
        covbat_pve=covbat_pve,
        covbat_max_components=covbat_max_components,
        gam_n_knots=gam_n_knots,
        gmm_components=gmm_components,
        gmm_tol=gmm_tol,
        gmm_max_iter=gmm_max_iter,
    )
    QC.fit(ref_data, mov_data, False)
    site = mov_data['site'].unique()[0]
    rwp_str = "RWP" if rwp else "NoRWP"

    mov_data = remove_covariates_effects2(mov_data)

    # Find outliers
    outliers_idx = []
    if args.robust in ['Z_SCORE_METRIC']:
        outliers_idx += find_outlier(mov_data, threshold=threshold)
    else:
        for bundle in QC.bundle_names:
            data = mov_data.query("bundle == @bundle")
            outliers_idx += find_outlier(data, threshold=threshold)
        
    if not rwp and (site.endswith('viz')):
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

def find_outliers_IQR(data, threshold=1.5):
    """
    Détecte les outliers selon la méthode de l'IQR,
    en s'assurant de laisser au moins 2 données dans l'ensemble.

    Retourne :
        list: Indices des outliers.
    """
    if len(data) <= 2:
        return []

    Q1 = data['mean_no_cov'].quantile(0.25)
    Q3 = data['mean_no_cov'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Masque des outliers
    outlier_mask = (data['mean_no_cov'] < lower_bound) | (data['mean_no_cov'] > upper_bound)
    outlier_indices = data[outlier_mask].index.to_list()
    # S'assurer qu'il reste au moins 2 données
    if len(data) - len(outlier_indices) < 2:
        bob = len(outlier_indices)
        # Trier les outliers par distance à la borne la plus proche
        distances = data.loc[outlier_mask, 'mean_no_cov'].apply(
            lambda x: min(abs(x - lower_bound), abs(x - upper_bound))
        )
        sorted_outliers = distances.sort_values(ascending=False).index

        # Garder seulement ceux nécessaires pour conserver au moins 2 données
        n_to_remove = len(data) - 2
        outlier_indices = sorted_outliers[:n_to_remove].to_list()
        

    return outlier_indices

def find_outliers_ZSCORE_BUNDLE(data, threshold=3.0):
    """
    Détecte les outliers avec le Z-score (|z| > threshold),
    tout en laissant au moins 2 valeurs dans l’ensemble.

    Paramètres
    ----------
    data : pandas.DataFrame
        Doit contenir la colonne 'mean_no_cov'.
    threshold : float, optionnel
        Threshold absolu du Z-score (par défaut 3.0).

    Retour
    ------
    list
        Indices des outliers.
    """
    # Pas assez de données pour juger
    if len(data) <= 2:
        return []

    col = data['mean_no_cov']
    mu = col.mean()
    sigma = col.std(ddof=0)  # écart-type population
    # Si pas de variance, aucun outlier possible
    if sigma == 0:
        return []

    z_scores = (col - mu) / sigma
    outlier_mask = z_scores.abs() > threshold
    outlier_indices = data[outlier_mask].index.to_list()

    # S’assurer qu’il reste au moins 2 points
    if len(data) - len(outlier_indices) < 2:
        # Trier les candidats par distance absolue (|z|)
        distances = z_scores.abs()[outlier_mask]
        sorted_outliers = distances.sort_values(ascending=False).index
        n_to_remove = len(data) - 2
        outlier_indices = sorted_outliers[:n_to_remove].to_list()

    return outlier_indices

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

    while len(data) > 2:
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


def find_outliers_VS2(data):
    """
    Équilibre les valeurs autour de la médiane en supprimant les valeurs les plus éloignées
    jusqu'à ce que la somme des écarts à droite et à gauche de la médiane soit équilibrée.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.

    Retourne :
        list: Une liste des indices des valeurs supprimées.
    """
    # --- Classement des métriques (comme décidé plus tôt) -------------
    METRICS_HIGH = {'md', 'mdt', 'rd', 'rdt', 'fw', 'ad', 'adt'}  # patho ↑
    METRICS_LOW  = {'fa', 'fat', 'afd'}                           # patho ↓
    # ------------------------------------------------------------------
    column='mean_no_cov'
    metric_name = data['metric'].iloc[0]

    if metric_name in METRICS_HIGH:
        side = 'right'   # on enlève les plus grosses valeurs
    elif metric_name in METRICS_LOW:
        side = 'left'    # on enlève les plus petites valeurs
    else:
        return []        # métrique non classée → on ne fait rien

    outliers_idx = []
    median = data[column].median()

    while len(data) > 2:
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

def find_outliers_MAD(data, threshold=3.5):
    """
    Détecte les valeurs aberrantes dans un DataFrame à l'aide de la méthode MAD,
    en s'assurant de laisser au moins 2 données dans le DataFrame.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        threshold (float): Valeur limite au-delà de laquelle une observation est considérée
            comme outlier. Par défaut : 3.5.

    Retourne :
        list: Une liste des indices des valeurs aberrantes, tout en conservant au moins deux données valides.
    """
    column='mean_no_cov'
    if len(data) <= 2:
        return []

    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))

    if mad == 0:
        print("MAD is zero, all values will appear as non-outliers.")
        return []

    modified_z_scores = 0.6745 * (data[column] - median) / mad
    outlier_mask = np.abs(modified_z_scores) > threshold
    outlier_indices = data[outlier_mask].index.to_list()

    # S'assurer qu'au moins 2 données restent
    if len(data) - len(outlier_indices) < 2:
        # Trier les outliers par score absolu décroissant (ceux les plus extrêmes d'abord)
        scores = np.abs(modified_z_scores[outlier_mask])
        sorted_outliers = scores.sort_values(ascending=False).index

        # Réduire le nombre d'outliers à retirer
        n_to_remove = len(data) - 2
        outlier_indices = sorted_outliers[:n_to_remove].to_list()

    return outlier_indices

def find_outliers_ZSCORE_METRIC(df, threshold=3.0):
    if df["sid"].nunique() <= 2:
        return []
    work = df.copy()

    def _z(x):
        mu = x.mean()
        sigma = x.std(ddof=0)
        if sigma == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - mu) / sigma

    work["_z"] = work.groupby("bundle")["mean_no_cov"].transform(_z)
    mean_abs_z = work.groupby("sid")["_z"].apply(lambda s: s.abs().mean())
    outlier_sids = mean_abs_z[mean_abs_z > threshold].index

    if work["sid"].nunique() - len(outlier_sids) < 2:
        to_kick = mean_abs_z.sort_values(ascending=False).index
        n_remove = work["sid"].nunique() - 2
        outlier_sids = to_kick[:n_remove]

    outlier_idx = work[work["sid"].isin(outlier_sids)].index.to_list()
    return outlier_idx


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

    while len(data) > 2:
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


def cheat(data):
    return data[data['disease'] != 'HC'].index.to_list()

def zscore_IQR(data):
    zs = flagged(data, method='Z_SCORE')
    iqr = find_outliers_IQR(data)
    return list(set(zs) | set(iqr))

def zscore_MAD(data):
    zs = flagged(data, method='Z_SCORE')
    mad = find_outliers_MAD(data)
    return list(set(zs) | set(mad))

def mad_vs(data):
    mad_idx = find_outliers_MAD(data)
    subset = data.drop(mad_idx)
    return find_outliers_VS(subset) 

def flagged(data, method, threshold: float = 1.0):
    """
    Retourne la liste des index dont la colonne `method` dépasse `threshold`.
    Par défaut, reproduit l'ancien comportement en cherchant les entrées égales à 1.
    """
    if method not in data.columns:
        raise KeyError(f"La colonne '{method}' est absente des données.")

    values = pd.to_numeric(data[method], errors="coerce")
    mask = values >= threshold
    mask = mask.fillna(False)
    return values[mask].index.to_list()

def rien(data):
    return []

def mlp25_all_mad(data, threshold=3.5):
    mlp = flagged(data, method='MLP2_ALL_5')
    mad = find_outliers_MAD(data, threshold=threshold)
    return list(set(mlp) | set(mad))

def mlp26_all_mad(data, threshold=3.5):
    mlp = flagged(data, method='MLP2_ALL_6')
    mad = find_outliers_MAD(data, threshold=threshold)
    return list(set(mlp) | set(mad))

# ------------------------------------------------------------
# Sn : médiane des médianes des distances, facteur 1.1926
# ------------------------------------------------------------
def find_outliers_SN(data: pd.DataFrame,
                     threshold: float = 3.0) -> list:
    """
    Détecte les outliers avec l'estimateur Sn de Rousseeuw et Croux.
    """
    column='mean_no_cov'
    if len(data) <= 2:
        return []

    x = data[column].to_numpy()
    med = np.median(x)

    # Matrice des distances absolues |xi - xj|
    diffs = np.abs(x[:, None] - x[None, :])
    # Médiane par ligne puis médiane globale
    Sn = 1.1926 * np.median(np.median(diffs, axis=1))

    if Sn == 0:
        print("Sn vaut zéro, impossible de normaliser.")
        return []

    robust_scores = np.abs(x - med) / Sn
    outlier_mask = robust_scores > threshold
    outlier_idx = data.index[outlier_mask].to_list()

    # On garde au moins deux points
    if len(data) - len(outlier_idx) < 2:
        scores = robust_scores[outlier_mask]
        sorted_idx = data.index[outlier_mask][np.argsort(scores)[::-1]]
        outlier_idx = sorted_idx[: len(data) - 2].to_list()

    return outlier_idx


# ------------------------------------------------------------
# Qn : 1er quartile des paires, facteur 2.2219
# ------------------------------------------------------------
def find_outliers_QN(data: pd.DataFrame,
                     threshold: float = 3.0) -> list:
    """
    Détecte les outliers avec l'estimateur Qn de Rousseeuw et Croux.
    """
    column='mean_no_cov'
    if len(data) <= 2:
        return []

    x = data[column].to_numpy()
    med = np.median(x)

    # Distances pour toutes les paires (upper-triangle)
    diffs = np.abs(x[:, None] - x[None, :])
    pairwise = diffs[np.triu_indices(len(x), k=1)]

    Qn = 2.2219 * np.percentile(pairwise, 25)

    if Qn == 0:
        print("Qn vaut zéro, impossible de normaliser.")
        return []

    robust_scores = np.abs(x - med) / Qn
    outlier_mask = robust_scores > threshold
    outlier_idx = data.index[outlier_mask].to_list()

    if len(data) - len(outlier_idx) < 2:
        scores = robust_scores[outlier_mask]
        sorted_idx = data.index[outlier_mask][np.argsort(scores)[::-1]]
        outlier_idx = sorted_idx[: len(data) - 2].to_list()

    return outlier_idx

def _auto_contamination(x: np.ndarray,
                        pval_weight: float = 0.5,
                        tail_weight: float = 0.5,
                        min_c: float = 0.03,
                        max_c: float = 0.3) -> float:
    """
    Retourne un taux d'outliers basé sur normalité (p-val Shapiro)
    et masse dans les queues robustes (z > 3).
    """
    n = len(x)
    # 1) p-valeur Shapiro (1 = très normal, 0 = très non-normal)
    try:
        p_val = shapiro(x).pvalue
    except Exception:
        p_val = 0.0          # si Shapiro plante (n > 5000) on force non-normal

    # 2) proportion de points au-delà de 3 σ robustes
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826  # MAD->σ
    if mad == 0:
        tail_frac = 0.0
    else:
        tail_frac = np.mean(np.abs(x - med) / mad > 3)

    # Combinaison linéaire
    score = (1 - p_val) * pval_weight + tail_frac * tail_weight
    contamination = min_c + score * (max_c - min_c)
    return float(np.clip(contamination, min_c, max_c))

def find_outliers_LOF(data: pd.DataFrame,
                           k: int = 20) -> list:
    """
    Détecte les outliers en 1-D avec LOF, en estimant automatiquement
    la proportion d'outliers selon la « gaussianité » des données.
    """
    column='mean_no_cov'
    if len(data) <= 2:
        return []

    x = data[column].to_numpy()
    contamination = _auto_contamination(x)


    lof = LocalOutlierFactor(n_neighbors=min(k, len(x) - 1),
                             contamination=contamination)
    preds = lof.fit_predict(x.reshape(-1, 1))   # -1 = outlier
    outlier_mask = preds == -1
    outlier_idx = data.index[outlier_mask].to_list()

    # On garde au moins deux points
    if len(data) - len(outlier_idx) < 2:
        scores = lof.negative_outlier_factor_[outlier_mask]
        sorted_idx = data.index[outlier_mask][np.argsort(scores)]
        outlier_idx = sorted_idx[: len(data) - 2].to_list()

    return outlier_idx

def _make_threshold_wrapper(func, default=None, param="threshold", **extra_kwargs):
    """
    Retourne un wrapper qui transmet une valeur de `threshold` optionnelle au paramètre `param`.
    Sans valeur fournie (ou si le `threshold` par défaut est None), la fonction est appelée
    sans paramètre additionnel.
    """
    accepts_param = param in inspect.signature(func).parameters

    def wrapper(data, threshold=None):
        value = default if threshold is None else threshold
        kwargs = dict(extra_kwargs)
        if value is not None and accepts_param:
            kwargs[param] = value
        return func(data, **kwargs)
    return wrapper


def _make_flagged_wrapper(method: str, default_threshold: float):
    def wrapper(data, threshold=None):
        value = default_threshold if threshold is None else threshold
        return flagged(data, method=method, threshold=value)
    return wrapper


ROBUST_METHODS = {
    "IQR": _make_threshold_wrapper(find_outliers_IQR, default=1.5),
    "MAD": _make_threshold_wrapper(find_outliers_MAD, default=3.5),
    "MAD_VS": _make_threshold_wrapper(mad_vs, default=None),
    "SN": _make_threshold_wrapper(find_outliers_SN, default=3.0),
    "QN": _make_threshold_wrapper(find_outliers_QN, default=3.0),
    "LOF": _make_threshold_wrapper(find_outliers_LOF, default=None),
    "MMS": _make_threshold_wrapper(reject_outliers_until_mad_equals_mean, default=0.001),
    "VS": _make_threshold_wrapper(find_outliers_VS, default=None),
    "VS2": _make_threshold_wrapper(find_outliers_VS2, default=None),
    "Z_SCORE": _make_flagged_wrapper("Z_SCORE", default_threshold=1.0),
    "Z_SCORE_IQR": _make_threshold_wrapper(zscore_IQR, default=None),
    "Z_SCORE_MAD": _make_threshold_wrapper(zscore_MAD, default=None),
    "Z_SCORE_BUNDLE": _make_threshold_wrapper(find_outliers_ZSCORE_BUNDLE, default=3.0),
    "Z_SCORE_METRIC": _make_threshold_wrapper(find_outliers_ZSCORE_METRIC, default=3.0),
    "CHEAT": _make_threshold_wrapper(cheat, default=None),
    "FLIP": _make_threshold_wrapper(rien, default=None),
    "MLP_ALL": _make_flagged_wrapper("MLP_ALL", default_threshold=0.5),
    "MLP2_ALL": _make_flagged_wrapper("MLP2_ALL", default_threshold=0.5),
    "MLP3_ALL": _make_flagged_wrapper("MLP3_ALL", default_threshold=0.5),
    "MLP4_ALL_5": _make_flagged_wrapper("MLP4_ALL_5", default_threshold=0.5),
}


from clinical_combat.harmonization.QuickCombat import QuickCombat

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

def remove_covariates_effects2(df):
    df = df.sort_values(by=["site", "sid", "bundle"])
    ignore_handedness = True
    ignore_sex = False
    if df['sex'].nunique() == 1:
        ignore_sex = True
    if df['handedness'].nunique() == 1:
        ignore_handedness = True
    design, y = get_design_matrices(df, ignore_handedness, ignore_sex)
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

    
    site = df_before['site'].unique()[0]
    disease = site.split('_')[0]
    metric = df_before['metric'].unique()[0]
    error = df_before['error'].unique()[0]
    nasty_bundle = df_before['nasty_bundle'].unique()[0]
    print ("bundle is ", nasty_bundle)
    
    test_index = site[:-3].split("_")[-1]

    bundle_column = 'bundle'
    bundle = nasty_bundle

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
    mean_sick_before = df_b_before[df_b_before['disease'] != 'HC'][y_column].mean()

    axes[0].axvline(mean_hc_before, color='blue', linestyle=':', linewidth=1.5, label='Moyenne HC')
    axes[0].axvline(mean_all_before, color='black', linestyle='-.', linewidth=1.5, label='Moyenne Tous')
    axes[0].axvline(mean_sick_before, color='orange', linestyle='--', linewidth=1.5, label='Moyenne Malades')
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
    sns.kdeplot(df_b_before[df_b_before['disease'] == 'HC'][y_column], label='HC', ax=axes[1], linewidth=2)
    sns.kdeplot(df_b_after[df_b_after['disease'] != 'HC'][y_column], label='Malades', ax=axes[1], linewidth=2)
    sns.kdeplot(df_b_after[y_column], label='Tous', linestyle='--', ax=axes[1], linewidth=2)

    mean_hc_after = df_b_before[df_b_before['disease'] == 'HC'][y_column].mean()
    std_hc_after = df_b_before[df_b_before['disease'] == 'HC'][y_column].std()
    mean_all_after = df_b_after[y_column].mean()
    std_all_after = df_b_after[y_column].std()
    mean_sick_after = df_b_after[df_b_after['disease'] != 'HC'][y_column].mean()

    axes[1].axvline(mean_hc_after, color='blue', linestyle=':', linewidth=1.5, label='Moyenne HC')
    axes[1].axvline(mean_all_after, color='black', linestyle='-.', linewidth=1.5, label='Moyenne Tous')
    axes[1].axvline(mean_sick_after, color='orange', linestyle='--', linewidth=1.5, label='Moyenne Malades')
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
    plt.savefig(os.path.join(output_dir, f"{test_index}_{robust_method}_{float(error):.2f}_{site}_{bundle}.png"))
    plt.close()
