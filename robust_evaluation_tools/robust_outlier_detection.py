import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

def find_outliers(df, robust_method, args = []):
    # Find outliers
    outliers_idx = []

    if robust_method in ['IQR', 'MAD', 'MAD_MEAN']:
        for metric in df['metric'].unique():
            for bundle in df['bundle'].unique():
                data = df[(df['metric'] == metric) & (df['bundle'] == bundle)]
                outliers_idx += use_robust_method(data, robust_method)
    elif robust_method == 'kmeans':
        outliers_idx = use_robust_method(data, robust_method)
    return outliers_idx

def use_robust_method(data, robust_method, args = []):
    if robust_method == 'IQR':
        return find_outliers_iqr(data)
    elif robust_method == 'MAD':
        return find_outliers_mad(data, args)
    elif robust_method == 'MAD_MEAN':
        return reject_outliers_until_mad_equals_mean(data, args)
    else:
        raise ValueError("Invalid robust method. Choose between 'iqr' and 'mad'.")
    
def find_outliers_iqr(data, threshold=3.5):
    Q1 = data['mean_no_cov'].quantile(0.25)
    Q3 = data['mean_no_cov'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrer les valeurs aberrantes
    outliers = data[(data['mean_no_cov'] < lower_bound) | (data['mean_no_cov'] > upper_bound)]

    return outliers.index.to_list()

def find_outliers_mad(data, args):
    threshold = args[0] if len(args) == 1 else 3.5
        
    column='mean_no_cov'
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

def reject_outliers_until_mad_equals_mean(data):
    column = 'mean_no_cov'
    while True:
        median = data[column].median()
        mad = np.median(np.abs(data[column] - median))
        mean = data[column].mean()

        if mad >= mean:
            break

        # Find the index of the maximum value
        max_idx = data[column].idxmax()
        # Drop the row with the maximum value
        data = data.drop(max_idx)

    return data.index.to_list()


def get_metrics(outliers_idx, mov_data):
    
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

