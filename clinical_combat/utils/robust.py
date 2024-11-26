from clinical_combat.harmonization import from_model_name
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import json
import os


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

    # Process movement data
    design_mov, y_mov = QC.get_design_matrices(mov_data)
    y_no_cov = QC.remove_covariate_effect(design_mov, y_mov)
    y_no_cov_flat = np.array(y_no_cov).flatten()
    mov_data.insert(3, "mean_no_cov", y_no_cov_flat, True)

    # Find outliers
    outliers_idx = []
    for bundle in QC.bundle_names:
        data = mov_data.query("bundle == @bundle")
        outliers_idx += find_outlier(data)

    # Calculate and save metrics
    site = mov_data['site'].unique()[0]
    metrics = get_metrics(outliers_idx, mov_data)
    metrics['site'] = site

    mov_data = mov_data.drop(columns=['mean_no_cov'])


    rwp_str = "RWP" if rwp else "NoRWP"
    metrics_filename = os.path.join(args.out_dir,f"metrics_{site}_{args.robust}_{rwp_str}.csv")
    metrics.to_csv(metrics_filename, index=True)

    # Save outliers
    outliers_filename = os.path.join(args.out_dir,f"outliers_{site}_{args.robust}_{rwp_str}.csv")
    mov_data.loc[outliers_idx].to_csv(outliers_filename, index=True)

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


def get_metrics(outliers_idx, mov_data):
    
    metrics_list = []
    mov_data['is_malade'] = mov_data['disease'].apply(lambda x: 0 if x == 'HC' else 1)
    mov_data['is_outlier'] = 0
    mov_data.loc[outliers_idx, 'is_outlier'] = 1
    
    for bundle in mov_data['bundle'].unique():
        bundle_data = mov_data[mov_data['bundle'] == bundle]
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
    return metrics_df.T

    # mov_data['is_malade'] = mov_data['disease'].apply(lambda x: 0 if x == 'HC' else 1)
    # outliers = mov_data.loc[outliers_idx]
    # outliers_sid = outliers['sid'].unique().tolist() 


    # mov_data['is_outlier'] = mov_data['sid'].apply(lambda x: 1 if x in outliers_sid else 0)

    # patients = mov_data.drop_duplicates(subset='sid')

    # y_true = patients['is_malade'].tolist()
    
    # y_pred = patients['is_outlier'].tolist()

    # # Calcul de la matrice de confusion pour obtenir les faux positifs et faux négatifs
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # # Calcul de la précision et du rappel (recall)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # taux_faux_positifs = fp / (fp + tn) if (fp + tn) != 0 else 0
    # f1 = f1_score(y_true, y_pred)

    # # Affichage des résultats
    # print(f"Précision : {tp} / {tp+fp} = {precision:.3f}")
    # print(f"Rappel (Recall) :{tp} / {tp+fn} = {recall:.3f}")
    # print(f"Taux de faux positifs : {fp} / {fp+tn} = {taux_faux_positifs:.3f}")
    # print(f"F1 score : {f1:.3f}")
    
    # metrics = [{
    #     'true_positives': tp,
    #     'false_positives': fp,
    #     'true_negatives': tn,
    #     'false_negatives': fn,
    #     'precision': precision,
    #     'recall': recall,
    #     'taux_faux_positifs': taux_faux_positifs,
    #     'f1_score': f1
        
    # }]

    # return pd.DataFrame(metrics)


def find_outliers_IQR(data):

    Q1 = data['mean_no_cov'].quantile(0.25)
    Q3 = data['mean_no_cov'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrer les valeurs aberrantes
    outliers = data[(data['mean_no_cov'] < lower_bound) | (data['mean_no_cov'] > upper_bound)]

    return outliers.index.to_list()

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

ROBUST_METHODS = {
    "IQR": find_outliers_IQR,
    "MAD": find_outliers_MAD
}

def create_plots(mov_data, QC):
    foldername = f"PLOTS/{mov_data['site'].unique()[0]}/"
    mov_data_HC = mov_data.query("disease == 'HC'")
    mov_data_SICK = mov_data.query("disease != 'HC'")
    
    design_mov, y_mov = QC.get_design_matrices(mov_data)
    y_no_cov = QC.remove_covariate_effect(design_mov, y_mov)

    design_mov_HC, y_mov_HC = QC.get_design_matrices(mov_data_HC)
    y_no_cov_HC = QC.remove_covariate_effect(design_mov_HC, y_mov_HC)

    design_mov_SICK, y_mov_SICK = QC.get_design_matrices(mov_data_SICK)
    y_no_cov_SICK = QC.remove_covariate_effect(design_mov_SICK, y_mov_SICK)

    plt.scatter(design_mov_HC[0][3], y_no_cov_HC[0],color='blue')
    plt.scatter(design_mov_SICK[0][3], y_no_cov_SICK[0],color='red')
    plt.savefig(foldername+"no_cov.png")
    plt.clf()
    plt.scatter(design_mov_HC[0][3], y_mov_HC[0],color='blue')
    plt.scatter(design_mov_SICK[0][3], y_mov_SICK[0],color='red')
    plt.savefig(foldername+"yes_cov.png")
    plt.clf()
    plot = sns.distplot(y_no_cov[0])
    fig = plot.get_figure()
    fig.savefig(foldername + "distribution.png") 
    plt.clf()


    plt.scatter(design_mov_HC[0][3], y_no_cov_HC[0])
    plt.savefig(foldername + "no_cov_HC.png")
    plt.clf()
    plt.scatter(design_mov_HC[0][3], y_mov_HC[0])
    plt.savefig(foldername +"yes_cov_HC.png")
    plt.clf()
    plot = sns.distplot(y_no_cov_HC[0])
    fig = plot.get_figure()
    fig.savefig(foldername + "distribution_HC.png") 
    plt.clf()


def plot_outliers_data(data):
    colors = {
        (0, 0): 'blue',
        (0, 1): 'red',
        (1, 0): 'yellow',
        (1, 1): 'green'
    }

    data['color'] = data.apply(lambda row: colors[(row['is_malade'], row['is_outlier'])], axis=1)

    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data['mean_no_cov'], c=data['color'])
    plt.xlabel('Index')
    plt.ylabel('Mean No Covariate')
    plt.title('Scatter Plot of Data with Outliers')
    plt.show()
