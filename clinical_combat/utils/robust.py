from clinical_combat.harmonization import from_model_name
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
REMOVE_WHOLE_PATIENT = True

def remove_outliers(ref_data, mov_data, args):
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
    QC.fit(ref_data, mov_data,False)
    
    design_mov, y_mov = QC.get_design_matrices(mov_data)
    y_no_cov = QC.remove_covariate_effect(design_mov, y_mov)
    y_no_cov_flat = np.array(y_no_cov).flatten()
    
    mov_data.insert(3,"mean_no_cov", y_no_cov_flat,True)
    outliers_idx = []
    for i, bundle in enumerate(QC.bundle_names):
        data = mov_data.query("bundle == @bundle")

        outliers_idx += find_outliers_IQR(data)

    QC.metrics = get_metrics(outliers_idx, mov_data)

    if REMOVE_WHOLE_PATIENT:
        outlier_patients_ids = mov_data.loc[outliers_idx]['sid'].unique().tolist()
        mov_data = mov_data[~mov_data['sid'].isin(outlier_patients_ids)]
    else :
        mov_data = mov_data.drop(outliers_idx)

    mov_data = mov_data.drop(columns=['mean_no_cov'])
    return mov_data

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

def get_metrics(outliers_idx, mov_data):


    mov_data['is_malade'] = mov_data['disease'].apply(lambda x: 0 if x == 'HC' else 1)
    outliers = mov_data.loc[outliers_idx]
    outliers_sid = outliers['sid'].unique().tolist() 


    mov_data['is_outlier'] = mov_data['sid'].apply(lambda x: 1 if x in outliers_sid else 0)

    patients = mov_data.drop_duplicates(subset='sid')

    y_true = patients['is_malade'].tolist()
    
    y_pred = patients['is_outlier'].tolist()

    # Calcul de la matrice de confusion pour obtenir les faux positifs et faux négatifs
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calcul de la précision et du rappel (recall)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    taux_faux_positifs = fp / (fp + tn) if (fp + tn) != 0 else 0
    f1 = f1_score(y_true, y_pred)

    # Affichage des résultats
    print(f"Précision : {tp} / {tp+fp} = {precision:.3f}")
    print(f"Rappel (Recall) :{tp} / {tp+fn} = {recall:.3f}")
    print(f"Taux de faux positifs : {fp} / {fp+tn} = {taux_faux_positifs:.3f}")
    print(f"F1 score : {f1:.3f}")
    
    metrics = {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'taux_faux_positifs': taux_faux_positifs,
        'f1_score': f1,
        'outliers': outliers
        
    }

    return metrics
    



def find_outliers_IQR(data):

    Q1 = data['mean_no_cov'].quantile(0.25)
    Q3 = data['mean_no_cov'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrer les valeurs aberrantes
    outliers = data[(data['mean_no_cov'] < lower_bound) | (data['mean_no_cov'] > upper_bound)]

    return outliers.index.to_list()
