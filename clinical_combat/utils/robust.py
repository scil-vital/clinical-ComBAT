from clinical_combat.harmonization import from_model_name
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


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

    plt.scatter(design_mov[0][3], y_no_cov[0])
    plt.savefig("PLOTS/no_cov.png")
    plt.clf()
    plt.scatter(design_mov[0][3], y_mov[0])
    plt.savefig("PLOTS/yes_cov.png")
    plt.clf()
    plot = sns.distplot(y_no_cov[0])
    fig = plot.get_figure()
    fig.savefig("PLOTS/distribution.png") 
    plt.clf()

    mov_data_HC = mov_data.query("disease == 'HC'")
    design_mov_HC, y_mov_HC = QC.get_design_matrices(mov_data_HC)
    y_no_cov_HC = QC.remove_covariate_effect(design_mov_HC, y_mov_HC)

    plt.scatter(design_mov_HC[0][3], y_no_cov_HC[0])
    plt.savefig("PLOTS/no_cov_HC.png")
    plt.clf()
    plt.scatter(design_mov_HC[0][3], y_mov_HC[0])
    plt.savefig("PLOTS/yes_cov_HC.png")
    plt.clf()
    plot = sns.distplot(y_no_cov_HC[0])
    fig = plot.get_figure()
    fig.savefig("PLOTS/distribution_HC.png") 
    plt.clf()

    return mov_data

