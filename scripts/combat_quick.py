# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script to compute and apply the transfer function from a moving site to a reference site.

This script calls combat_quick_fit.py, combat_quick_apply.py, combat_visualize_model.py and
combat_visualize_harmonization.py. The exact commands are printed in the terminal.

Harmonization methods:
    classic:
            uses both moving and reference data to fit the covariates regression parameters
            (Beta_mov). Fortin et al., 2017 method, see https://pubmed.ncbi.nlm.nih.gov/28826946/
    clinic (default):
            uses a priori from the reference site to fit the moving site
            (Beta_mov, variance)
    covbat:
            first applies classic ComBat and then aligns covariance across sites following
            the CovBat procedure (Chen et al., 2021)
    gam:
            replaces the linear age model by a spline-based fit (ComBat-GAM)
    gmm:
            models standardized residuals using Gaussian mixtures to better match higher-order moments

NOTE: the harmonization parameters (regul, degree, nu, tau) are preset according to the
      harmonization method chosen. See default settings.
      If the reference site is MRC-CBSU_Siemens_3T_2, it's renamed CamCAN in the figures.

Examples:
# Harmonized with the Clinic method with un polynomial degree of 2
combat_quick.py reference_site.raw.csv.gz moving_site.raw.csv.gz --degree 2

# Harmonized with the Classic method (i.e. Fortin et al., (2017) method)
combat_quick.py reference_site.raw.csv.gz moving_site.raw.csv.gz --method classic

"""
import argparse
import logging
import os
import subprocess

import pandas as pd

from clinical_combat.utils.scilpy_utils import add_overwrite_arg, add_verbose_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "ref_data",
        help="Path to the reference site data.",
    )
    p.add_argument(
        "mov_data",
        help="Path to the moving site data.",
    )
    p.add_argument(
        "--out_dir",
        help="Output directory. [%(default)s]",
        default="./",
    )
    p.add_argument(
        "--output_model_filename",
        help="Output CSV model filename. "
        + "['ref_site-moving-site.model.metric_name.csv']",
        default="",
    )

    p.add_argument(
        "--output_results_filename",
        help="Output CSV of the harmonized data filename. "
        + "['ref_site-moving-site.metric_name.model.res.csv']",
        default="",
    )

    p.add_argument(
        "-m",
        "--method",
        default="clinic",
        choices=["classic", "clinic", "covbat", "gam", "gmm"],
        help="Harmonization method.",
    )
    p.add_argument(
        "--ignore_sex",
        action="store_true",
        help="If set, ignore the sex covariate in the data.",
    )
    p.add_argument(
        "--ignore_handedness",
        action="store_true",
        help="If set, ignore the handedness covariate in the data.",
    )
    p.add_argument(
        "--limit_age_range",
        action="store_true",
        help="If set, exclude reference site subjects with age outside the range of the moving "
        + "site subject ages.",
    )
    p.add_argument(
        "--no_empirical_bayes",
        action="store_true",
        help="If set, skip empirical Bayes estimator for alpha and sigma estimation.",
    )
    p.add_argument(
        "--robust",
        default="MLP4_ALL_5",
        help="If set, use combat robust. This tries "
        + "identifying/rejecting non-HC subjects.",
    )
    p.add_argument(
        "--regul_ref",
        type=float,
        default=0,
        help="Regularization parameter for the reference site data. [%(default)s]",
    )
    p.add_argument(
        "--regul_mov",
        type=float,
        help="Regularization parameter for the moving site data. Set to '-1' for automatic tuning "
        + "[default=0 for classic; -1 for clinic]",
    )
    p.add_argument(
        "--degree",
        type=int,
        help="Degree of the polynomial fit in Combat. "
        + "[default=1 for classic; 2 for clinic].",
    )
    p.add_argument(
        "--nu",
        type=float,
        default=5,
        help="Combat Clinic hyperparameter for the standard deviation estimation of the moving "
        + "site data. It must be >=0. [%(default)s]",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=2,
        help="Combat Clinic hyperparameter for the covariate fit of the moving site data. "
        "It must be >= 1. [%(default)s]",
    )
    p.add_argument(
        "--covbat_pve",
        type=float,
        default=0.95,
        help="Minimum proportion of variance explained to retain for CovBat adjustments. [%(default)s]",
    )
    p.add_argument(
        "--covbat_max_components",
        type=int,
        help="Maximum number of principal components to harmonize with CovBat.",
    )
    p.add_argument(
        "--gam_n_knots",
        type=int,
        default=7,
        help="Number of knots used for the natural spline age term in ComBat-GAM. [%(default)s]",
    )
    p.add_argument(
        "--gmm_components",
        type=int,
        default=2,
        help="Number of Gaussian components to use for ComBat-GMM. [%(default)s]",
    )
    p.add_argument(
        "--gmm_tol",
        type=float,
        default=1e-4,
        help="Tolerance on log-likelihood improvement for the ComBat-GMM EM algorithm. [%(default)s]",
    )
    p.add_argument(
        "--gmm_max_iter",
        type=int,
        default=50,
        help="Maximum number of EM iterations for ComBat-GMM. [%(default)s]",
    )
    p.add_argument(
        "--bundles",
        nargs="+",
        help="List of bundle to use for figures. To plot all bundles use "
        "--bundles all. ['mni_IIT_mask_skeletonFA'].",
    )
    p.add_argument(
        "--degree_qc",
        type=int,
        help="Degree for QC fit. By default it uses the same as the model.",
        default=0,
    )
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    ref_data = pd.read_csv(args.ref_data)
    mov_data = pd.read_csv(args.mov_data)

    # Check if moving site is a string
    if mov_data.site.dtype != "str":
        mov_data.site = mov_data.site.astype(str)
    if ref_data.site.dtype != "str":
        ref_data.site = ref_data.site.astype(str)

    all_bundles = list(ref_data.bundle.unique())
    if args.bundles is None:
        args.bundles = ["mni_AC"]
    elif args.bundles == ["all"]:
        args.bundles = all_bundles
    for b in args.bundles:
        if b not in all_bundles:
            args.bundles.remove(b)
            logging.warning("Bundle %s not founded in the data.", b)
    if len(args.bundles) == 0:
        args.bundles = all_bundles[0:1]
        logging.warning("No valid input bundle. Selecting bundle %s", args.bundles)

    # model file name
    if len(args.output_model_filename) == 0:
        args.output_model_filename = (
            mov_data.site.unique()[0]
            + "-"
            + ref_data.site.unique()[0]
            + "."
            + ref_data.metric.unique()[0]
            + "."
            + args.method
            + ".model.csv"
        )
    # output data filename
    if len(args.output_results_filename) == 0:
        args.output_results_filename = (
            mov_data.site.unique()[0]
            + "."
            + ref_data.metric.unique()[0]
            + "."
            + args.method
            + ".csv.gz"
        )
    ###########
    ### fit ###
    ###########
    print(
        "\n ComBAT Harmonization -> \n     Reference site : ",
        os.path.basename(args.ref_data),
        "\n\n     Moving site : ",
        os.path.basename(args.mov_data),
    )
    print("\n     Fit model : ", args.output_model_filename)
    cmd = (
        "combat_quick_fit.py"
        + " "
        + args.ref_data
        + " "
        + args.mov_data
        + " --out_dir "
        + args.out_dir
        + " --output_model_filename "
        + args.output_model_filename
        + " --method "
        + args.method
        + " --regul_ref "
        + str(args.regul_ref)
        + " --nu "
        + str(args.nu)
        + " --tau "
        + str(args.tau)
        + " -v "
        + str(args.verbose)
    )
    if args.regul_mov:
        cmd += " --regul_mov " + str(args.regul_mov)
    if args.degree:
        cmd += " --degree " + str(args.degree)
    if args.ignore_sex:
        cmd += " --ignore_sex"
    if args.ignore_handedness:
        cmd += " --ignore_handedness"
    if args.limit_age_range:
        cmd += " --limit_age_range"
    if args.no_empirical_bayes:
        cmd += " --no_empirical_bayes"
    if args.robust:
        cmd += " --robust " + str(args.robust)
    if args.covbat_pve is not None:
        cmd += " --covbat_pve " + str(args.covbat_pve)
    if args.covbat_max_components is not None:
        cmd += " --covbat_max_components " + str(args.covbat_max_components)
    if args.gam_n_knots is not None:
        cmd += " --gam_n_knots " + str(args.gam_n_knots)
    if args.gmm_components is not None:
        cmd += " --gmm_components " + str(args.gmm_components)
    if args.gmm_tol is not None:
        cmd += " --gmm_tol " + str(args.gmm_tol)
    if args.gmm_max_iter is not None:
        cmd += " --gmm_max_iter " + str(args.gmm_max_iter)
    if args.overwrite:
        cmd += " -f"
    logging.info(cmd)
    subprocess.call(cmd, shell=True)

    #############
    ### apply ###
    #############
    print("\n     Harmonized site : ", args.output_results_filename)
    cmd = (
        "combat_quick_apply.py"
        + " "
        + args.mov_data
        + " "
        + os.path.join(args.out_dir, args.output_model_filename)
        + " --out_dir "
        + args.out_dir
        + " --output_results_filename "
        + args.output_results_filename
        + " -v "
        + str(args.verbose)
    )
    if args.overwrite:
        cmd += " -f"
    logging.info(cmd)
    subprocess.call(cmd, shell=True)

    ###############
    ### figures ###
    ###############
    print("\n\n\n Visualize Harmonization :")
    print("\n     Model (DataModels_*) :", args.output_model_filename)

    bundles = ""
    for curr_bundle in args.bundles:
        bundles += str(curr_bundle) + " "
    cmd = (
        "combat_visualize_model.py"
        + " "
        + args.ref_data
        + " "
        + args.mov_data
        + " "
        + os.path.join(args.out_dir, args.output_model_filename)
        + " --out_dir "
        + args.out_dir
        + " --bundles "
        + bundles
        + " -v "
        + str(args.verbose)
    )
    if args.overwrite:
        cmd += " -f"
    logging.info(cmd)
    subprocess.call(cmd, shell=True)

    print("\n     Harmonized data (AgeCurve_*) : ", args.output_results_filename)
    cmd = (
        "combat_visualize_harmonization.py"
        + " "
        + args.ref_data
        + " "
        + args.mov_data
        + " "
        + os.path.join(args.out_dir, args.output_results_filename)
        + " --out_dir "
        + args.out_dir
        + " --bundles "
        + bundles
        + " -v "
        + str(args.verbose)
    )
    if args.overwrite:
        cmd += " -f"
    logging.info(cmd)
    subprocess.call(cmd, shell=True)

    ##########
    ### QC ###
    ##########
    print("\n\n\n Quality control :")
    print("\n   Raw data ")

    cmd = (
        "combat_quick_QC.py"
        + " "
        + args.ref_data
        + " "
        + args.mov_data
        + " "
        + os.path.join(args.out_dir, args.output_model_filename)
        + " -v "
        + str(args.verbose)
        + " --degree_qc " 
        + str(args.degree_qc)
        + " --out_dir "
        + args.out_dir
    )
    if args.overwrite:
        cmd += " -f"

    logging.info(cmd)
    # subprocess.call(cmd, shell=True)

    print("\n   Harmonized data ")
    cmd = (
        "combat_quick_QC.py"
        + " "
        + args.ref_data
        + " "
        + os.path.join(args.out_dir, args.output_results_filename)
        + " "
        + os.path.join(args.out_dir, args.output_model_filename)
        + " -v "
        + str(args.verbose)
        + " --degree_qc " 
        + str(args.degree_qc)
        + " --out_dir "
        + args.out_dir
    )
    if args.overwrite:
        cmd += " -f"

    logging.info(cmd)
    # subprocess.call(cmd, shell=True)
    print("\n ")


if __name__ == "__main__":
    main()
