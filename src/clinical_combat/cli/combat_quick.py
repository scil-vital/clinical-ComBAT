#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute and apply the transfer function from
a moving site to a reference site.

This script calls combat_quick_fit, combat_quick_apply,
combat_visualize_model and combat_visualize_harmonization.
The exact commands are printed in the terminal.

Harmonization methods:
    pairwise:
        uses both moving and reference data to fit the
        covariates regression parameters (Beta_mov).
        Fortin et al., 2017 method,
        see https://pubmed.ncbi.nlm.nih.gov/28826946/
    clinic (default):
        uses a priori from the reference site to fit the moving site
        (Beta_mov, variance)

NOTE: the harmonization parameters (regul, degree, nu, tau) are preset
      according to the harmonization method chosen. See default settings.
      If the reference site is MRC-CBSU_Siemens_3T_2,
      it's renamed CamCAN in the figures.

Examples:
# Harmonized with the Clinic method with un polynomial degree of 2
combat_quick reference_site.raw.csv.gz moving_site.raw.csv.gz --degree 2

# Harmonized with the Pairwise method (i.e. Fortin et al., (2017) method)
combat_quick reference_site.raw.csv.gz moving_site.raw.csv.gz --method pairwise

"""
import argparse
import logging
import os
import subprocess

import pandas as pd

from clinical_combat.utils.scilpy_utils import (add_overwrite_arg,
                                                add_verbose_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("ref_data",
                   help="Path to the reference site data.")
    p.add_argument("mov_data",
                   help="Path to the moving site data.")
    p.add_argument("--out_dir",
                   default="./",
                   help="Output directory. [%(default)s]")
    p.add_argument("--output_model_filename",
                   default="",
                   help="Output CSV model filename. "
                        "['ref_site-moving-site.metric_name.model.csv']")

    p.add_argument("--output_results_filename",
                   default="",
                   help="Output CSV of the harmonized data filename. "
                        "['ref_site-moving-site.metric_name.model.raw/harmonized.csv']")

    p.add_argument("-m", "--method",
                   default="clinic",
                   choices=["pairwise", "clinic"],
                   help="Harmonization method.")
    p.add_argument("--ignore_sex",
                   action="store_true",
                   help="If set, ignore the sex covariate in the data.")
    p.add_argument("--ignore_handedness",
                   action="store_true",
                   help="If set, ignore the handedness covariate in the data.")
    p.add_argument("--limit_age_range",
                   action="store_true",
                   help="If set, exclude reference site subjects with "
                        "age outside the range of "
                        "the moving site subject ages.")
    p.add_argument("--no_empirical_bayes",
                   action="store_true",
                   help="If set, skip empirical Bayes estimator for "
                        "alpha and sigma estimation.")
    p.add_argument("--robust",
                   action="store_true",
                   help="If set, use combat robust. This tries "
                        "identifying/rejecting non-HC subjects.")
    p.add_argument("--regul_ref",
                   type=float,
                   default=0,
                   help="Regularization parameter for "
                        "the reference site data. [%(default)s]")
    p.add_argument("--regul_mov",
                   type=float,
                   help="Regularization parameter for "
                        "the moving site data. Set to '-1' for "
                        "automatic tuning "
                        "[default=0 for pairwise; -1 for clinic]")
    p.add_argument("--degree",
                   type=int,
                   help="Degree of the polynomial fit in Combat. "
                        "[default=1 for pairwise; 2 for clinic].")
    p.add_argument("--nu",
                   type=float,
                   default=5,
                   help="Combat Clinic hyperparameter for "
                        "the standard deviation estimation of the moving "
                        "site data. It must be >=0. [%(default)s]")
    p.add_argument("--tau",
                   type=float,
                   default=2,
                   help="Combat Clinic hyperparameter for "
                        "the covariate fit of the moving site data. "
                        "It must be >= 1. [%(default)s]")
    p.add_argument("--bundles",
                   nargs="+",
                   help="List of bundle to use for figures. "
                        "To plot all bundles use --bundles all. "
                        "By default, it takes the second bundle.")
    p.add_argument("--degree_qc",
                   type=int,
                   default=0,
                   help="Degree for QC fit. "
                        "By default it uses the same as the model.")
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
        args.bundles = all_bundles[1:2]
    elif args.bundles == ["all"]:
        args.bundles = all_bundles
    for b in args.bundles:
        if b not in all_bundles:
            args.bundles.remove(b)
            logging.warning("Bundle %s not founded in the data.", b)
    if len(args.bundles) == 0:
        args.bundles = all_bundles[0:1]
        logging.warning("No valid input bundle. "
                        "Selecting bundle %s", args.bundles)

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
            + ".harmonized.csv.gz"
        )
    # Fit
    print(
        "\n ComBAT Harmonization -> \n\n     Reference site : ",
        os.path.basename(args.ref_data),
        "\n\n     Moving site : ",
        os.path.basename(args.mov_data),
    )
    print("\n     Fit model : ", args.output_model_filename)
    cmd = (
        "combat_quick_fit"
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

    if args.regul_mov is not None:
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
        cmd += " --robust"
    if args.overwrite:
        cmd += " -f"
    logging.info(cmd)
    subprocess.call(cmd, shell=True)

    # Apply
    print("\n     Harmonized site : ", args.output_results_filename)
    cmd = (
        "combat_quick_apply"
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

    # Figures
    print("\n\n Visualize Harmonization :")
    print("\n     Model (DataModels_*) :", args.output_model_filename)

    bundles = ""
    for curr_bundle in args.bundles:
        bundles += str(curr_bundle) + " "
    cmd = (
        "combat_visualize_model"
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

    print("\n     Harmonized data (AgeCurve_*) : ",
          args.output_results_filename)
    cmd = (
        "combat_visualize_harmonization"
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

    # QC
    print("\n\n Quality control :")
    print("\n   Raw data ")

    cmd = (
        "combat_quick_QC"
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
    subprocess.call(cmd, shell=True)

    print("\n   Harmonized data ")
    cmd = (
        "combat_quick_QC"
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
    subprocess.call(cmd, shell=True)
    print("\n ")


if __name__ == "__main__":
    main()
