#!/usr/bin/env python3
"""
Script to compute and apply the transfer function from a moving site to a reference site.

This script calls combat_quick_fit.py, combat_quick_apply.py, combat_visualize_model.py and
combat_visualize_harmonization.py. The exact commands are printed in the terminal.

Harmonization method:
    vanilla:
            uses both moving and reference data to fit the covariates regression parameters
            (Beta_mov). Fortin et al., 2017 method, see https://pubmed.ncbi.nlm.nih.gov/28826946/

NOTE: the harmonization parameters are preset. See default settings.
      If the reference site is MRC-CBSU_Siemens_3T_2, it's renamed CamCAN in the figures.

Example:

# Harmonized with the Vanilla method (i.e. Fortin et al., (2017) method)
combat_quick.py reference_site.raw.csv.gz moving_site.raw.csv.gz

"""
import argparse
import logging
import os
import subprocess

import pandas as pd

from vanilla_combat.utils.scilpy_utils import add_overwrite_arg, add_verbose_arg


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
        "--bundles",
        nargs="+",
        help="List of bundle to use for figures. To plot all bundles use "
        "--bundles all. ['mni_IIT_mask_skeletonFA'].",
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
        args.bundles = ["mni_IIT_mask_skeletonFA"]
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
            + ".model.csv"
        )
    # output data filename
    if len(args.output_results_filename) == 0:
        args.output_results_filename = (
            mov_data.site.unique()[0]
            + "."
            + ref_data.metric.unique()[0]
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
        + " -v "
        + str(args.verbose)
    )
    if args.ignore_sex:
        cmd += " --ignore_sex"
    if args.ignore_handedness:
        cmd += " --ignore_handedness"
    if args.limit_age_range:
        cmd += " --limit_age_range"
    if args.no_empirical_bayes:
        cmd += " --no_empirical_bayes"
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
        + " --out_dir "
        + args.out_dir
    )
    if args.overwrite:
        cmd += " -f"

    logging.info(cmd)
    subprocess.call(cmd, shell=True)

    print("\n   Harmonized data ")
    cmd = (
        "combat_quick_QC.py"
        + " "
        + args.ref_data
        + " "
        + os.path.join(args.out_dir, args.output_results_filename)
        + " "
        + os.path.join(args.out_dir, args.output_model_filename)
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
