#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute the transfer function from a moving site to a reference site.

Harmonization methods:
    classic: uses both moving and reference data to fit the covariate
             regression parameters (Beta_mov).
    clinic: uses a priori from the reference site to fit the moving site
            (Beta_mov, variance)

Examples:
# Use the classic method to harmonize the moving site data to the reference site data 
# (linear)
combat_quick_fit.py reference_site.raw.csv.gz moving_site.raw.csv.gz --method classic

# Use the clinic method to harmonize the moving site data to the reference site data (non-linear)
combat_quick_fit.py reference_site.raw.csv.gz moving_site.raw.csv.gz --method clinic
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from clinical_combat.harmonization import from_model_name
from clinical_combat.utils.robust import remove_outliers
from clinical_combat.utils.scilpy_utils import (
    add_overwrite_arg,
    add_verbose_arg,
    assert_outputs_exist,
)


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
        help="Output directory.[%(default)s]",
        default="./",
    )
    p.add_argument(
        "-o",
        "--output_model_filename",
        help="Output CSV model filename."
        + "['ref_site-moving-site.model.metric_name.csv']",
        default="",
    )
    p.add_argument(
        "-m",
        "--method",
        default="clinic",
        choices=["classic", "clinic"],
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
        help="Degree of the polynomial fit in Combat. Default is linear "
        + "[default=1 for classic; 2 for clinic].",
    )
    p.add_argument(
        "--nu",
        type=float,
        default=5,
        help="Combat Clinic hyperparameter for the standard deviation estimation of the moving "
        + "site data. It must be >=0.  [%(default)s]",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=2,
        help="Combat Clinic hyperparameter for the covariate fit of the moving site data. "
        "It must be >= 1. [%(default)s]",
    )
    p.add_argument(
        "--hc",
        action="store_true",
        help="Will keep only HC subjects in the data.",
    )
    p.add_argument(
        "--robust",
        default="No",
        choices=["No", "IQR", "MAD","MMS", 'VS', 'VS2','TOP5', 'TOP10', 'TOP20', 'TOP30', 'TOP40', 'TOP50', 'CHEAT', 'FLIP', 'Z_SCORE'],
        help="If set, use combat robust. This tries "
        + "identifying/rejecting non-HC subjects.",
    )
    p.add_argument(
        "--rwp",
        action="store_true",
        help="Will remove whole patient if is outlierin one bundle",
    )
    p.add_argument(
        "--ignore_bundles",
        nargs="+",
        help="List of bundle to ignore.",
        default=['left_ventricle', 'right_ventricle']
    )

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if args.regul_mov is None:
        if args.method in ["classic"]:
            args.regul_mov = 0
        else:
            args.regul_mov = -1

    if args.degree is None:
        if args.method in ["classic"]:
            args.degree = 1
        else:
            args.degree = 2

    ref_data = pd.read_csv(args.ref_data)
    ref_data = ref_data[~ref_data['bundle'].isin(args.ignore_bundles)]
    mov_data = pd.read_csv(args.mov_data)
    mov_data = mov_data[~mov_data['bundle'].isin(args.ignore_bundles)]

    logging.info("Bundles: %s will be ignored.", args.ignore_bundles)

    # Check if moving site is a string
    if mov_data.site.dtype != "str":
        mov_data.site = mov_data.site.astype(str)
    if ref_data.site.dtype != "str":
        ref_data.site = ref_data.site.astype(str)

    if len(np.unique(ref_data["site"])) != 1:
        raise AssertionError("The reference data contains more than one site.")
    if len(np.unique(mov_data["site"])) != 1:
        raise AssertionError("The moving data contains more than one site.")
    if np.unique(ref_data["metric"]) != np.unique(mov_data["metric"]):
        raise AssertionError("Data file have different metrics.")

    if args.output_model_filename == "":
        output_filename = os.path.join(
            args.out_dir,
            str(np.unique(mov_data["site"])[0])
            + "-"
            + str(np.unique(ref_data["site"])[0])
            + "."
            + str(np.unique(ref_data["metric"])[0])
            + "."
            + args.method.lower()
            + ".model.csv",
        )
    else:
        output_filename = os.path.join(args.out_dir, args.output_model_filename)
    os.makedirs(args.out_dir, exist_ok=True)
    assert_outputs_exist(parser, args, output_filename, check_dir_exists=True)

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
    
    if args.robust != 'No':
        mov_data = remove_outliers(ref_data, mov_data, args)
    cols = ['Z_SCORE']
    cols_to_drop = [c for c in cols if c in mov_data.columns]
    mov_data =  mov_data.drop(columns=cols_to_drop)
    QC.robust = args.robust
    QC.fit(ref_data, mov_data, args.hc)

    logging.info("Saving file: %s", output_filename)
    QC.save_model(output_filename)


if __name__ == "__main__":
    main()
