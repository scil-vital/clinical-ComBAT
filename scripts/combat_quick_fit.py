#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute the transfer function from a moving site to a reference site.

Harmonization method:
    classic: uses both moving and reference data to fit the covariate
             regression parameters (Beta_mov).

Examples:
# Use the classic method to harmonize the moving site data to the reference site data 
# (linear)
combat_quick_fit.py reference_site.raw.csv.gz moving_site.raw.csv.gz

"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from classic_combat.harmonization import QuickCombatClassic
from classic_combat.utils.scilpy_utils import (
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
            + ".model.csv",
        )
    else:
        output_filename = os.path.join(args.out_dir, args.output_model_filename)
    os.makedirs(args.out_dir, exist_ok=True)
    assert_outputs_exist(parser, args, output_filename, check_dir_exists=True)

    QC = QuickCombatClassic(
        ignore_handedness_covariate=args.ignore_handedness,
        ignore_sex_covariate=args.ignore_sex,
        use_empirical_bayes=not args.no_empirical_bayes,
        limit_age_range=args.limit_age_range,
    )

    QC.fit(ref_data, mov_data)

    logging.info("Saving file: %s", output_filename)
    QC.save_model(output_filename)


if __name__ == "__main__":
    main()
