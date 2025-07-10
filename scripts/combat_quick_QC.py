#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute the quality control of the transfer function from a moving site to a
reference site using Bhattacharyya distance.

# Usage :
# Use the pairwise method to harmonize the moving site data to the reference site data 
# (linear)
combat_quick_QC.py reference_site.raw.csv.gz moving_site.raw.csv.gz \
                  reference_site-moving_site.model.metric_name.csv

"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from clinical_combat.harmonization import from_model_filename, from_model_name
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
        help="Path to the reference data.",
    )
    p.add_argument(
        "mov_data",
        help="Path to the moving data.",
    )
    p.add_argument("model", help="Combat CSV model parameters.")
    p.add_argument(
        "--out_dir",
        help="Output directory.[%(default)s]",
        default="./",
    )
    p.add_argument(
        "--degree_qc",
        type=int,
        help="Degree for model fit. By default it uses the input model degree.",
        default=0,
    )
    p.add_argument(
        "--print_only",
        action="store_true",
        help="If set, do not save the distance to a text file.",
    )
    p.add_argument(
        "-o",
        "--output_results_filename",
        help="Output txt results filename. ['mov_data.bhattacharrya.txt']",
        default="",
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

    ref_data = pd.read_csv(args.ref_data).query("disease == 'HC'") 
    ref_data = ref_data[~ref_data['bundle'].isin(args.ignore_bundles)]
    mov_data = pd.read_csv(args.mov_data).query("disease == 'HC'")
    mov_data = mov_data[~mov_data['bundle'].isin(args.ignore_bundles)]

    logging.info("Bundles: %s will be ignored.", args.ignore_bundles)


    model = from_model_filename(args.model)

    if args.degree_qc == 0:
        args.degree_qc = model.degree

    QC = from_model_name(
        "pairwise",
        ignore_handedness_covariate=model.ignore_handedness_covariate,
        ignore_sex_covariate=model.ignore_sex_covariate,
        use_empirical_bayes=False,
        limit_age_range=False,
        degree=args.degree_qc,
        regul_ref=0,
        regul_mov=0,
        nu=0,
        tau=2,
    )
    QC.fit(ref_data, ref_data)

    metric_name = QC.model_params["metric_name"]
    ref_site = QC.model_params["ref_site"]    

    # Check if moving site is a string
    if mov_data.site.dtype != "str":
        mov_data.site = mov_data.site.astype(str)
    if ref_data.site.dtype != "str":
        ref_data.site = ref_data.site.astype(str)

    if len(np.unique(mov_data["site"])) != 1:
        raise AssertionError("The moving data contains more than one site.")
    if metric_name != np.unique(mov_data["metric"]):
        raise AssertionError("Data file have different metrics.")
    if ref_site != np.unique(ref_data["site"]):
        logging.warning("Model site and reference data site don't match.")

    dists = QC.get_bundles_bhattacharyya_distance(ref_data, mov_data)

    count = len(mov_data.sid.unique())

    if args.output_results_filename == "":
        output_filename = os.path.join(
            args.out_dir,
            os.path.basename(args.mov_data).split(".csv")[0] + ".bhattacharrya.txt",
        )
    else:
        output_filename = os.path.join(args.out_dir, args.output_results_filename)

    print(
        "      Mean Bhattacharrya distance: %f (min: %f, max: %f)"
        % (np.mean(dists), np.min(dists), np.max(dists))
    )

    if not args.print_only:
        assert_outputs_exist(parser, args, output_filename, check_dir_exists=True)
        logging.info("Saving file: %s", output_filename)
        header = "HC"
        for curr_bundle in QC.bundle_names:
            header += " " + curr_bundle
        np.savetxt(
            output_filename,
            np.array([count] + dists)[np.newaxis, :],
            header=header,
            fmt="%s",
            comments="",
        )


if __name__ == "__main__":
    main()
