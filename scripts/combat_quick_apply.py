#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a precompute harmonization model to a site data.

Example:
combat_quick_apply.py moving_site.raw.csv.gz moving_site.metric.model.csv


"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

from clinical_combat.harmonization import from_model_filename
from clinical_combat.utils.combatio import save_quickcombat_data_to_csv
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
        "-o",
        "--output_results_filename",
        help="Output CSV of the harmonized data filename."
        + "['moving-site.metric_name.model.res.csv']",
        default="",
    )
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    QC = from_model_filename(args.model)

    metric_name = QC.model_params["metric_name"]
    mov_site = str(QC.model_params["mov_site"])
    mov_data = pd.read_csv(args.mov_data)

    # Check if moving site is a string
    if mov_data.site.dtype != "str":
        mov_data.site = mov_data.site.astype(str)

    if len(np.unique(mov_data["site"])) != 1:
        raise AssertionError("The moving data contains more than one site.")
    if metric_name != np.unique(mov_data["metric"]):
        raise AssertionError("Data file have different metrics.")
    if mov_site != np.unique(mov_data["site"])[0]:
        logging.warning("Model and data site don't match.")

    y_harm = QC.apply(mov_data)

    if args.output_results_filename == "":
        output_filename = os.path.join(
            args.out_dir,
            str(mov_site)
            + "."
            + metric_name
            + "."
            + QC.model_params["name"]
            + ".csv.gz",
        )
    else:
        output_filename = os.path.join(args.out_dir, args.output_results_filename)
    os.makedirs(args.out_dir, exist_ok=True)
    assert_outputs_exist(parser, args, output_filename, check_dir_exists=True)

    logging.info("Saving file: %s", output_filename)
    save_quickcombat_data_to_csv(
        mov_data,
        y_harm,
        QC.bundle_names,
        metric_name,
        QC.model_params["name"],
        args.model,
        output_filename,
    )


if __name__ == "__main__":
    main()
