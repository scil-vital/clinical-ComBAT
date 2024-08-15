#!/usr/bin/env python3

"""
Generates an altered/modified version of the input site data based on three parameters:
        slope, add factor and multiply factor (expressed as a percentage).
        100% corresponds to the initial value of the site data.
        These parameters support positive (50) or negative (-50) values.

The script returns the site data to 0 and then applies the slope, add and mult parameters.

# Uasage Examples :
combat_corrupt_data.py input.csv output.csv --mult 50 --add 30 --slope 10
combat_corrupt_data.py input.csv output.csv --mult 150 --add 80 --slope -30 --nbr_sub 100
combat_corrupt_data.py input.csv output.csv --mult 95 --add 125 --slope -40 --nbr_sub 50 \
                       --site_name "new_site_name"
"""

import argparse
import random
import os

import numpy as np
import pandas as pd

from clinical_combat.harmonization import from_model_name
from clinical_combat.harmonization.QuickCombat import QuickCombat

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
    p.add_argument("in_file", help="Input CSV site file.")
    p.add_argument("out_file", help="Output corrupted CSV site file.")
    p.add_argument(
        "--mult",
        type=float,
        default=100,
        help="Multiplicative bias in percent. Adjust the variance of the data. "
        "100 is not changing it. [%(default)s]",
    )
    p.add_argument(
        "--add",
        type=float,
        default=100,
        help="Additive bias in percent. Adjust the intercept of the data. "
        "100 is not changing it. [%(default)s]",
    )
    p.add_argument(
        "--slope",
        type=float,
        default=100,
        help="Slope bias in percent. Adjust the slope of the data. "
        "100 is not changing it. [%(default)s]",
    )
    p.add_argument(
        "--nbr_sub",
        type=int,
        default=-1,
        help="Mx number of subject to select. By default, all subjets are used.",
    )
    p.add_argument(
        "--site_name",
        type=str,
        help="Change the site name. By default the site name is unchanged.",
    )
    p.add_argument(
        "--HC",
        action="store_true",
        help="Only select HC.",
    )
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    random.seed(0)

    data = pd.read_csv(args.in_file)

    if args.HC:
        data = data.query("disease == 'HC'")

    if args.nbr_sub > 0:
        all_sids = list(data.sid.unique())
        random.shuffle(all_sids)
        sids = all_sids[: args.nbr_sub]
        data = data.query("sid in @sids")

    data = data.sort_values(by=["site", "sid", "bundle"])

    if args.site_name is None:
        args.site_name = str(data.site.unique()) + "_corrupted"
    # TODO ici on fait le test avec p[airwise, donc je ne sais pas si on peut juste enlever ce script
    QC = from_model_name(
        ignore_handedness_covariate=True,
        ignore_sex_covariate=True,
        use_empirical_bayes=False,
        limit_age_range=False,
    )
    QC.bundle_names = data.bundle.unique()

    design, y = QC.get_design_matrices(data)
    alpha, beta = QuickCombat.get_alpha_beta(design, y)
    sigma = QuickCombat.get_sigma(design, y, alpha, beta)

    y_cor = []
    for i in range(len(design)):
        cov_effect = np.dot(design[i][1:, :].transpose(), beta[i])
        y_normed_i = (y[i] - alpha[i] - cov_effect) / sigma[i]
        y_cor.append(
            y_normed_i * sigma[i] * args.mult / 100
            + alpha[i] * args.add / 100
            + cov_effect * args.slope / 100
        )

    data["site"] = args.site_name

    _dir = os.path.dirname(args.out_file)
    if len(_dir) > 0:
        os.makedirs(_dir, exist_ok=True)
    assert_outputs_exist(parser, args, args.out_file, check_dir_exists=True)

    save_quickcombat_data_to_csv(
        data,
        np.array(y_cor),
        QC.bundle_names,
        np.unique(data["metric"]),
        "raw",
        "raw",
        args.out_file,
    )


if __name__ == "__main__":
    main()
