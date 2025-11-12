#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fmt: off
"""
Harmonizes data from a subject (moving) to a reference site (target) using the following formula
for each bundle and subject independently:

    bundle harmonized = ((bundle value - average moving) *
                        (variability target/variability moving)) + average target

Average and variability can be computed with mean/std or median/mad depending on the method
argument.

Method :
    'Mean': Use mean and standard deviation to compute the average and variability across bundles.
    'Median': Use median and median absolute deviation (MAD, a robust alternative to std)
              to compute the average and variability.

Options --in_ref_harmonized
    Option designed to compare single subject harmonization with a reference harmonization.
    Compares the harmonization given as input with a harmonization considered as a reference.
    The distance is evaluated with the difference between the averages obtained with the two
    harmonized methods. Four measures are added in input DataFrame:
        - difference_harmonized: difference between the mean of input and reference harmonization
        - uncertainty: standard deviation of the difference_harmonized
        - err_lower: absolute value of the negative difference_harmonized
        - err_upper: value of the positive difference_harmonized
    Uncertainty, err_lower and err_upper are used to plot error bars in the figure with option
    --display_errors in the combat_visualize_age_curve script.

Examples:
# Compute single subject harmonization with mean and standard deviation (default)
combat_single_subject_harmonization /path/to/input.csv
# Compute single subject harmonization with median and median absolute deviation
combat_single_subject_harmonization /path/to/input.csv --method median
# Compute single subject harmonization with mean and standard deviation and compares with a reference
combat_single_subject_harmonization /path/to/input.csv \
    --in_ref_harmonization /path/to/reference.csv

"""
import os
import argparse
import pandas as pd
import numpy as np

from clinical_combat.visualization.viz import (generate_query,
                                               compute_reference_average_variability,
                                               compute_distance_between_harmonization)

import warnings
warnings.filterwarnings("ignore")


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("in_file", help="Json or CSV file with raw input data")

    h = p.add_argument_group(title='Options for Harmonization')
    h.add_argument("--outname",
                   help="Output file name. The input file name with suffix "
                        "_single_subject_harmonization.csv is used by default.")
    h.add_argument("--method",
                   default="mean",
                   choices=["mean", "median"],
                   help="Method to compute average and variability across bundles "
                        "[%(default)s].")
    h.add_argument("--save_bundles_data",
                   action="store_true",
                   help="Save the average and variability of the bundles for each "
                        "subject in *.bundle_data.csv.")

    r = p.add_argument_group(title='Options for Reference site')
    r.add_argument("--reference_site",
                   default="MRC-CBSU_Siemens_3T_2",
                   help="Tag for reference site [%(default)s].")

    wdw = p.add_argument_group(title='Window options for the moving average mean and standard '
                               'deviation computation')
    wdw.add_argument("--window_size",
                     type=int,
                     default=4,
                     help="Window size in years [%(default)s].")
    wdw.add_argument("--window_count",
                     type=int,
                     default=5,
                     help="Minimum number of subjects per window [%(default)s].")
    wdw.add_argument("--window_update",
                     action="store_true",
                     help="Update the window size to have a minimum "
                          "number of subjects per window [%(default)s].")
    wdw.add_argument("--ages",
                     nargs=2,
                     default=(20, 90),
                     metavar=('MIN', 'MAX'),
                     help="Range of ages to use [%(default)s].")

    data = p.add_argument_group(title='Options for data selection')
    data.add_argument("--sites", nargs='+',
                      help="List of sites to use [%(default)s].")
    data.add_argument("--bundles", nargs='+',
                      help="List of bundle to use [%(default)s].")
    data.add_argument("--metrics", nargs='+',
                      help="List of metric to use [%(default)s].")
    data.add_argument("--sexes", nargs='+',
                      help="List of sex to use [%(default)s].")
    data.add_argument("--handednesses", nargs='+',
                      help="List of handednesses to use [%(default)s].")
    data.add_argument("--diseases", nargs='+',
                      help="List of diseases to use [%(default)s].")

    error = p.add_argument_group(title='Options to plot error bars')
    error.add_argument("--in_ref_harmonization",
                       help="Path to json or csv results file used as reference for compute error"
                            " bars for plot. \nUsed to compare single subject harmonization results.")
    error.add_argument("--harmonization_name", default="combat",
                       help="Row criteria corresponding to harmonization type in harmonization "
                            "column [%(default)s].\nUsed to compare single subject harmonization "
                            "results.")

    return p


def compute_single_subject_harmonization(subject_bundle_value, average_moving, variability_moving,
                                         average_target, variability_target):
    """
    Compute the single subject harmonization for a given bundle from average and variability of
    across bundles.
    Args:
        subject_bundle_value: float, mean or median value for one bundle and one subject
        average_moving: float, average of the moving site across bundles
        variability_moving: float, standard deviation of the moving site across bundles
        average_target: float, average of the target site across bundles
        variability_target: float, standard deviation of the target site across bundles
    Returns:
        bundle_harmonized: float, harmonized value of the bundle for the subject
    """
    bundle_harmonized = ((subject_bundle_value - average_moving) *
                         (variability_target/variability_moving)) + average_target
    return bundle_harmonized


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.in_file)

    if args.outname is None:
        args.outname = os.path.splitext(os.path.basename(args.in_file))[0]
    args.outname = args.outname + '_' + args.method + '_wdw_' + str(args.window_size)

    if args.metrics is None:
        args.metrics = list(np.unique(df.metric))

    if args.bundles is None:
        args.bundles = list(np.unique(df.bundle))

    if args.sexes is None:
        args.sexes = list(np.unique(df.sex))

    if args.handednesses is None:
        args.handednesses = list(np.unique(df.handedness))

    if args.diseases and "HC" in args.diseases:
        args.diseases.remove("HC")

    if args.sites is None:
        args.sites = list(np.unique(df["site"]))

    if args.reference_site in args.sites:
        args.sites.remove(args.reference_site)

    # Extract reference site data and check if it is not empty
    df_reference_site = df.query(generate_query(args.metrics, args.bundles, ['HC'],
                                                site=args.reference_site,
                                                sex=args.sexes, handedness=args.handednesses))
    if len(df_reference_site) == 0:
        raise ValueError("No reference data found to data options")

    # Extract sites data and check if it is not empty
    df_moving_sites = df.query(generate_query(args.metrics, args.bundles, ['HC'],
                                              site=args.sites,
                                              sex=args.sexes, handedness=args.handednesses))
    if len(df_moving_sites) == 0:
        raise ValueError("No moving data are found to data options")

    # Compute harmonized single subject for each metric, bundle and subject
    for metric in args.metrics:
        ss_harmonized_df_per_subject, ss_bundles_data_per_subject = [], []
        for subject in df_moving_sites.sid.unique():
            # Compute for each subject the average and variability across bundles
            curr_subject = df_moving_sites.query("sid == @subject & metric == @metric"
                                                 ).reset_index(drop=True)
            if args.method == 'mean':
                average_bundles_subject = curr_subject['mean'].mean()
                variability_bundles_subject = curr_subject['mean'].std()
            elif args.method == 'median':
                average_bundles_subject = curr_subject['mean'].median()
                variability_bundles_subject = np.median(np.absolute(curr_subject['mean'] -
                                                                    average_bundles_subject))

            average_bundles_ref, variability_bundles_ref = compute_reference_average_variability(
                df_reference_site, curr_subject.age.values[0], windows_size=args.window_size,
                method=args.method)

            # Compute for each bundle the single subject harmonized value
            ss_harmonized_bundles_values = []
            for bundle in args.bundles:
                bundle_subject_val = curr_subject.query("bundle == @bundle")['mean'].values[0]
                bundle_harmonized = compute_single_subject_harmonization(
                    bundle_subject_val, average_bundles_subject, variability_bundles_subject,
                    average_bundles_ref, variability_bundles_ref)
                ss_harmonized_bundles_values.append(bundle_harmonized)

            # Add single subject harmonized values for all bundles to the subject DataFrame
            curr_subject['mean'] = ss_harmonized_bundles_values
            # Store the harmonized data for each subject
            ss_harmonized_df_per_subject.append(curr_subject)

            if args.save_bundles_data:
                # Add subjects average and variability values for all bundles
                ss_bundles_data_per_subject.append([subject,
                                                    average_bundles_subject,
                                                    variability_bundles_subject])

        # Save single subject harmonization data :
        # concatenate POST single_subject (ss) harmonization data and reference data
        ss_harmonized_df = pd.concat(ss_harmonized_df_per_subject[:], ignore_index=True)
        ss_harmonized_df['harmonization'] = 'single_subject'
        ss_orig_shape = len(ss_harmonized_df)

        # Compute uncertainty/errors between two harmonization methods
        if args.in_ref_harmonization:
            df_ref_harmonized = pd.read_csv(args.in_ref_harmonization)
            df_ref_harmonized = df_ref_harmonized.query("harmonization == @args.harmonization_name")

            # Compute uncertainty and errors for each bundle
            ss_harmonized_df_with_errors = []
            for bundle in ss_harmonized_df.bundle.unique():
                curr_ss_harmonized = ss_harmonized_df.query(generate_query(
                    metric, bundle, ['HC'], site=args.sites, sex=args.sexes))

                # Deals with the absence of a bundle in harmonized reference data
                if bundle not in df_ref_harmonized.bundle.unique():
                    ss_harmonized_df_with_errors.append(curr_ss_harmonized)
                else:
                    curr_ref_harmonized = df_ref_harmonized.query(generate_query(
                        metric, bundle, ['HC'], site=args.sites, sex=args.sexes))

                    # Compute uncertainty/errors for each bundle
                    ss_harmonized_df_with_errors.append(compute_distance_between_harmonization(
                        curr_ss_harmonized, curr_ref_harmonized))
            # Concatenate all errors df in one DataFrame, add NaN for df with no distance values.
            ss_harmonized_df = pd.concat(ss_harmonized_df_with_errors[:], ignore_index=True)
            # Checks for distance computation and concatenation errors
            if len(ss_harmonized_df) != ss_orig_shape:
                raise ValueError('\nComparison with harmonized reference data has modified \
                                 the original data.')

        # Add reference data copy with single_subject tag to easier figure plotting
        df_reference_site_for_ss = df_reference_site.copy()
        df_reference_site_for_ss['harmonization'] = 'single_subject'
        # Concatenate in order PRE and POST data for reference and moving sites in one DataFrame
        final_ss_harmonized_df = pd.concat([df_reference_site, df_moving_sites,
                                            df_reference_site_for_ss, ss_harmonized_df],
                                           ignore_index=True)
        # Save DataFrame in csv file
        final_ss_harmonized_df.to_csv(args.outname + '_single_subject_harmonization.csv',
                                      index=False)

        # Save bundles data if not empty in CSV file
        bundles_tmp = []
        if len(ss_bundles_data_per_subject) > 0:
            bundles_tmp.append(pd.DataFrame(ss_bundles_data_per_subject,
                                            columns=['sid', 'average', 'variability']))
            bundles_data = pd.concat(bundles_tmp, ignore_index=True)
            bundles_data.to_csv(args.outname + '_single_subject_harmonization.bundle_data.csv',
                                index=False)


if __name__ == "__main__":
    main()
