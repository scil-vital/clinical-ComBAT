#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fmt: off
"""
Visualizes the difference between the mean or standard deviation per site and the
reference site for all bundles before and after harmonization, using a box plot and sliding window.
Similarly, the effect size is assessed for each site using Cohen's D. The smaller the Cohen's D
value, the smaller the difference between the 2 sites, and vice versa.

Each box corresponds to a site and represents the mean error, STD or effect of size (Cohen's D)
for all bundles (group mean) before and after harmonization.


example:
combat_visualize_errors_cohensd adni_md.csv
"""

import argparse
import warnings
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns

from clinical_combat.utils.combatio import load_sites_data
from clinical_combat.visualization.plots import generate_boxplot_error_cohend
from clinical_combat.visualization.viz import (
    add_manufacturers_to_df, compute_site_errors_and_effectsize,
    convert_matrix_to_dataframe, generate_query, get_valid_age_windows,
    load_metaverse_results, viz_identify_valid_sites)

warnings.filterwarnings("ignore")


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("in_files", help="Path to csv sites data.", nargs='+')

    reference = p.add_argument_group(title='Options for Reference site')
    reference.add_argument("--reference_site", default="MRC-CBSU_Siemens_3T_2",
                           help="Reference site [%(default)s].")
    reference.add_argument("--rename_reference_site",
                            help="Rename the reference site to this name.")
    reference.add_argument("--min_subject_per_site", type=int, default=10,
                           help="Exclude site fewer subjects than min_subject_per_site."
                           " [%(default)s].")
    reference.add_argument("--meta",
                           help="Metaverse csv file.")

    wdw = p.add_argument_group(title='Window options for the moving average mean and standard '
                               'deviation computation')
    wdw.add_argument("--window_size", type=int, default=10,
                     help="Window size in years. [%(default)s].")
    wdw.add_argument("--window_count", type=int, default=5,
                     help="Minimum number of subjects per window. [%(default)s].")
    wdw.add_argument("--window_update", action="store_true",
                     help="Update the window size to have a minimum number of subjects per window."
                     " [%(default)s].")
    wdw.add_argument("--ages", nargs=2, default=(10, 90),  metavar=('MIN', 'MAX'),
                     help="Range of ages to use min,max [%(default)s].")

    data = p.add_argument_group(title='Options for data selection')
    data.add_argument("--sites", nargs='+',
                      help="List of sites to use [%(default)s].")
    data.add_argument("--bundles", nargs='+',
                      help="List of bundle to use for error plots [%(default)s].")
    data.add_argument("--metrics", nargs='+',
                      help="List of metric to use [%(default)s].")
    data.add_argument("--sexes", nargs='+',
                      help="List of sex to use [%(default)s].")
    data.add_argument("--handednesses", nargs='+',
                      help="List of handednesses to use [%(default)s].")
    data.add_argument("--diseases", nargs='+',
                      help="List of diseases to use [%(default)s].")

    plot = p.add_argument_group(title='Options for figures')
    plot.add_argument("--y_axis_percentile", nargs=2, default=(1, 99), metavar=('MIN', 'MAX'),
                      help="Range of metric value to use for ymin,ymax in percentile [%(default)s].")
    plot.add_argument("--percentiles", nargs='+', default=(5, 25, 50, 75, 95),
                      help="Number of percentile use to add percentile on curve.")
    plot.add_argument("--line_widths", nargs='+', default=(0.25, 1, 2, 1, 0.25),
                      help="Line width for percentile delimitation.")
    plot.add_argument("--palette", default=sns.color_palette("Spectral"),
                      help="Color palette, the palette can be a color from the Seaborn palette"
                      " (default), or  a list of specific colors. [%(default)s].")
    plot.add_argument("--increase_ylim", nargs=1, default=5,
                      help="Percentage of increase and decrease of y-axis limit [%(default)s].")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    palette = args.palette

    df = load_sites_data(args.in_files)

    if args.meta:
        manufacturer_dict = load_metaverse_results(args.meta, args.reference_site)
        df = add_manufacturers_to_df(df, parse_dict=manufacturer_dict)
    else:
        df = add_manufacturers_to_df(df, reference_site=args.reference_site,
                                     reference_name=args.rename_reference_site)
    df = df.sort_values(["manufacturer", "site"])

    if not len(df.metric.unique()) == 1:
        raise ValueError('The input data contains more then 1 metric.')
    metric = df.metric.unique()[0]

    if args.bundles is None:
        args.bundles = list(np.unique(df["bundle"]))

    if args.sexes is None:
        args.sexes = list(np.unique(df.sex))

    if args.handednesses is None:
        args.handednesses = list(np.unique(df.handedness))

    if args.diseases and "HC" in args.diseases:
        args.diseases.remove("HC")

    if args.sites is None:
        args.sites = df["site"].unique().tolist()

    site_list_with_minN = viz_identify_valid_sites(df, args.sites, args.min_subject_per_site,
                                                   metric=args.metrics, bundle=args.bundles)

    nbr_ref_subs = df.query(generate_query(args.metrics, args.bundles, ['HC'],
                                           site=args.reference_site,
                                           harmonization='raw', model="raw",
                                           sex=args.sexes, handedness=args.handednesses)).shape[0]

    if nbr_ref_subs == 0:
        print("No data for the reference site")
        return

    # Set age and percentile limits
    min_age, max_age = float(args.ages[0]), float(args.ages[1])
    y_min_percentile, y_max_percentile = float(
        args.y_axis_percentile[0]), float(args.y_axis_percentile[1])

    ### Moving Average ###
    target_site_list = [site for site in site_list_with_minN if site not in args.reference_site]

    # Create empty matrix to store mean and std error - all data
    matrix_size = (len(args.bundles), len(target_site_list),
                    len(df.harmonization.unique()))
    matrix_mean_error = np.zeros(matrix_size)
    matrix_std_error = np.zeros(matrix_size)
    matrix_cohenD = np.zeros(matrix_size)

    # Create empty dataframe to store mean and std error for each bundle
    mean_error_df, std_error_df, cohenD_df = [], [], []

    for bundle_idx, bundle in enumerate(args.bundles):
        print("Processing: ", bundle)

        df_average_reference = df.query(
                generate_query(metric, bundle, ['HC'], site=args.reference_site,
                               sex=args.sexes, harmonization="raw", model="raw"))
        if len(df_average_reference) == 0:
            continue
        reference_window_age, reference_window_mean, reference_window_std = [], [], []

        window_ref_df, age_reference_site = get_valid_age_windows(
            df_average_reference, min_age, max_age, 10, use_dynamic_window_size=False,
            min_n_subjects=args.window_count)
        # Reference - Compute moving average and standard deviation for each age window
        for ref_curr_window_df, ref_curr_age in zip(window_ref_df, age_reference_site):
            reference_window_age.append(ref_curr_age)
            reference_window_mean.append(
                ref_curr_window_df['mean'].mean())
            reference_window_std.append(
                ref_curr_window_df['mean'].std())

        harmonization_idx = -1
        for harmonization in np.unique(df["harmonization"]):
            df_ = df.query(generate_query(metric, bundle, ['HC'],
                                          site=target_site_list,
                                          sex=args.sexes,
                                          harmonization=harmonization,
                                          model=None))

            if len(df_.model.unique())>1:
                print("More than 1 model per harmonization is not supported.")
            if len(df_) == 0:
                continue

            harmonization_idx += 1

            # Moving sites data
            # Extract data for all sites and compute mean and std error
            for site_idx, curr_site in enumerate(target_site_list):
                mean_errors, std_errors, window_cohend = [], [], []

                # Extract data for site
                df_site_hc = df.query(generate_query(metric, bundle, ['HC'], site=curr_site,
                                                        sex=args.sexes,
                                                        harmonization=harmonization,
                                                        model=None))
                if len(df_site_hc) == 0:
                    continue
                min_age_site, max_age_site = df_site_hc['age'].min(), df_site_hc['age'].max()
                # Site - Compute moving average and standard deviation for each age window
                window_ref_age_dfs, reference_age = get_valid_age_windows(
                    df_site_hc, min_age_site, max_age_site, args.window_size,
                    use_dynamic_window_size=args.window_update, min_n_subjects=args.window_count)

                for site_curr_window_df, site_curr_age in zip(window_ref_age_dfs,
                                                                reference_age):
                    mean_errors, std_errors, window_cohend = compute_site_errors_and_effectsize(
                        site_curr_window_df, site_curr_age, reference_window_age,
                        reference_window_mean, reference_window_std)

                # Store mean, std error and cohen's D for each site, bundle and harmonization
                if len(mean_errors) > 0:
                    matrix_mean_error[bundle_idx, site_idx, harmonization_idx] = np.mean(
                        np.abs(mean_errors))
                    matrix_std_error[bundle_idx, site_idx, harmonization_idx] = np.mean(
                        np.abs(std_errors))
                    matrix_cohenD[bundle_idx, site_idx, harmonization_idx] = np.mean(
                        np.abs(window_cohend))

            # Append and convert site matrix to dataframe
            mean_error_df.append(
                convert_matrix_to_dataframe(matrix_mean_error[bundle_idx, :, harmonization_idx],
                                            target_site_list, bundle, harmonization))
            std_error_df.append(
                convert_matrix_to_dataframe(matrix_std_error[bundle_idx, :, harmonization_idx],
                                            target_site_list, bundle, harmonization))
            cohenD_df.append(
                convert_matrix_to_dataframe(matrix_cohenD[bundle_idx, :, harmonization_idx],
                                            target_site_list, bundle, harmonization))

    # Concatenate dataframes for each bundle
    df_mean_error = pd.concat(mean_error_df[:]).reset_index(drop=True)
    df_std_error = pd.concat(std_error_df[:]).reset_index(drop=True)
    df_cohenD = pd.concat(cohenD_df[:]).reset_index(drop=True)

    df_mean_error = add_manufacturers_to_df(
        df_mean_error, reference_site=args.reference_site)
    df_mean_error = df_mean_error.sort_values(["manufacturer", "site"])
    plot_order = df_mean_error.site.unique().tolist()

    ### Error Plot ###
    ymin_mean = min([matrix_mean_error.min(), matrix_std_error.min()])
    ymax_mean = max([matrix_mean_error.max(), matrix_std_error.max()])
    ymin_mean = ymin_mean - (ymin_mean * args.increase_ylim / 100)
    ymax_mean = ymax_mean + (ymax_mean * args.increase_ylim / 100)


    for harmonization in np.unique(df["harmonization"]):

        curr_error = df_mean_error.query("harmonization == @harmonization")
        curr_std = df_std_error.query("harmonization == @harmonization")

        if len(curr_error) == 0:
            continue

        generate_boxplot_error_cohend(
            curr_error, curr_std, "site", "mean", metric, "site", order=plot_order,
            ylim=(ymin_mean, ymax_mean), prefix_title=harmonization.upper(),
            ax1_title='Mean Error', ax2_title='STD Error', colors=palette,
            save_prefix="ErrorBoxplots_" + harmonization + "_",
            title=" Data\n" + "Differences with Reference Bundles")

        ### Cohen's D  ###
        if harmonization == "raw":
            continue

        ymin_cd = min(matrix_cohenD.flatten())
        ymax_cd = max(matrix_cohenD.flatten())
        ymin_cd = ymin_cd - (ymin_cd * args.increase_ylim / 100)
        ymax_cd = ymax_cd + (ymax_cd * args.increase_ylim / 100)

        cohenD_pre = df_cohenD[df_cohenD.harmonization == 'raw']
        cohenD_combat = df_cohenD[df_cohenD.harmonization == harmonization]

        generate_boxplot_error_cohend(
            cohenD_pre, cohenD_combat, "site", "mean", metric, "site", order=plot_order,
            title=" Cohens D\n", ax1_title='Raw Data',
            ax2_title=harmonization + ' Harmonization', ylim=(ymin_cd, ymax_cd), colors=palette,
            save_prefix="EffectSizeBoxplots_" + harmonization + "_")


if __name__ == "__main__":
    main()
