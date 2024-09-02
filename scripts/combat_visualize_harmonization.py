#!/usr/bin/env python3
"""
Visualizes reference site (target) and moving site before (raw) and after (combat) harmonization.
To view several sites at once, use combat_quick_visualize.py.

By default, the script displays the mean value (mean) and standard deviations (SD) per percentile,
per metric and for the AF left bundle (mni_AF_L) for the reference and moving sites before (raw)
and after (harmonized, ComBAT) harmonization, using a dynamic sliding window.

Input:
    Reference corresponds to data from the site used as reference.
    Moving data - provide 2 inputs: raw data csv (*.raw.csv) and
                                    harmonized data csv (*.result.csv from qc_apply script).

Output:
    Outname: Corresponds to the prefix used to save generated figures.
             By default/option: {AgeCurve_name/outname}_method_name_metric_bundle_{+/- suffix}.png
                                with name = reference_moving, outname replaces 'AgeCurve_name'
             Possibility to add a suffix if needed with the --add_suffix option.

Dynamic window:
    The sliding window is defined by the arguments:
        window_size, i.e. the age range to calculate mean/SD
        window_count, i.e. the minimum number of subjects required to calculate mean/SD.
    By default, the script applies a dynamic window, i.e. increases the age range until
    window_count is reached. To disable it, use --no_dynamic_window, the mean/SD will be
    calculated on the number of subjects corresponding to the age range without adaptation
    (not recommended).

Display options:
    The --display options allow you to add/change plot elements, and the --hide options
    allow you to remove elements such as disease data.
    The --display_point option will display data moving with a scatterplot rather than a curve.
    With the --display_errors option, it displays the uncertainty value (uncertainty)
    or the error bars (bound) in the post-harmonization graph, ONLY with the “single_subject”
    harmonization. For the moment, this option is only available for the --display_point option.

# Default usage:
combat_visualize_harmonization.py ref.raw.csv.gz moving.raw.csv.gz harmonization.results.csv

--------------------------------

Usage examples:
# Output options:
combat_visualize_harmonization.py ref.raw.csv.gz moving.raw.csv.gz harmonization.results.csv
                            --outname AgeCurve_AF_L --out_dir ./figures/ --add_suffix test

# Display data for all bundles:
combat_visualize_harmonization.py ref.raw.csv.gz moving.raw.csv.gz harmonization.results.csv
                            --bundles all

# Display data for n bundles:
combat_visualize_harmonization.py ref.raw.csv.gz moving.raw.csv.gz harmonization.results.csv
                            --bundles mni_AF_L mni_CC_L mni_CST_L

# Display data without percentiles and disease:
combat_visualize_harmonization.py ref.raw.csv.gz moving.raw.csv.gz harmonization.results.csv
                            --hide_disease --hide_percentiles

# Display data moving with a scatterplot:
combat_visualize_harmonization.py ref.raw.csv.gz moving.raw.csv.gz harmonization.results.csv
                            --display_point

"""

import argparse
import logging
import random
from itertools import product

import numpy as np
import pandas as pd

from vanilla_combat.utils.combatio import load_sites_data
from vanilla_combat.utils.scilpy_utils import add_overwrite_arg, add_verbose_arg
from vanilla_combat.visualization.plots import (
    add_errorbars_to_plot,
    add_reference_percentiles_to_curve,
    add_scatterplot_to_curve,
    add_site_curve_to_reference_curve,
    initiate_joint_marginal_plot,
    update_global_figure_style_and_save,
)
from vanilla_combat.visualization.viz import (
    compute_reference_windows_and_percentiles_by_windows,
    compute_site_curve,
    custom_palette,
    generate_query,
    get_valid_age_windows,
    line_style,
    viz_identify_valid_sites,
)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument("in_reference", help="Path to CSV for reference site (raw data).")
    p.add_argument(
        "in_movings",
        nargs="+",
        help="Path to 2 CSVs file for moving site: Raw and harmonized.",
    )

    out = p.add_argument_group(title="Options for output figure.")
    out.add_argument(
        "--out_dir",
        help="Output directory.[%(default)s]",
        default="./",
    )
    out.add_argument("--outname", help="Filename to save figure.")
    out.add_argument(
        "--add_suffix", help="Add suffix to figure title and output PNG filename."
    )

    wdw = p.add_argument_group(
        title="Window options for the moving average mean and standard "
        "deviation computation"
    )
    wdw.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Window size in years. [%(default)s].",
    )
    wdw.add_argument(
        "--window_count",
        type=int,
        default=10,
        help="Minimum number of subjects per window. [%(default)s].",
    )
    wdw.add_argument(
        "--no_dynamic_window",
        action="store_true",
        help="Use to avoid updating the window size to have a minimum number of "
        "subjects per window. Not recommended.",
    )
    wdw.add_argument(
        "--min_subject_per_site",
        type=int,
        default=10,
        help="Exclude site fewer subjects than min_subject_per_site [%(default)s].",
    )

    data = p.add_argument_group(title="Options for data selection")
    data.add_argument(
        "--ages",
        nargs=2,
        default=(20, 90),
        metavar=("MIN", "MAX"),
        help="Range of ages to use min, max. Affects only the reference site."
        " [%(default)s].",
    )
    data.add_argument(
        "--bundles",
        nargs="+",
        help="List of bundle to use for figures. To plot all bundles use "
        "--bundles all. ['mni_IIT_mask_skeletonFA'].",
    )
    data.add_argument("--sexes", nargs="+", help="List of sex to use. All by default.")
    data.add_argument(
        "--handednesses", nargs="+", help="List of handednesses to use. All by default."
    )
    data.add_argument(
        "--diseases", nargs="+", help="List of diseases to use. All by default."
    )

    viz = p.add_argument_group(title="Display options")
    viz.add_argument(
        "--display_point",
        action="store_true",
        help="Show moving site with a scatterplot rather than a curve.",
    )
    viz.add_argument(
        "--display_marginal_hist",
        action="store_true",
        help="Add marginal histograms to plot.",
    )
    viz.add_argument(
        "--hide_disease",
        action="store_true",
        help="Deletes data corresponding to diseases.",
    )
    viz.add_argument(
        "--hide_percentiles",
        action="store_true",
        help="Show data with SD rather than SD per percentiles for both reference and"
        " moving sites.",
    )

    plot = p.add_argument_group(title="Options for plot.")
    plot.add_argument(
        "--randomize_line",
        action="store_true",
        help="Will choose a random color and linestyle for site moving.",
    )
    plot.add_argument(
        "--increase_ylim",
        type=int,
        default=5,
        help="Percentage of increase and decrease of y-axis limit [%(default)s].",
    )
    plot.add_argument(
        "--fixed_ylim",
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Fixed values for y-axis limit.",
    )
    plot.add_argument(
        "--y_axis_percentile",
        nargs=2,
        default=(1, 99),
        metavar=("MIN", "MAX"),
        help="Range of metric value to use for min,max in percentile [%(default)s].",
    )
    plot.add_argument(
        "--percentiles",
        nargs="+",
        default=(5, 25, 50, 75, 95),
        help="Number of percentile use to add percentile on curve.",
    )
    plot.add_argument(
        "--line_widths",
        nargs="+",
        default=(0.25, 1, 2, 1, 0.25),
        help="Line width for percentile delimitation.",
    )
    plot.add_argument(
        "--line_style",
        help="Line style for moving site data only. Default is dashed style.",
    )

    error = p.add_argument_group(
        title="Options to plot error bars."
        "Designed for single-subject harmonization data."
    )
    error.add_argument(
        "--display_errors", action="store_true", help="Display error bars on the plot."
    )
    error.add_argument(
        "--error_metric",
        default="uncertainty",
        choices=["uncertainty", "bounds"],
        help="Error metric used to display error bar [%(default)s].",
    )
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Set color and linestyle for moving site
    if args.randomize_line:
        moving_palette = random.choice(custom_palette[1:])
        moving_linestyle = random.choice(line_style[1:])
    elif args.line_style:
        moving_linestyle = args.line_style
    else:
        moving_palette = custom_palette[1]
        moving_linestyle = line_style[1]

    # Load CSV files corresponding to reference site (raw) and moving site (raw + harmonized)
    df_ref, df_moving = load_sites_data([args.in_reference]), load_sites_data(
        args.in_movings
    )
    ref_site, moving_site = df_ref.site.unique(), df_moving.site.unique()
    df = pd.concat([df_ref, df_moving]).reset_index(drop=True)

    if args.display_errors and "uncertainty" not in df.columns.tolist():
        raise ValueError(
            "The distance/errors data are missing in the harmonization file."
        )

    if not len(df.metric.unique()) == 1:
        raise ValueError("The input data contains more then 1 metric.")
    metric = df.metric.unique()[0]

    all_bundles = np.intersect1d(df_ref.bundle.unique(), df_moving.bundle.unique())
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

    if args.sexes is None:
        args.sexes = list(np.unique(df.sex))

    if args.handednesses is None:
        args.handednesses = list(np.unique(df.handedness))

    if args.diseases is None:
        args.diseases = list(np.unique(df.disease))

    if args.diseases and "HC" in args.diseases:
        args.diseases.remove("HC")
    # List to check all data
    all_disease = ["HC"] + args.diseases if args.diseases is not None else ["HC"]

    # Check the number of subject per site and filter out sites with too few subjects
    valid_site_list = viz_identify_valid_sites(
        df, df.site.unique().tolist(), args.min_subject_per_site, bundle=args.bundles[0]
    )

    if ref_site not in valid_site_list:
        raise ValueError("No valid data found for the reference site.")

    # Generated joint plot Age Curve Figures for each bundle
    min_age, max_age = float(args.ages[0]), float(args.ages[1])
    y_min_percentile, y_max_percentile = float(args.y_axis_percentile[0]), float(
        args.y_axis_percentile[1]
    )

    for bundle in args.bundles:
        logging.info("Processing: %s", bundle)
        df_vals = df.query(
            generate_query(
                metric,
                bundle,
                all_disease,
                site=valid_site_list,
                age=(min_age, max_age),
                sex=args.sexes,
                handedness=args.handednesses,
            )
        )

        if len(df_vals) == 0:
            logging.info("No data found for metric %s and bundle %s.", metric, bundle)
            continue

        # Extract reference site data for bundle
        df_ref_bundle = df.query(
            generate_query(
                metric,
                bundle,
                ["HC"],
                age=(min_age, max_age),
                sex=args.sexes,
                handedness=args.handednesses,
                harmonization="raw",
                model="raw",
                site=ref_site,
            )
        )

        # Set y-axis limits according to reference site data or plot option
        if args.fixed_ylim:
            ymin, ymax = float(args.fixed_ylim[0]), float(args.fixed_ylim[1])
        else:
            ymin, ymax = np.percentile(
                df_ref_bundle["mean"], [y_min_percentile, y_max_percentile]
            )
            ymin = ymin - (
                df_ref_bundle["mean"].min() * (0.05 + args.increase_ylim / 100)
            )
            ymax = ymax + (
                df_ref_bundle["mean"].max() * (0.05 + args.increase_ylim / 100)
            )

        # Generate figure for each harmonization type
        for harmonization, model in product(
            np.unique(df.harmonization), np.unique(df.model)
        ):
            # Extract dataframe corresponding to the bundle for each harmonization type
            df_bundle = df.query(
                generate_query(
                    metric,
                    bundle,
                    ["HC"],
                    age=(min_age, max_age),
                    sex=args.sexes,
                    handedness=args.handednesses,
                    harmonization=harmonization,
                    model=model,
                )
            )
            if len(df_bundle) == 0:
                continue
            # Merge reference site data to the harmonized moving site data for plot
            df_bundle = pd.concat([df_bundle, df_ref_bundle])

            # Initialize the joint plot using all sites data - for bundle
            g, ax = initiate_joint_marginal_plot(
                df_bundle,
                "age",
                "mean",
                "site",
                (ymin, ymax),
                legend_title="Sites",
                xlim=(min_age, max_age),
                marginal_hist=args.display_marginal_hist,
                hist_hur_order=valid_site_list,
                hist_palette=custom_palette[: len(valid_site_list)],
            )

            ## Reference data
            logging.info(
                "Harmonization method: %s - Display data for reference site: %s ",
                harmonization,
                ref_site,
            )

            # Extract data for reference site for plotting
            df_average_reference = df.query(
                generate_query(
                    metric,
                    bundle,
                    ["HC"],
                    site=ref_site,
                    sex=args.sexes,
                    harmonization="raw",
                    model="raw",
                )
            )

            # Plot without percentile display
            if args.hide_percentiles:
                logging.info("The percentiles are not displayed on the plot.")
                # Reference - Compute moving average and standard deviation for each age window
                windows_reference_per_age, reference_age = get_valid_age_windows(
                    df_average_reference,
                    min_age,
                    max_age,
                    args.window_size,
                    no_dynamic_window_size=args.no_dynamic_window,
                    min_n_subjects=args.window_count,
                )
                site_window_age, site_window_mean, site_window_std = compute_site_curve(
                    windows_reference_per_age, reference_age
                )

                # Plot - Add moving average and standard deviation to joint plot
                if len(site_window_mean) > 0:
                    label = windows_reference_per_age[0]["site"].unique()[0]
                    ax = add_site_curve_to_reference_curve(
                        ax,
                        site_window_age,
                        site_window_mean,
                        site_window_std,
                        label_site=label,
                        ylim=(ymin, ymax),
                        color=custom_palette[0],
                    )
            else:
                # The percentiles are displayed on the plot
                # Compute percentiles for each age window
                reference_percentiles, age_reference_site, _ = (
                    compute_reference_windows_and_percentiles_by_windows(
                        df_average_reference,
                        args.percentiles,
                        (min_age, max_age),
                        args.window_size,
                        args.window_count,
                        dynamic=args.no_dynamic_window,
                    )
                )

                # Plot - Add percentiles computed from reference to joint plot
                ax = add_reference_percentiles_to_curve(
                    ax,
                    age_reference_site,
                    reference_percentiles,
                    args.percentiles,
                    args.line_widths,
                )

            ## Moving site data
            logging.info(
                "Harmonization method: %s - Display data for moving site: %s ",
                harmonization,
                moving_site,
            )
            # Add moving site data POINTS to plot
            if args.display_point:
                logging.info("Display data points for %s site.", moving_site)
                df_site = df.query(
                    generate_query(
                        metric,
                        bundle,
                        ["HC"],
                        site=moving_site,
                        age=(min_age, max_age),
                        handedness=args.handednesses,
                        sex=args.sexes,
                        harmonization=harmonization,
                        model=model,
                    )
                )
                if len(df_site) > 0:
                    ax = add_scatterplot_to_curve(
                        ax,
                        df_site,
                        "age",
                        "mean",
                        "site",
                        hue_order=moving_site,
                        alpha=0.8,
                        palette=[moving_palette],
                    )

                # Add error bars to the plot for post harmonization figures
                if args.display_errors and harmonization != "raw":
                    error_data = df_site["uncertainty"].values
                    if args.error_metric == "bounds":
                        error_data = [
                            df_site["err_lower"].values,
                            df_site["err_upper"].values,
                        ]
                    # Check if error_data is not empty
                    if np.isnan(error_data).all() == True:
                        raise ValueError(
                            "No error data found for: {}, {}".format(bundle, metric)
                        )
                    # Add uncertainty or error bars to point
                    ax = add_errorbars_to_plot(
                        ax,
                        df_site["age"],
                        df_site["mean"],
                        error_data,
                        [moving_palette],
                        label=moving_site[0],
                    )

            # Add site data CURVES to plot - Default
            else:
                if args.display_errors:
                    logging.warning(
                        "The display_errors option is not implemented for curve."
                    )

                moving_site_hc = df.query(
                    generate_query(
                        metric,
                        bundle,
                        ["HC"],
                        site=moving_site,
                        sex=args.sexes,
                        harmonization=harmonization,
                        model=model,
                    )
                )
                min_age_site, max_age_site = (
                    moving_site_hc["age"].min(),
                    moving_site_hc["age"].max(),
                )

                if args.hide_percentiles:
                    logging.info("The percentiles are not displayed on the plot.")
                    # Moving - Compute moving average and standard deviation for each age window
                    windows_reference_per_age, reference_age = get_valid_age_windows(
                        moving_site_hc,
                        min_age_site,
                        max_age_site,
                        args.window_size,
                        no_dynamic_window_size=args.no_dynamic_window,
                        min_n_subjects=args.window_count,
                    )
                    site_window_age, site_window_mean, site_window_std = (
                        compute_site_curve(windows_reference_per_age, reference_age)
                    )

                    # Plot - Add moving average and standard deviation to joint plot
                    if len(site_window_mean) > 0:
                        label = windows_reference_per_age[0]["site"].unique()[0]
                        ax = add_site_curve_to_reference_curve(
                            ax,
                            site_window_age,
                            site_window_mean,
                            site_window_std,
                            label_site=label,
                            ylim=(ymin, ymax),
                            color=moving_palette,
                        )

                else:
                    site_percentiles, age_site_site, _ = (
                        compute_reference_windows_and_percentiles_by_windows(
                            moving_site_hc,
                            args.percentiles,
                            (min_age_site, max_age_site),
                            args.window_size,
                            args.window_count,
                            dynamic=args.no_dynamic_window,
                        )
                    )

                    # Plot - Add percentiles computed from reference to joint plot
                    ax = add_reference_percentiles_to_curve(
                        ax,
                        age_site_site,
                        site_percentiles,
                        args.percentiles,
                        args.line_widths,
                        set_color=moving_palette,
                        line_style=moving_linestyle,
                    )

            # Add disease scatterplot to Curve plot
            if args.hide_disease:
                pass
            elif len(args.diseases) > 0:
                logging.info(
                    "Harmonization %s - Display diseases data points.", harmonization
                )
                df_disease = df.query(
                    generate_query(
                        metric,
                        bundle,
                        args.diseases,
                        site=valid_site_list,
                        sex=args.sexes,
                        handedness=args.handednesses,
                        harmonization=harmonization,
                        model=model,
                        age=(min_age, max_age),
                    )
                )
                ax = add_scatterplot_to_curve(
                    ax,
                    df_disease,
                    "age",
                    "mean",
                    "disease",
                    marker="^",
                    alpha=0.8,
                    hue_order=args.diseases,
                    legend="auto",
                    palette=custom_palette[::-1][: len(args.diseases)],
                )

            # Save figure
            # Set prefix to save figure
            prefix = "AgeCurve_{}-{}".format(
                ref_site[0], moving_site[0].replace("_", "")
            )
            # Set suffix to save figure
            suffix = ""
            if args.display_point:
                suffix += "_scatter"
            if args.add_suffix is not None:
                suffix += "_" + args.add_suffix

            # Update aspect and save figure in PNG.
            update_global_figure_style_and_save(
                g,
                ax,
                args,
                parser,
                metric,
                bundle,
                harmonization,
                model,
                suffix_save=suffix,
                prefix_save=prefix,
                title=" \n" + " Age Curve - ",
                outpath=args.out_dir,
                outname=args.outname,
            )


if __name__ == "__main__":
    main()
