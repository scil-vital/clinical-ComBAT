#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizes with scatterplot a list of raw or harmonized CSV files.

# Default usage
combat_visualize_data.py reference_site.raw.csv.gz moving_site1.raw.csv.gz moving_site2.raw.csv.gz
                moving_siteN.raw.csv.gz

# To display all bundles
combat_visualize_data.py reference_site.raw.csv.gz moving_site1.raw.csv.gz moving_site2.raw.csv.gz
                         moving_siteN.raw.csv.gz --bundles all

# To display only set of bundles
combat_visualize_data.py reference_site.raw.csv.gz moving_site1.raw.csv.gz moving_site2.raw.csv.gz
                            moving_siteN.raw.csv.gz --bundles mni_AF_L mni_AF_R
"""

import argparse
import logging
import os

from clinical_combat.utils.combatio import load_sites_data
from clinical_combat.utils.scilpy_utils import add_overwrite_arg, add_verbose_arg
from clinical_combat.visualization.plots import (
    add_scatterplot_to_curve,
    initiate_joint_marginal_plot,
    update_global_figure_style_and_save,
)
from clinical_combat.visualization.viz import custom_palette, markers_style


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument("in_files", help="Path to csv sites data.", nargs="+")

    p.add_argument(
        "--bundles",
        nargs="+",
        help="List of bundle to use for figures. To plot all bundles use "
        "--bundles all. ['mni_IIT_mask_skeletonFA'].",
    )

    out = p.add_argument_group(title="Options for output figure.")
    out.add_argument("--out_dir", default="./", help="Output directory.[%(default)s]")
    out.add_argument("--outname", help="Filename to save figure.")
    out.add_argument(
        "--add_suffix", help="Add suffix to figure title and output PNG filename."
    )

    viz = p.add_argument_group(title="Display options")
    viz.add_argument(
        "--hide_disease",
        action="store_true",
        help="Deletes data corresponding to diseases.",
    )
    viz.add_argument(
        "--display_marginal_hist",
        action="store_true",
        help="Add marginal histograms to plot.",
    )

    plot = p.add_argument_group(title="Options for plot visualization")
    plot.add_argument(
        "--no_background",
        action="store_true",
        help="Save figure with empty background.",
    )
    plot.add_argument(
        "--fixed_ylim",
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Fixed value for y-axis limit.",
    )
    plot.add_argument(
        "--xlim",
        nargs=2,
        default=(20, 90),
        metavar=("MIN", "MAX"),
        help="X-axis limit, usually range of ages to use min, max [%(default)s].",
    )

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Load data from CSV files and set basename from file paths
    df = load_sites_data(args.in_files)
    file_basenames = [os.path.basename(file) for file in args.in_files]

    if not len(df.metric.unique()) == 1:
        raise ValueError("The input data contains more then 1 metric.")
    metric = df.metric.unique()[0]

    all_disease = list(df.disease.unique())
    if "HC" in all_disease:
        all_disease.remove("HC")
    all_dataset = df.input.unique()

    # set disease and sites color
    disease_palette = custom_palette[0]
    sites_palette = custom_palette[1:]

    all_bundles = list(df.bundle.unique())
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

    for bundle in args.bundles:
        logging.info("Processing: %s", bundle)
        df_vals = df.query("bundle == @bundle & disease == 'HC'")

        if args.fixed_ylim:
            ymin, ymax = float(args.fixed_ylim[0]), float(args.fixed_ylim[1])
        else:
            ymin = df_vals["mean"].min() - df_vals["mean"].min() * 0.05
            ymax = df_vals["mean"].max() + df_vals["mean"].max() * 0.05

        # Initialize the joint plot
        g, ax = initiate_joint_marginal_plot(
            df_vals,
            "age",
            "mean",
            "input",
            (ymin, ymax),
            legend_title="Dataset",
            marginal_hist=args.display_marginal_hist,
            xlim=(args.xlim[0], args.xlim[1]),
            hist_hur_order=file_basenames,
            hist_palette=sites_palette[0 : len(all_dataset) + 1],
        )

        # Add data to build figure frame (white)
        ax = add_scatterplot_to_curve(
            ax,
            df_vals,
            "age",
            "mean",
            "input",
            hue_order=args.in_files[0],
            alpha=0,
            palette=["#FFFFFF"],
            legend=False,
        )

        # Add scatter point to plot
        ax = add_scatterplot_to_curve(
            ax,
            df_vals,
            "age",
            "mean",
            "input",
            hue_order=file_basenames,
            alpha=0.8,
            palette=sites_palette[0 : len(all_dataset) + 1],
        )

        if not args.hide_disease:
            # Add the legend
            for idx_disease, disease in enumerate(all_disease):
                df_disease = df.query("bundle == @bundle & disease == @disease")
                if len(df_disease) > 0:
                    ax = add_scatterplot_to_curve(
                        ax,
                        df_disease,
                        "age",
                        "mean",
                        "disease",
                        marker=markers_style[idx_disease],
                        alpha=0.8,
                        palette=[disease_palette],
                        legend=True,
                    )

            # Add point corresponding to disease data
            for idx_disease, disease in enumerate(all_disease):
                for idx, d in enumerate(all_dataset):
                    df_disease = df.query(
                        "bundle == @bundle & disease == @disease & input == @d"
                    )
                    if len(df_disease) > 0:
                        ax = add_scatterplot_to_curve(
                            ax,
                            df_disease,
                            "age",
                            "mean",
                            "disease",
                            marker=markers_style[idx_disease],
                            alpha=0.8,
                            palette=sites_palette[idx : idx + 1],
                            legend=False,
                        )

        # Save figure
        suffix = ""
        prefix = "Dataset_{}-sites".format(len(args.in_files))
        if args.add_suffix is None:
            args.add_suffix = ""
        if len(args.add_suffix) > 0:
            prefix += "_" + args.add_suffix

        suffix = ""
        if args.add_suffix is None:
            args.add_suffix = ""

        # Update aspect and save figure in PNG.
        update_global_figure_style_and_save(
            g,
            ax,
            args,
            parser,
            metric,
            bundle,
            "",
            "all",
            suffix_save=suffix,
            prefix_save=prefix,
            empty_background=args.no_background,
            title=" Dataset\n" + args.add_suffix,
            outpath=args.out_dir,
            outname=args.outname,
        )


if __name__ == "__main__":
    main()
