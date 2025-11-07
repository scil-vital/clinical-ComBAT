#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizes the harmonization model between a reference site and a moving site.
The plot is a scatterplot, with each color representing a
ite (reference = black; moving = blue, default colors)
and its corresponding model (same colors).

# Default usage
combat_visualize_model reference_site.raw.csv.gz moving_site.raw.csv.gz \
                       --models harmonization.model.csv

--------------------------------

Usage examples:
# Set fixed color for reference and moving sites
combat_visualize_model reference_site.raw.csv.gz moving_site.raw.csv.gz \
                       --models harmonization.model.csv \
                       --fixed_color r b

# Show only models
combat_visualize_model reference_site.raw.csv.gz moving_site.raw.csv.gz \
                       --models harmonization.model.csv \
                       --only_models --no_background
"""

import argparse
import logging

import matplotlib
import numpy as np
import pandas as pd

from clinical_combat.harmonization import from_model_filename
from clinical_combat.utils.combatio import load_sites_data
from clinical_combat.utils.scilpy_utils import (add_overwrite_arg,
                                                add_verbose_arg)
from clinical_combat.visualization.plots import (
    add_models_to_plot,
    add_scatterplot_to_curve,
    initiate_joint_marginal_plot,
    scale_color,
    update_global_figure_style_and_save,
)
from clinical_combat.visualization.viz import (custom_palette,
                                               generate_query,
                                               line_style)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument("in_reference",
                   help="Path to CSV for target site (raw data).")
    p.add_argument("in_moving",
                   help="Path to CSV file for moving site.")
    p.add_argument("in_model",
                   help="Path to CSV for harmonization parameter.")
    p.add_argument("--bundles",
                   nargs="+",
                   help="List of bundle to use for figures. "
                        "To plot all bundles use --bundles all. "
                        "['mni_IIT_mask_skeletonFA'].")

    out = p.add_argument_group(title="Options for output figure.")
    out.add_argument("--out_dir",
                     default="./",
                     help="Output directory.[%(default)s]")
    out.add_argument("--outname",
                     help="Filename to save figure. "
                          "['movSite'-'refSite'_'modelName'_model_'metric'_'bundle'.png]")
    out.add_argument("--add_suffix",
                     help="Add suffix to figure title and "
                          " output PNG filename.")

    viz = p.add_argument_group(title="Display options")
    viz.add_argument("--hide_disease",
                     action="store_true",
                     help="Deletes data corresponding to diseases.")
    viz.add_argument("--display_marginal_hist",
                     action="store_true",
                     help="Add marginal histograms to plot.")

    plot = p.add_argument_group(title="Options for plot visualization")
    plot.add_argument("--no_background",
                      action="store_true",
                      help="Save figure with empty background.")
    plot.add_argument("--fixed_ylim",
                      nargs=2,
                      metavar=("MIN", "MAX"),
                      help="Fixed value for y-axis limit.")
    plot.add_argument("--xlim",
                      nargs=2,
                      default=(20, 90),
                      metavar=("MIN", "MAX"),
                      help="X-axis limit, usually range of ages "
                           " to use min, max [%(default)s].")
    plot.add_argument("--fixed_color",
                      nargs=2,
                      metavar=("REFERENCE", "MOVING"),
                      help="Use to set color for each site.")

    line = p.add_argument_group(title="Options for regression line plot")
    line.add_argument("--only_models",
                      action="store_true",
                      help="Show only regression line on the plot.")
    line.add_argument("--line_width",
                      type=float,
                      default=2.5,
                      help="Width of regression lines from models.")
    line.add_argument("--lightness",
                      type=float,
                      default=1,
                      help="Use to scale the color; "
                           " <1 to darker and >1 to lighter.")
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    in_files = [args.in_reference, args.in_moving]

    df_target = load_sites_data([args.in_reference])
    df_moving = load_sites_data([args.in_moving])
    df = pd.concat([df_target, df_moving]).reset_index(drop=True)

    df = load_sites_data(in_files)

    if not len(df.metric.unique()) == 1:
        raise ValueError("The input data contains more then 1 metric.")
    metric = df.metric.unique()[0]

    all_disease = list(df.disease.unique())
    if "HC" in all_disease:
        all_disease.remove("HC")

    all_bundles = np.intersect1d(df_target.bundle.unique(),
                                 df_moving.bundle.unique())
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
        logging.warning("No valid input bundle. "
                        "Selecting bundle %s", args.bundles)

    # Set colors for reference and moving sites
    curr_palette = custom_palette[:2]
    if args.fixed_color is not None:
        ref_color, moving_color = args.fixed_color[0], args.fixed_color[1]
        curr_palette = [ref_color, moving_color]
    else:
        ref_color = matplotlib.colors.ColorConverter.to_rgb(curr_palette[0])
        ref_color = scale_color(ref_color, args.lightness)
        moving_color = matplotlib.colors.ColorConverter.to_rgb(curr_palette[1])
        moving_color = scale_color(moving_color, args.lightness)

    # Load model parameters
    QC = from_model_filename(args.in_model)
    mov_site = QC.model_params["mov_site"]
    ref_site = QC.model_params["ref_site"]

    if ref_site != df.site.unique()[0]:
        raise ValueError("Model site and reference data site don't match.")
    if mov_site != df.site.unique()[1]:
        raise ValueError("Model site and moving data site don't match.")

    # Generate plots for each bundle
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
            hist_hur_order=df_vals.input.unique().tolist(),
            hist_palette=curr_palette,
        )

        # Add data to build figure frame (white)
        ax = add_scatterplot_to_curve(
            ax,
            df_vals,
            "age",
            "mean",
            "input",
            hue_order=df_vals.input.unique().tolist(),
            alpha=0,
            palette=curr_palette,
            legend=False,
        )

        # Add regression line from models to the plot
        age_min, age_max = df_vals.age.min(), df_vals.age.max()

        # Reference regression curve
        ax = add_models_to_plot(
            ax,
            QC,
            bundle,
            age_min,
            age_max,
            moving_site=False,
            color=ref_color,
            lightness=args.lightness,
            line_width=args.line_width,
            line_style=line_style[0],
        )

        # Moving regression curve
        ax = add_models_to_plot(
            ax,
            QC,
            bundle,
            age_min,
            age_max,
            moving_site=True,
            color=moving_color,
            lightness=args.lightness,
            line_width=args.line_width,
            line_style=line_style[0],
        )

        # Add scatter point and regression line from models
        if not args.only_models:
            # Add scatter point to plot
            ax = add_scatterplot_to_curve(
                ax,
                df_vals,
                "age",
                "mean",
                "input",
                hue_order=df_vals.input.unique().tolist(),
                alpha=0.8,
                palette=curr_palette,
                legend=False,
            )

            if not args.hide_disease:
                # Add point corresponding to disease data
                df_disease = df.query(generate_query(metric, bundle,
                                                     all_disease))
                ax = add_scatterplot_to_curve(
                    ax,
                    df_disease,
                    "age",
                    "mean",
                    "disease",
                    marker="^",
                    alpha=0.8,
                    hue_order=all_disease,
                    legend="auto",
                    palette=custom_palette[::-1][: len(all_disease)],
                )

        # Add legend to the plot
        handle, labels = ax.get_legend_handles_labels()
        ax.legend(handle, [ref_site, mov_site])

        # Save figure
        # Set prefix to save figure
        prefix = "DataModels_{}-{}".format(
            ref_site.replace("_", ""), mov_site.replace("_", "")
        )

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
            QC.model_params["name"],
            QC.model_params["name"],
            suffix_save=suffix,
            prefix_save=prefix,
            empty_background=args.no_background,
            legend_title="Models - Sites",
            title=" \n" + " Data Models - ",
            outpath=args.out_dir,
            outname=args.outname,
        )


if __name__ == "__main__":
    main()
