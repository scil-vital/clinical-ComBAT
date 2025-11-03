#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fmt: off
"""
Visualizes the distribution of sites before and after harmonization using Boxplot.
Each box corresponding to a site represents the average value of a metric for all
bundles (average of bundles) before and after harmonization.

example:
combat_visualize_sites_boxplot fodf_afd_total.json
combat_visualize_sites_boxplot fodf_afd_total.json --bundles mni_AF_L --diseases 'AD' \
    --rename_reference_site Cam-CAN --meta sample/metaverse.csv
"""

import argparse
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clinical_combat.utils.combatio import load_sites_data
from clinical_combat.visualization.plots import generate_boxplot
from clinical_combat.visualization.viz import (add_manufacturers_to_df,
                                              generate_query,
                                              load_metaverse_results,
                                              viz_identify_valid_sites)

warnings.filterwarnings("ignore")


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("in_files", help="Path to csv sites data.", nargs='+')

    p.add_argument("--out_name",
                   help="Filename prefix used to save PNG figure.")

    reference = p.add_argument_group(title='Options for Reference site')
    reference.add_argument("--reference_site", default="MRC-CBSU_Siemens_3T_2",
                           help="Reference site [%(default)s].")
    reference.add_argument("--rename_reference_site",
                            help="Rename the reference site to this name.")
    reference.add_argument("--meta",
                           help="Metaverse csv file.")
    reference.add_argument("--min_subject_per_site", type=int, default=10,
                           help="Exclude site fewer subjects than min_subject_per_site"
                           "[%(default)s].")
    reference.add_argument("--ages", nargs=2, default=(10, 90),  metavar=('MIN', 'MAX'),
                           help="Range of ages to use min,max [%(default)s].")

    data = p.add_argument_group(title='Options for data selection')
    data.add_argument("--sites", nargs='+',
                      help="List of sites to use [%(default)s].")
    data.add_argument("--bundles", nargs='+',
                      help="List of bundle to use for the sliding window figures [%(default)s].")
    data.add_argument("--sexes", nargs='+',
                      help="List of sex to use [%(default)s].")
    data.add_argument("--handednesses", nargs='+',
                      help="List of handednesses to use [%(default)s].")
    data.add_argument("--diseases", nargs='+',
                      help="List of diseases to use [%(default)s].")

    plot = p.add_argument_group(title='Options for plot visualization')
    plot.add_argument("--y_axis_percentile", nargs=2, default=(1, 99), metavar=('MIN', 'MAX'),
                      help="Range of metric value to use for ymin,ymax in percentile [%(default)s].")
    plot.add_argument("--increase_ylim", nargs=1, default=6,
                      help="Percentage of increase and decrease of y-axis limit [%(default)s].")
    plot.add_argument("--palette", default=sns.color_palette("Spectral"),
                      help="Color palette, the palette can be a color from the Seaborn palette"
                      " (default), or  a list of specific colors. [%(default)s].")

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    palette = args.palette
    df = load_sites_data(args.in_files)

    if args.meta:
        manufacturer_dict = load_metaverse_results(args.meta, args.reference_site)
        df = add_manufacturers_to_df(df, parse_dict=manufacturer_dict,
                                     reference_name=args.rename_reference_site)
    else:
        df = add_manufacturers_to_df(df, reference_site=args.reference_site,
                                     reference_name=args.rename_reference_site)
    df = df.sort_values(["manufacturer", "site"])

    if not len(df.metric.unique()) == 1:
        raise ValueError('The input data contains more then 1 metric.')
    metric = df.metric.unique()[0]

    if args.bundles is None:
        args.bundles = list(np.unique(df.bundle))

    if args.sexes is None:
        args.sexes = list(np.unique(df.sex))

    if args.handednesses is None:
        args.handednesses = list(np.unique(df.handedness))

    if args.sites is None:
        args.sites = list(np.unique(df["site"]))

    if args.reference_site != 'None':
        args.sites = [args.reference_site] + args.sites

    # Check the number of subject per site and filter out sites with too few subjects
    site_list_with_minN = viz_identify_valid_sites(df, args.sites, args.min_subject_per_site,
                                                   metric=metric, bundle=args.bundles[0])

    final_site_list = site_list_with_minN
    min_age, max_age = float(args.ages[0]), float(args.ages[1])

    ### Site Boxplot ###
    sns.set(rc={"figure.figsize": (15, 8)})
    sns.set_style("white")
    for bundle in args.bundles:
        vals = df.query(generate_query(metric, bundle, args.diseases, site=final_site_list,
                                        age=(min_age, max_age), sex=args.sexes,
                                        handedness=args.handednesses))['mean']
        ymin, ymax = np.percentile(vals, [float(args.y_axis_percentile[0]),
                                            float(args.y_axis_percentile[1])])

        ymin = ymin - (ymin * args.increase_ylim / 100)
        ymax = ymax + (ymax * args.increase_ylim / 100)

        if args.reference_site is not None:
            df_ref_bundle = df.query(generate_query(metric, bundle, ['HC'],
                                                    age=(min_age, max_age),
                                                    sex=args.sexes,
                                                    handedness=args.handednesses,
                                                    harmonization='raw',
                                                    model='raw',
                                                    site=args.reference_site)
                                    ).sort_values(["manufacturer", "site"])

        for harmonization, model in product(np.unique(df.harmonization),
                                                 np.unique(df.model)):
            curr_df_sites = df.query(
                generate_query(metric, bundle, ['HC'], site=final_site_list, sex=args.sexes,
                                handedness=args.handednesses, harmonization=harmonization,
                                age=(min_age, max_age), model=model)
            ).sort_values(["manufacturer", "site"])

            if len(curr_df_sites) == 0:
                continue

            curr_df = curr_df_sites
            if args.reference_site is not None:
                curr_df = pd.concat([df_ref_bundle, curr_df_sites])
            else:
                curr_df = curr_df_sites

            xlabels = curr_df["label_site"].unique().tolist()

            g, ax = generate_boxplot(curr_df, "site", "mean", "manufacturer",
                                     (ymin, ymax), metric, bundle, palette=palette,
                                     prefix_title=harmonization.upper(),
                                     title=" Data", x_ticklabels=xlabels)

            if args.diseases:
                for disease in args.diseases:
                    df_disease_site = df.query(generate_query(metric, bundle, disease,
                                                              sex=args.sexes, site=final_site_list,
                                                              handedness=args.handednesses,
                                                              harmonization=harmonization,
                                                              age=(min_age, max_age)))
                    if len(df_disease_site) > 0:
                        sns.stripplot(df_disease_site, x="site", y="mean", hue="disease",
                                        palette='Set2', marker=".", ax=ax, legend=True, s=10)

            # Set the legend properties
            handles, labels = ax.get_legend_handles_labels()
            box_len = len(curr_df.manufacturer.unique())
            legend_box = ax.legend(handles[:box_len], labels[:box_len], loc=2, ncol=1,
                                   frameon=False, borderaxespad=0, title="Manufacturers",
                                   fontsize=12, title_fontsize=14, bbox_to_anchor=(1.01, 1))
            bbox_artist = [legend_box]
            # Add second legend axe for diseases, this requires adding the previous
            if args.diseases:
                legend_data = ax.legend(handles[box_len:], labels[box_len:], loc=2, ncol=1,
                                        borderaxespad=0, frameon=False, title="Diseases",
                                        fontsize=12, title_fontsize=14,
                                        bbox_to_anchor=(1.01, 0.5),)
                ax.add_artist(legend_box)
                bbox_artist.append(legend_data)

            method_name = harmonization
            if harmonization != model:
                method_name += "_" + model

            if args.out_name:
                out_filename = args.out_name + "_" + method_name + "_" \
                    + metric.replace("_", "") + "_" + \
                    bundle.replace("_", "") + ".png"
            else:
                out_filename = "SiteBoxplot_" + method_name  + "_" + \
                    metric.replace("_", "") + "_" + \
                    bundle.replace("_", "") + ".png"

            plt.savefig(out_filename, dpi=300,
                        bbox_inches="tight", bbox_extra_artists=bbox_artist)
            plt.close('all')


if __name__ == "__main__":
    main()
