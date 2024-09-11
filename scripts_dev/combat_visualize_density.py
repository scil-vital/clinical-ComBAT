#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fmt: off
"""
Visualize density plot (KDE) per site for all bundles (HC).

example:
combat_visualize_density.py adni_md.csv --bundles mni_AF_L mni_AF_R
"""
import argparse
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns

from clinical_combat.utils.combatio import load_sites_data
from clinical_combat.visualization.plots import (
    add_kde_to_joinplot, add_scatterplot_to_curve,
    initiate_joint_marginal_plot, update_global_figure_style_and_save)
from clinical_combat.visualization.viz import (custom_palette, generate_query,
                                              viz_identify_valid_sites)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("in_files", help="Path to csv sites data.", nargs='+')

    reference = p.add_argument_group(title='Options for Reference site')
    reference.add_argument("--reference_site",
                           default="MRC-CBSU_Siemens_3T_2",
                           help="Reference site [%(default)s].")
    reference.add_argument("--min_subject_per_site", type=int, default=10,
                           help="Exclude site fewer subjects than "
                           "min_subject_per_site [%(default)s].")
    reference.add_argument("--ages", nargs=2, default=(10, 90),
                           metavar=('MIN', 'MAX'),
                           help="Range of ages to use min,max [%(default)s].")

    data = p.add_argument_group(title='Options for data selection')
    data.add_argument("--sites", nargs='+',
                      help="List of sites to use [%(default)s].")
    data.add_argument("--bundles", nargs='+',
                      help="List of bundle to use for figures [%(default)s].")
    data.add_argument("--sexes", nargs='+',
                      help="List of sex to use [%(default)s].")
    data.add_argument("--handednesses", nargs='+',
                      help="List of handednesses to use [%(default)s].")
    data.add_argument("--diseases", nargs='+',
                      help="List of diseases to use [%(default)s].")

    plot = p.add_argument_group(title='Options for plot visualization')
    plot.add_argument("--y_axis_percentile", nargs=2, default=(0, 100),
                      metavar=('MIN', 'MAX'),
                      help="Range of metric value to use for ymin, "
                      "ymax in percentile [%(default)s].")
    plot.add_argument("--show_marginal_hist", action="store_true",
                      help="Add marginal histograms to plot.")
    plot.add_argument("--increase_ylim", nargs=1, type=int, default=1,
                      help="Percentage of increase and decrease of y-axis"
                      " limit [%(default)s].")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = load_sites_data(args.in_files)

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
        args.sites = list(np.unique(df["site"]))

    site_list_with_minN = viz_identify_valid_sites(df, args.sites, args.min_subject_per_site,
                                                   metric=metric, bundle=args.bundles)

    nbr_ref_subs = df.query(generate_query(metric, args.bundles, ['HC'],
                                           site=args.reference_site, harmonization='raw',
                                           model='raw', sex=args.sexes,
                                           handedness=args.handednesses)).shape[0]

    if nbr_ref_subs > 0:
        final_site_list = site_list_with_minN
        final_site_list.remove(args.reference_site)
        final_site_list = [args.reference_site] + final_site_list
        palette = custom_palette
    else:
        print("No subject found for reference site " + args.reference_site)
        final_site_list = site_list_with_minN
        palette = custom_palette[1:]
        args.reference_site = None

    # Set age and percentile limits
    min_age, max_age = float(args.ages[0]), float(args.ages[1])
    y_min_percentile, y_max_percentile = float(
        args.y_axis_percentile[0]), float(args.y_axis_percentile[1])

    ### KDE Plot ###
    for bundle in args.bundles:
        print("\nProcessing: ", bundle)
        df_vals = df.query(generate_query(metric, bundle, args.diseases,
                                          site=final_site_list,
                                          age=(min_age, max_age),
                                          sex=args.sexes,
                                          handedness=args.handednesses))
        # Set y limit for plot
        ymin, ymax = np.percentile(df_vals["mean"],
                                   [y_min_percentile, y_max_percentile])
        ymin, ymax = ymin - (ymin * args.increase_ylim /
                             100), ymax + (ymax * args.increase_ylim/100)

        kdeplot_query = generate_query(metric, bundle, ['HC'],
                                       site=final_site_list,
                                       sex=args.sexes,
                                       handedness=args.handednesses)

        if args.reference_site is not None:
            df_ref_bundle = df.query(generate_query(metric, bundle, ['HC'],
                                                    age=(min_age, max_age),
                                                    sex=args.sexes,
                                                    handedness=args.handednesses,
                                                    harmonization='raw',
                                                    model='raw',
                                                    site=args.reference_site))

        for harmonization, model in product(np.unique(df.harmonization),
                                                 np.unique(df.model)):

            df_kde = df.query(kdeplot_query +
                     " & harmonization == @harmonization \
                         & model == @model")
            if len(df_kde) == 0:
                continue
            if args.reference_site is not None:
                df_kde = pd.concat([df_kde, df_ref_bundle])
                df_kde.drop_duplicates(inplace=True)
            g, ax = initiate_joint_marginal_plot(
                df_kde,
                "age", "mean", "site", (ymin, ymax),
                legend_title='Sites', marginal_hist=args.show_marginal_hist,
                hist_hur_order=final_site_list,
                hist_palette=palette[: len(final_site_list)])

            g.plot_joint(sns.kdeplot, fill=False, alpha=0.33,
                         common_norm=False, legend=True,
                         palette=palette[:len(final_site_list)],
                         hue_order=final_site_list)

            ax = g.ax_joint
            ax = add_kde_to_joinplot(
                ax, pd.concat([df_kde]),
                "age", "mean", "site",
                palette=palette[:len(final_site_list)], fill=True,
                hue_order=final_site_list, alpha=0.5)

            if args.reference_site is not None:
                ax = add_kde_to_joinplot(ax, df_ref_bundle,
                                        "age", "mean", "site",
                                        palette=palette[:1],
                                        alpha=0.3)

            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.25, 1),
                            borderaxespad=0,
                            frameon=False, title="Sites")
            g.ax_marg_y.legend([], [], frameon=False)

            ### disease ###
            # Add disease datapoints to density plot
            if args.diseases:
                df_disease = df.query(generate_query(metric, bundle, args.diseases,
                                                     site=final_site_list,
                                                     age=(min_age, max_age),
                                                     sex=args.sexes,
                                                     handedness=args.handednesses,
                                                     harmonization=harmonization,
                                                     model=model))
                ax = add_scatterplot_to_curve(ax, df_disease, "age", "mean",
                                              "disease",
                                              palette=palette[::-1][: len(
                                                  args.diseases)], marker=".",
                                              hue_order=args.diseases,
                                              alpha=0.8, legend=True)
            suffix = ''
            if args.diseases:
                suffix = "_diseases"

            update_global_figure_style_and_save(g, ax, metric, bundle,
                                                harmonization, model,
                                                prefix_save="Density",
                                                suffix_save=suffix,
                                                title=" Data\n Density - ",
                                                move_legend=True)


if __name__ == '__main__':
    main()
