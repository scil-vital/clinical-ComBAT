#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize mean value per site/manufacturer.

example:
combat_json_viz_manufacturer ../merged_jsons/fodf_afd_total.json
updated_metaverse.csv
--min_subject_per_site 5 --bundle mni_IIT_mask_skeletonFA --ages 50,100

"""

import argparse
import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from clinical_combat.utils.combatio import load_results_json


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_json", help="DWI Nifti image.")
    p.add_argument("in_csv", help="metaverse csv file")

    p.add_argument(
        "--ref",
        type=str,
        default="MRC-CBSU_Siemens_3T_2",
        help="Reference site [%(default)s].",
    )
    p.add_argument(
        "--sites", type=str, default="all", help="List of sites to use [%(default)s]."
    )
    p.add_argument(
        "--min_subject_per_site",
        type=int,
        default=20,
        help="Exclude site fewer subjects than min_subject_per_site" + "[%(default)s].",
    )
    p.add_argument(
        "--bundles",
        type=str,
        default="all",
        help="List of bundle to use [%(default)s].",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="List of metric to use [%(default)s].",
    )
    p.add_argument(
        "--sexes", type=str, default="all", help="List of sex to use [%(default)s]."
    )
    p.add_argument(
        "--handednesses",
        type=str,
        default="all",
        help="List of handednesses to use [%(default)s].",
    )
    p.add_argument(
        "--fieldStrength",
        type=str,
        default="all",
        help="List of field_strength to use [%(default)s].",
    )
    p.add_argument(
        "--multishell",
        type=str,
        default="all",
        help=" True, False or all [%(default)s].",
    )
    p.add_argument(
        "--diseases",
        type=str,
        default="all",
        help="List of diseases to use [%(default)s].",
    )
    p.add_argument(
        "--ages",
        type=str,
        default="0,100",
        help="Range of ages to use min,max [%(default)s].",
    )

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    palette = [
        "#000000",
        "#0000FF",
        "#00FF00",
        "#FF0000",
        "#FFFF00",
        "#00FFFF",
        "#FF00FF",
        "#00BF32",
        "#BF9430",
        "#2C3D82",
        "#248F40",
        "#A61300",
        "#A67400",
        "#071C71",
        "#007C21",
        "#FF5640",
        "#FFC640",
        "#4965D6",
        "#38DF64",
        "#FF8373",
        "#FFD573",
        "#6F83D6",
        "#64DF85",
        "#FF5600",
        "#FF7C00",
        "#04859D",
        "#00AA72",
        "#BF4030",
        "#BF6030",
        "#BF7630",
        "#206876",
        "#207F60",
        "#1533AD" "#A63800",
        "#A65100",
        "#015666",
        "#006E4A",
        "#FF8040",
        "#FF9D40",
        "#37B6CE",
        "#35D4A0",
        "#FFA273",
        "#FFB773",
        "#5FBDCE",
    ]

    df = load_results_json(args.in_json)

    csv = np.loadtxt(args.in_csv, dtype="str", skiprows=1, delimiter=",")

    # manufacturer_dict = {"MRC-CBSU_Siemens_3T_2": ["*Ref-Siemens"]}
    # for s in csv:
    #    if "ADNI" in s[1] and "ADNIDOD" not in s[1]:
    #        if s[12].isnumeric():
    #            ss = str(int(s[12]))
    #        else:
    #            ss = s[12]
    #        if not s[13] == "":
    #            r = manufacturer_dict.get(ss, [])
    #            r.append(s[13])
    #            manufacturer_dict[ss] = list(np.unique(r))

    manufacturer_dict = {"MRC-CBSU_Siemens_3T_2": "*Ref-Siemens"}
    for s in csv:
        if s[12].isnumeric():
            ss = str(int(s[12]))
        else:
            ss = s[12]
        if s[13] == "":
            mm = "unknown"
        else:
            mm = s[13]
        manufacturer_dict[ss] = mm

    sites = np.array(df["site"])
    manufacturer = []
    for s in sites:
        if s in manufacturer_dict.keys():
            manufacturer.append(manufacturer_dict[s])
        elif "philips" in s.lower():
            manufacturer.append("Philips")
        elif "siemens" in s.lower():
            manufacturer.append("Siemens")
        elif "ge" in s.lower():
            manufacturer.append("GE")
        else:
            manufacturer.append("unknown")

    # manufacturer = [manufacturer_dict[s] for s in sites]

    # manufacturer = [manufacturer_dict[s] for s in sites]

    manufacturer_list = np.unique(manufacturer)
    manufacturer_list = np.delete(
        manufacturer_list, np.where(manufacturer_list == "*Ref-Siemens")
    )
    manufacturer_list = np.delete(
        manufacturer_list, np.where(manufacturer_list == "unknown")
    )
    # manufacturer_list = np.delete(
    #    manufacturer_list, np.where(manufacturer_list == "Philips"))
    all_manufacturer_list = np.array(["*Ref-Siemens"] + list(manufacturer_list))

    df.insert(1, "manufacturer", manufacturer, True)

    harmonization_list = np.unique(df["harmonization"])

    if args.metrics == "all":
        metric_list = list(np.unique(df["metric"]))
    else:
        metric_list = list(args.metrics.split(","))

    if args.bundles == "all":
        bundle_list = list(np.unique(df["bundle"]))
    else:
        bundle_list = list(args.bundles.split(","))

    if args.sexes == "all":
        sex_list = list(np.unique(df["sex"]))
    else:
        sex_list = list(args.sexes.split(","))

    min_age = float(args.ages.split(",")[0])
    max_age = float(args.ages.split(",")[1])

    if args.handednesses == "all":
        handedness_list = list(np.unique(df["handedness"]))
    else:
        handedness_list = list(args.handednesses.split(","))

    if args.fieldStrength == "all":
        field_strength_list = list(np.unique(df["field_strength"]))
    else:
        field_strength_list = list(args.fieldStrength.split(","))

    if args.multishell == "all":
        multishell_list = list(np.unique(df["multi-shell"]))
    else:
        multishell_list = list(args.multishell.split(","))

    if args.diseases == "all":
        all_disease_list = list(np.unique(df["disease"]))
    else:
        all_disease_list = list(args.diseases.split(","))
    if "HC" in all_disease_list:
        disease_list = all_disease_list
        disease_list.remove("HC")

    if args.sites == "all":
        site_list_tmp = list(np.unique(df["site"]))
        # for s in ["3", "7", "16", "33", "52", "94", "109", "128", "131"]:
        #    site_list_tmp.remove(s)
    else:
        site_list_tmp = list(args.sites.split(","))

    site_list = []
    nbr_subs = dict()
    for s in site_list_tmp:
        nbr_subjects = df.query(
            "site == @s \
                                & harmonization == 'pre' \
                                & bundle == @bundle_list[0] \
                                & metric == @metric_list[0] \
                                & disease == 'HC' \
                                & age >= @min_age \
                                & age <= @max_age \
                                & field_strength in @field_strength_list \
                                & `multi-shell` in @multishell_list \
                                & handedness in @handedness_list"
        ).shape[0]
        nbr_subs[s] = nbr_subjects
        if nbr_subjects >= args.min_subject_per_site:
            site_list.append(s)

    if args.ref in site_list:
        site_list.remove(args.ref)
    all_site_list = [args.ref] + site_list

    ############
    ### SITE ###
    ############
    sns.set(rc={"figure.figsize": (20, 10)})
    for metric in metric_list:
        for bundle in bundle_list:
            means = df.query(
                "metric == @metric \
                & bundle == @bundle \
                & site in @all_site_list  \
                & age >= @min_age \
                & age <= @max_age \
                & sex in @sex_list \
                & disease == @all_disease_list \
                & handedness in @handedness_list"
            )["mean"]
            ymax = np.nan_to_num(means.max()) * 1.1
            ymin = np.nan_to_num(means.min()) / 1.1
            for h in np.unique(df["harmonization"]):
                query = "metric == @metric \
                       & bundle == @bundle \
                       & disease == 'HC' \
                       & sex in @sex_list \
                       & site in @all_site_list \
                       & handedness in @handedness_list \
                       & harmonization == @h \
                       & age >= @min_age \
                       & age <= @max_age"
                figure()
                ax = sns.boxplot(
                    df.query(query).sort_values(["manufacturer", "site"]),
                    x="site",
                    y="mean",
                    hue="manufacturer",
                    dodge=False,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_ylim(0, ymax)
                ax.set_ylabel("Mean " + metric.upper(), fontsize=24)
                ax.set_xlabel("SITES [#sub]", fontsize=24)
                ax.set_title(
                    bundle.replace("mni_", "").upper() + " - " + h.upper(), fontsize=30
                )
                ax.tick_params(axis="both", which="major", labelsize=18)

                labels = [item.get_text() for item in ax.get_xticklabels()]
                ax.set_xlim(-0.8, len(labels))
                query = "metric == @metric \
                       & bundle == @bundle \
                       & disease == @d \
                       & sex in @sex_list \
                       & site in @s \
                       & handedness in @handedness_list \
                       & harmonization == @h \
                       & age >= @min_age \
                       & age <= @max_age"

                for j, d in enumerate(disease_list):
                    scatter([-1], [-1], color=palette[-(j + 1)], marker=".", label=d)
                    for i, s in enumerate(labels):
                        df_ = df.query(query)
                        if len(df_) > 0:
                            scatter(
                                [i] * len(df_),
                                df_["mean"],
                                color=palette[-(j + 1)],
                                marker=".",
                            )
                legend()

                new_labels = [l + "   [" + str(nbr_subs[l]) + "]" for l in labels]
                new_labels[0] = new_labels[0].replace("MRC-CBSU_Siemens_3T_2", "CamCan")
                ax.set_xticklabels(new_labels)

                savefig(
                    "metric-"
                    + metric.replace("_", "")
                    + "_harmonization-"
                    + h
                    + "_bundle-"
                    + bundle.replace("mni_", "")
                    + "_sites_boxplot.png",
                    dpi=300,
                    bbox_inches="tight",
                )

                # import pdb
                # pdb.set_trace()


if __name__ == "__main__":
    main()
