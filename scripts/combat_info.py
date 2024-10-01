#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print info on the data available in the CSV file.
"""

import argparse

import numpy as np
import pandas as pd

from clinical_combat.utils.combatio import load_sites_data


def count_by_category(df, column_name, bundles, metrics):
    """
    Function to count the number of elements corresponding to a column
    in the dataframe.

    Args:
        df (pd.DataFrame): CSV file with data
        column_name (str): Column name to count
        bundles (list): List of bundles
        metrics (str): Metrics name

    Returns:
        txt_category (str): Text with the count of each category
    """

    if df[column_name].dtype == int:
        df[column_name] = df[column_name].astype(str)
    category_id = list(np.unique(df[column_name]))
    txt_category = ""
    for category in category_id:
        qry = "bundle == @bundles[0] & metric == @metrics & {} == '{}'".format(
            column_name, category
        )
        count = df.query(qry).shape[0]
        txt_category += str(category) + "(n=" + str(count) + "), "
    if txt_category[-2:] == ", ":
        txt_category = txt_category[:-2]
    return txt_category


def get_age_range(df, column_name, bundles, metrics):
    """
    Function to get the age range for each category in the dataframe.

    Args:
        df (pd.DataFrame): CSV file with data
        column_name (str): Column name to count
        bundles (list): List of bundles
        metrics (str): Metrics name

    Returns:
        txt_category (str): Text with the age range for each category
    """
    category_id = list(np.unique(df[column_name]))
    txt_category = ""
    for category in category_id:
        qry = "bundle == @bundles[0] & metric == @metrics & {} == '{}'".format(
            column_name, category
        )
        df_age = df.query(qry)
        txt_category += (
            str(category)
            + "(age="
            + str(df_age["age"].min())
            + "-"
            + str(df_age["age"].max())
            + "), "
        )
    if txt_category[-2:] == ", ":
        txt_category = txt_category[:-2]
    return txt_category


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_file", help="CSV site data file.")

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load dataframe
    df = load_sites_data([args.in_file])

    # Extract the unique values for each column
    site = df["site"].unique()[0]
    bundles = list(np.unique(df["bundle"]))
    sexes = list(np.unique(df["sex"]))
    handedness = list(np.unique(df["handedness"]))
    disease = list(np.unique(df["disease"]))
    metrics = df["metric"].unique()[0]
    sid_unique = df.sid.unique().tolist()

    # Compute global Infos
    nbr_subjects = df.query("bundle == @bundles[0] & metric == @metrics").shape[0]
    df["age"] = df["age"].astype(int)
    gl_age_text = str(df["age"].min()) + " - " + str(df["age"].max())

    # print generic infos
    print("\n   Sites: " + str(site).replace(" ", ""))
    print("   Metrics: " + str(metrics).replace(" ", ""))
    print("   Mean Age: " + str(np.round(df["age"].mean(), 2)) + "\n")

    # Merge the global info lists in a single variable and set column names
    global_pop = [nbr_subjects, gl_age_text, disease, handedness, sexes]
    varname_global = [
        "Number of Subject",
        "Age Range",
        "Disease",
        "Handedness",
        "Sexes",
    ]

    # Details for disease, sex, hand
    diseases_str = count_by_category(df, "disease", bundles, metrics)
    txt_disease_age = get_age_range(df, "disease", bundles, metrics)
    txt_sex = count_by_category(df, "sex", bundles, metrics)
    txt_hand = count_by_category(df, "handedness", bundles, metrics)
    # Merge the three detail info lists in a single variable and set column names
    details_pop = [len(sid_unique), txt_disease_age, diseases_str, txt_sex, txt_hand]
    varname_detail = [
        "N Unique Sid",
        "Age by",
        "Disease (n)",
        "Handedness (n)",
        "Sexes (n)",
    ]

    # Create the dataframe with the results
    df = pd.DataFrame([varname_global, global_pop, varname_detail, details_pop]).T
    idx_col = "Site : {}".format(site)
    df.columns = [idx_col, "GlobalInfos", "Categories", "DetailInfos"]
    df = df.set_index(idx_col)

    # Split bundles list in Left, Right and bilateral bundle.
    left_bundles = [
        bdl_l for bdl_l in bundles if any(left in bdl_l for left in ["right_", "_R"])
    ]
    left_bundles.sort()
    right_bundles = [
        bdl_r for bdl_r in bundles if any(right in bdl_r for right in ["left_", "_L"])
    ]
    right_bundles.sort()
    bilateral = [
        bdl_bi
        for bdl_bi in bundles
        if not any(bilat in bdl_bi for bilat in ["left_", "_L", "right_", "_R"])
    ]
    bilateral.sort()
    # Merge the three bundle lists in a single dataframe
    tmp_unilateral = pd.DataFrame(left_bundles, right_bundles).reset_index()
    tmp_bilateral = pd.DataFrame(bilateral)
    df_bundles_list = pd.concat([tmp_unilateral, tmp_bilateral], axis=1).fillna("")
    df_bundles_list.columns = ["Left_bundles", "Right_bundles", "Bilaretal_bundles"]

    # Print resulting dataframes
    print(df)
    print("\nNumber of Bundles: " + str(len(bundles)))
    print(df_bundles_list)


if __name__ == "__main__":
    main()
