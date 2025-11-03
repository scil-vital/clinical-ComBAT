#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataframe computation and manipulation functions for viewing harmonization and raw data.

Includes 3 dictionaries: color palette, linestyle palette and marker palette.
Also includes a set of functions for manipulating dataframes (pandas) and computing a
number of elements used by plots functions (plots.py).

"""
import logging

import numpy as np
import pandas as pd

custom_palette = [
    "#000000",
    "#de3838",
    "#5ecbf3",
    "#6ce66c",
    "#cc971b",
    "#9400D3",
    "#a5bc0e",
    "#39618f",
    "#31984d",
    "#b75c50",
    "#c08f07",
    "#5f6ca7",
    "#3CB371",
    "#FF5640",
    "#FFC640",
    "#4965D6",
    "#00FF7F",
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
    "#5F9EA0",
    "#20B2AA",
    "#193dcc",
    "#A63800",
    "#A65100",
    "#015666",
    "#006E4A",
    "#FF8040",
    "#37B6CE",
    "#556B2F",
    "#FFB773",
    "#6A5ACD",
] * 10


line_style = [
    "-",
    "--",
    "-.",
    ":",
    (0, (1, 10)),
    (0, (1, 1)),
    (5, (10, 3)),
    (0, (5, 10)),
    (0, (5, 1)),
    (0, (5, 5)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 1, 5, 1, 5)),
] * 10

markers_style = ["^", "*", "s", "p", "1", "2", "3", "4"]


def add_manufacturers_to_df(
    df, reference_site=None, reference_name=None, parse_dict=None
):
    """
    Add manufacturer meta data to dataframe.
    Args:
        df (pd.DataFrame): dataframe.
        reference_site (str, optional): reference site name.
        reference_name (str, optional): Custom name for reference site.
        parse_dict (dict): dictionary of data.
                {keys: "site", value: "manufacturer_dict"}
    Returns:
        pd.DataFrame: dataframe with manufacturer and site meta data.
    """
    manufacturers, sites = [], []
    for curr_site in df["site"]:
        if parse_dict is not None and curr_site in parse_dict.keys():
            manufacturers.append(parse_dict[curr_site])
            if reference_name is not None:
                sites.append(reference_name)
            elif (
                "mrc-cbsu_siemens_3t_2" in curr_site.lower()
                or "camcan" in curr_site.lower()
            ):
                manufacturers.append("Siemens")
                sites.append(curr_site)
            else:
                sites.append(curr_site.replace("_" + parse_dict[curr_site], ""))

        elif reference_site is not None and curr_site == reference_site:
            manufacturers.append("Reference")
            if reference_name is not None:
                curr_site = reference_name
            sites.append(curr_site)
        elif "philips" in curr_site.lower():
            manufacturers.append("Philips")
            sites.append(curr_site.lower().replace("_philips", "").replace("3t", "3T"))
        elif "siemens" in curr_site.lower():
            manufacturers.append("Siemens")
            sites.append(curr_site.lower().replace("_siemens", "").replace("3t", "3T"))
        elif "ge" in curr_site.lower():
            manufacturers.append("GE")
            sites.append(curr_site.lower().replace("_ge", "").replace("3t", "3T"))
        else:
            manufacturers.append("unknown")
            sites.append(curr_site)

    df.insert(1, "manufacturer", manufacturers, True)
    df.insert(1, "label_site", sites, True)

    return df


def load_metaverse_results(in_file, reference_site):
    """
    Load metaverse data and set manufacturer meta data.
    Args:
        in_file (str): input file.
        reference_site (str): reference site. Usually MRC-CBSU_Siemens_3T_2..
    Returns:
        dict: dictionary of data. {keys: "site", value: "manufacturer_dict"}
    """
    manufacturer_dict = {reference_site: "Reference"}
    csv = np.loadtxt(in_file, dtype="str", skiprows=1, delimiter=",")
    for line in csv:
        if line[13] == "":
            manufacturer = "unknown"
        else:
            manufacturer = line[13]
        manufacturer_dict[line[15]] = manufacturer
    return manufacturer_dict


def convert_matrix_to_dataframe(
    matrix, sites_list, bundle, harmonization, column_title="mean"
):
    """
    Convert matrix to dataframe.
    Args:
        matrix (np.array): matrix.
        sites_list (list): list of sites.
        bundle (str): bundle.
        harmonization (str): harmonization.
        column_title (list, optional): column title. Defaults is ['mean'].
    Returns:
        pd.DataFrame: dataframe
    """
    df = pd.DataFrame(matrix, columns=[column_title])
    df["site"] = sites_list
    df["harmonization"] = harmonization
    df["bundle"] = bundle
    return df


def generate_query(
    metric,
    bundle,
    disease,
    site=None,
    age=None,
    harmonization=None,
    sex=None,
    handedness=None,
    model=None,
):
    """
    Generate query for dataframe.
    Args:
        metric (str or list): metric.
        bundle (str or list): bundle.
        disease (str or list): disease.
        site (str or list, optional): site. Defaults is any.
        age (list, optional): age range. Defaults is any.
        harmonization (str or list, optional): harmonization. Defaults is any.
        sex (list, optional): harmonization. Defaults is any.
        handedness (list, optional): harmonization. Defaults is any.
        model (list, optional: model. Default is any.

    Returns:
        str: query

    """
    args = locals()
    args = dict((k, v) for k, v in args.items() if v is not None)
    if "age" in args.keys() and not None:
        del args["age"]
    query = " & ".join(
        [
            (
                f'{col} == "{row}"'
                if type(row) == str
                else (
                    f"{col} == {row}"
                    if type(row) == int or float
                    else f"" if row is None else f"{col} in {row}"
                )
            )
            for col, row in args.items()
        ]
    )
    if age is not None:
        query += f" & age >= {age[0]} & age <= {age[1]}"
    return query.replace("'", '"')


def get_valid_age_windows(
    df,
    min_age,
    max_age,
    window_size,
    no_dynamic_window_size=False,
    min_n_subjects=5,
    fixed_n_subjects=False,
    n_subjects=20,
    filter_percentiles=[5, 95],
):
    """
    Gets the valid age windows for sliding windows.

    By default, the function returns the dataframe corresponding to the selected age range.

    Option: no_dynamic_window_size at True
        If the minimum number of subjects is not respected (min_n_subjects), the function not iterates
        to reach the minimum number of subjects (this can be a larger number than min_n_subjects).
        The window size is not updated by incrementing the window size by the increment value.

    Option: fixed_n_subjects at True
         If the number of subjects is > to the minimum number of subjects (min_n_subjects),
         the function filters the dataframe corresponding to the age range with the extreme
         percentiles (filter_percentiles). If the dataframe is > the minimum number of subjects
         required, the function randomly selects a number of subjects corresponding to n_subjects.

    For the moment, the combination of the 2 options is not available.

    Args
    df: dataframe
        Dataframe containing the data to be used for the sliding window.
    min_age: float
        Minimum age to use for the sliding window.
    max_age: float
        Maximum age to use for the sliding window.
    window_size: int
        Size of the sliding window in years.
    no_dynamic_window_size: bool
        If True, the window size is not updated to have a minimum number of subjects per window.
        Default is False.
    increment: int
        Increment to use for the window size update.
    min_n_subjects: int
        Minimum number of subjects per window.
    max_n_subjects: int
        Maximum number of subjects per window.
    fixed_n_subjects: bool
        If True, the number of subjects per window is fixed to n_subjects.
    n_subjects: int
        Number of subjects per window. Used with fixed_n_subjects.
    filter_percentiles: list of int
        Percentiles to use for filtering the data.  Used with fixed_n_subjects.

    Returns
    dataframe: list of dataframe
        List of dataframe containing the data for each window.
    reference_age: list of float
        List of age corresponding to each window.
    """
    dataframe, reference_age = [], []
    for current_age in range(int(min_age), int(max_age)):
        window_age_min = current_age - window_size / 2
        window_age_max = current_age + window_size / 2
        current_age_df = df.query("age >= @window_age_min & age <= @window_age_max")

        if current_age_df.shape[0] > min_n_subjects and fixed_n_subjects == False:
            dataframe.append(current_age_df)
            reference_age.append(current_age)

        elif current_age_df.shape[0] > min_n_subjects and fixed_n_subjects == True:
            curr_percentiles = np.percentile(current_age_df["mean"], filter_percentiles)
            current_age_df = current_age_df.query(
                "mean > @curr_percentiles[0] & mean < @curr_percentiles[1]"
            )
            if current_age_df.shape[0] > n_subjects:
                current_age_df = current_age_df.sample(
                    n_subjects, random_state=n_subjects
                )
            dataframe.append(current_age_df)
            reference_age.append(current_age)

        elif (
            current_age_df.shape[0] < min_n_subjects and no_dynamic_window_size == True
        ):
            logging.warning("No use of a dynamic window is not recommended.")
            dataframe.append(current_age_df)
            reference_age.append(current_age)

        elif current_age_df.shape[0] < min_n_subjects:
            count = 0
            while current_age_df.shape[0] < min_n_subjects and count < 50:
                window_age_min -= 1
                window_age_max += 1
                current_age_df = df.query(
                    "age >= @window_age_min & age <= @window_age_max"
                )
                count += 1

            dataframe.append(current_age_df)
            reference_age.append(current_age)
            logging.info(
                "Window size is update for age: %d and reach minimum N of %i after "
                "%i iterations.",
                current_age,
                min_n_subjects,
                count,
            )

    return dataframe, reference_age


def compute_reference_windows_and_percentiles_by_windows(
    df_reference,
    percentiles,
    ages_lim,
    window_size,
    min_n_subjects,
    dynamic=False,
    mean_col="mean",
):
    """
    Compute the reference windows and reference percentiles for each age.
    Args:
        df_reference: DataFrame, reference site data
        percentiles: list, lits of percentiles value to compute
        ages_lim: tuple, range of ages to use (min_age, max_age)
        window_size: int, size of the window in years
        min_n_subjects: int, minimum number of subjects per window
        dynamic: bool, if True, update the window size to have a minimum number of subjects per
                 window
        mean_col: str, name of the column containing the mean value in DataFrame
    Returns:
        reference_percentiles: array, reference data percentiles for each age window
        age_reference_site: array, age of the reference site
        window_reference_df: list, reference site DataFrame for each age window
    """
    window_reference_df, age_reference_site = get_valid_age_windows(
        df_reference,
        ages_lim[0],
        ages_lim[1],
        window_size,
        no_dynamic_window_size=dynamic,
        min_n_subjects=min_n_subjects,
    )
    # Reference - Compute moving average and standard deviation for each age window
    reference_percentiles = []
    for ref_curr_window_df in window_reference_df:
        reference_percentiles.append(
            np.percentile(ref_curr_window_df[mean_col], percentiles)
        )
    return np.array(reference_percentiles), age_reference_site, window_reference_df


def compute_site_curve(windows_reference_per_age, reference_age):
    """
    Compute the mean and standard deviation for each age window.
    This function is design to take the output of get_valid_age_windows function.

    Args:
        windows_reference_per_age (list): list of dataframes for each age window.
        reference_age (list): list of age for each window.
    Returns:
        site_window_age (np.array): age for each window.
        site_window_mean (np.array): mean for each window.
        site_window_std (np.array): standard deviation for each window.
    """
    site_window_age, site_window_mean, site_window_std = [], [], []
    for site_curr_window_df, site_curr_age in zip(
        windows_reference_per_age, reference_age
    ):
        curr_mean, curr_std = (
            site_curr_window_df["mean"].mean(),
            site_curr_window_df["mean"].std(),
        )
        # Append the mean and standard deviation to the list per window
        site_window_age.append(site_curr_age)
        site_window_mean.append(curr_mean)
        site_window_std.append(curr_std)
    return (
        np.array(site_window_age),
        np.array(site_window_mean),
        np.array(site_window_std),
    )


def compute_distance_between_harmonization(
    df_harmonized, df_ref_harmonized, mean_col="mean"
):
    """
    Compute the distance between the harmonized data (df_harmonized) and the reference harmonized
    data (df_ref_harmonized). Mainly used to display error bars in the estimation of the given
    harmonization compared to the reference harmonization.
    The distance is evaluated with the difference between the averages obtained with the two
    harmonized methods. Four measures are added in DataFrame df_harmonized:
        - difference_harmonized: difference between the mean of df_harmonized and df_ref_harmonized
        - uncertainty: standard deviation of the difference_harmonized
        - err_lower: absolute value of the negative difference_harmonized
        - err_upper: value of the positive difference_harmonized

    Args:
        df_harmonized: DataFrame, harmonized data
        df_ref_harmonized: DataFrame, reference harmonized data
        mean_col: str, name of the column containing the mean value in DataFrame
    Returns:
        df: DataFrame, harmonized data with the difference, distance (lower/upper bound) and
            the uncertainty
    """
    # Rename the mean column of the reference harmonized data to avoid issues with merge
    df_ref_harmonized.rename(columns={mean_col: "mean_ref_harmonized"}, inplace=True)
    # Drop the columns that are in both dataframes, except sid and site
    rm_list = [
        col
        for col in df_ref_harmonized.columns.tolist()
        if col in df_harmonized.columns.tolist() and col != "sid" and col != "site"
    ]
    df_ref_harmonized.drop(rm_list, axis=1, inplace=True)
    # Merge the two dataframes
    df = df_harmonized.merge(df_ref_harmonized, on=["sid", "site"])
    # Compute metrics of the difference between the two harmonized methods
    df["difference_harmonized"] = df["mean"] - df["mean_ref_harmonized"]
    df["err_lower"] = np.absolute(
        df["difference_harmonized"].apply(lambda x: x if x < 0 else 0)
    )
    df["err_upper"] = df["difference_harmonized"].apply(lambda x: x if x > 0 else 0)
    # Check if there is any negative value in the upper error or positive value in the lower error
    if (df["err_upper"].values < 0).any() == True or (
        df["err_lower"].values < 0
    ).any() == True:
        raise ValueError(
            "\nError in computing errors: Negative value in upper or positive value \
                          in lower."
        )
    # Compute uncertainty for each site
    df_with_uncertainty = []
    for site in df.site.unique():
        df_site = df.query("site == @site")
        df_site["uncertainty"] = df_site["difference_harmonized"].std()
        df_with_uncertainty.append(df_site)
    df = pd.concat(df_with_uncertainty)

    return df


def compute_reference_average_variability(
    df_reference, subject_age, windows_size=4, method="mean"
):
    """
    Compute the average and variability of the reference site for the subject_age
    Args:
        df_reference: DataFrame, reference site data
        subject_age: int/float, age of the subject
        windows_size: int, size of the window in years
        method: str, method to compute the average and variability, 'mean' or 'median'
    Returns:
        reference_average: float, average of the reference site for the subject_age
        reference_variability: float, variability of the reference site for the subject_age
    """
    # Compute mean error between single subject and reference site
    windows_reference_min = subject_age - windows_size / 2
    windows_reference_max = subject_age + windows_size / 2
    mean_ref = df_reference.query(
        "age >= @windows_reference_min & age <= @windows_reference_max"
    )["mean"]
    if method == "median":
        reference_average = mean_ref.median()
        reference_variability = np.median(np.absolute(mean_ref - reference_average))
    else:
        reference_average = mean_ref.mean()
        reference_variability = mean_ref.std()
    return reference_average, reference_variability


def compute_site_errors_and_effectsize(
    df_site, age_site, reference_age, reference_mean, reference_std
):
    """
    Compute the mean error, std error and Cohen's D for a specific site and age window from
    a reference mean and age window.

    Args:
        df_site (dataframe): dataframe containing the site data corresponding to age_site
        age_site (float): age of the site associated to the df_site
        reference_age (list): list of age for the reference data corresponding to age_site
        reference_mean (list): list of mean for the reference data corresponding to age_site
        reference_std (list): list of std for the reference data corresponding to age_site

    Returns:
        mean_errors (list): list of mean errors
        std_errors (list): list of std errors
        cohensd (list): list of Cohen's D effect size
    """
    site_window_age, site_window_mean, site_window_std = [], [], []
    mean_errors, std_errors, cohensd = [], [], []
    site_mean = df_site["mean"].mean()
    site_std = df_site["mean"].std()

    # Compute mean, std error and cohen's D
    if age_site in reference_age:
        idx = reference_age.index(age_site)
        mean_errors.append(reference_mean[idx] - site_mean)
        std_errors.append(reference_std[idx] - site_std)

        diff_mean = reference_mean[idx] - site_mean
        pooled_std = np.sqrt((reference_std[idx] ** 2 + site_std**2) / 2)
        cohensd.append(diff_mean / pooled_std)

    return mean_errors, std_errors, cohensd


def viz_identify_valid_sites(
    df,
    full_site_list=None,
    min_subject_per_site=10,
    harmonisation="raw",
    metric=None,
    bundle=None,
    disease="HC",
    handedness=None,
    sex=None,
):
    """Identify site with enough subjects (min_subject_per_site)

    Args:
        df (DataFrame): pd.DataFrame
        in_site_list (list, optional): list of sites. Defaults is all site.
        min_subject_per_site (int, optional): Minimum HC per site.
                                              Defaults to 10.
        metric (str, optional): metric. Defaults is any.
        bundle (str, optional): bundle. Defaults is any.

    Returns:
        list: list of valid sites
    """
    if full_site_list is None:
        full_site_list = list(np.unique(df["site"]))
    if metric is None:
        metric = list(np.unique(df["metric"]))[0]
    if bundle is None:
        bundle = list(np.unique(df["bundle"]))[0]

    validate_site_list = []
    for curr_site in full_site_list:
        nbr_subjects = df.query(
            generate_query(
                metric,
                bundle,
                [disease],
                site=[curr_site],
                sex=sex,
                handedness=handedness,
                harmonization=harmonisation,
            )
        ).shape[0]
        if nbr_subjects >= min_subject_per_site:
            validate_site_list.append(curr_site)
        else:
            logging.info(
                "Site %s was removed (%i HC subjects)", curr_site, nbr_subjects
            )

    if len(validate_site_list) < 1:
        raise ValueError("No moving sites.")
    else:
        return validate_site_list


def viz_get_ylim(
    df,
    metric,
    bundle,
    site_list=None,
    sex_list=None,
    disease_list=None,
    handedness_list=None,
    min_age=0,
    max_age=100,
    y_min_percentile=1,
    y_max_percentile=99,
):

    if site_list is None:
        site_list = list(np.unique(df["site"]))
    if sex_list is None:
        sex_list = list(np.unique(df["sex"]))
    if disease_list is None:
        disease_list = list(np.unique(df["disease"]))
    if handedness_list is None:
        handedness_list = list(np.unique(df["handedness"]))

    query = "metric == @metric \
        & site in @site_list  \
        & bundle == @bundle \
        & sex in @sex_list \
        & age >= @min_age \
        & age <= @max_age \
        & disease == @disease_list \
        & handedness in @handedness_list"
    vals = df.query(query)["mean"]

    if len(vals) == 0:
        logging.info("No data found for metric %s and bundle %s." + metric, bundle)
        return None, None

    return np.percentile(vals, [y_min_percentile, y_max_percentile])
