# -*- coding: utf-8 -*-
import os
from os.path import basename

import numpy as np
import pandas as pd


def load_sites_data(csv_files: list) -> pd.DataFrame:
    """
    Load a list of site data csv files.

    Args:
        csv_files (list): path to the CSV harmonization files.

    Returns:
        df (Pandas DataFrame): concatenated dataframes.
    """

    dfs = []
    for csv in csv_files:
        dfs.append(load_results_csv(csv))
    df = pd.concat(dfs)

    df.drop_duplicates(
        subset=[
            "site",
            "harmonization",
            "bundle",
            "age",
            "sex",
            "disease",
            "handedness",
            "sid",
            "metric",
            "model",
            "input",
        ],
        inplace=True,
    )
    return df


def load_results_csv(in_csv: str) -> pd.DataFrame:
    """
    Load a harmonization csv result file.

    Args:
        in_csv (str): path to the CSV harmonization file.

    Returns:
        df (Pandas DataFrame): harmonization data.
    """

    df = pd.read_csv(in_csv)
    df["disease"] = df["disease"].replace("NC", "HC")
    df = df.astype({"site": str})
    if "harmonization" not in df.keys():
        # assumes harmonization='raw'
        df.insert(1, "harmonization", "raw", allow_duplicates=False)

    df["harmonization"] = df["harmonization"].replace("PRE", "raw")
    df["harmonization"] = df["harmonization"].replace("pre", "raw")

    if "model" not in df.keys():
        # assumes model=harmonization
        df.insert(1, "model", df["harmonization"], allow_duplicates=False)
    df.insert(
        1,
        "input",
        [os.path.basename(in_csv)] * len(df["harmonization"]),
        allow_duplicates=False,
    )
    df.drop_duplicates(
        subset=[
            "site",
            "harmonization",
            "bundle",
            "age",
            "sex",
            "disease",
            "handedness",
            "sid",
            "metric",
            "model",
        ],
        inplace=True,
    )
    df.dropna(inplace=True)  # remove rows with nans
    return df


def save_quickcombat_data_to_csv(
    mov_data,
    y_harm,
    bundle_names,
    metric_name,
    method_name,
    model_name,
    out_filename="results.res.csv",
):
    """
    Save the harmonized data to a CSV file.

    Args:
        mov_data (Pandas DataFrame): harmonization data.
        y_harm (numpy array): harmonized data.
        bundle_names (list): list of bundle names.
        metric_name (str): metric name.
        method_name (str): harmonization method name.
        model_name (str): harmonization model name.
        out_filename (str): output file name.
    
    Returns:
        CSV file with the harmonized data.

    """
    header = (
        "sid,bundle,metric,mean,site,age,sex,handedness,disease,harmonization,model"
    )

    if ".model.csv" in model_name:
        model_name = basename(model_name).replace(".model", "")
        model_name = basename(model_name).replace(".csv", "")

    res = np.empty((11, 0))

    for i, bundle in enumerate(bundle_names):
        data = mov_data.query("bundle == @bundle")
        sids = data["sid"].to_numpy()
        y = data["mean"].to_numpy()
        ages = data["age"].to_numpy()
        sexes = data["sex"].to_numpy()
        handednesses = data["handedness"].to_numpy()
        diseases = data["disease"].to_numpy()
        sites = data["site"].to_numpy()
        bundles = data["bundle"].to_numpy()
        metrics = np.array([metric_name] * y.shape[0]).reshape((-1))
        harmonizations_method = np.array([method_name] * y.shape[0]).reshape((-1))
        harmonizations_method_model = np.array([model_name] * y.shape[0]).reshape((-1))

        mov_combat = np.vstack(
            [
                sids,
                bundles,
                metrics,
                y_harm[i],
                sites,
                ages,
                sexes,
                handednesses,
                diseases,
                harmonizations_method,
                harmonizations_method_model,
            ]
        )
        res = np.hstack([res, mov_combat])

    np.savetxt(out_filename, res.T, delimiter=",", header=header, fmt="%s", comments="")



def save_quickcombat_data_to_best(
    mov_data,
    y_harm,
    bundle_names,
    metric_name,
    method_name,
    model_name,
):
    """
    Save the harmonized data to a CSV file.

    Args:
        mov_data (Pandas DataFrame): harmonization data.
        y_harm (numpy array): harmonized data.
        bundle_names (list): list of bundle names.
        metric_name (str): metric name.
        method_name (str): harmonization method name.
        model_name (str): harmonization model name.
        out_filename (str): output file name.
    
    Returns:
        CSV file with the harmonized data.

    """

    if ".model.csv" in model_name:
        model_name = basename(model_name).replace(".model", "")
        model_name = basename(model_name).replace(".csv", "")

    res = np.empty((9, 0))

    mov_data["site"] = "adni_compilation"

    for i, bundle in enumerate(bundle_names):
        data = mov_data.query("bundle == @bundle")
        sids = data["sid"].to_numpy()
        y = data["mean"].to_numpy()
        ages = data["age"].to_numpy()
        sexes = data["sex"].to_numpy()
        handednesses = data["handedness"].to_numpy()
        diseases = data["disease"].to_numpy()
        sites = data["site"].to_numpy()
        bundles = data["bundle"].to_numpy()
        metrics = np.array([metric_name] * y.shape[0]).reshape((-1))

        mov_combat = np.vstack(
            [
                sids,
                bundles,
                metrics,
                y_harm[i],
                sites,
                ages,
                sexes,
                handednesses,
                diseases,
            ]
        )
        res = np.hstack([res, mov_combat])

    return res
