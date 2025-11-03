#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import pandas as pd

import numpy as np
import numpy.testing as npt

from clinical_combat import COMBAT_ROOT

data_path = os.path.join(COMBAT_ROOT, "src/clinical_combat/data/")

def test_quick_combat_pairwise():
    out = os.path.join(COMBAT_ROOT, "src/clinical_combat/cli/tests/out/QuickCombat_pairwise")
    if os.path.exists(out):
        shutil.rmtree(out)

    cmd = (
        "combat_quick "
        + data_path
        + "CamCAN.md.raw.csv.gz "
        + data_path
        + "ShamCamCAN.md.raw.csv.gz "
        + "-m pairwise "
        + "--out_dir "
        + out
    )
    subprocess.call(cmd, shell=True)

    model = os.path.join(out, "ShamCamCAN-CamCAN.md.pairwise.model.csv")
    data = os.path.join(out, "ShamCamCAN.md.pairwise.csv.gz")
    fig1 = os.path.join(
        out, "AgeCurve_CamCAN-ShamCamCAN_raw_md_mniIITmaskskeletonFA.png"
    )
    fig2 = os.path.join(
        out, "AgeCurve_CamCAN-ShamCamCAN_pairwise_md_mniIITmaskskeletonFA.png"
    )
    fig3 = os.path.join(
        out, "DataModels_CamCAN-ShamCamCAN_pairwise_md_mniIITmaskskeletonFA.png"
    )
    dist1 = os.path.join(out, "ShamCamCAN.md.pairwise.bhattacharrya.txt")
    dist2 = os.path.join(out, "ShamCamCAN.md.raw.bhattacharrya.txt")

    npt.assert_(os.path.exists(model), msg="Model file not generated.")
    npt.assert_(os.path.exists(data), msg="Harmonized data file not generated.")
    npt.assert_(os.path.exists(fig1), msg="Raw AgeCurve figure not generated.")
    npt.assert_(os.path.exists(fig2), msg="Harmonized AgeCurve figure not generated.")
    npt.assert_(os.path.exists(fig3), msg="Model figure not generated.")
    npt.assert_(
        os.path.exists(dist1), msg="Harmonized Bhattacharrya file not generated."
    )
    npt.assert_(os.path.exists(dist2), msg="Raw Bhattacharrya file not generated.")

    model_ = os.path.join(
        COMBAT_ROOT,
        "src/clinical_combat/cli/tests/target_out/QuickCombat_pairwise",
        "ShamCamCAN-CamCAN.md.pairwise.model.csv",
    )
    a = np.loadtxt(model, dtype=str, delimiter=",")
    b = np.loadtxt(model_, dtype=str, delimiter=",")
    npt.assert_array_almost_equal(a[2:, 1:].astype("float"), b[2:, 1:].astype("float"))

    data_ = os.path.join(
        COMBAT_ROOT,
        "src/clinical_combat/cli/tests/target_out/QuickCombat_pairwise",
        "ShamCamCAN.md.pairwise.csv.gz",
    )
    a = pd.read_csv(data)["mean"].to_numpy()
    b = pd.read_csv(data_)["mean"].to_numpy()
    npt.assert_array_almost_equal(a, b)


def test_quick_combat_clinic():
    out = os.path.join(COMBAT_ROOT, "src/clinical_combat/cli/tests/out/QuickCombat_clinic")

    if os.path.exists(out):
        shutil.rmtree(out)

    cmd = (
        "combat_quick "
        + data_path
        + "CamCAN.md.raw.csv.gz "
        + data_path
        + "ShamCamCAN.md.raw.csv.gz "
        + "-m clinic "
        + "--out_dir "
        + out
    )
    subprocess.call(cmd, shell=True)

    model = os.path.join(out, "ShamCamCAN-CamCAN.md.clinic.model.csv")
    data = os.path.join(out, "ShamCamCAN.md.clinic.csv.gz")
    fig1 = os.path.join(
        out, "AgeCurve_CamCAN-ShamCamCAN_raw_md_mniIITmaskskeletonFA.png"
    )
    fig2 = os.path.join(
        out, "AgeCurve_CamCAN-ShamCamCAN_clinic_md_mniIITmaskskeletonFA.png"
    )
    fig3 = os.path.join(
        out, "DataModels_CamCAN-ShamCamCAN_clinic_md_mniIITmaskskeletonFA.png"
    )
    dist1 = os.path.join(out, "ShamCamCAN.md.clinic.bhattacharrya.txt")
    dist2 = os.path.join(out, "ShamCamCAN.md.raw.bhattacharrya.txt")

    npt.assert_(os.path.exists(model), msg="Model file not generated.")
    npt.assert_(os.path.exists(data), msg="Harmonized data file not generated.")
    npt.assert_(os.path.exists(fig1), msg="Raw AgeCurve figure not generated.")
    npt.assert_(os.path.exists(fig2), msg="Harmonized AgeCurve figure not generated.")
    npt.assert_(os.path.exists(fig3), msg="Model figure not generated.")
    npt.assert_(
        os.path.exists(dist1), msg="Harmonized Bhattacharrya file not generated."
    )
    npt.assert_(os.path.exists(dist2), msg="Raw Bhattacharrya file not generated.")

    model_ = os.path.join(
        COMBAT_ROOT,
        "src/clinical_combat/cli/tests/target_out/QuickCombat_clinic",
        "ShamCamCAN-CamCAN.md.clinic.model.csv",
    )
    a = np.loadtxt(model, dtype=str, delimiter=",")
    b = np.loadtxt(model_, dtype=str, delimiter=",")
    npt.assert_array_almost_equal(a[2:, 1:].astype("float"), b[2:, 1:].astype("float"))

    data_ = os.path.join(
        COMBAT_ROOT,
        "src/clinical_combat/cli/tests/target_out/QuickCombat_clinic",
        "ShamCamCAN.md.clinic.csv.gz",
    )
    a = pd.read_csv(data)["mean"].to_numpy()
    b = pd.read_csv(data_)["mean"].to_numpy()
    npt.assert_array_almost_equal(a, b)


def test_visualize_data():
    out = os.path.join(COMBAT_ROOT, "src/clinical_combat/cli/tests/out/visualized_data")
    
    if os.path.exists(out):
        shutil.rmtree(out)

    cmd = (
        "combat_visualize_data "
        + data_path
        + "CamCAN.md.raw.csv.gz "
        + data_path
        + "ShamCamCAN.md.raw.csv.gz "
        + "--out_dir "
        + out
        + " --display_marginal_hist -f"
    )
    subprocess.call(cmd, shell=True)
    fig1 = os.path.join(out, "Dataset_2-sites_md_mniIITmaskskeletonFA.png")
    npt.assert_(os.path.exists(fig1), msg="combat_visualize_data fig not generated.")

    if os.path.exists(out):
        shutil.rmtree(out)

    cmd = (
        "combat_visualize_data "
        + data_path
        + "CamCAN.md.raw.csv.gz "
        + data_path
        + "ShamCamCAN.md.raw.csv.gz "
        + "--out_dir "
        + out
        + " --display_marginal_hist --hide_disease"
    )
    subprocess.call(cmd, shell=True)

    fig1 = os.path.join(out, "Dataset_2-sites_md_mniIITmaskskeletonFA.png")
    npt.assert_(os.path.exists(fig1), msg="combat_visualize_data fig not generated.")


def test_corrupt_data():
    out = os.path.join(COMBAT_ROOT, "src/clinical_combat/cli/tests/out/corrupt_data")

    if os.path.exists(out):
        shutil.rmtree(out)

    cmd = (
        "combat_corrupt_data "
        + data_path
        + "CamCAN.md.raw.csv.gz "
        + out
        + "/corruptCamCAN.md.raw.csv.gz "
        + "--mult 90 --add 110 --slope 75 --nbr_sub 0 --site_name corruptcamcan --HC"
    )
    subprocess.call(cmd, shell=True)
    data = os.path.join(out, "corruptCamCAN.md.raw.csv.gz")
    npt.assert_(os.path.exists(data), msg="corrupt data not generated.")

    data_ = os.path.join(
        COMBAT_ROOT,
        "src/clinical_combat/cli/tests/target_out/corrupt_data",
        "corruptCamCAN.md.raw.csv.gz",
    )
    a = pd.read_csv(data)["mean"].to_numpy()
    b = pd.read_csv(data_)["mean"].to_numpy()
    npt.assert_array_almost_equal(a, b)


def test_info():
    out = os.path.join(COMBAT_ROOT, "src/clinical_combat/cli/tests/target_out/info")

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    cmd = (
        "combat_info"
        + data_path
        + "CamCAN.md.raw.csv.gz "
        + "> "
        + out
        + "/output.txt"
    )
    subprocess.call(cmd, shell=True)

    output_ = os.path.join(
        COMBAT_ROOT,
        "src/clinical_combat/cli/tests/target_out/info",
        "output.txt",
    )

    out1 = open(output_, "r").read()
    out2 = open(out + "/output.txt", "r").read()
    npt.assert_(out1 == out2, msg="combat_info has an invalid output.")
