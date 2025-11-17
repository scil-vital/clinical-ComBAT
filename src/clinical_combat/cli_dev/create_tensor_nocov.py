#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from os import listdir
from os.path import join

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
from genericpath import isdir
from tqdm import tqdm

from clinical_combat.utils.combatio import assert_covar_file, pickle_save, save_data

matplotlib.use("Agg")


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("input_folder", help="Input folder")
    p.add_argument("bundle_path", help="Path to bundle images")
    p.add_argument("reference", help="Path to reference images")
    p.add_argument("output", help="Output folder")
    p.add_argument(
        "-cl",
        "--clip_value",
        help="Number of STD to clip images, if negative, it " + "doesn't clip images.",
        type=float,
        default=3.0,
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print(args)

    os.makedirs(args.output, exist_ok=True)

    inputs = listdir(args.input_folder)

    brain_mask_img = nib.load(args.reference)
    brain_mask = brain_mask_img.get_fdata("unchanged") > 0

    print("Loading bundles ...")
    bundles_brain_voxels = np.empty((0, np.count_nonzero(brain_mask)))
    bundle_files = list(np.sort(os.listdir(args.bundle_path)))
    bundle_names = []
    for b in tqdm(bundle_files):
        bundle_names.append(b.split(".")[0])
        bundle = (
            nib.load(join(args.bundle_path, b)).get_fdata("unchanged") > 0
        ).astype(np.float64)
        bundle_brain_voxel = bundle[np.nonzero(brain_mask)]
        bundles_brain_voxels = np.vstack((bundles_brain_voxels, bundle_brain_voxel))

    save_data(args.output, bundles_brain_voxels, "bundles_brain_voxels.npy")
    pickle_save(args.output, "bundle_name.pkl", bundle_names)

    metrics_dict = {}
    for sid in tqdm(inputs):
        metrics_path = join(args.input_folder, sid)
        if not os.path.exists(metrics_path) or not os.path.isdir(metrics_path):
            continue

        metrics = listdir(metrics_path)
        for m in metrics:
            if not m[-7:] == ".nii.gz":
                continue

            key = m.split(".")[0]
            print(key)
            m_path = join(metrics_path, m)
            metric = nib.load(m_path).get_fdata("unchanged")
            metric_brain_voxel = metric[np.nonzero(brain_mask)]

            max_clip = metric_brain_voxel.mean() + args.clip_value * np.std(
                metric_brain_voxel, ddof=1
            )
            metric_brain_voxel = np.clip(metric_brain_voxel, 0, max_clip)

            bundle_vector = np.zeros(shape=(len(bundle_names)))
            for i, b_name in enumerate(bundle_names):
                bundle_voxel = metric_brain_voxel[
                    bundles_brain_voxels[bundle_names.index(b_name), :].astype(bool)
                ]
                bundle_vector[i] = np.mean(bundle_voxel)

            if key in metrics_dict.keys():
                metrics_dict[key][0].append(bundle_vector)
                metrics_dict[key][1].append(sid)
            else:
                metrics_dict[key] = ([bundle_vector], [sid])

    for kk, vv in metrics_dict.items():
        with open(join(args.output, f"data_{kk}.pkl"), "wb") as handle:
            pickle.dump(
                (bundle_names, vv[1], np.array(vv[0])),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


if __name__ == "__main__":
    main()