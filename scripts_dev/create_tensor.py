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
from joblib import Parallel, delayed
from tqdm import tqdm

from clinical_combat.utils.combatio import pickle_save, save_data

matplotlib.use("Agg")


def process_files(args, sid, brain_mask, bundle_names, bundles_brain_voxels):
    metrics_dict = {}
    if args.preregistered:
        metrics_path = join(args.input_folder, sid)
    else:
        metrics_path = join(args.result_folder, sid, "Apply_Transform")

    if not os.path.exists(metrics_path):
        return

    metrics = listdir(metrics_path)
    for m in metrics:
        key = m.split(".")[0]
        m_path = join(metrics_path, m)
        try:
            metric = nib.load(m_path).get_fdata("unchanged")
        except Exception as e:
            print(e)
            print(f"Cannot load {m_path}, {key}")
            continue
        metric_brain_voxel = metric[np.nonzero(brain_mask)]
        print(f"Clipping value data to {args.clip_value}")
        # min_clip = brain_voxel.mean() - args.clip_value * np.std(brain_voxel, ddof=1)
        max_clip = metric_brain_voxel.mean() + args.clip_value * np.std(
            metric_brain_voxel, ddof=1
        )

        # clip images to 0, 3 std
        metric_brain_voxel = np.clip(metric_brain_voxel, 0, max_clip)
        bundle_vector = np.zeros(shape=(len(bundle_names)))
        for i, b_name in enumerate(bundle_names):
            bundle_voxel = metric_brain_voxel[
                bundles_brain_voxels[bundle_names.index(b_name), :].astype(bool)
            ]
            bundle_vector[i] = np.mean(bundle_voxel)

        metrics_dict[key] = bundle_vector

    if not bool(metrics_dict):
        return

    return sid, metrics_dict


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("input_folder", help="Input folder")
    p.add_argument("result_folder", help="Path to various sites images")
    p.add_argument("bundle_path", help="Path to bundle images")
    p.add_argument("output", help="Output folder")
    p.add_argument(
        "--preregistered",
        action="store_true",
        help="If True, input folder metrics will be used.",
    )
    p.add_argument(
        "-cl",
        "--clip_value",
        help="Number of STD to clip images, if negative, it doesn't clip " + "images.",
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

    if "reference.nii.gz" in inputs:
        reference_image = join(args.input_folder, "reference.nii.gz")
    else:
        raise ValueError("No reference.nii.gz in input folder")

    covariates = {}
    if "covariates" in inputs:
        covariates_files = listdir(join(args.input_folder, "covariates"))

        for f in covariates_files:
            covariates_file = pd.read_csv(
                join(args.input_folder, "covariates", f), delimiter=",", dtype=str
            )
            # assert_covar_file(covariates_file)

            if f in covariates:
                covariates[f].append(covariates_file)
            else:
                covariates[f] = [
                    covariates_file,
                ]

    else:
        raise ValueError("No covariates in input folder")

    if "covariates_apply" in inputs:
        covariates_files = listdir(join(args.input_folder, "covariates_apply"))

        for f in covariates_files:
            covariates_file = pd.read_csv(
                join(args.input_folder, "covariates_apply", f), delimiter=",", dtype=str
            )
            # assert_covar_file(covariates_file)

            if f in covariates:
                covariates[f].append(covariates_file)
            else:
                covariates[f] = [
                    covariates_file,
                ]

    else:
        raise ValueError("No covariates_apply in input folder")

    for k, v in covariates.items():
        concat_df = pd.concat(v)
        covariates[k] = concat_df.drop_duplicates(subset="sid")

    brain_mask_img = nib.load(reference_image)
    brain_mask = brain_mask_img.get_fdata("unchanged") > 0

    print("Loading bundles ...")
    bundles_brain_voxels = np.empty((0, np.count_nonzero(brain_mask)))
    bundle_files = os.listdir(args.bundle_path)
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

    for k, v in tqdm(covariates.items()):
        loadings = Parallel(n_jobs=-1)(
            delayed(process_files)(
                args, sid, brain_mask, bundle_names, bundles_brain_voxels
            )
            for sid in v["sid"]
        )
        to_save = {}
        site_name = k.split(".")[0]
        loadings = [l for l in loadings if l is not None]
        for sid, metrics_dict in loadings:
            for kk, vv in metrics_dict.items():
                if kk in to_save:
                    to_save[kk][0].append(sid)
                    to_save[kk][1].append(vv)
                else:
                    to_save[kk] = [
                        [
                            sid,
                        ],
                        [
                            vv,
                        ],
                    ]

        for kkk, (s_list, values) in to_save.items():
            with open(join(args.output, f"{site_name}_{kkk}.pkl"), "wb") as handle:
                pickle.dump(
                    (bundle_names, s_list, np.vstack(values)),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )


if __name__ == "__main__":
    main()
