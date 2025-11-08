# -*- coding: utf-8 -*-
"""ComBat-GAM harmonization built on top of QuickCombatClassic."""
from typing import Dict

import numpy as np
import pandas as pd

from clinical_combat.harmonization.QuickCombatClassic import QuickCombatClassic


class QuickCombatGam(QuickCombatClassic):
    """Quick ComBat implementation that models covariates with natural cubic splines.

    This approximates the ComBat-GAM variant by replacing the polynomial age fit with
    a natural cubic spline basis whose knots are shared across reference and moving data.
    """

    def __init__(
        self,
        bundle_names=None,
        model_params=None,
        ignore_sex_covariate: bool = False,
        ignore_handedness_covariate: bool = False,
        use_empirical_bayes: bool = True,
        limit_age_range: bool = False,
        degree: int = 1,
        regul_ref: float = 0,
        regul_mov: float = 1e-3,
        alpha=None,
        beta=None,
        sigma=None,
        gamma_ref=None,
        delta_ref=None,
        gamma_mov=None,
        delta_mov=None,
        gam_n_knots: int = 7,
    ):
        super().__init__(
            bundle_names=bundle_names,
            model_params=model_params,
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            gamma_ref=gamma_ref,
            delta_ref=delta_ref,
            gamma_mov=gamma_mov,
            delta_mov=delta_mov,
        )
        self.gam_n_knots = max(int(gam_n_knots), 4)
        self.gam_knots: Dict[str, np.ndarray] = {}
        self.gam_feature_counts: Dict[str, int] = {}
        self.gam_keep_indices: Dict[str, np.ndarray] = {}
        self.gam_requested_features: Dict[str, int] = {}

    def initialize_from_model_params(self, model_filename):
        super().initialize_from_model_params(model_filename)
        gam_params = self.model_params.get("gam")
        if gam_params:
            self.gam_n_knots = int(gam_params.get("n_knots", self.gam_n_knots))
            self.gam_knots = {
                str(bundle): np.array(knots, dtype=float)
                for bundle, knots in gam_params.get("knots", {}).items()
            }
            self.gam_feature_counts = {
                str(bundle): int(count)
                for bundle, count in gam_params.get("feature_counts", {}).items()
            }
            self.gam_keep_indices = {
                str(bundle): np.array(indices, dtype=int)
                for bundle, indices in gam_params.get("keep_indices", {}).items()
            }
            self.gam_requested_features = {
                str(bundle): int(count)
                for bundle, count in gam_params.get("requested_features", {}).items()
            }

    def save_model(self, model_filename):
        self.model_params["name"] = "gam"
        self.model_params["gam"] = {
            "n_knots": int(self.gam_n_knots),
            "knots": {bundle: knots.tolist() for bundle, knots in self.gam_knots.items()},
            "feature_counts": {bundle: int(count) for bundle, count in self.gam_feature_counts.items()},
            "keep_indices": {bundle: indices.tolist() for bundle, indices in self.gam_keep_indices.items()},
            "requested_features": {bundle: int(count) for bundle, count in self.gam_requested_features.items()},
        }
        super().save_model(model_filename)

    def set_model_fit_params(self, ref_data, mov_data):
        super().set_model_fit_params(ref_data, mov_data)
        self.model_params["name"] = "gam"
        self.model_params["gam"] = {
            "n_knots": int(self.gam_n_knots),
            "knots": {bundle: knots.tolist() for bundle, knots in self.gam_knots.items()},
            "feature_counts": {bundle: int(count) for bundle, count in self.gam_feature_counts.items()},
            "keep_indices": {bundle: indices.tolist() for bundle, indices in self.gam_keep_indices.items()},
            "requested_features": {bundle: int(count) for bundle, count in self.gam_requested_features.items()},
        }

    def prepare_data(self, ref_data, mov_data, HC_only=True):
        ref_data, mov_data = super().prepare_data(ref_data, mov_data, HC_only)
        self._compute_knots(ref_data, mov_data)
        return ref_data, mov_data

    def _compute_knots(self, ref_data: pd.DataFrame, mov_data: pd.DataFrame):
        combined = pd.concat([ref_data, mov_data], ignore_index=True)
        self.gam_knots = {}
        self.gam_feature_counts = {}
        self.gam_keep_indices = {}
        self.gam_requested_features = {}

        base_cols = 1
        if not self.ignore_sex_covariate:
            base_cols += 1
        if not self.ignore_handedness_covariate:
            base_cols += 1

        for bundle in self.bundle_names:
            bundle_mask = combined["bundle"] == bundle
            ages = combined.loc[bundle_mask, "age"].to_numpy(dtype=float)
            knots = self._select_knots(ages)
            self.gam_knots[bundle] = knots

            n_total = int(bundle_mask.sum())
            available = 1 + max(0, knots.size - 2)
            usable = max(1, n_total - base_cols)
            requested = max(1, min(available, usable))
            raw_basis = self._raw_spline_basis(ages, bundle, requested)

            if raw_basis.shape[0] > 1:
                variances = raw_basis.var(axis=1)
                keep_mask = variances > 1e-12
                keep_mask[0] = True
            else:
                keep_mask = np.ones(raw_basis.shape[0], dtype=bool)

            if not np.any(keep_mask):
                keep_mask = np.array([True])

            self.gam_requested_features[bundle] = raw_basis.shape[0]
            self.gam_keep_indices[bundle] = np.where(keep_mask)[0]
            self.gam_feature_counts[bundle] = int(np.sum(keep_mask))

    def _select_knots(self, ages: np.ndarray) -> np.ndarray:
        ages = ages[np.isfinite(ages)]
        if ages.size == 0:
            return np.array([0.0, 0.33, 0.66, 1.0])

        unique = np.unique(ages)
        if unique.size < 2:
            base = float(unique[0])
            return np.array([base - 1.0, base - 0.33, base + 0.33, base + 1.0])

        n_knots = min(self.gam_n_knots, unique.size, len(ages))
        if n_knots < 4:
            n_knots = 4
        quantiles = np.linspace(0, 1, n_knots)
        knots = np.quantile(ages, quantiles)
        knots = np.unique(knots)
        while knots.size < 4:
            jitter = np.linspace(-0.5, 0.5, 4)
            knots = np.unique(np.concatenate([knots, ages.min() + jitter * 0.01]))
            knots.sort()
        if np.isclose(knots[-1], knots[-2]):
            knots[-1] += 1e-3
        return knots

    def get_design_matrices(self, df):
        design = []
        Y = []
        for bundle in self.bundle_names:
            data = df.query("bundle == @bundle")
            n_samples = len(data["sid"])
            if n_samples == 0:
                continue

            rows = [np.ones(n_samples)]
            if not self.ignore_sex_covariate:
                rows.append(self.to_category(data["sex"]))
            if not self.ignore_handedness_covariate:
                rows.append(self.to_category(data["handedness"]))

            ages = data["age"].to_numpy(dtype=float)
            max_features = self.gam_feature_counts.get(bundle, 1)
            features = self._build_spline_features(ages, bundle, max_features)
            if features.size > 0:
                rows.extend(features)

            design.append(np.array(rows))
            Y.append(data["mean"].to_numpy())
        return design, Y

    def get_beta_labels(self):
        labels = []
        if not self.ignore_sex_covariate:
            labels.append("beta_sex")
        if not self.ignore_handedness_covariate:
            labels.append("beta_handedness")

        feature_count = 1
        if self.gam_feature_counts:
            feature_count = int(next(iter(self.gam_feature_counts.values())))
        else:
            gam_params = self.model_params.get("gam") if hasattr(self, "model_params") else None
            if gam_params and gam_params.get("feature_counts"):
                feature_count = int(next(iter(gam_params["feature_counts"].values())))

        for idx in range(feature_count):
            labels.append(f"beta_age_spline_{idx}")
        return labels

    def _raw_spline_basis(self, ages: np.ndarray, bundle: str, max_features: int) -> np.ndarray:
        knots = self.gam_knots.get(bundle)
        if knots is None:
            knots = self._select_knots(ages)
            self.gam_knots[bundle] = knots

        if max_features <= 0:
            return np.empty((0, ages.size))

        features = [ages]
        # Number of additional spline basis columns beyond the linear age term.
        remaining = max_features - 1
        if knots.size >= 4:
            last = knots[-1]
            before_last = knots[-2]
            denom = last - before_last
            if np.isclose(denom, 0.0):
                denom = 1.0

            def d(x, knot):
                return np.maximum(0.0, x - knot) ** 3

            internal_knots = knots[1:-1]
            if remaining < internal_knots.size:
                internal_knots = internal_knots[:remaining]
            for knot in internal_knots:
                basis = (
                    d(ages, knot)
                    - d(ages, before_last) * ((last - knot) / denom)
                    + d(ages, last) * ((before_last - knot) / denom)
                )
                features.append(basis)
                remaining -= 1
                if remaining <= 0:
                    break

        return np.array(features, dtype=float)

    def _build_spline_features(self, ages: np.ndarray, bundle: str, max_features: int) -> np.ndarray:
        requested = self.gam_requested_features.get(bundle, max_features)
        raw_basis = self._raw_spline_basis(ages, bundle, requested)

        if raw_basis.size == 0:
            return raw_basis

        keep_indices = self.gam_keep_indices.get(bundle)
        if keep_indices is None:
            keep_indices = np.arange(min(raw_basis.shape[0], max_features))

        keep_indices = keep_indices[keep_indices < raw_basis.shape[0]]
        if keep_indices.size == 0:
            keep_indices = np.array([0])

        feature_array = raw_basis[keep_indices].astype(float)
        feature_array -= feature_array.mean(axis=1, keepdims=True)
        return feature_array
