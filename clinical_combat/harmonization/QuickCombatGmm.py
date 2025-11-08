# -*- coding: utf-8 -*-
"""ComBat-GMM harmonization method built on top of QuickCombatClassic."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from clinical_combat.harmonization.QuickCombatClassic import QuickCombatClassic


class QuickCombatGmm(QuickCombatClassic):
    """Two-stage ComBat variant using a hidden GMM batch before site harmonization."""

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
        regul_mov: float = 0,
        gmm_components: int = 2,
        gmm_tol: float = 1e-4,
        gmm_max_iter: int = 50,
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
        )
        self.gmm_components = max(1, int(gmm_components))
        self.gmm_tol = float(gmm_tol)
        self.gmm_max_iter = max(10, int(gmm_max_iter))

        self.stage1_model: Optional[QuickCombatClassic] = None
        self.gmm_best_feature: Optional[str] = None
        self.gmm_weights: Optional[np.ndarray] = None
        self.gmm_means: Optional[np.ndarray] = None
        self.gmm_covariances: Optional[np.ndarray] = None
        self.gmm_reference_label: Optional[int] = None
        self._fit_apply_result: Optional[List[np.ndarray]] = None

    # ---------------------------------------------------------------------
    # Serialization helpers
    # ---------------------------------------------------------------------
    def initialize_from_model_params(self, model_filename):
        super().initialize_from_model_params(model_filename)
        gmm_params = self.model_params.get("gmm")
        if not gmm_params:
            return

        self.gmm_components = int(gmm_params.get("components", self.gmm_components))
        self.gmm_tol = float(gmm_params.get("tol", self.gmm_tol))
        self.gmm_max_iter = int(gmm_params.get("max_iter", self.gmm_max_iter))
        self.gmm_best_feature = gmm_params.get("best_feature")

        weights = gmm_params.get("weights")
        means = gmm_params.get("means")
        covariances = gmm_params.get("covariances")
        self.gmm_weights = np.array(weights, dtype=float) if weights else None
        self.gmm_means = np.array(means, dtype=float) if means else None
        self.gmm_covariances = np.array(covariances, dtype=float) if covariances else None
        self.gmm_reference_label = gmm_params.get("reference_label")

        stage1 = gmm_params.get("stage1")
        if stage1:
            self.stage1_model = QuickCombatClassic(
                bundle_names=stage1.get("bundle_names"),
                ignore_sex_covariate=self.ignore_sex_covariate,
                ignore_handedness_covariate=self.ignore_handedness_covariate,
                use_empirical_bayes=self.use_empirical_bayes,
                limit_age_range=self.limit_age_range,
                degree=self.degree,
                regul_ref=self.regul_ref,
                regul_mov=self.regul_mov,
            )
            self.stage1_model.bundle_names = np.array(stage1.get("bundle_names", []), dtype=str)
            self.stage1_model.alpha = np.array(stage1.get("alpha", []), dtype=float)
            self.stage1_model.beta = np.array(stage1.get("beta", []), dtype=float)
            self.stage1_model.sigma = np.array(stage1.get("sigma", []), dtype=float)
            self.stage1_model.gamma_ref = np.array(stage1.get("gamma_ref", []), dtype=float)
            self.stage1_model.delta_ref = np.array(stage1.get("delta_ref", []), dtype=float)
            self.stage1_model.gamma_mov = np.array(stage1.get("gamma_mov", []), dtype=float)
            self.stage1_model.delta_mov = np.array(stage1.get("delta_mov", []), dtype=float)
            self.stage1_model.model_params = stage1.get("model_params", {})
        else:
            self.stage1_model = None

    def save_model(self, model_filename):
        self.model_params["name"] = "gmm"
        self.model_params["gmm"] = {
            "components": int(self.gmm_components),
            "tol": float(self.gmm_tol),
            "max_iter": int(self.gmm_max_iter),
            "best_feature": self.gmm_best_feature,
            "weights": self.gmm_weights.tolist() if self.gmm_weights is not None else [],
            "means": self.gmm_means.tolist() if self.gmm_means is not None else [],
            "covariances": self.gmm_covariances.tolist() if self.gmm_covariances is not None else [],
            "reference_label": self.gmm_reference_label,
            "stage1": self._serialize_stage1_model(),
        }
        super().save_model(model_filename)

    def set_model_fit_params(self, ref_data, mov_data):
        super().set_model_fit_params(ref_data, mov_data)
        self.model_params["name"] = "gmm"
        self.model_params["gmm"] = {
            "components": int(self.gmm_components),
            "tol": float(self.gmm_tol),
            "max_iter": int(self.gmm_max_iter),
            "best_feature": self.gmm_best_feature,
            "weights": self.gmm_weights.tolist() if self.gmm_weights is not None else [],
            "means": self.gmm_means.tolist() if self.gmm_means is not None else [],
            "covariances": self.gmm_covariances.tolist() if self.gmm_covariances is not None else [],
            "reference_label": self.gmm_reference_label,
            "stage1": self._serialize_stage1_model(),
        }

    def _serialize_stage1_model(self) -> Optional[Dict]:
        if not self.stage1_model:
            return None
        return {
            "bundle_names": list(self.stage1_model.bundle_names),
            "alpha": self.stage1_model.alpha.tolist(),
            "beta": self.stage1_model.beta.tolist(),
            "sigma": self.stage1_model.sigma.tolist(),
            "gamma_ref": self.stage1_model.gamma_ref.tolist(),
            "delta_ref": self.stage1_model.delta_ref.tolist(),
            "gamma_mov": self.stage1_model.gamma_mov.tolist(),
            "delta_mov": self.stage1_model.delta_mov.tolist(),
            "model_params": self.stage1_model.model_params,
        }

    # ---------------------------------------------------------------------
    # Core algorithm
    # ---------------------------------------------------------------------
    def fit(self, ref_data, mov_data, HC_only=True):
        ref_data, mov_data = self.prepare_data(ref_data, mov_data, HC_only)
        ref_data["sid"] = ref_data["sid"].astype(str)
        mov_data["sid"] = mov_data["sid"].astype(str)
        ref_data["site"] = ref_data["site"].astype(str)
        mov_data["site"] = mov_data["site"].astype(str)
        self._fit_apply_result = None

        combined = pd.concat([ref_data, mov_data], ignore_index=True)
        pivot = combined.pivot(index="bundle", columns="sid", values="mean")
        pivot.columns = pivot.columns.astype(str)

        try:
            best_feature, assignments, gmix = self._fit_hidden_batch(pivot)
        except RuntimeError:
            # fallback to classic ComBat
            super().fit(ref_data, mov_data, HC_only)
            self.stage1_model = None
            self.gmm_best_feature = None
            self.gmm_weights = None
            self.gmm_means = None
            self.gmm_covariances = None
            self.gmm_reference_label = None
            self._fit_apply_result = super().apply(mov_data)
            return self._fit_apply_result

        self.gmm_best_feature = best_feature
        self.gmm_weights = gmix.weights_.astype(float)
        self.gmm_means = gmix.means_.astype(float)
        self.gmm_covariances = gmix.covariances_.astype(float)

        assignments_series = pd.Series(assignments, index=pivot.columns)
        ref_site = str(ref_data["site"].iloc[0])
        ref_subjects = combined.loc[combined["site"] == ref_site, "sid"].astype(str)
        counts = assignments_series.loc[ref_subjects].value_counts()
        if counts.empty:
            # Should not happen, but fall back if all ref subjects filtered out
            super().fit(ref_data, mov_data, HC_only)
            self.stage1_model = None
            self.gmm_reference_label = None
            self.gmm_weights = None
            self.gmm_means = None
            self.gmm_covariances = None
            self._fit_apply_result = super().apply(mov_data)
            return self._fit_apply_result

        self.gmm_reference_label = int(counts.idxmax())
        moving_label = 1 - self.gmm_reference_label

        ref_cluster_ids = assignments_series.index[assignments_series == self.gmm_reference_label]
        mov_cluster_ids = assignments_series.index[assignments_series == moving_label]

        cluster_ref_df = combined[combined["sid"].isin(ref_cluster_ids)].copy()
        cluster_mov_df = combined[combined["sid"].isin(mov_cluster_ids)].copy()

        cluster_ref_df["sid"] = cluster_ref_df["sid"].astype(str)
        cluster_mov_df["sid"] = cluster_mov_df["sid"].astype(str)

        if cluster_mov_df.empty or cluster_ref_df.empty:
            super().fit(ref_data, mov_data, HC_only)
            self.stage1_model = None
            self.gmm_weights = None
            self.gmm_means = None
            self.gmm_covariances = None
            self.gmm_reference_label = None
            self._fit_apply_result = super().apply(mov_data)
            return self._fit_apply_result

        # Stage 1: harmonize between hidden clusters using classic ComBat
        stage1_model = QuickCombatClassic(
            bundle_names=None,
            ignore_sex_covariate=self.ignore_sex_covariate,
            ignore_handedness_covariate=self.ignore_handedness_covariate,
            use_empirical_bayes=self.use_empirical_bayes,
            limit_age_range=self.limit_age_range,
            degree=self.degree,
            regul_ref=self.regul_ref,
            regul_mov=self.regul_mov,
        )
        cluster_ref_prepared, cluster_mov_prepared = stage1_model.prepare_data(
            cluster_ref_df, cluster_mov_df, HC_only
        )
        stage1_model.fit(cluster_ref_prepared, cluster_mov_prepared, HC_only)
        mov_harm_arrays = stage1_model.apply(cluster_mov_prepared)
        cluster_mov_harmonized = self._replace_mean_values(
            cluster_mov_prepared.copy(), stage1_model.bundle_names, mov_harm_arrays
        )
        combined_stage1 = pd.concat([cluster_ref_prepared, cluster_mov_harmonized], ignore_index=True)

        # Stage 2: standard site-based ComBat on the stage1 data
        ref_stage1 = combined_stage1[combined_stage1["site"] == ref_site].copy()
        mov_stage1 = combined_stage1[
            combined_stage1["site"] == str(mov_data["site"].iloc[0])
        ].copy()

        ref_stage1_prepared, mov_stage1_prepared = self.prepare_data(ref_stage1, mov_stage1, HC_only)
        ref_stage1_prepared["sid"] = ref_stage1_prepared["sid"].astype(str)
        mov_stage1_prepared["sid"] = mov_stage1_prepared["sid"].astype(str)
        super().fit(ref_stage1_prepared, mov_stage1_prepared, HC_only)
        self._fit_apply_result = super().apply(mov_stage1_prepared)

        # Persist stage-1 model parameters for later application
        self.stage1_model = stage1_model
        return self._fit_apply_result

    # ------------------------------------------------------------------
    def apply(self, data):
        if self.stage1_model is None or self.gmm_best_feature is None:
            return super().apply(data)

        data = data.copy()
        data["sid"] = data["sid"].astype(str)
        data = data.sort_values(["sid", "bundle"]).reset_index(drop=True)

        data_stage1 = self._apply_stage1(data)
        data_stage1 = data_stage1.sort_values(["sid", "bundle"]).reset_index(drop=True)
        return super().apply(data_stage1)

    # ------------------------------------------------------------------
    def _fit_hidden_batch(self, pivot: pd.DataFrame) -> Tuple[str, np.ndarray, GaussianMixture]:
        caseno = pivot.columns
        best_feature = None
        best_model: Optional[GaussianMixture] = None
        best_preds: Optional[np.ndarray] = None
        best_aic = np.inf

        for feature in pivot.index:
            values = pivot.loc[feature].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                # Skip features with NaNs or infs
                continue
            duplicated = np.column_stack((values, values))
            try:
                gmix = GaussianMixture(
                    n_components=min(self.gmm_components, 2),
                    tol=self.gmm_tol,
                    max_iter=self.gmm_max_iter,
                    covariance_type="full",
                )
                gmix.fit(duplicated)
            except ValueError:
                continue

            preds = gmix.predict(duplicated)
            counts = np.bincount(preds, minlength=2)
            if counts.min() <= 0.25 * len(caseno):
                continue

            aic = gmix.aic(duplicated)
            if aic < best_aic:
                best_aic = aic
                best_feature = feature
                best_model = gmix
                best_preds = preds

        if best_feature is None or best_model is None or best_preds is None:
            raise RuntimeError("No valid Gaussian mixture split found.")

        return best_feature, best_preds, best_model

    def _apply_stage1(self, data: pd.DataFrame) -> pd.DataFrame:
        pivot = data.pivot(index="bundle", columns="sid", values="mean")
        if self.gmm_best_feature not in pivot.index:
            return data

        values = pivot.loc[self.gmm_best_feature].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            return data
        duplicated = np.column_stack((values, values))
        labels = self._predict_gmm_labels(duplicated)
        moving_label = 1 - int(self.gmm_reference_label)

        subjects = pivot.columns
        moving_ids = [sid for sid, label in zip(subjects, labels) if label == moving_label]
        if not moving_ids:
            return data

        mov_subset = data[data["sid"].isin(moving_ids)].copy()
        mov_subset = mov_subset.sort_values(["sid", "bundle"]).reset_index(drop=True)
        harm_arrays = self.stage1_model.apply(mov_subset)
        mov_harmonized = self._replace_mean_values(
            mov_subset, self.stage1_model.bundle_names, harm_arrays
        )

        remaining = data[~data["sid"].isin(moving_ids)].copy()
        combined = pd.concat([mov_harmonized, remaining], ignore_index=True)
        return combined

    def _predict_gmm_labels(self, X: np.ndarray) -> np.ndarray:
        if self.gmm_weights is None or self.gmm_means is None or self.gmm_covariances is None:
            raise AssertionError("GMM parameters are not loaded.")
        eps = 1e-12
        log_probs = []
        for weight, mean, cov in zip(self.gmm_weights, self.gmm_means, self.gmm_covariances):
            cov = np.asarray(cov, dtype=float)
            inv_cov = np.linalg.inv(cov)
            log_det = np.linalg.slogdet(cov)[1]
            diff = X - mean
            exponent = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
            log_prob = np.log(weight + eps) - 0.5 * (
                exponent + log_det + X.shape[1] * np.log(2 * np.pi)
            )
            log_probs.append(log_prob)
        log_probs = np.vstack(log_probs)
        return np.argmax(log_probs, axis=0)

    @staticmethod
    def _replace_mean_values(
        data: pd.DataFrame, bundle_names: Sequence[str], harmonized_values: Sequence[np.ndarray]
    ) -> pd.DataFrame:
        updated = data.copy()
        for bundle, values in zip(bundle_names, harmonized_values):
            mask = updated["bundle"] == bundle
            updated.loc[mask, "mean"] = values
        return updated

    # ------------------------------------------------------------------
    def get_fit_apply_result(self) -> Optional[List[np.ndarray]]:
        """Return the harmonized values computed during the last fit call."""
        return self._fit_apply_result


__all__ = ["QuickCombatGmm"]
