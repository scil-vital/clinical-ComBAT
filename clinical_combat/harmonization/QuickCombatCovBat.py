# -*- coding: utf-8 -*-
"""CovBat harmonization method built on top of QuickCombatClassic."""
import logging
from typing import Dict, Optional

import numpy as np

from clinical_combat.harmonization.QuickCombatClassic import QuickCombatClassic


class QuickCombatCovBat(QuickCombatClassic):
    """Quick CovBat implementation following Chen et al. (2021).

    The method first applies classic ComBat to align means and variances, then
    adjusts the covariance structure in a shared principal component space.
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
        regul_mov: float = 0,
        alpha=None,
        beta=None,
        sigma=None,
        gamma_ref=None,
        delta_ref=None,
        gamma_mov=None,
        delta_mov=None,
        covbat_pve: float = 0.95,
        covbat_max_components: Optional[int] = None,
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
        self.covbat_pve = covbat_pve
        self.covbat_max_components = covbat_max_components
        self.covbat_components: Optional[np.ndarray] = None
        self.covbat_feature_mean: Optional[np.ndarray] = None
        self.covbat_site_means: Dict[str, np.ndarray] = {}
        self.covbat_site_stds: Dict[str, np.ndarray] = {}
        self.covbat_target_site: Optional[str] = None
        self.covbat_eigenvalues: Optional[np.ndarray] = None

    def initialize_from_model_params(self, model_filename):
        super().initialize_from_model_params(model_filename)
        covbat_params = self.model_params.get("covbat")
        if covbat_params is None:
            raise AssertionError(
                "CovBat parameters not found in the model header; the model cannot be loaded."
            )

        self.covbat_pve = covbat_params.get("pve", self.covbat_pve)
        self.covbat_feature_mean = np.array(
            covbat_params.get("feature_mean", []), dtype=float
        )
        components = covbat_params.get("components", [])
        self.covbat_components = np.array(components, dtype=float)
        site_means = {}
        for site, values in covbat_params.get("site_means", {}).items():
            site_means[str(site)] = np.array(values, dtype=float)
        self.covbat_site_means = site_means
        site_stds = {}
        for site, values in covbat_params.get("site_stds", {}).items():
            site_stds[str(site)] = np.array(values, dtype=float)
        self.covbat_site_stds = site_stds
        self.covbat_target_site = covbat_params.get("target_site")
        self.covbat_eigenvalues = np.array(
            covbat_params.get("eigenvalues", []), dtype=float
        )

    def save_model(self, model_filename):
        if self.covbat_components is None or self.covbat_feature_mean is None:
            raise AssertionError(
                "CovBat parameters are missing; fit must be run before saving the model."
            )
        covbat_params = {
            "pve": float(self.covbat_pve),
            "feature_mean": self.covbat_feature_mean.tolist(),
            "components": self.covbat_components.tolist(),
            "site_means": {k: v.tolist() for k, v in self.covbat_site_means.items()},
            "site_stds": {k: v.tolist() for k, v in self.covbat_site_stds.items()},
            "target_site": self.covbat_target_site,
            "eigenvalues": []
            if self.covbat_eigenvalues is None
            else self.covbat_eigenvalues.tolist(),
        }
        self.model_params["covbat"] = covbat_params
        self.model_params["name"] = "covbat"
        super().save_model(model_filename)

    def fit(self, ref_data, mov_data, HC_only=True):
        super().fit(ref_data, mov_data, HC_only)

        # Reuse the cleaned/ordered data produced during the base fit so that
        # the CovBat computations stay consistent with the fitted parameters.
        ref_data, mov_data = self.prepare_data(ref_data, mov_data, HC_only)

        ref_site = str(np.unique(ref_data["site"])[0])
        mov_site = str(np.unique(mov_data["site"])[0])
        design_ref, y_ref = self.get_design_matrices(ref_data)
        design_mov, y_mov = self.get_design_matrices(mov_data)

        residual_ref, _ = self._compute_residual_and_expected(design_ref, y_ref, "ref")
        residual_mov, _ = self._compute_residual_and_expected(design_mov, y_mov, "mov")

        combined_residual = np.vstack([residual_ref, residual_mov])
        if combined_residual.size == 0:
            logging.warning("CovBat skipped due to empty residual matrix; falling back to classic ComBat.")
            self.covbat_components = np.zeros((len(self.bundle_names), 0))
            self.covbat_feature_mean = np.zeros(len(self.bundle_names))
            self.covbat_site_means = {ref_site: np.array([]), mov_site: np.array([])}
            self.covbat_site_stds = {ref_site: np.array([]), mov_site: np.array([])}
            self.covbat_target_site = ref_site
            self.covbat_eigenvalues = np.array([])
            return

        feature_mean = combined_residual.mean(axis=0)
        centered = combined_residual - feature_mean

        if centered.shape[0] <= 1:
            logging.warning("CovBat skipped because covariance cannot be estimated (<=1 subject).")
            self.covbat_components = np.zeros((len(self.bundle_names), 0))
            self.covbat_feature_mean = feature_mean
            self.covbat_site_means = {ref_site: np.array([]), mov_site: np.array([])}
            self.covbat_site_stds = {ref_site: np.array([]), mov_site: np.array([])}
            self.covbat_target_site = ref_site
            self.covbat_eigenvalues = np.array([])
            return

        cov_mat = np.cov(centered, rowvar=False, ddof=1)
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        total_var = np.sum(eigvals)
        if total_var <= 0:
            logging.warning("CovBat skipped because covariance has zero variance.")
            self.covbat_components = np.zeros((len(self.bundle_names), 0))
            self.covbat_feature_mean = feature_mean
            self.covbat_site_means = {ref_site: np.array([]), mov_site: np.array([])}
            self.covbat_site_stds = {ref_site: np.array([]), mov_site: np.array([])}
            self.covbat_target_site = ref_site
            self.covbat_eigenvalues = np.array([])
            return

        explained = eigvals / total_var
        cumulative = np.cumsum(explained)
        n_features = len(self.bundle_names)
        max_components = self.covbat_max_components or n_features
        q = np.searchsorted(cumulative, self.covbat_pve) + 1
        q = min(max(q, 0), max_components)
        if q <= 0:
            logging.info("CovBat threshold yielded zero components; falling back to classic ComBat.")
            self.covbat_components = np.zeros((len(self.bundle_names), 0))
            self.covbat_feature_mean = feature_mean
            self.covbat_site_means = {ref_site: np.array([]), mov_site: np.array([])}
            self.covbat_site_stds = {ref_site: np.array([]), mov_site: np.array([])}
            self.covbat_target_site = ref_site
            self.covbat_eigenvalues = np.array([])
            return

        components = eigvecs[:, :q]
        self.covbat_components = components
        self.covbat_feature_mean = feature_mean
        self.covbat_target_site = ref_site
        self.covbat_eigenvalues = eigvals[:q]

        ref_centered = residual_ref - feature_mean
        mov_centered = residual_mov - feature_mean
        ref_scores = ref_centered @ components
        mov_scores = mov_centered @ components

        self.covbat_site_means = {
            ref_site: ref_scores.mean(axis=0) if ref_scores.size else np.zeros(q),
            mov_site: mov_scores.mean(axis=0) if mov_scores.size else np.zeros(q),
        }
        self.covbat_site_stds = {
            ref_site: self._safe_std(ref_scores),
            mov_site: self._safe_std(mov_scores),
        }

        self.model_params["name"] = "covbat"
        self.model_params["covbat"] = {
            "pve": float(self.covbat_pve),
            "feature_mean": self.covbat_feature_mean.tolist(),
            "components": self.covbat_components.tolist(),
            "site_means": {k: v.tolist() for k, v in self.covbat_site_means.items()},
            "site_stds": {k: v.tolist() for k, v in self.covbat_site_stds.items()},
            "target_site": self.covbat_target_site,
            "eigenvalues": self.covbat_eigenvalues.tolist(),
        }

    def apply(self, data):
        if self.covbat_components is None or self.covbat_feature_mean is None:
            raise AssertionError("CovBat model parameters are not loaded.")
        if self.covbat_components.size == 0:
            return super().apply(data)

        site_values = np.unique(data["site"].astype(str))
        # if len(site_values) != 1:
        #     raise AssertionError(f"CovBat apply expects data from a single site. Sites in data: {site_values}")
        site = site_values[0]
        if site not in self.covbat_site_means:
            logging.warning(
                "Site %s not seen during CovBat fit; falling back to classic ComBat for this data.",
                site,
            )
            return super().apply(data)

        design, Y = self.get_design_matrices(data)
        residual, expected = self._compute_residual_and_expected(design, Y, "ref" if site == self.covbat_target_site else "mov")
        centered = residual - self.covbat_feature_mean
        scores = centered @ self.covbat_components
        orthogonal = centered - scores @ self.covbat_components.T

        target_mean = self.covbat_site_means[self.covbat_target_site]
        target_std = self.covbat_site_stds[self.covbat_target_site]
        source_mean = self.covbat_site_means[site]
        source_std = self.covbat_site_stds[site]

        target_std = np.where(target_std <= 0, 1.0, target_std)
        source_std = np.where(source_std <= 0, 1.0, source_std)

        adjusted_scores = (scores - source_mean) / source_std * target_std + target_mean
        adjusted_centered = adjusted_scores @ self.covbat_components.T + orthogonal
        adjusted_residual = adjusted_centered + self.covbat_feature_mean
        harmonized = adjusted_residual + expected

        # Reformat to match QuickCombatClassic.apply output: list of arrays per bundle.
        harmonized_list = [harmonized[:, i] for i in range(harmonized.shape[1])]
        return harmonized_list

    def _compute_residual_and_expected(self, design, Y, site):
        n_bundles = len(self.bundle_names)
        if n_bundles == 0:
            return np.empty((0, 0)), np.empty((0, 0))

        n_subjects = len(Y[0]) if Y else 0
        residual = np.zeros((n_subjects, n_bundles))
        expected = np.zeros((n_subjects, n_bundles))

        for i in range(n_bundles):
            cov_effect = np.dot(design[i][1:, :].transpose(), self.beta[i]) if design[i].shape[0] > 1 else np.zeros(n_subjects)
            z = (Y[i] - self.alpha[i] - cov_effect) / self.sigma[i]
            if site == "ref":
                gamma_std = self.gamma_ref[i] / self.sigma[i]
                delta_site = self.delta_ref[i]
            elif site == "mov":
                gamma_std = self.gamma_mov[i] / self.sigma[i]
                delta_site = self.delta_mov[i]
            else:
                raise AssertionError(f"Unknown site type '{site}' for CovBat residual computation.")

            z_star = (z - gamma_std) / delta_site
            residual[:, i] = z_star * self.sigma[i] * self.delta_ref[i]
            expected[:, i] = self.alpha[i] + cov_effect + self.gamma_ref[i]

        return residual, expected

    @staticmethod
    def _safe_std(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return np.array([])
        std = scores.std(axis=0, ddof=1) if scores.shape[0] > 1 else np.zeros(scores.shape[1])
        std = np.where(std <= 0, 1.0, std)
        return std
