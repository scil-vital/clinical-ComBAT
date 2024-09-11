# -*- coding: utf-8 -*-
import logging

import numpy as np

from clinical_combat.harmonization.QuickCombat import QuickCombat
from clinical_combat.harmonization.QuickCombatPairwise import QuickCombatPairwise


class QuickCombatClinic(QuickCombatPairwise):
    """
    Quick ComBat: Harmonize the moving site to the reference site.
    Each site regression parameters is fitted independently.
    """

    def __init__(
        self,
        bundle_names=None,
        model_params=None,
        ignore_sex_covariate=False,
        ignore_handedness_covariate=False,
        alpha_ref=None,
        beta_ref=None,
        sigma_ref=None,
        alpha_mov=None,
        beta_mov=None,
        sigma_mov=None,
        gamma=None,
        delta=None,
        use_empirical_bayes=True,
        limit_age_range=False,
        degree=1,
        regul_ref=0,
        regul_mov=0,
        nu=0,
        tau=1,
    ):
        """
        alpha_ref: Array
            Covariates intercept parameter of the reference site.
        beta_ref: Array
            Covariates slope parameters of the reference site.
        sigma_ref: Array
            Standard deviation of the reference site.
        alpha_mov: Array
            Covariates intercept parameter of the moving site.
        beta_mov: Array
            Covariates slope parameters of the moving site.
        sigma_mov: Array
            Standard deviation of the moving site.
        gamma: Array
            Additive bias between the moving and the reference sites.
        delta: Array
            Multiplicative bias between the moving and the reference sites.
        use_empirical_bayes: bool
            Uses empirical Bayes estimator for alpha and sigma estimation.
        limit_age_range: bool
            Remove reference data with age outside the range of the moving site.
        degree: int
            Polynomial degree of the age fit.
        regul_ref: float
            Regularization parameter for the reference site data.
        regul_mov: float
            Regularization parameter for the moving site data.
        nu: float
            Hyperparameter for the standard deviation estimation of the moving site data.
        tau: float
            Hyperparameter for the covariate fit of the moving site data.

        """
        super().__init__(
            bundle_names,
            model_params,
            ignore_sex_covariate,
            ignore_handedness_covariate,
            alpha_ref=alpha_ref,
            beta_ref=beta_ref,
            sigma_ref=sigma_ref,
            alpha_mov=alpha_mov,
            beta_mov=beta_mov,
            sigma_mov=sigma_mov,
            gamma=gamma,
            delta=delta,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov,
        )
        self.nu = nu
        self.tau = tau
        if self.nu < 0:
            raise AssertionError("nu must be greater or equal to 0.")
        if self.tau < 1:
            raise AssertionError("tau must be greater or equal to 1.")

    def initialize_from_model_params(self, model_filename):
        """
        Initialize the object from a model file

        model_filename: str
            Model filename

        """
        super().initialize_from_model_params(model_filename)
        self.nu = self.model_params["nu"]

    def set_model_fit_params(self, ref_data, mov_data):
        """
        Set the model parameter given the input data used for the fit.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.

        """
        super().set_model_fit_params(ref_data, mov_data)
        self.model_params["nu"] = self.nu
        self.model_params["name"] = "clinic"

    def regularization_parameter_tuning(
        self, ref_data, mov_data, step=1.5, max_reg=1e10
    ):
        age_min = int(np.floor(ref_data.age.min()))
        age_max = int(np.ceil(ref_data.age.max()))
        x = np.arange(age_min, age_max + 1, 1)

        ages = set(
            np.ceil(mov_data.age.to_numpy()).astype(int).tolist()
            + np.floor(mov_data.age.to_numpy()).astype(int).tolist()
        )

        mov_mask = np.zeros(len(x))
        for i, xx in enumerate(x):
            if xx in ages:
                mov_mask[i] = 1

        # fit intercept and covariates of the reference site using the reference site data
        design_ref, y_ref = self.get_design_matrices(ref_data)
        self.alpha_ref, self.beta_ref = QuickCombat.get_alpha_beta(
            design_ref, y_ref, self.regul_ref
        )
        ref_models = []
        for b in self.bundle_names:
            ref_models.append(self.predict(x, b, moving_site=False))

        design_mov, y_mov = self.get_design_matrices(mov_data)
        current_reg = 0.01
        while current_reg < max_reg:
            self.alpha_mov, self.beta_mov = QuickCombat.get_alpha_beta(
                design_mov,
                y_mov,
                current_reg,
                np.hstack([self.alpha_ref[:, None], self.beta_ref]),
            )
            mov_models = []
            for b in self.bundle_names:
                mov_models.append(self.predict(x, b, moving_site=True))

            error = self.eval_fit(ref_models, mov_models, mov_mask)
            logging.debug("Current reg: %d  Minimizing term: %d", current_reg, error)
            if error == 0:
                # found the smallest working lambda
                logging.info("Optimal reg term found: %d", current_reg)
                return current_reg

            current_reg *= step

        # failed to find a working lambda
        logging.warning("No optimal reg term found, set reg to: %d", max_reg)
        return max_reg

    def eval_fit(self, ref_models, mov_models, mov_mask):

        tau1 = 1 / self.tau
        tau2 = self.tau
        evals = []
        for ref_model, mov_model in zip(ref_models, mov_models):
            distances = ref_model - mov_model
            d1 = np.abs(np.min(distances))
            d2 = np.abs(np.max(distances))
            dmin = np.abs(np.min(distances[mov_mask > 0]))
            dmax = np.abs(np.max(distances[mov_mask > 0]))
            v = np.sign(dmin * tau1 - d1) + np.sign(d2 - dmax * tau2) + 2
            evals.append(v)

        return np.sum(evals)

    def fit(self, ref_data, mov_data):
        """
        Combat Clinic fit. The moving site beta and alpha are fitted using the moving site data.
        The reference site alpha and beta is fitted using the reference site data.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.
        """

        ref_data, mov_data = self.prepare_data(ref_data, mov_data)
        if self.regul_mov == -1:
            self.regul_mov = self.regularization_parameter_tuning(ref_data, mov_data)

        # fit intercept and covariates of the reference site using the reference site data
        design_ref, y_ref = self.get_design_matrices(ref_data)
        self.alpha_ref, self.beta_ref = QuickCombat.get_alpha_beta(
            design_ref, y_ref, self.regul_ref
        )
        self.sigma_ref = QuickCombat.get_sigma(
            design_ref, y_ref, self.alpha_ref, self.beta_ref
        )
        # fit intercept and covariates of the moving site using the moving site data
        design_mov, y_mov = self.get_design_matrices(mov_data)
        self.alpha_mov, self.beta_mov = QuickCombat.get_alpha_beta(
            design_mov,
            y_mov,
            self.regul_mov,
            np.hstack([self.alpha_ref[:, None], self.beta_ref]),
        )
        self.sigma_mov = QuickCombat.get_sigma(
            design_mov, y_mov, self.alpha_mov, self.beta_mov
        )

        z = self.standardize_moving_data(design_mov, y_mov)

        self.gamma = np.array([np.mean(x) for x in z])
        self.delta = np.array(
            [np.std(x - self.gamma.reshape(-1, 1), ddof=1) for x in z]
        )

        if self.use_empirical_bayes:
            new_delta = []
            for i in range(len(self.sigma_mov)):
                N = len(y_mov[i])
                # The target normalized std is 1 (self.nu * target_std = self.nu)
                new = (self.delta[i] * N + self.nu) / (N + self.nu)
                new_delta.append(new)
            self.delta = np.array(new_delta)

        self.set_model_fit_params(ref_data, mov_data)
        return
