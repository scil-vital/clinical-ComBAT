# -*- coding: utf-8 -*-
import numpy as np

from clinical_combat.harmonization.QuickCombat import QuickCombat


class QuickCombatPairwise(QuickCombat):
    """
    Quick ComBat: Harmonize the moving site to the reference site.
    Each site regression parameters is fitted independently.
    """

    def set_model_fit_params(self, ref_data, mov_data):
        """
        Set the model parameter given the input data used for the fit.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.

        """
        super().set_model_fit_params(ref_data, mov_data)
        self.model_params["name"] = "pairwise"

    def standardize_moving_data(self, X, Y):
        """
        Standardize the data (Y). Combat Pairwise standardize the moving site data with the
        moving site intercept. Because the data are harmonize to the reference site, sigma is
        obtained from the reference site data.

        .. math::
        S_Y = (Y - X^T B - alpha_{mov}) / sigma_{ref}

        X: array
            The design matrix of the covariates.
        Y: array
            The values corresponding to the design matrix.
        """
        s_y = []
        for i in range(len(X)):
            covariate_effect = np.dot(X[i][1:, :].transpose(), self.beta_mov[i])
            s_y.append(
                (Y[i] - self.alpha_mov[i] - covariate_effect) / (self.sigma_ref[i])
            )
        return s_y

    def fit(self, ref_data, mov_data):
        """
        Combat Pairwise fit. The moving site beta and alpha are fitted using the moving site data.
        The reference site alpha and beta is fitted using the reference site data.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.
        """

        ref_data, mov_data = self.prepare_data(ref_data, mov_data)

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
            design_mov, y_mov, self.regul_mov
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
            self.gamma, self.delta = QuickCombat.emperical_bayes_estimate(
                z,
                self.gamma,
                self.delta,
            )
        self.set_model_fit_params(ref_data, mov_data)
        return
