# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from clinical_combat.harmonization.QuickCombat import QuickCombat


class QuickCombatClassic(QuickCombat):
    """
    Quick ComBat: Harmonize the moving site to the reference site.
    Regression parameters are jointly fitted as in Fortin et al. 2017.
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
        self.model_params["name"] = "classic"

    def standardize_moving_data(self, X, Y):
        """
        Standardize the data (Y). Combat standardize the data with 
        the jointly estimated covariate effect, intercept and standard deviation. 

        .. math::
        S_Y = (Y - X^T B - alpha) / sigma

        X: array
            The design matrix of the covariates.
        Y: array
            The values corresponding to the design matrix.
        """
        s_y = []
        for i in range(len(X)):
            covariate_effect = np.dot(X[i][1:, :].transpose(), self.beta_mov[i])
            s_y.append(
                (Y[i] - self.alpha_mov[i] - covariate_effect) / self.sigma_mov[i]
            )
        return s_y

    def fit(self, ref_data, mov_data):
        """
        Combat Classic fit. The moving site beta and alpha are fitted using all 
        data.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.
        """
        ref_data, mov_data = self.prepare_data(ref_data, mov_data)
 
        # fit intercept and covariates of the moving site using all data
        all_data = pd.concat([ref_data, mov_data])

        design_mov, y_mov = self.get_design_matrices(mov_data)
        design_ref, y_ref = self.get_design_matrices(ref_data)
        design_all, y_all = self.get_design_matrices(all_data)
        self.alpha_mov, self.beta_mov = QuickCombat.get_alpha_beta(design_all, y_all)
        self.sigma_mov = QuickCombat.get_sigma(
            design_all,
            y_all,
            self.alpha_mov,
            self.beta_mov,
        )
        self.sigma_ref = self.sigma_mov
        self.alpha_ref = self.alpha_mov
        self.beta_ref = self.beta_mov

        z = self.standardize_moving_data(design_mov, y_mov)
        self.gamma_mov = np.array([np.mean(x) for x in z])
        self.delta_mov = np.array([np.var(x, ddof=1) for x in z])

        if self.use_empirical_bayes:
            self.gamma_mov, self.delta_mov = QuickCombat.emperical_bayes_estimate(
                z,
                self.gamma_mov,
                self.delta_mov,
            )
        self.gamma_mov *= self.sigma_mov
        
        z_ref = self.standardize_moving_data(design_ref, y_ref)
        self.gamma_ref = np.array([np.mean(x) for x in z_ref])
        self.delta_ref = np.array([np.var(x, ddof=1) for x in z_ref])
        self.gamma_ref *= self.sigma_mov
        
        self.set_model_fit_params(ref_data, mov_data)
        return


    def apply(self, data):
        """
        Apply the harmonization fitted model to data.

        data: df
            Dataframe representing the data to harmonized.

        Returns
        -------
        harm_y: array
            Harmonized data values.
        """
        if (
            self.alpha_ref is None
            or self.beta_ref is None
            or self.sigma_ref is None
            or self.gamma_ref is None
            or self.delta_ref is None
            or self.alpha_mov is None
            or self.beta_mov is None
            or self.sigma_mov is None
            or self.gamma_mov is None
            or self.delta_mov is None
        ):
            raise AssertionError("Model parameters are not fitted.")

        design, Y = self.get_design_matrices(data)
        harm_y = []

        for i in range(len(design)):
            covariate_effect = np.dot(
                design[i][1:, :].transpose(), self.beta_mov[i]
            )

            harm_y.append(
                (self.delta_ref[i] / self.delta_mov[i]) 
                * (Y[i] - self.alpha_mov[i] - covariate_effect - self.gamma_mov[i])
                + self.gamma_ref[i]
                + self.alpha_mov[i]              
                + covariate_effect
            )
            
        return harm_y
