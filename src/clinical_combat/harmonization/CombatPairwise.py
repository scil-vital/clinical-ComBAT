# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from clinical_combat.harmonization.Combat import Combat


class CombatPairwise(Combat):
    """
    ComBat: Harmonize the moving site to the reference site.
    Regression parameters are jointly fitted as in Fortin et al. 2017.
    """

    def __init__(
            self,
            bundle_names=None,
            model_params=None,
            ignore_sex_covariate=False,
            ignore_handedness_covariate=False,
            use_empirical_bayes=True,
            limit_age_range=False,
            degree=1,
            regul=0,
            alpha=None,
            beta=None,
            sigma=None,
            gamma_ref=None,
            delta_ref=None,
            gamma_mov=None,
            delta_mov=None,
            ):
        """
        regul: float
            Regularization parameter.
        alpha: Array
            Covariates intercept parameter.
        beta: Array
            Covariates slope parameters.
        sigma: Array
            Standard deviation.
        gamma_ref: Array
            Additive bias of the reference site.
        delta_ref: Array
            Multiplicative bias of the reference site.
        gamma_mov: Array
            Additive bias of the moving site.
        delta_mov: Array
            Multiplicative bias of the moving site.

        """
        super().__init__(
            bundle_names,
            model_params,
            ignore_sex_covariate,
            ignore_handedness_covariate,
            use_empirical_bayes,
            limit_age_range,
            degree
        )
        self.regul = regul
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.gamma_ref = gamma_ref
        self.delta_ref = delta_ref
        self.gamma_mov = gamma_mov
        self.delta_mov = delta_mov

    def initialize_from_model_params(self, model_filename):
        """
        Initialize the object from a model file

        model_filename: str
            Model filename

        """
        super().initialize_from_model_params(model_filename)
        nb = len(self.get_beta_labels())

        params = np.loadtxt(model_filename, delimiter=",",
                            dtype=str, skiprows=1)
        self.regul = self.model_params["regul"]
        self.bundle_names = params[0, 1:]
        self.alpha = params[1, 1:].astype("float64").transpose()
        self.beta = params[2: 2 + nb, 1:].astype("float64").transpose()
        self.sigma = params[2 + nb, 1:].astype("float64").transpose()
        self.gamma_ref = params[3 + nb, 1:].astype("float64").transpose()
        self.delta_ref = params[4 + nb, 1:].astype("float64").transpose()
        self.gamma_mov = params[5 + nb, 1:].astype("float64").transpose()
        self.delta_mov = params[6 + nb, 1:].astype("float64").transpose()

    def save_model(self, model_filename):
        """
        Save the harmonization model to file.

        model_filename: str
            Model filename.

        """
        params = np.hstack(
            [
                self.bundle_names.reshape(-1, 1),
                self.alpha.reshape(-1, 1),
                self.beta,
                self.sigma.reshape(-1, 1),
                self.gamma_ref.reshape(-1, 1),
                self.delta_ref.reshape(-1, 1),
                self.gamma_mov.reshape(-1, 1),
                self.delta_mov.reshape(-1, 1),
            ]
        ).transpose()

        param_labels = ["bundle_names", "intercept"]
        param_labels.extend(self.get_beta_labels())
        param_labels.append("sigma")
        for site in ["ref", "mov"]:
            param_labels.append(site + "_gamma")
            param_labels.append(site + "_delta")

        param_labels = np.array(param_labels).reshape([-1, 1])

        params = np.hstack([param_labels, params])
        header = str(self.model_params)
        np.savetxt(model_filename, params, delimiter=",",
                   fmt="%s", header=header)

    def set_model_fit_params(self, ref_data, mov_data):
        """
        Set the model parameter given the input data used for the fit.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.

        """
        super().set_model_fit_params(ref_data, mov_data)
        self.model_params["regul"] = self.regul
        self.model_params["name"] = "pairwise"

    def standardize_moving_data(self, X, Y):
        """
        Standardize the data (Y). Combat standardize the data with
        the jointly estimated covariate effect,
        intercept and standard deviation.

        .. math::
        S_Y = (Y - X^T B - alpha) / sigma

        X: array
            The design matrix of the covariates.
        Y: array
            The values corresponding to the design matrix.
        """
        s_y = []
        for i in range(len(X)):
            covariate_effect = np.dot(X[i][1:, :].transpose(), self.beta[i])
            s_y.append(
                (Y[i] - self.alpha[i] - covariate_effect) / self.sigma[i]
            )
        return s_y

    def fit(self, ref_data, mov_data):
        """
        Combat Pairwise fit. The moving site beta and alpha
        are fitted using all data.

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
        self.alpha, self.beta = Combat.get_alpha_beta(design_all, y_all,
                                                           regul=self.regul)
        self.sigma = Combat.get_sigma(design_all, y_all,
                                           self.alpha, self.beta)

        z = self.standardize_moving_data(design_mov, y_mov)
        self.gamma_mov = np.array([np.mean(x) for x in z])
        self.delta_mov = np.array([np.std(x, ddof=1) for x in z])

        if self.use_empirical_bayes:
            self.gamma_mov, self.delta_mov = Combat.emperical_bayes_estimate(
                z,
                self.gamma_mov,
                self.delta_mov**2,
            )
        self.gamma_mov *= self.sigma

        z_ref = self.standardize_moving_data(design_ref, y_ref)
        self.gamma_ref = np.array([np.mean(x) for x in z_ref])
        self.delta_ref = np.array([np.std(x, ddof=1) for x in z_ref])
        self.gamma_ref *= self.sigma

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
            self.alpha is None
            or self.beta is None
            or self.sigma is None
            or self.gamma_ref is None
            or self.delta_ref is None
            or self.gamma_mov is None
            or self.delta_mov is None
        ):
            raise AssertionError("Model parameters are not fitted.")

        design, Y = self.get_design_matrices(data)
        harm_y = []

        for i in range(len(design)):
            covariate_effect = np.dot(
                design[i][1:, :].transpose(), self.beta[i]
            )

            harm_y.append(
                (self.delta_ref[i] / self.delta_mov[i])
                * (Y[i] - self.alpha[i] - covariate_effect - self.gamma_mov[i])
                + self.gamma_ref[i]
                + self.alpha[i]
                + covariate_effect
            )
        return harm_y

    def predict(self, ages, bundle, moving_site=True):
        """
        Use the model to predict values.

        ages: array
            Age use to do the prediction.

        bundle: str
            Bundle to use.

        Returns
        -------
        y: array
            Model-predicted value for the input ages.
        """

        design = []
        design.append(np.ones(len(ages)))  # intercept

        if not self.ignore_sex_covariate:
            design.append(np.ones(len(ages)) * 0.5)

        if not self.ignore_handedness_covariate:
            design.append(np.ones(len(ages)) * 0.5)

        # Elevate to a polynomial of degree the age data
        for degree in np.arange(1, self.degree + 1):
            design.append(ages**degree)

        design = np.array(design)

        idx = list(self.bundle_names).index(bundle)

        # There is single alpha/beta
        if moving_site:
            B = np.hstack([self.alpha[idx], self.beta[idx]])
        else:
            B = np.hstack([self.alpha[idx], self.beta[idx]])

        y = np.dot(design.transpose(), B)
        return y
