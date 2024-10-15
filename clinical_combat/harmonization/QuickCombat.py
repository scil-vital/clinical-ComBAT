# -*- coding: utf-8 -*-
import logging
import sys

import numpy as np
from matplotlib.pyplot import *

from clinical_combat.harmonization.QuickHarmonizationMethod import (
    QuickHarmonizationMethod,
)


class QuickCombat(QuickHarmonizationMethod):
    """
    Quick ComBat: Abstract class.
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
        gamma_ref=None,
        delta_ref=None,
        alpha_mov=None,
        beta_mov=None,
        sigma_mov=None,
        gamma_mov=None,
        delta_mov=None,
        use_empirical_bayes=True,
        limit_age_range=False,
        degree=1,
        regul_ref=0,
        regul_mov=0,
    ):
        """
        alpha_ref: Array
            Covariates intercept parameter of the reference site.
        beta_ref: Array
            Covariates slope parameters of the reference site.
        sigma_ref: Array
            Standard deviation of the reference site.
        gamma_ref: Array
            Additive bias of the reference site.
        delta_ref: Array
            Multiplicative bias of the reference site.
        alpha_mov: Array
            Covariates intercept parameter of the moving site.
        beta_mov: Array
            Covariates slope parameters of the moving site.
        sigma_mov: Array
            Standard deviation of the moving site.
        gamma_mov: Array
            Additive bias of the moving site.
        delta_mov: Array
            Multiplicative bias of the moving site.
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

        """
        super().__init__(
            bundle_names,
            model_params,
            ignore_sex_covariate,
            ignore_handedness_covariate,
        )
        self.alpha_ref = alpha_ref
        self.beta_ref = beta_ref
        self.sigma_ref = sigma_ref
        self.gamma_ref = gamma_ref
        self.delta_ref = delta_ref
        self.alpha_mov = alpha_mov
        self.beta_mov = beta_mov
        self.sigma_mov = sigma_mov
        self.gamma_mov = gamma_mov
        self.delta_mov = delta_mov
        self.use_empirical_bayes = use_empirical_bayes
        self.limit_age_range = limit_age_range
        self.degree = degree
        self.regul_ref = regul_ref
        self.regul_mov = regul_mov

        if self.degree < 0:
            raise AssertionError("Degree must be greater than 1.")
        if self.regul_ref < 0:
            raise AssertionError("regul_ref must be greater or equal to 0.")
        if self.regul_mov < 0 and not self.regul_mov == -1:
            raise AssertionError("regul_mov must be greater or equal to 0, or -1.")


    def standardize_data(self, X, Y):
        """
        Abstract function.
        """
        pass

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

        if moving_site:
            B = np.hstack([self.alpha_mov[idx], self.beta_mov[idx]])
        else:
            B = np.hstack([self.alpha_ref[idx], self.beta_ref[idx]])

        y = np.dot(design.transpose(), B)
        return y

    def prepare_data(self, ref_data, mov_data):
        """
        Validate and prepare the input sites data before the model fit.
        Data are sorted, the limit age range is applied and 'HC' are selected.

        ref_data: dataframe
            Reference site input data.

        mov_data: dataframe
            Moving site input data.

        Returns
        -------
        ref_data: dataframe
            Prepared reference site input data.

        mov_data: dataframe
            Prepared moving site input data.
        """
        self.assert_data(ref_data)
        ref_data = ref_data.sort_values(by=["site", "sid", "bundle"])

        self.assert_data(mov_data)
        mov_data = mov_data.sort_values(by=["site", "sid", "bundle"])

        if self.limit_age_range:
            min_age = np.min(mov_data["age"])
            max_age = np.max(mov_data["age"])
            ref_data = ref_data.query("age >= @min_age & age <= @max_age")

        mov_data = mov_data.query("disease == 'HC'")

        self.bundle_names = np.intersect1d(
            ref_data.bundle.unique(), mov_data.bundle.unique()
        )

        if len(self.bundle_names) == 0:
            raise AssertionError(
                "Bundles in the reference site do not match with bundles in "
                "the moving site"
            )
        for b in mov_data.bundle.unique():
            if b not in self.bundle_names:
                logging.info(
                    "Bundle %s was not found in the reference site and will be ignored."
                    % b
                )
        return ref_data, mov_data

    def get_mean_bhattacharyya_distance(self, ref_data, mov_data):
        """
        Returns the mean Bhattacharyya distance across all bundles.

        ref_data: dataframe
            Reference site input data.

        mov_data: dataframe
            Moving site input data.

        """
        return np.mean(self.get_bundles_bhattacharyya_distance(ref_data, mov_data))

    def get_bundles_bhattacharyya_distance(self, ref_data, mov_data):
        """
        Returns the Bhattacharyya distance for all bundles.

        ref_data: dataframe
            Reference site input data.

        mov_data: dataframe
            Moving site input data.
        """
        dists = []
        for bundle in self.bundle_names:
            dists.append(
                self.get_bundle_bhattacharyya_distance(ref_data, mov_data, bundle)
            )
        return dists

    def get_bundle_bhattacharyya_distance(self, ref_data, mov_data, bundle_name):
        """
        Returns the Bhattacharyya distance for one bundle.

        ref_data: dataframe
            Reference site input data.

        mov_data: dataframe
            Moving site input data.

        bundle_name: str
            Name of the bundle.

        """

        bundle_idx = list(self.bundle_names).index(bundle_name)
        ref_data, mov_data = self.prepare_data(ref_data, mov_data)

        design_ref, y_ref = self.get_design_matrices(ref_data)

        covariate_effect_ref = np.dot(
            design_ref[bundle_idx][1:, :].transpose(), self.beta_ref[bundle_idx]
        )
        ref_dist = y_ref[bundle_idx] - self.alpha_ref[bundle_idx] - covariate_effect_ref

        design_mov, y_mov = self.get_design_matrices(mov_data)
        covariate_effect_mov = np.dot(
            design_mov[bundle_idx][1:, :].transpose(), self.beta_ref[bundle_idx]
        )
        mov_dist = y_mov[bundle_idx] - self.alpha_ref[bundle_idx] - covariate_effect_mov

        return QuickCombat.bhattacharyya_distance(ref_dist, mov_dist)

    @staticmethod
    def bhattacharyya_distance(target_dist, moving_dist):
        """
        Compute the Bhattacharyya distance from two 1D gaussian distributions.

        target_dist: array
            Target distribution.

        moving_dist: array
            Moving distribution.

        """
        target_mean = np.mean(target_dist)
        target_std = np.std(target_dist)
        moving_mean = np.mean(moving_dist)
        moving_std = np.std(moving_dist)

        distance_terme1 = (
            1 / 4 * (target_mean - moving_mean) ** 2 / (target_std**2 + moving_std**2)
        )
        distance_terme2 = (
            1
            / 2
            * np.log((target_std**2 + moving_std**2) / (2 * target_std * moving_std))
        )
        return distance_terme1 + distance_terme2

    def initialize_from_model_params(self, model_filename):
        """
        Initialize the object from a model file

        model_filename: str
            Model filename

        """
        super().initialize_from_model_params(model_filename)

        params = np.loadtxt(model_filename, delimiter=",", dtype=str, skiprows=1)

        self.degree = self.model_params["degree"]
        self.regul_ref = self.model_params["regul_ref"]
        self.regul_mov = self.model_params["regul_mov"]
        self.model_params["nbr_beta_params"] = len(self.get_beta_labels())
        nb = self.model_params["nbr_beta_params"]
        self.ignore_handedness_covariate = self.model_params[
            "ignore_handedness_covariate"
        ]
        self.ignore_sex_covariate = self.model_params["ignore_sex_covariate"]

        self.bundle_names = params[0, 1:]
        self.alpha_ref = params[1, 1:].astype("float64").transpose()
        self.beta_ref = params[2 : 2 + nb, 1:].astype("float64").transpose()
        self.sigma_ref = params[2 + nb, 1:].astype("float64").transpose()
        self.gamma_ref = params[3 + nb, 1:].astype("float64").transpose()
        self.delta_ref = params[4 + nb, 1:].astype("float64").transpose()
        self.alpha_mov = params[5 + nb, 1:].astype("float64").transpose()
        self.beta_mov = params[6 + nb : 6 + nb + nb, 1:].astype("float64").transpose()
        self.sigma_mov = params[6 + nb + nb, 1:].astype("float64").transpose()
        self.gamma_mov = params[-2, 1:].astype("float64").transpose()
        self.delta_mov = params[-1, 1:].astype("float64").transpose()

    def save_model(self, model_filename):
        """
        Save the harmonization model to file.

        model_filename: str
            Model filename.

        """
        params = np.hstack(
            [
                self.bundle_names.reshape(-1, 1),
                self.alpha_ref.reshape(-1, 1),
                self.beta_ref,
                self.sigma_ref.reshape(-1, 1),
                self.gamma_ref.reshape(-1, 1),
                self.delta_ref.reshape(-1, 1),
                self.alpha_mov.reshape(-1, 1),
                self.beta_mov,
                self.sigma_mov.reshape(-1, 1),
                self.gamma_mov.reshape(-1, 1),
                self.delta_mov.reshape(-1, 1),
            ]
        ).transpose()

        beta_labels = self.get_beta_labels()
        param_labels = ["bundle_names"]

        for site in ["ref", "mov"]:
            param_labels.append(site + "_intercept")
            for l in beta_labels:
                param_labels.append(site + "_" + l)
            param_labels.append(site + "_std")
            param_labels.append(site + "_gamma")
            param_labels.append(site + "_delta")

        param_labels = np.array(param_labels).reshape([-1, 1])

        params = np.hstack([param_labels, params])
        header = str(self.model_params)
        np.savetxt(model_filename, params, delimiter=",", fmt="%s", header=header)

    def set_model_fit_params(self, ref_data, mov_data):
        """
        Set the model parameter given the input data used for the fit.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.

        """
        super().set_model_fit_params(ref_data, mov_data)
        self.model_params["use_empirical_bayes"] = self.use_empirical_bayes
        self.model_params["limit_age_range"] = self.limit_age_range
        self.model_params["min_age"] = np.min(mov_data["age"])
        self.model_params["max_age"] = np.max(mov_data["age"])
        self.model_params["nbr_beta_params"] = len(self.get_beta_labels())
        self.model_params["degree"] = self.degree
        self.model_params["regul_ref"] = self.regul_ref
        self.model_params["regul_mov"] = self.regul_mov

    def get_beta_labels(self):
        """
        Get the list of labels of the regression parameters.

        Returns
        -------
        beta_labels: list
            Labels of the regression parameters.
        """
        beta_labels = []
        if not self.ignore_sex_covariate:
            beta_labels.append("beta_sex")
        if not self.ignore_handedness_covariate:
            beta_labels.append("beta_handedness")
        for degree in np.arange(1, self.degree + 1):
            beta_labels.append("beta_age" + str(degree))
        return beta_labels

    def get_design_matrices(self, df):
        """
        Compute the design matrices of all bundles for the input data.

        df: dataframe
            Input data

        Returns
        -------
        design: array
            Design matrices for all bundles.
        Y: array
            Values assossiated with the design matrices
        """
        design = []
        Y = []

        for bundle in self.bundle_names:
            data = df.query("bundle == @bundle")
            hstack_list = []
            hstack_list.append(np.ones(len(data["sid"])))  # intercept

            if not self.ignore_sex_covariate:
                hstack_list.append(QuickCombat.to_category(data["sex"]))

            if not self.ignore_handedness_covariate:
                hstack_list.append(QuickCombat.to_category(data["handedness"]))

            # Elevate to a polynomial of degree the age data
            ages = data["age"].to_numpy()
            for degree in np.arange(1, self.degree + 1):
                hstack_list.append(ages**degree)

            design.append(np.array(hstack_list))
            Y.append(data["mean"].to_numpy())

        return design, Y

    @staticmethod
    def to_category(values):
        """
        Change categorical values 1, 2 to 0, 1.

        values: array
            The values to categorise.

        Returns
        -------
        cat: array
            Categorical values.
        """
        cat = np.zeros(len(values))
        cat[values == 1] = 0
        cat[values == 2] = 1
        return cat

    @staticmethod
    def get_alpha_beta(X, Y, regul=0, reference_Bs=None):
        """
        Fit the regression parameters of the covariates. This may include Age, Sex and Handedness.
        The age may be a linear or quadratic fit. See `get_design_matrices(.)`.
        Subjects with nans are removed from the estimation.

        X: array
            The design matrix of the covariates.
        Y: array
            The values corresponding to the design matrix.
        regul: float
            Regularisation term (r)
        reference_Bs: array
            Reference regression parameters to be use as prior.

        .. math::
        B = (X^T X + r*I)^{-1} X^T Y

        Returns
        -------
        alpha: array
            The intercept value for each bundle (B[0]).
        beta: array
            The regression parameters.
        """
        Bs = []
        for i in range(len(X)):
            mod = X[i].transpose()
            yy = Y[i]
            wh = np.isfinite(yy)
            mod = mod[wh, :]
            yy = yy[wh]
            mod_transpose_mod = np.dot(mod.T, mod)

            if reference_Bs is not None:
                ref_w = reference_Bs[i].T + sys.float_info.epsilon
            else:
                ref_w = np.ones(mod_transpose_mod.shape[0]) * sys.float_info.epsilon

            # The amplitude of each term should be proportional of the reference weights.
            regul_mat = (
                regul
                * np.abs(ref_w[0] / ref_w)
                * np.identity(mod_transpose_mod.shape[0])
            )
            # no regul for the bias term
            regul_mat[0, 0] = 0

            mat = mod_transpose_mod + regul_mat
            vec = np.dot(regul_mat, ref_w) + np.dot(mod.T, yy)

            B = np.linalg.solve(mat, vec)
            Bs.append(B)
        Bs = np.array(Bs)
        return Bs[:, 0], Bs[:, 1:]

    @staticmethod
    def get_sigma(X, Y, alpha, beta):
        """
        Calculate the standard deviation of the data after removing the covariate effect.

        X: array
            The design matrix of the covariates.
        Y: array
            The values corresponding to the design matrix.
        alpha: array
            The intercept value for each bundle.
        beta: array
            The regression parameters for each bundle.

        .. math::
        std = std(Y-alpha-XB))

        Return
        ------
        sigma: standard deviation of the data (each bundle)
        """
        stds = []
        for i in range(len(X)):
            s = np.std(
                Y[i] - alpha[i] - np.dot(X[i][1:, :].transpose(), beta[i]),
                ddof=1,
            )
            stds.append(s)
        return np.array(stds)

    @staticmethod
    def estimate_a_prior(delta_hat):
        """
        Estimate the `a` prior. `a` is the shape parameter of an inverse gamma.
        `delta` is the estimate of the multiplicative bias.

        detla_hat: array
            Estimate of the multiplicative bias.

        .. math::
        m <-- mean of delta_hat
        v <-- variance of delta_hat
        b = (2 * v + m^2) / v

        Return
        ------
        a: float
            The shape parameter of the inverse gamma.
        """
        m = np.mean(delta_hat)
        v = np.var(delta_hat, ddof=1)
        return (2 * v + m**2) / v

    @staticmethod
    def estimate_b_prior(delta_hat):
        """
        Estimate the `b` prior. `b` is the scale parameter of the inverse gamma.
        `delta` is the estimate of the multiplicative bias.

        detla_hat: array
            Estimate of the multiplicative bias.

        .. math::
        m <-- mean of delta_hat
        v <-- variance of delta_hat
        b = (m * v + m^3) / v

        Return
        ------
        b: float
            The scale parameter of the inverse gamma.
        """
        m = np.mean(delta_hat)
        v = np.var(delta_hat, ddof=1)
        return (m * v + m**3) / v

    @staticmethod
    def emperical_bayes_estimate(sdat, gamma_hat, delta_hat, conv=0.0001):
        """
        Expectation-Maximization function to find Bayes biases
        sdat: array
            Standardized data
        gamma_hat: array
            Standardized estimate of the additive bias. No empirical bayes.
        delta_hat:
            Standardized estimate of the multiplicative bias. No empirical bayes.
        conv: flaot
            Convergence, after this the loop will stop

        Returns
        -------

        g_new: array
            Standardized estimate of the additive bias, using empirical bayes
        d_new: array
            Standardized estimate of the multiplicative bias, using empirical bayes

        """
        # g_bar: mean of the standardized estimate of the additive bias
        # without empirical bayes
        gamma_bar = np.mean(gamma_hat)

        # t2: tau ** 2, variance of the standardized estimate of the additive bias
        # without empirical bayes
        t2 = np.var(gamma_hat, ddof=1)

        # a: shape of the multplicative bias (shape in an inver-gamma)
        a = QuickCombat.estimate_a_prior(delta_hat)
        # b: scale of the multplicative bias (scale in an inver-gamma)
        b = QuickCombat.estimate_b_prior(delta_hat)

        n = len(sdat)
        g_old = gamma_hat.copy()
        d_old = delta_hat.copy()

        change = 1
        count = 0
        while change > conv or count > 1000:
            # postmean
            g_new = (t2 * n * gamma_hat + d_old * gamma_bar) / (t2 * n + d_old)

            sum2 = []
            for i in range(len(sdat)):
                sum2.append(np.sum((sdat[i] - g_new[i]) ** 2))
            sum2 = np.array(sum2)

            # postvar
            d_new = (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

            change = max(
                (abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max()
            )
            g_old = g_new
            d_old = d_new
            count = count + 1
        return g_new, d_new
