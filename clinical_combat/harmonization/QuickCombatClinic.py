# -*- coding: utf-8 -*-
import logging

import numpy as np

from clinical_combat.harmonization.QuickCombat import QuickCombat


class QuickCombatClinic(QuickCombat):
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
        gamma_mov=None,
        delta_mov=None,
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
        gamma_mov: Array
            Additive bias of the moving site.
        delta_mov: Array
            Multiplicative bias of the reference site.
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

        self.alpha_ref = alpha_ref
        self.beta_ref = beta_ref
        self.sigma_ref = sigma_ref
        self.alpha_mov = alpha_mov
        self.beta_mov = beta_mov
        self.sigma_mov = sigma_mov
        self.gamma_mov = gamma_mov
        self.delta_mov = delta_mov


    def initialize_from_model_params(self, model_filename):
        """
        Initialize the object from a model file.

        model_filename: str
            Model filename

        """
        super().initialize_from_model_params(model_filename)
        self.nu = self.model_params["nu"]
        self.tau = self.model_params["tau"]

        params = np.loadtxt(model_filename, delimiter=",", dtype=str, skiprows=1)
        nb = len(self.get_beta_labels())
        self.bundle_names = params[0, 1:]
        self.alpha_ref = params[1, 1:].astype("float64").transpose()
        self.beta_ref = params[2 : 2 + nb, 1:].astype("float64").transpose()
        self.sigma_ref = params[2 + nb, 1:].astype("float64").transpose()        
        self.alpha_mov = params[3 + nb, 1:].astype("float64").transpose()
        self.beta_mov = params[4 + nb : 4 + nb + nb, 1:].astype("float64").transpose()
        self.sigma_mov = params[4 + nb + nb, 1:].astype("float64").transpose()
        self.gamma_mov = params[5 + nb + nb, 1:].astype("float64").transpose()
        self.delta_mov = params[6 + nb + nb, 1:].astype("float64").transpose()


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
        self.model_params["tau"] = self.tau
        self.model_params["name"] = "clinic"


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
        param_labels.append("mov_gamma")
        param_labels.append("mov_delta")

        param_labels = np.array(param_labels).reshape([-1, 1])

        params = np.hstack([param_labels, params])
        header = str(self.model_params)
        np.savetxt(model_filename, params, delimiter=",", fmt="%s", header=header)


    def standardize_moving_data(self, X, Y):
        """
        Standardize the data (Y). Combat Clinic standardize the moving site data with 
        the moving site intercept. Because the data are harmonize to the reference site, 
        sigma is obtained from the reference site data.

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

 
    def fit(self, ref_data, mov_data, HC_only=True):
        """
        Combat Clinic fit. 
        The moving site beta and alpha are fitted using the moving site data.
        The reference site alpha and beta is fitted using the reference site data.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.
        """

        ref_data, mov_data = self.prepare_data(ref_data, mov_data, HC_only)

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
        self.gamma_mov = np.array([np.mean(x) for x in z])
        self.delta_mov = np.array([np.std(x, ddof=1) for x in z])

        if self.use_empirical_bayes:
            new_delta = []
            for i in range(len(self.sigma_mov)):
                N = len(y_mov[i])
                # The target normalized std is 1 (self.nu * target_std = self.nu)
                new = (self.delta_mov[i] * N + self.nu) / (N + self.nu)
                new_delta.append(new)
            self.delta_mov = np.array(new_delta)

        self.set_model_fit_params(ref_data, mov_data)
        #import pdb; pdb.set_trace()
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
            or self.alpha_mov is None
            or self.beta_mov is None
            or self.sigma_mov is None
            or self.gamma_mov is None
            or self.delta_mov is None
        ):
            raise AssertionError("Model parameters are not fitted.")

        design, Y = self.get_design_matrices(data)
        z = self.standardize_moving_data(design, Y)

        harm_y = []

        for i in range(len(design)):
            covariate_effect_ref = np.dot(
                design[i][1:, :].transpose(), self.beta_ref[i]
            )

            harm_y.append(
                self.sigma_ref[i] / self.delta_mov[i] * (z[i] - self.gamma_mov[i])
                + self.alpha_ref[i]
                + covariate_effect_ref
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
            B = np.hstack([self.alpha_mov[idx], self.beta_mov[idx]])
        else:
            B = np.hstack([self.alpha_ref[idx], self.beta_ref[idx]])

        y = np.dot(design.transpose(), B)
        return y


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


 