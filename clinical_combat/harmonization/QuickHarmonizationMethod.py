# -*- coding: utf-8 -*-
import ast
import logging

import numpy as np


class QuickHarmonizationMethod:
    """
    Abstract class.
    """

    def __init__(
        self,
        bundle_names=None,
        model_params=None,
        ignore_sex_covariate=False,
        ignore_handedness_covariate=False,
    ):
        """
        bundle_names: Array
            List of bundle names.
        model_params: Dict
            Model parameters and data info.
        ignore_sex_covariate: bool
            If true, do not use sex as a covariate
        ignore_handedness_covariate: bool
            If true, do not use handedness as a covariate

        """

        self.bundle_names = bundle_names
        self.ignore_sex_covariate = ignore_sex_covariate
        self.ignore_handedness_covariate = ignore_handedness_covariate
        if model_params:
            self.model_params = model_params
        else:
            self.model_params = {}

    def initialize_from_model_params(self, model_filename):
        """
        Initialize the object from a model file

        model_filename: str
            Model filename

        """

        with open(model_filename) as f:
            self.model_params = ast.literal_eval(f.readline()[2:])

        self.use_empirical_bayes = self.model_params["use_empirical_bayes"]
        self.limit_age_range = self.model_params["limit_age_range"]
        self.ignore_sex_covariate = self.model_params["ignore_sex_covariate"]
        self.ignore_handedness_covariate = self.model_params[
            "ignore_handedness_covariate"
        ]
        return

    def assert_data(self, df):
        """
        Check that all required columns are present in the data frame

        df: Dataframe
        """

        for c in [
            "sid",
            "bundle",
            "mean",
            "age",
            "disease",
            "handedness",
            "sex",
            "site",
            "metric",
        ]:
            if c not in df.columns:
                raise AssertionError("Missing column " + c + " in data.")

        for bundle in df.bundle.unique():
            df1 = df.query("bundle == @bundle and disease == 'HC'")
            if not self.ignore_sex_covariate:
                for v in df1.sex.unique():
                    if v not in [1, 2]:
                        raise AssertionError(
                            str(v)
                            + " is an invalid value for the sex covariate. Set ignore_sex_covariate "
                            + "to harmonize this data."
                        )
                if len(df1.sex.unique()) == 1:
                    self.ignore_sex_covariate = True
                    logging.warning(
                        "A single sex covariate was found. The sex covariate will be ignored."
                    )

            if not self.ignore_handedness_covariate:
                for v in df1.handedness.unique():
                    if v not in [1, 2]:
                        raise AssertionError(
                            str(v)
                            + " is an invalid value for the handedness covariate. "
                            + "Set ignore_handedness_covariate to harmonize this data."
                        )
                if len(df1.handedness.unique()) == 1:
                    self.ignore_handedness_covariate = True
                    logging.warning(
                        "A single handedness covariate was found. "
                        + "The handedness covariate will be ignored."
                    )

    def set_model_fit_params(self, ref_data, mov_data):
        """
        Set the model parameter given the input data used for the fit.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.

        """
        self.model_params["name"] = ""
        self.model_params["ignore_handedness_covariate"] = (
            self.ignore_handedness_covariate
        )
        self.model_params["ignore_sex_covariate"] = self.ignore_sex_covariate
        self.model_params["mov_site"] = np.unique(mov_data["site"])[0]
        self.model_params["ref_site"] = np.unique(ref_data["site"])[0]
        self.model_params["metric_name"] = np.unique(ref_data["metric"])[0]

    def fit(self, ref_data, mov_data, HC_only=True):
        pass

    def apply(self, data):
        pass
