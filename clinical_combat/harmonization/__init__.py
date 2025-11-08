# -*- coding: utf-8 -*-
from __future__ import absolute_import

import ast

from clinical_combat.harmonization.QuickCombatClinic import QuickCombatClinic
from clinical_combat.harmonization.QuickCombatClassic import QuickCombatClassic
from clinical_combat.harmonization.QuickCombatCovBat import QuickCombatCovBat
from clinical_combat.harmonization.QuickCombatGam import QuickCombatGam
from clinical_combat.harmonization.QuickCombatGmm import QuickCombatGmm


def from_model_name(
    name,
    ignore_sex_covariate=False,
    ignore_handedness_covariate=False,
    use_empirical_bayes=True,
    limit_age_range=False,
    degree=1,
    regul_ref=0,
    regul_mov=0,
    nu=0,
    tau=1,
    robust='No',
    covbat_pve=0.95,
    covbat_max_components=None,
    gam_n_knots=7,
    gmm_components=2,
    gmm_tol=1e-4,
    gmm_max_iter=200,
):

    if name == "classic":
        QC = QuickCombatClassic(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov
        )
    elif name == "clinic":
        QC = QuickCombatClinic(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov,
            nu=nu,
            tau=tau,
        )
    elif name == "covbat":
        QC = QuickCombatCovBat(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov,
            covbat_pve=covbat_pve,
            covbat_max_components=covbat_max_components,
        )
    elif name == "gam":
        QC = QuickCombatGam(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov,
            gam_n_knots=gam_n_knots,
        )
    elif name == "gmm":
        QC = QuickCombatGmm(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov,
            gmm_components=gmm_components,
            gmm_tol=gmm_tol,
            gmm_max_iter=gmm_max_iter,
        )
    else:
        raise AssertionError(
            name + " is an invalid value for the harmonization method."
        )
    return QC


def from_model_filename(model_filename):
    with open(model_filename) as f:
        model_params = ast.literal_eval(f.readline()[2:])

    model = from_model_name(model_params["name"])
    model.initialize_from_model_params(model_filename)
    return model
