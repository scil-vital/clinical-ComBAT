from __future__ import absolute_import

import ast

from clinical_combat.harmonization.QuickCombatClinic import QuickCombatClinic
from clinical_combat.harmonization.QuickCombatPairwise import \
    QuickCombatPairwise
from clinical_combat.harmonization.QuickCombatVanilla import QuickCombatVanilla



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
    tau=1
):

    if name == "vanilla":
        QC = QuickCombatVanilla(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov
        )
    elif name == "pairwise":
        QC = QuickCombatPairwise(
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
    else:
        raise AssertionError(
            name + " is an invalid value for the harmonization method."
        )
    return QC


def from_model_filename(model_filename):
    with open(model_filename) as f:
        model_params = ast.literal_eval(f.readline()[2:])

    QC = from_model_name(model_params["name"])
    QC.initialize_from_model_params(model_filename)
    return QC
