from __future__ import absolute_import

import ast

from clinical_combat.harmonization.QuickCombatVanilla import QuickCombatVanilla



def from_model_name(
    ignore_sex_covariate=False,
    ignore_handedness_covariate=False,
    use_empirical_bayes=True,
    limit_age_range=False,
    degree=1,
    regul_ref=0,
):
    QC = QuickCombatVanilla(
        ignore_sex_covariate=ignore_sex_covariate,
        ignore_handedness_covariate=ignore_handedness_covariate,
        use_empirical_bayes=use_empirical_bayes,
        limit_age_range=limit_age_range,
        degree=degree,
        regul_ref=regul_ref
    )

    return QC


def from_model_filename(model_filename):
    with open(model_filename) as f:
        model_params = ast.literal_eval(f.readline()[2:])

    QC = from_model_name(model_params["name"])
    QC.initialize_from_model_params(model_filename)
    return QC
