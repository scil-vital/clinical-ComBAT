from __future__ import absolute_import

import ast

from clinical_combat.harmonization.QuickCombatVanilla import QuickCombatVanilla

def from_model_filename(model_filename):
    with open(model_filename) as f:
        model_params = ast.literal_eval(f.readline()[2:])

    QC = QuickCombatVanilla()
    QC.initialize_from_model_params(model_filename)
    return QC
