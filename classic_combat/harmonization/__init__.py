from __future__ import absolute_import

import ast

from classic_combat.harmonization.QuickCombatClassic import QuickCombatClassic

def from_model_filename(model_filename):
    with open(model_filename) as f:
        model_params = ast.literal_eval(f.readline()[2:])

    QC = QuickCombatClassic()
    QC.initialize_from_model_params(model_filename)
    return QC
