# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def leave_one_out_error(ref_data, mov_data, QC):
    """
    Compute the leave-one-out error for a given QC object and data.

    Args:
        ref_data (pd.DataFrame): Reference data.
        mov_data (pd.DataFrame): Data to be corrected.
        QC (QuickCombat): QuickCombat object.
    
    Returns:
        error (np.float32): Mean squared error of the leave-one-out error.
    """
    errors = []
    for sid in mov_data.sid.unique():
        print(sid)
        data = mov_data.query("sid != @sid")
        QC.fit(ref_data, data)
        sub_data = mov_data.query("sid == @sid")
        age = sub_data["age"].iloc[0]
        # sex = sub_data.sex[0].iloc[0]
        # handedness = sub_data.handedness[0].iloc[0]
        for bundle in QC.bundle_names:
            v = sub_data.query("bundle == @bundle")["mean"].iloc[0]
            p = QC.predict([age], bundle)
            errors.append((v - p) ** 2)
    return np.mean(errors)


def find_mov_regul(ref_data, mov_data, QC):
    """
    Find the optimal regularization term for QuickCombat.

    Args:
        ref_data (pd.DataFrame): Reference data.
        mov_data (pd.DataFrame): Data to be corrected.
        QC (QuickCombat): QuickCombat object.
    
    Returns:
        reg (np.float32): Optimal regularization term.
    """
    regs = np.arange(0, 10.5, 0.5) ** 2
    print(regs)
    errors = []
    for reg in regs:
        QC.regul_mov = reg
        errors.append(leave_one_out_error(ref_data, mov_data, QC))

    # import pdb; pdb.set_trace()

    plt.plot(regs, errors)
    plt.scatter(regs, errors)
    plt.title("QuickCombat Fitting Error\nAll Bundles")
    plt.xlabel("Regularization Term")
    plt.ylabel("MSE")
    plt.show()
    plt.savefig("errors.png")

    return regs[np.argmin(errors)]
