import numpy as np


def get_cohensd(data, group1, group2):
    """
    Compute the Cohen's d statistic between subjects of two groups [1].

    Args:
        data (np.array): Data with shape (features, subject)
        group1 (list): Subject keys for the first group.
        group2 (list): Subject keys for the second group.

    Returns:
        cohens_d (list of np.float32): The Cohen's d statistic for metric `metric_key` 
                                       between subjects of group1 and group2.

    References :
        [1] : https://www.statisticshowto.com/cohens-d/
    """

    # Handle case where bundle is absent for all subjects
    n = min(len(group1), len(group2))
    if n > 1:
        m1 = np.mean(data[:, group1])
        m2 = np.mean(data[:, group2])
        s1 = np.std(np.mean(data[:, group1], axis=0))
        s2 = np.std(np.mean(data[:, group2], axis=0))
        std_pooled = np.sqrt((s1**2 + s2**2) / 2)
        cohensd = np.abs(m1 - m2) / std_pooled

        # Correction factor for small samples
        if 50 > n > 2:
            corr_factor = ((n - 3) / (n - 2.25)) * np.sqrt((n - 2) / n)
            cohensd *= corr_factor

        return cohensd
    else:
        return np.float32(0.0)
