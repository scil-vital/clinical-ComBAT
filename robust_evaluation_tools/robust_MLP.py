import numpy as np
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import json

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
MODEL_DIR = Path("Pytorch_models")  # tous les artefacts seront enregistrés ici

def _compute_z(df, value_col="mean_no_cov"):
    # Calcule le z-score par metric_bundle
    stats = (df.groupby("metric_bundle")[value_col]
               .agg(['mean', 'std'])
               .rename(columns={'mean': 'gmean', 'std': 'gstd'}))
    stats["gstd"] = stats["gstd"].replace(0, 1e-6)
    df = df.merge(stats, on="metric_bundle", how="left")
    df["zscore"] = (df[value_col] - df["gmean"]) / df["gstd"]
    return df.drop(columns=["gmean", "gstd"])


def _pivot_features(df, value_col="zscore", bundle_col="metric_bundle"):
    # Transforme en matrice (sid × features), sans 'disease'
    return df.pivot(index="sid", columns=bundle_col, values=value_col)

# ------------------------------------------------------------------
# Fonction principale
# ------------------------------------------------------------------
def predict_malades_MLP(df, run_name, threshold=0.5):
    # Charge les artefacts

    state_dict = torch.load(MODEL_DIR / f"{run_name}_weights.pt", map_location="cpu")
    # 1) recharge le JSON d’hyperparams

    with open(MODEL_DIR / f"{run_name}_params.json") as fp:
        hp = json.load(fp)

# 2) reconstruis le modèle avec ces hyperparams
    model = PatientMLP(
        hidden_dims=(hp["h1"], hp["h2"], hp["h3"]),
        drop       = hp["dropout"]
    )
    model.load_state_dict(state_dict)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device ="cpu"
    model = model.to(device).eval()

    # 1. Retire les ventricules
    df = df[~df["bundle"].isin(["left_ventricle", "right_ventricle"])].copy()

    # 2. z-score
    df_z = _compute_z(df, value_col="mean_no_cov")

    # 3. pivot en matrice features
    mat = _pivot_features(df_z, value_col="zscore")
    sid_order = mat.index
    X = mat.values.astype(np.float32)


    # 5. Inférence
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        proba = torch.sigmoid(logits).cpu().numpy()

    # 6. Liste des sids malades
    malades = sid_order[proba >= threshold].tolist()
    return malades

# ------------------------------------------------------------------
# Modèle
# ------------------------------------------------------------------
class PatientMLP(nn.Module):
    def __init__(self, in_features=430, hidden_dims=(256, 128, 64), drop=0.3):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU(),
                       nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(1)