import numpy as np
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

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
def predict_malades_MLP(df, run_name):
    # Charge les artefacts

    state_dict = torch.load(MODEL_DIR / f"{run_name}_weights.pt", map_location="cpu")
    # 1) recharge le JSON d’hyperparams

    with open(MODEL_DIR / f"{run_name}_params.json") as fp:
        hp = json.load(fp)

    # 2) reconstruit le modèle avec ces hyperparams
    # Compatibilité ascendante: accepte soit h1/h2/h3, soit hidden_dims (liste)
    if "hidden_dims" in hp:
        hidden_dims = tuple(int(x) for x in hp["hidden_dims"])  # e.g. [256,128,64]
    else:
        # Ancienne convention (Optuna): h1/h2/h3
        hidden_dims = (int(hp["h1"]), int(hp["h2"]), int(hp["h3"]))

    model = PatientMLP(
        hidden_dims=hidden_dims,
        drop=hp.get("dropout", 0.5),
        activation=hp.get("activation", "relu"),
        batch_norm=hp.get("batch_norm", True),
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
        proba = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    # 6. Retourne un DataFrame (sid, probabilité) pour seuil ultérieur
    return pd.DataFrame({
        "sid": sid_order.to_numpy(),
        "prob_maladie": proba.astype(float),
    })

# ------------------------------------------------------------------
# Modèle
# ------------------------------------------------------------------
class PatientMLP(nn.Module):
    def __init__(
        self,
        in_features: int = 430,
        hidden_dims: Sequence[int] = (256, 128, 64),
        drop: Union[float, Sequence[float]] = 0.3,
        activation: str = "relu",
        batch_norm: bool = True,
        config: Optional[dict] = None,
    ):
        """MLP configurable pour classification binaire.

        Paramètres (legacy et/ou via config):
        - in_features: nombre d'entrées (par défaut 430)
        - hidden_dims: liste/tuple des tailles de couches cachées
        - drop: float unique ou liste de dropouts par couche
        - activation: "relu" | "gelu" | "leaky_relu" | "elu" | "tanh"
        - batch_norm: insère une BatchNorm1d entre Linear et activation
        - config: dict prioritaire pouvant contenir les mêmes clés que ci-dessus
        """
        super().__init__()

        if config is not None:
            in_features = config.get("in_features", in_features)
            hidden_dims = config.get("hidden_dims", hidden_dims)
            drop = config.get("dropout", config.get("drop", drop))
            activation = config.get("activation", activation)
            batch_norm = config.get("batch_norm", batch_norm)

        def _act(name: str) -> nn.Module:
            name = (name or "relu").lower()
            if name == "relu":
                return nn.ReLU()
            if name == "gelu":
                return nn.GELU()
            if name == "leaky_relu":
                return nn.LeakyReLU(negative_slope=0.01)
            if name == "elu":
                return nn.ELU()
            if name == "tanh":
                return nn.Tanh()
            # Défaut raisonnable
            return nn.ReLU()

        # Normalise drop à une liste de même longueur que hidden_dims
        if isinstance(drop, (int, float)):
            drop_list = [float(drop)] * len(hidden_dims)
        else:
            drop_list = [float(d) for d in drop]
            if len(drop_list) != len(hidden_dims):
                raise ValueError("La liste 'drop' doit avoir la même longueur que 'hidden_dims'.")

        layers: List[nn.Module] = []
        prev = int(in_features)
        for h, pdrop in zip(hidden_dims, drop_list):
            h = int(h)
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_act(activation))
            if pdrop and pdrop > 0:
                layers.append(nn.Dropout(pdrop))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ------------------------------------------------------------------
# Config par défaut et builder utilitaire
# ------------------------------------------------------------------
DEFAULT_MODEL_CONFIG = {
    "in_features": 430,
    "hidden_dims": [256, 128, 64],
    "activation": "relu",
    "dropout": 0.3,
    "batch_norm": True,
}


def build_mlp_from_config(cfg: Optional[dict] = None) -> PatientMLP:
    """Construit un PatientMLP à partir d'un dict de configuration.

    Exemple d'usage:
        cfg = {"hidden_dims": [512,256], "activation": "gelu", "dropout": 0.4}
        model = build_mlp_from_config(cfg)
    """
    merged = dict(DEFAULT_MODEL_CONFIG)
    if cfg:
        merged.update(cfg)
    return PatientMLP(
        in_features=merged.get("in_features", 430),
        hidden_dims=merged.get("hidden_dims", (256, 128, 64)),
        drop=merged.get("dropout", 0.3),
        activation=merged.get("activation", "relu"),
        batch_norm=merged.get("batch_norm", True),
    )
