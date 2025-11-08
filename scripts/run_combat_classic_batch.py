#!/usr/bin/env python3
"""
Batch helper to run `combat_quick_fit.py` in classic mode for multiple sites.

For each CSV located in an input directory (default: AD_30), this script
identifies the moving site, creates an output directory dedicated to that site,
and launches `combat_quick_fit.py` with CamCAN as the reference cohort.

Example usage (from the repository root):
    python scripts/run_combat_classic_batch.py

To see the planned commands without executing them:
    python scripts/run_combat_classic_batch.py --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _slugify(text: str) -> str:
    """Return a filesystem-friendly version of the site name."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_") or "unknown_site"


def _infer_site(csv_path: Path) -> str:
    """Read the site identifier from a CSV, ensuring that only one site is present."""
    df = pd.read_csv(csv_path, usecols=["site"])
    sites = df["site"].dropna().astype(str).unique()
    if len(sites) != 1:
        raise ValueError(
            f"{csv_path} contient {len(sites)} sites distincts ({sites.tolist()}); "
            "le script attend un seul site par fichier."
        )
    return sites[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch ComBat classic vs CamCAN for all CSVs in a directory."
    )
    parser.add_argument(
        "--input-dir",
        default="TBI_30_afd",
        help="Répertoire contenant les CSV des sites à harmoniser. [%(default)s]",
    )
    parser.add_argument(
        "--reference-data",
        default="DONNES/CamCAN/CamCAN.afd.raw.csv.gz",
        help="Chemin vers les données de référence CamCAN. [%(default)s]",
    )
    parser.add_argument(
        "--output-root",
        default="TBI_30_results_rob",
        help="Répertoire racine où placer les résultats (un sous-dossier par site). "
        "[%(default)s]",
    )
    parser.add_argument(
        "--method",
        default="classic",
        choices=["classic"],
        help="Méthode ComBat utilisée pour le fit. (Valeurs futures : classic) [%(default)s]",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        help="Limiter le traitement à certains noms de site (exact match).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les commandes sans les exécuter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Passer l'option --overwrite à combat_quick_fit.py.",
    )
    parser.add_argument(
        "--quick-fit-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Arguments supplémentaires à transmettre à combat_quick_fit.py. "
            "Utiliser après un '--', par ex.: -- --ignore_sex"
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (repo_root / args.input_dir).resolve()
    reference_path = (repo_root / args.reference_data).resolve()
    output_root = (repo_root / args.output_root).resolve()
    quick_fit_script = (repo_root / "scripts" / "combat_quick.py").resolve()

    if not input_dir.is_dir():
        parser.error(f"Répertoire d'entrée introuvable: {input_dir}")
    if not reference_path.is_file():
        parser.error(f"Fichier de référence introuvable: {reference_path}")
    if not quick_fit_script.is_file():
        parser.error(f"Script combat_quick.py introuvable: {quick_fit_script}")

    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        parser.error(f"Aucun fichier CSV trouvé dans {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)

    # Ensure clinical_combat can be imported when the subprocess runs.
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(repo_root)
        if env.get("PYTHONPATH") is None
        else str(repo_root) + os.pathsep + env["PYTHONPATH"]
    )

    for csv_path in csv_paths:
        site_name = _infer_site(csv_path)
        if args.sites and site_name not in args.sites:
            continue

        safe_site = _slugify(site_name)
        site_out_dir = output_root / safe_site
        site_out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(quick_fit_script),
            str(reference_path),
            str(csv_path.resolve()),
            "--method",
            args.method,
            "--out_dir",
            str(site_out_dir),
            "-f"
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.quick_fit_args:
            cmd.extend(args.quick_fit_args)

        printable_cmd = " ".join(map(str, cmd))
        if args.dry_run:
            print(f"[DRY-RUN] {printable_cmd}")
            continue

        print(f"[RUN] {printable_cmd}")
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
