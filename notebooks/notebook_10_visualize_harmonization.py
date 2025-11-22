"""
Notebook 10: Harmonization visualization utility.

This script collects harmonized outputs for multiple robust methods and builds
per-site visualizations for a given bundle and harmonization method. It expects
harmonized data organized on disk using the following convention::

    <input_root>/<bundle>/<robust_method>/<harmonization_method>/<site>/

Each site directory should contain at least one CSV file with a column matching
``metric``. The script will pick the first CSV that provides the metric and use
it to generate a simple visualization saved under::

    <output_root>/<bundle>/<site>/<metric>_<robust_method>.png

The saved file name explicitly lists the robust method so that downstream users
can quickly identify which method generated the image.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class VisualizationConfig:
    """Configuration parameters for notebook 10."""

    metric: str
    bundle: str
    harmonization_method: str
    robust_methods: List[str]
    input_root: Path
    output_root: Path


class HarmonizationVisualizer:
    """Load harmonized data and write metric visualizations per site."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def run(self) -> None:
        sites = self._collect_sites()
        if not sites:
            logging.warning("No sites discovered for bundle '%s'.", self.config.bundle)
            return

        for site in sorted(sites):
            self._process_site(site)

    def _collect_sites(self) -> set[str]:
        """Return the union of site directory names across all robust methods."""

        sites: set[str] = set()
        for method in self.config.robust_methods:
            site_root = self._method_site_root(method)
            if not site_root.exists():
                logging.warning(
                    "Skipping robust method '%s' because %s is missing.",
                    method,
                    site_root,
                )
                continue
            for child in site_root.iterdir():
                if child.is_dir():
                    sites.add(child.name)
        return sites

    def _method_site_root(self, robust_method: str) -> Path:
        return (
            self.config.input_root
            / self.config.bundle
            / robust_method
            / self.config.harmonization_method
        )

    def _process_site(self, site: str) -> None:
        logging.info("Processing site '%s'", site)
        for method in self.config.robust_methods:
            self._visualize_method_site(method, site)

    def _visualize_method_site(self, robust_method: str, site: str) -> None:
        data_path = self._find_metric_file(robust_method, site)
        if data_path is None:
            logging.warning(
                "No data for metric '%s' using method '%s' at site '%s'.",
                self.config.metric,
                robust_method,
                site,
            )
            return

        output_dir = self.config.output_root / self.config.bundle / site
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.config.metric}_{robust_method}.png"

        df = pd.read_csv(data_path)
        if self.config.metric not in df.columns:
            logging.warning(
                "File %s does not contain metric '%s'.", data_path, self.config.metric
            )
            return

        values = df[self.config.metric].dropna()
        if values.empty:
            logging.warning(
                "Metric '%s' is empty in %s for method '%s' at site '%s'.",
                self.config.metric,
                data_path,
                robust_method,
                site,
            )
            return

        self._plot_metric(values, robust_method, site, output_path)

    def _find_metric_file(self, robust_method: str, site: str) -> Optional[Path]:
        site_root = self._method_site_root(robust_method) / site
        if not site_root.exists():
            return None

        for candidate in sorted(site_root.iterdir()):
            if candidate.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(candidate, nrows=1)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.debug("Failed reading %s: %s", candidate, exc)
                    continue
                if self.config.metric in df.columns:
                    return candidate
        return None

    def _plot_metric(
        self, values: pd.Series, robust_method: str, site: str, output_path: Path
    ) -> None:
        plt.figure(figsize=(8, 4))
        plt.hist(values, bins=20, color="#4477aa", edgecolor="black", alpha=0.7)
        plt.title(
            f"{self.config.metric} - {robust_method}\n"
            f"Bundle: {self.config.bundle} | Site: {site} | Harmonization: {self.config.harmonization_method}"
        )
        plt.xlabel(self.config.metric)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logging.info("Saved visualization to %s", output_path)


def parse_args(argv: Optional[Iterable[str]] = None) -> VisualizationConfig:
    parser = argparse.ArgumentParser(description="Notebook 10 harmonization visualization")
    parser.add_argument("metric", help="Metric column to visualize (e.g., MAE, RMSE)")
    parser.add_argument("bundle", help="Bundle identifier used in the input directory structure")
    parser.add_argument(
        "harmonization_method",
        help="Harmonization method name used inside the input directory structure",
    )
    parser.add_argument(
        "robust_methods",
        nargs="+",
        help="List of robust methods to include (e.g., HC RAW our_method)",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/harmonized"),
        help="Root directory containing harmonized outputs organized by bundle/method/harmonization/site",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/visualizations"),
        help="Directory where visualizations will be written",
    )

    args = parser.parse_args(argv)

    return VisualizationConfig(
        metric=args.metric,
        bundle=args.bundle,
        harmonization_method=args.harmonization_method,
        robust_methods=args.robust_methods,
        input_root=args.input_root,
        output_root=args.output_root,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    config = parse_args(argv)
    visualizer = HarmonizationVisualizer(config)
    visualizer.run()


if __name__ == "__main__":
    main()
