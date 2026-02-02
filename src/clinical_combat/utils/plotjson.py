"""
Mostly used to visualize the harmonization results simply in
QC reports, we provide this class which saves/load inputs of
plots in a single JSON file.
"""

import json
from enum import Enum
from collections import defaultdict

class PlotJson:
    class Type(Enum):
        SCATTER = "scatter"
        LINE = "line"

    def __init__(self, bundle, metric):
        self.bundle = bundle
        self.metric = metric
        self.data = {}

    def add_plot(self, plot_name, plot_class: Type, **kwargs):
        """Add data for a specific plot.

        Args:
            plot_name (str): The name of the plot.
            plot_data (dict): The data associated with the plot.
        """
        kwargs['plot_class'] = plot_class.value
        self.data[plot_name] = kwargs

    def to_json(self):
        """Convert the stored plot data to a JSON-compatible dictionary.

        Returns:
            dict: The JSON-compatible representation of the plot data.
        """
        return json.dumps(self.data, indent=4)
    
    def save_json(self, filepath):
        """Save the stored plot data to a JSON file.

        Args:
            filepath (str): The path to the file where the JSON data will be saved.
        """
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=4)

class PlotJsonAggregator:
    def __init__(self):
        self.aggregated_data = defaultdict(dict)

    @property
    def data(self):
        return self.aggregated_data
    
    @classmethod
    def from_file(cls, filepath):
        aggregator = cls()
        aggregator.add_plot_json_aggregator_from_file(filepath)
        return aggregator
    
    @classmethod
    def from_json(cls, json_data: str):
        aggregator = cls()

        if not isinstance(json_data, list):
            json_data = [json_data]

        for j in json_data:
            data = json.loads(j)
            for bundle, metrics in data.items():
                for metric, plots in metrics.items():
                    aggregator.aggregated_data[bundle][metric] = plots
        return aggregator

    def add_plot_json(self, plot_json: PlotJson):
        self.aggregated_data[plot_json.bundle][plot_json.metric] = plot_json.data
    
    def add_plot_json_aggregator(self, other_aggregator):
        for bundle, metrics in other_aggregator.aggregated_data.items():
            for metric, plots in metrics.items():
                self.aggregated_data[bundle][metric] = plots
    
    def add_plot_json_aggregator_from_file(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        for bundle, metrics in data.items():
            for metric, plots in metrics.items():
                self.aggregated_data[bundle][metric] = plots

    def save_aggregated_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.aggregated_data, f, indent=4)
    