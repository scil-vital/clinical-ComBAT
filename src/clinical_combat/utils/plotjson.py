"""
Mostly used to visualize the harmonization results simply in
QC reports, we provide this class which saves/load inputs of
plots in a single JSON file.
"""

import json
from enum import Enum
from collections import defaultdict

class PlotJson:
    """
    A utility class to store and manage plot data in JSON format for easy
    serialization and deserialization.
    """
    class Type(Enum):
        SCATTER = "scatter"
        LINE = "line"

    def __init__(self, bundle, metric):
        self.bundle = bundle
        self.metric = metric
        self.data = {}

    def add_plot(
        self,
        plot_name,
        plot_class: Type,
        data_x: list,
        data_y: list,
        x_label: str = "",
        y_label: str = "",
        plot_group: str = None,
        **kwargs
    ):
        """Add data for a specific plot.

        Args:
            plot_name (str): The name of the plot.
            plot_class (Type): The type of the plot.
            data_x (list): The data for the x-axis.
            data_y (list): The data for the y-axis.
            x_label (str, optional): The label for the x-axis. Defaults to "".
            y_label (str, optional): The label for the y-axis. Defaults to "".
            **kwargs: Additional keyword arguments containing plot data and styling.
        """
        kwargs['plot_class'] = plot_class.value
        kwargs['data_x'] = data_x
        kwargs['data_y'] = data_y
        kwargs['x_label'] = x_label
        kwargs['y_label'] = y_label

        if plot_group is not None:
            if plot_group not in self.data:
                self.data[plot_group] = {}
            self.data[plot_group][plot_name] = kwargs
        else:
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
    """
    Aggregates multiple PlotJson instances into a single JSON structure
    for easier management and storage.
    """
    def __init__(self):
        self.aggregated_data = defaultdict(dict)

    @property
    def data(self):
        return self.aggregated_data
    
    @classmethod
    def from_file(cls, filepath):
        """
        Create a PlotJsonAggregator instance from a JSON file.
        Args:
            filepath (str): The path to the JSON file.
        """
        aggregator = cls()
        aggregator.add_plot_json_aggregator_from_file(filepath)
        return aggregator
    
    @classmethod
    def from_json(cls, json_data: str):
        """
        Create a PlotJsonAggregator instance from a JSON string.
        Args:
            json_data (str): The JSON string.
        """
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
        """
        Add a PlotJson instance to the aggregator.
        Args:
            plot_json (PlotJson): The PlotJson instance to add.
        """
        self.aggregated_data[plot_json.bundle][plot_json.metric] = plot_json.data
    
    def add_plot_json_aggregator(self, other_aggregator):
        """
        Add another PlotJsonAggregator's data to this aggregator.
        Args:
            other_aggregator (PlotJsonAggregator): The other aggregator to merge.
        """
        for bundle, metrics in other_aggregator.aggregated_data.items():
            for metric, plots in metrics.items():
                self.aggregated_data[bundle][metric] = plots
    
    def add_plot_json_aggregator_from_file(self, filepath):
        """
        Add plot data from a JSON file to the aggregator.
        This is mostly used to load previously saved plot JSON files.
        Args:
            filepath (str): The path to the JSON file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        for bundle, metrics in data.items():
            for metric, plots in metrics.items():
                self.aggregated_data[bundle][metric] = plots

    def save_aggregated_json(self, filepath):
        """
        Save the aggregated plot data to a JSON file.
        Args:
            filepath (str): The path to the file where the JSON data will be saved.
        """
        with open(filepath, 'w') as f:
            json.dump(self.aggregated_data, f, indent=4)
    