#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting functions to harmonization.

Includes a set of functions that generate plots (scatter, box, line, kde, error_bars)
and ancillary functions to lightness a color or select a subset of a dataframe.
"""

import colorsys
import logging
import os
import warnings
from os.path import join

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from clinical_combat.utils.scilpy_utils import assert_outputs_exist

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


def initiate_joint_marginal_plot(
    df,
    x,
    y,
    hue,
    ylim=None,
    xlim=None,
    alpha=0.5,
    marginal_hist=False,
    hist_bin=20,
    hist_palette=None,
    hist_hur_order=None,
    hist_legend=False,
    legend_title="Sites",
):
    """
    Initiate a joint plot with marginal histogram.

    Args:
        df (DataFrame): pd.DataFrame
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        hue (str): Column name for hue color
        ylim (tuple): Y-axis limits (ymin, ymax)
        alpha (float): Transparency (default: 0.5)
        marginal_hist (bool): If True, plot marginal histogram (default: False)
        hist_bin (int): Number of bins for histogram (default: 20)
        hist_palette (list): List of colors for histogram (default: None)
        hist_hur_order (list): List of hue order for histogram (default: None)
        hist_legend (bool): If True, plot histogram legend (default: False)
        legend_title (str): Title of legend (default: Sites)

    Returns:
        fig (JointGrid): seaborn JointGrid Plot
        ax (Axes): matplotlib Axes

    """
    fig = sns.JointGrid(df, x=x, y=y, hue=hue, ylim=ylim)
    if marginal_hist:
        fig.plot_marginals(
            sns.histplot,
            bins=hist_bin,
            palette=hist_palette,
            hue_order=hist_hur_order,
            alpha=alpha,
            stat="percent",
            common_bins=True,
            common_norm=False,
            legend=hist_legend,
        )
    else:
        fig.ax_marg_x.remove()
        fig.ax_marg_y.remove()

    ax = fig.ax_joint
    fig.ax_marg_y.legend([], [], frameon=False)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    if hist_legend:
        sns.move_legend(
            fig.ax_marg_x,
            "upper left",
            bbox_to_anchor=(1.25, 1),
            borderaxespad=0,
            frameon=False,
            title=legend_title,
        )

    return fig, ax


def add_reference_percentiles_to_curve(
    ax,
    ref_age,
    ref_percentiles,
    percentiles,
    set_line_widths,
    line_style="solid",
    set_color="#000000",
    add_grid=False,
):
    """
    Adds percentile data to the joint plot for the reference site.

    Args:
        ax (Axes): matplotlib Axes
        ref_age (np.array): Reference age vector (range from min_age to max_age)
        ref_percentiles (tuple): (index corresponding to each age, percentiles values in np.array)
        percentiles (list): List of percentiles to plot (default: [5, 25, 50, 75, 95])
        set_line_widths (list): List of line widths for percentiles (default: [1, 1, 2, 1, 1])
        set_color (str): Color of percentiles (default: #000000)
        add_grid (bool): If True, add grid to plot (default: True, color: #000000, alpha: 0.05)

    Returns:
        ax (Axes): matplotlib Axes with grey percentile curves

    """
    for curr_percentile_idx, curr_percentile in enumerate(percentiles):
        ax.plot(
            ref_age,
            ref_percentiles[:, curr_percentile_idx],
            color=set_color,
            linestyle=line_style,
            linewidth=set_line_widths[curr_percentile_idx],
            label=str(curr_percentile).zfill(2) + "th percentile",
        )
    ax.fill_between(
        ref_age,
        ref_percentiles[:, 0],
        ref_percentiles[:, -1],
        color=set_color,
        linestyle=line_style,
        alpha=0.20,
    )
    ax.fill_between(
        ref_age,
        ref_percentiles[:, 1],
        ref_percentiles[:, -2],
        color=set_color,
        linestyle=line_style,
        alpha=0.20,
    )
    ax.legend()

    if add_grid:
        ax.grid(color="#000000", alpha=0.05)

    return ax


def add_site_curve_to_reference_curve(
    ax,
    site_age,
    site_mean,
    site_std,
    line_width=2,
    ylim=None,
    label_site=None,
    color="r",
    add_grid=False,
    alpha=0.2,
    linestyle="-",
):
    """
    Adds percentile data to the joint plot for the reference site.

    Args:
        ax (Axes): matplotlib Axes
        site_age (np.array): Age vector for site (range from min_age to max_age)
        site_mean (np.array): Mean vector for site
        site_std (np.array): Standard deviation vector for site
        line_width (int): Line width for mean curve (default: 2)
        ylim (tuple): Y-axis limits (ymin, ymax)
        label_site (str): Label name for site (default: None)
        color (str): Color for curve sites (default: r)
        add_grid (bool): If True, add grid to plot (default: True, color: #000000, alpha: 0.05)
        alpha (float): Transparency (default: 0.2)

    Returns:
        ax (Axes): matplotlib Axes with site curves

    """
    sns.lineplot(
        x=site_age,
        y=site_mean,
        color=color,
        linewidth=line_width,
        legend=True,
        ax=ax,
        label=label_site,
        dashes=(2, 2),
    )
    ax.lines[0].set_linestyle(linestyle)
    ax.fill_between(
        site_age, site_mean - site_std, site_mean + site_std, color=color, alpha=alpha
    )
    ax.set_ylim(ylim)
    ax.legend()
    if add_grid:
        ax.grid(color="#000000", alpha=0.05)

    return ax


def add_scatterplot_to_curve(
    ax,
    df,
    x,
    y,
    hue=None,
    palette="Set3",
    hue_order=None,
    alpha=0.8,
    marker="o",
    marker_size=30,
    linewidth=0,
    legend=True,
):
    """
    Adds points corresponding to specific data to the joint plot.
    Data could be disease or site for example.

    Args:
        ax (Axes): matplotlib Axes
        df (DataFrame): pd.DataFrame
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        hue (str): Column name for hue color
        hur_order (list): List of hue order for data (default: None)
        alpha (float): Transparency (default: 0.5)
        marker (str): Marker shape (default: o)
        marker_size (int): Marker size (default: 30)
        linewidth (int): Thick of the line width (default: 0)
        legend (bool): If True, plot legend (default: True)

    Returns:
        ax (Axes): matplotlib Axes with data point corresponding to dataframe

    """
    sns.scatterplot(
        df,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        ax=ax,
        legend=legend,
        marker=marker,
        s=marker_size,
        linewidth=linewidth,
        palette=palette,
        markers=True,
        alpha=alpha,
    )
    if legend:
        ax.legend()

    return ax


def scale_color(rgb, lightness_scale):
    """
    Scale the lightness of the color.

    Args:
        rgb (tuple): RGB color
        lightness_scale (float): Lightness scale

    Returns:
        tuple: RGB color with scaled lightness
    """
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(hue, min(1, lightness * lightness_scale), s=saturation)


def add_models_to_plot(
    ax,
    df_model,
    bundle,
    age_min,
    age_max,
    moving_site=False,
    color="r",
    lightness=1,
    line_width=2.5,
    line_style="--",
):
    x = np.arange(age_min, age_max, 1)
    # Set the color for regression line
    curr_color = matplotlib.colors.ColorConverter.to_rgb(color)
    curr_color = scale_color(curr_color, lightness)

    y = df_model.predict(x, bundle, moving_site=moving_site)

    ax.plot(
        x,
        y,
        label=df_model.model_params["name"],
        color=curr_color,
        linewidth=line_width,
        linestyle=line_style,
    )

    return ax


def add_kde_to_joinplot(
    ax,
    df,
    x,
    y,
    hue,
    palette="Set3",
    hue_order=None,
    alpha=0.6,
    fill=False,
    common_norm=False,
    legend=False,
):
    """
    Adds kde plot corresponding to specific data to the joint plot.
    Data could be disease or site for example.

    Args:
        ax (Axes): matplotlib Axes
        df (DataFrame): pd.DataFrame
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        hue (str): Column name for hue color
        hur_order (list): List of hue order for data (default: None)
        alpha (float): Transparency (default: 0.5)
        fill (bool): If True, fill area under curve (default: False)
        common_norm (bool): If True, normalize all curves to the same height (default: False)
        linewidth (int): Thick of the line width (default: 0)
        legend (bool): If True, plot legend (default: False)

    Returns:
        ax (Axes): matplotlib Axes with data point corresponding to dataframe

    """
    sns.kdeplot(
        df,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        ax=ax,
        legend=legend,
        fill=fill,
        palette=palette,
        alpha=alpha,
        common_norm=common_norm,
    )

    return ax


def add_errorbars_to_plot(
    ax,
    x,
    y,
    y_error,
    palette,
    fmt="none",
    capsize_bar=0,
    label=None,
    alpha=0.4,
    line_width=0,
    linestyle="none",
):
    """
    Adds error bars to the plot. Only available for scatter plots.

    Args:
        ax (Axes): matplotlib Axes
        x (np.array): x-axis data, must be the same as already in the plot
        y (np.array): y-axis data, must be the same as already in the plot
        y_error (np.array or list): y-axis error data, could be np.array (uncertainty) or list
                                    of np.array for lower and upper bounds [lower, upper].
                                    Must be the same size as y.
        palette (str): Color palette
        fmt (str): Format of the error bars (default: none)
        capsize_bar (int): Size of the capsize bar (default: 0)
        label (str): Label for the error bars (default: None)
        alpha (float): Transparency (default: 0.4)
        line_width (int): Line width for error bars (default: 0)
        linestyle (str): Line style for error bars (default: none)

    Returns:
        ax (Axes): matplotlib Axes with error bars
    """
    # If is used to manage the curve case.
    if line_width > 0:
        ax.errorbar(
            x,
            y,
            yerr=y_error,
            fmt=fmt,
            capsize=capsize_bar,
            alpha=alpha,
            lw=line_width,
            linestyle=linestyle,
            zorder=1,
            color=palette,
            label=label,
        )
    else:
        ax.errorbar(
            x,
            y,
            yerr=y_error,
            color=palette,
            fmt=fmt,
            capsize=capsize_bar,
            alpha=alpha,
            zorder=1,
            label=label,
        )
    return ax


def generate_boxplot_error_cohend(
    df_ax1,
    df_ax2,
    x,
    y,
    metric,
    hue,
    order=None,
    ylim=None,
    prefix_title="",
    title="",
    colors=sns.color_palette("Spectral"),
    show_means=True,
    share_x=True,
    share_y=False,
    fig_size=(15, 5),
    ax1_title="",
    ax2_title="",
    save_prefix=None,
):
    """
    Generate boxplot with error bars and mean.

    Args:
        df_ax1 (DataFrame): pd.DataFrame for first subplot
        df_ax2 (DataFrame): pd.DataFrame for second subplot
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        metric (str): Metric name
        hue (str): Column name for hue color
        order (list): List of hue order for data (default: None)
        ylim (tuple): Y-axis limits (ymin, ymax)
        prefix_title (str): Prefix title (default: '')
        title (str): Title (default: '')
        colors (list): List of colors for data (default: sns.color_palette("Spectral"))
        show_means (bool): If True, show mean (default: True)
        share_x (bool): If True, share x-axis (default: True)
        share_y (bool): If True, share y-axis (default: False)
        fig_size (tuple): Figure size (default: (10, 5))
        ax1_title (str): Title for first subplot (default: '')
        ax2_title (str): Title for second subplot (default: '')
        save_prefix (str): Prefix to add to figure name (default: None)
        x_ticklabels (list): List of x-axis tick labels (default: None)

    Returns:
        PNG: Boxplot figure saved in current folder in PNG format.

    """
    fig, ax = plt.subplots(1, 2, figsize=fig_size, sharex=share_x, sharey=share_y)
    sns.boxplot(
        ax=ax[0],
        x=x,
        y=y,
        showmeans=show_means,
        hue=hue,
        palette=colors,
        data=df_ax1,
        order=order,
    )
    sns.boxplot(
        ax=ax[1],
        x=x,
        y=y,
        showmeans=show_means,
        hue=hue,
        palette=colors,
        data=df_ax2,
        order=order,
    )

    if ylim is not None:
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)

    ax[0].set_ylabel("Mean of " + metric.replace("_", " "), fontsize=14)
    x_ticklabels = [
        label.replace("adni_", "ADNI ").replace("_", " ") for label in order
    ]
    fig.text(0.5, 0.04, x, ha="center", fontsize=14)
    fig.suptitle(prefix_title + title, fontsize=16, y=1.05)
    ax[0].set_xticklabels(x_ticklabels, rotation=90)
    ax[1].set_xticklabels(x_ticklabels, rotation=90)
    ax[0].set_title(ax1_title, fontsize=14, y=1.1, pad=-14)
    ax[1].set_title(ax2_title, fontsize=14, y=1.1, pad=-14)

    plt.savefig(
        save_prefix + metric.replace("_", "") + ".png", dpi=300, bbox_inches="tight"
    )


def generate_boxplot(
    box_df,
    x,
    y,
    hue,
    ylim,
    metric,
    bundle,
    dodge=False,
    x_ticklabels=None,
    xlabel="Sites",
    prefix_title="",
    title="",
    legend=True,
    palette=sns.color_palette("Spectral"),
    hue_order=None,
    order=None,
):
    """
    Generate boxplot.

    Args:
        box_df (DataFrame): pd.DataFrame
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        hue (str): Column name for hue color
        ylim (tuple): Y-axis limits (ymin, ymax)
        metric (str): Metric name
        bundle (str): Bundle name
        dodge (bool): If True, dodge boxplot (default: False)
        x_ticklabels (list): List of x-axis tick labels (default: None)
        xlabel (str): X-axis label (default: Sites)
        prefix_title (str): Prefix title (default: '')
        title (str): Title (default: '')
        legend (bool): If True, show legend (default: True)
        palette (list): List of colors for data (default: sns.color_palette("Spectral"))
        hue_order (list): List of hue order for data (default: None)
        order (list): List of order for data (default: None)

    Returns:
        fig (Axes): matplotlib Axes
        ax (Axes): matplotlib Axes
    """
    fig = sns.boxplot(
        box_df,
        x=x,
        y=y,
        palette=palette,
        hue=hue,
        hue_order=hue_order,
        legend=legend,
        order=order,
    )
    ax = fig.axes
    ax.set_xticklabels(x_ticklabels, rotation=90)
    ax.set_ylim(ylim)
    ax.set_ylabel("Mean " + metric.upper(), fontsize=24)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_title(
        prefix_title.upper() + title + bundle.replace("mni_", " "), fontsize=30
    )
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlim(-0.8, len(ax.get_xticklabels()))

    return fig, ax


def update_global_figure_style_and_save(
    fig,
    ax,
    args,
    parser,
    metric,
    bundle,
    harmonization_type,
    move_legend=True,
    legend_location="upper left",
    legend_frameon=False,
    legend_anchor=(1.2, 1),
    legend_title="",
    x_label="Age",
    axis_fontsize=14,
    title="",
    title_fontsize=18,
    prefix_save="",
    suffix_save="",
    empty_background=False,
    dpi=300,
    outpath=None,
    outname=None,
):
    """
    Update figure for x and y axis labels, legend (position change) and adjust final figure size.
    Saves the figure in the current folder to PNG. Figure name depends on script parameters.

    Args:
        fig (JointGrid): seaborn JointGrid
        ax (Axes): matplotlib Axes
        args (Namespace): Namespace object from argparse
        parser (ArgumentParser): ArgumentParser object from argparse
        metric (str): Metric name
        bundle (str): Bundle name
        harmonization_type (str): Harmonization type (pre or combat)
        move_legend (bool): If True, move legend (default: True)
        legend_location (str): Location of legend (default: upper left)
        legend_frameon (bool): If True, add frame to legend (default: False)
        legend_anchor (tuple): Legend anchor (default: (1.2, 1))
        legend_title (str): Legend title (default: "")
        x_label (str): X-axis label (default: Age)
        axis_fontsize (int): Font size for axis labels (default: 14)
        title (str): Title (default: "")
        title_fontsize (int): Font size for title (default: 18)
        empty_background (bool): If True, save with transparent background (default: False)
        suffix_save (str): Suffix to add to figure name (default: "")
        dpi (int): DPI for figure, resolution (default: 300)
        outpath (Path): Path to save the figure
        outname (str): Filename of figure

    Returns:
        PNG: Final figure saved in PNG format.

    """
    if move_legend:
        sns.move_legend(
            ax,
            legend_location,
            bbox_to_anchor=legend_anchor,
            borderaxespad=0,
            frameon=legend_frameon,
            title=legend_title,
        )

    fig.set_axis_labels(
        x_label,
        metric.replace("_", " ").upper().replace(" RANSAC", ""),
        fontsize=axis_fontsize,
    )

    fig.fig.subplots_adjust(top=0.9)

    method_name = harmonization_type
    prefix_title = ""
    if harmonization_type != "raw":
        prefix_title = "ComBAT-"
        method_name += "_harmonized"

    fig.fig.suptitle(
        prefix_title
        + harmonization_type.capitalize()
        + title
        + bundle.replace("_", " ").replace("mni ", ""),
        fontsize=title_fontsize,
    )

    if outname is None:
        outname = "{}_{}_{}_{}{}.png".format(
            prefix_save,
            method_name,
            metric.replace("_", ""),
            bundle.replace("_", ""),
            suffix_save,
        )
    else:
        outname += "_{}_{}_{}{}.png".format(
            method_name, metric.replace("_", ""), bundle.replace("_", ""), suffix_save
        )
    output_filename = join(outpath, outname).replace("__", "_")
    os.makedirs(args.out_dir, exist_ok=True)
    assert_outputs_exist(parser, args, output_filename, check_dir_exists=True)
    logging.info("Saving file: %s", output_filename)
    fig.fig.savefig(
        output_filename,
        transparent=empty_background,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close("all")


def subsample_dataframe(df: pd.DataFrame, max_unique_sids: int, seed: int = 42):
    """
    Generate subsample from full dataframe.

    Args:
        df (DataFrame): pd.DataFrame
        max_unique_sids (int): Maximum of unique subject ids to keep
        seed (int): Random seed (default: 42)

    Returns:
        DataFrame: Subsampled DataFrame

    """
    rng = np.random.default_rng(seed)

    grouped = df.groupby("site")

    def keep_max_sids(group):
        unique_sids = group["sid"].unique()
        if len(unique_sids) > max_unique_sids:
            sids_to_keep = rng.choice(unique_sids, max_unique_sids, replace=False)
            return group[group["sid"].isin(sids_to_keep)]
        else:
            return group

    # Apply the function to each group
    return grouped.apply(keep_max_sids).reset_index(drop=True)
