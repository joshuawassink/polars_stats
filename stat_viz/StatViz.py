import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats
from typing import List, Optional, Dict, Callable, Any
import re
from airstyle.utils import set_font, set_prop_cycle
from utils.pl_stats import lazy_ci, aggregate_with_ci_lazy, compare_groups_lazy

# Apply airstyle formatting
set_font()
set_prop_cycle()


def create_bins(df: pl.DataFrame,
               column: str,
               bins: List[float],
               labels: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Create bins for a column in a polars DataFrame.
    
    Parameters:
    -----------
    df : polars.DataFrame
        Input dataframe
    column : str
        Column name to bin
    bins : List[float]
        List of bin edges
    labels : Optional[List[str]]
        Labels for bins. If None, labels will be generated automatically
        
    Returns:
    --------
    polars.DataFrame
        DataFrame with a new column '{column}_binned' containing the binned values
    """
    # If no labels provided, generate them
    if labels is None:
        labels = ['0-' + str(bins[0])]
        for i in range(len(bins) - 1):
            labels.append(f'{bins[i]}-{bins[i+1]}')
        labels.append(f'{bins[-1]}+')
    
    # Make sure we have the right number of labels
    if len(labels) != len(bins) + 1:
        raise ValueError(f"Number of labels ({len(labels)}) must be equal to number of bins + 1 ({len(bins) + 1})")
    alias = f"{column}_binned"
    # Create binned columns
    result_df = df.with_columns([
        pl.when(pl.col(column).is_null())
        .then(pl.lit(labels[-1]))
        .otherwise(
            pl.col(column).cut(bins, labels=labels)
        ).alias(alias)
    ])

    return result_df


def aggregate_with_ci(df: pl.DataFrame, 
                     grouper: str, 
                     int_columns: List[str], 
                     max_value: Optional[int] = 5,
                     bin_edges: Optional[List[float]] = None,
                     bin_labels: Optional[List[str]] = None, 
                     alpha: float = 0.05) -> pl.DataFrame:
    """
    Aggregate dataframe with confidence intervals using pure polars operations.
    
    Parameters:
    -----------
    df : polars.DataFrame
        Input dataframe
    grouper : str
        Column name to group by
    int_columns : list
        Column names to calculate statistics for
    max_value : int, optional
        Maximum value for the grouper column, values above will be capped
    confidence : float, optional
        Confidence level for the interval calculation (default: 0.95)
        
    Returns:
    --------
    polars.DataFrame
        Aggregated dataframe with confidence intervals
    """
    # Apply binning if bin_edges are specified
    if bin_edges is not None:
        df = create_bins(df, grouper, bin_edges, bin_labels)
        # Use the binned column as the grouper
        binned_grouper = f"{grouper}_binned"
    else:
        # Apply max_value cap if specified and binning is not used
        if max_value is not None:
            df = df.with_columns(
                pl.when(pl.col(grouper) < max_value)
                  .then(pl.col(grouper))
                  .otherwise(max_value)
                  .alias(grouper)
            )
        binned_grouper = grouper
    
    # Create aggregation expressions
    agg_exprs = []  # Updated from pl.count() to pl.len()
    
    # For each column, calculate mean and standard error
    for col in int_columns:
        agg_exprs.extend(lazy_ci(col, alpha))
    # Perform the aggregation
    result_df = df.group_by(binned_grouper).agg(agg_exprs)
        
    return result_df


def _validate_plot_inputs(list_of_dfs: List[pl.DataFrame], 
                         labels: List[str], 
                         grouper: str, 
                         int_columns: List[str],
                         return_data: bool = False) -> None:
    """
    Validate inputs for plotting functions.
    
    Parameters:
    -----------
    list_of_dfs : List[pl.DataFrame]
        List of DataFrames to analyze
    labels : List[str]
        Labels for each DataFrame
    grouper : str
        Column name to group by
    int_columns : List[str]
        Columns to calculate statistics for
        
    Raises:
    -------
    ValueError
        If inputs are invalid
    """
    if not isinstance(list_of_dfs, list) or len(list_of_dfs) == 0:
        raise ValueError("list_of_dfs must be a non-empty list of polars DataFrames")
    
    if not isinstance(labels, list) or len(labels) != len(list_of_dfs):
        raise ValueError(f"labels must be a list with the same length as list_of_dfs. "
                        f"Expected {len(list_of_dfs)} labels, got {len(labels)}")
    
    if not isinstance(grouper, str) or not grouper.strip():
        raise ValueError("grouper must be a non-empty string column name")
    
    if not isinstance(int_columns, list) or len(int_columns) == 0:
        raise ValueError("int_columns must be a non-empty list of column names")
    
    # Validate that all DataFrames have the required columns
    for i, df in enumerate(list_of_dfs):
        if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            raise ValueError(f"Item {i} in list_of_dfs is not a polars DataFrame or LazyFrame")
        
        # Handle both DataFrame and LazyFrame
        if isinstance(df, pl.LazyFrame):
            df_columns = df.collect_schema().names()
        else:
            df_columns = df.columns
            
        if grouper not in df_columns:
            raise ValueError(f"DataFrame {i} (label: '{labels[i]}') missing required grouper column '{grouper}'")
        
        missing_cols = [col for col in int_columns if col not in df_columns]
        if missing_cols:
            raise ValueError(f"DataFrame {i} (label: '{labels[i]}') missing required columns: {missing_cols}")


def _prepare_plot_data(list_of_dfs: List[pl.DataFrame], 
                      grouper: str, 
                      int_columns: List[str],
                      bin_edges: Optional[List[float]] = None) -> tuple[pl.DataFrame, str, List[str]]:
    """
    Prepare and aggregate data for plotting.
    
    Parameters:
    -----------
    list_of_dfs : List[pl.DataFrame]
        List of DataFrames to analyze
    grouper : str
        Column name to group by
    int_columns : List[str]
        Columns to calculate statistics for
    bin_edges : Optional[List[float]]
        Bin edges for grouping
        
    Returns:
    --------
    tuple[pl.DataFrame, str, List[str]]
        Combined aggregated dataframe, actual grouper column name, and sorted unique groups
    """
    results = []
    
    # Apply the aggregate_with_ci function to each DataFrame and store results
    for df in list_of_dfs:
        agg_df = aggregate_with_ci_lazy(df, grouper, int_columns)
        results.append(agg_df)

    # Add a df_index column to each DataFrame
    for i, df in enumerate(results):
        results[i] = df.with_columns(pl.lit(i).alias('df_index'))
    
    # Concatenate all DataFrames vertically
    combined_df = pl.concat(results)
    
    # Determine which column to use for grouping
    actual_grouper = f"{grouper}_binned" if bin_edges is not None else grouper
    
    # Get unique groups and sort them properly
    unique_groups_raw = combined_df.select(actual_grouper).unique()[actual_grouper].to_list()
    
    # Custom sorting that handles both numeric and binned data
    def sort_key(value):
        str_val = str(value)
        # Handle "plus" bins - put them at the end
        if "+" in str_val:
            # Extract number from plus bin for secondary sorting
            match = re.search(r"([0-9]+)\+", str_val)
            if match:
                return (1, int(match.group(1)))  # (1, number) puts plus bins after regular ones
            return (1, float('inf'))  # Plus bins without numbers go last
        
        # Handle range bins (e.g., "5-10")
        match = re.search(r"^([0-9]+(?:\.[0-9]+)?)-", str_val)
        if match:
            return (0, float(match.group(1)))
        
        # Handle pure numeric values
        try:
            return (0, float(str_val))
        except ValueError:
            # For non-numeric strings, sort alphabetically
            return (0, str_val)
    
    # Sort using the custom key and convert to strings for consistency
    unique_groups = [str(g) for g in sorted(unique_groups_raw, key=sort_key)]
    
    return combined_df, actual_grouper, unique_groups


def bar_plot_with_cis(combined_df: pl.DataFrame,
                     actual_grouper: str,
                     unique_groups: List[str],
                     list_of_dfs: List[pl.DataFrame],
                     labels: List[str],
                     int_columns: List[str],
                     ax,
                     show_error: bool = True,
                     **kwargs) -> None:
    """
    Create bar plots with confidence intervals.
    
    Parameters:
    -----------
    combined_df : pl.DataFrame
        Combined aggregated dataframe
    actual_grouper : str
        Column name used for grouping
    unique_groups : List[str]
        Sorted unique group values
    list_of_dfs : List[pl.DataFrame]
        Original list of DataFrames
    labels : List[str]
        Labels for each DataFrame
    int_columns : List[str]
        Columns to plot
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot on
    show_error : bool
        Whether to show error bars
    **kwargs : dict
        Additional arguments (unused but kept for compatibility)
    """
    # Get the color cycle from matplotlib
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    position_map = {group: i for i, group in enumerate(unique_groups)}
    
    # Plot the aggregated summary stats with capped CI bars
    for col_idx, col in enumerate(int_columns):
        for i in range(len(list_of_dfs)):
            group_df = combined_df.filter(pl.col('df_index') == i)
            
            # Sort by the actual grouper values to maintain consistency with unique_groups order
            group_df = group_df.with_columns(
                pl.col(actual_grouper).cast(pl.String).alias(f"{actual_grouper}_str")
            ).sort(f"{actual_grouper}_str")
            
            # Get base positions ensuring they match the sorted unique_groups order
            base_positions = [position_map[str(g)] for g in group_df[actual_grouper]]
            y_values = group_df[f'{col}_mean'].to_numpy()
            
            if show_error:
                # Calculate yerr values for error bars
                lower_yerr = group_df[f'{col}_mean'] - group_df[f'{col}_ci_lower']
                # Round negative lower CIs to 0
                lower_yerr = np.maximum(lower_yerr.to_numpy(), 0)
                yerr = [lower_yerr, (group_df[f'{col}_ci_upper'] - group_df[f'{col}_mean']).to_numpy()]
            else:
                yerr = None

            if col_idx == i:
                # For bar charts, we need to calculate width based on number of datasets
                total_width = 0.8  # Width for all bars in a group (0.8 leaves a 0.2 gap between groups)
                bar_width = total_width / len(list_of_dfs)
                
                # Calculate position for side-by-side bars
                adjusted_x = [x - (total_width/2) + (i+0.5)*bar_width for x in base_positions]
                
                # Use the color assigned to this column from the color cycle
                # Each column gets a consistent color from the color palette
                bar_color = color_cycle[col_idx % len(color_cycle)]
                
                # Plot the bars with color based on column
                bars = ax.bar(
                    adjusted_x,
                    y_values,
                    width=bar_width,
                    color=bar_color,
                    label=f'{labels[i]}',
                    alpha=0.7
                )
                
                # Add percentage labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height) and height > 0.001:
                        ax.text(bar.get_x() + bar.get_width()/1.3, height + 0.0005,
                                f'{height:.1f}%',
                                ha='center', fontsize=11)
                
                if show_error:
                    # Add error bars on top of the bars
                    ax.errorbar(
                        adjusted_x,
                        y_values,
                        yerr=yerr,
                        fmt='o',
                        color='black',
                        capsize=3,
                        capthick=1,
                        elinewidth=1.5,
                        alpha=0.8
                    )


def scatter_plot_with_cis(combined_df: pl.DataFrame,
                         actual_grouper: str,
                         unique_groups: List[str],
                         list_of_dfs: List[pl.DataFrame],
                         labels: List[str],
                         int_columns: List[str],
                         ax,
                         show_error: bool = True,
                         offset: float = 0.2,
                         **kwargs) -> None:
    """
    Create scatter plots with confidence intervals.
    
    Parameters:
    -----------
    combined_df : pl.DataFrame
        Combined aggregated dataframe
    actual_grouper : str
        Column name used for grouping
    unique_groups : List[str]
        Sorted unique group values
    list_of_dfs : List[pl.DataFrame]
        Original list of DataFrames
    labels : List[str]
        Labels for each DataFrame
    int_columns : List[str]
        Columns to plot
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot on
    show_error : bool
        Whether to show error bars
    offset : float
        Offset between points for different DataFrames
    **kwargs : dict
        Additional arguments (unused but kept for compatibility)
    """
    position_map = {group: i for i, group in enumerate(unique_groups)}
    
    # Plot the aggregated summary stats with capped CI bars
    for col in int_columns:
        for i in range(len(list_of_dfs)):
            group_df = combined_df.filter(pl.col('df_index') == i)
            
            # Sort by the actual grouper values to maintain consistency with unique_groups order
            group_df = group_df.with_columns(
                pl.col(actual_grouper).cast(pl.String).alias(f"{actual_grouper}_str")
            ).sort(f"{actual_grouper}_str")
            
            # Get base positions ensuring they match the sorted unique_groups order
            base_positions = [position_map[str(g)] for g in group_df[actual_grouper]]
            y_values = group_df[f'{col}_mean'].to_numpy()
            
            if show_error:
                # Calculate yerr values for error bars
                lower_yerr = group_df[f'{col}_mean'] - group_df[f'{col}_ci_lower']
                # Round negative lower CIs to 0
                lower_yerr = np.maximum(lower_yerr.to_numpy(), 0)
                yerr = [lower_yerr, (group_df[f'{col}_ci_upper'] - group_df[f'{col}_mean']).to_numpy()]
            else:
                yerr = None

            # For scatter plots, apply offset to base positions
            scatter_x = [x + i * offset - ((len(list_of_dfs) - 1) * offset / 2) for x in base_positions]
            
            # Plot points with error bars
            ax.errorbar(
                scatter_x,
                y_values,
                yerr=yerr,
                fmt='o',
                capsize=5,
                label=f'{labels[i]} - {col}' if len(int_columns) > 1 else f'{labels[i]}'
            )


def plot_summary_stats(list_of_dfs: List[pl.DataFrame], 
                      labels: List[str], 
                      grouper: str, 
                      int_columns: List[str], 
                      ax, 
                      plot_type: str = "scatter",
                      show_error: bool = True, 
                      offset: float = 0.2,
                      bin_edges: Optional[List[float]] = None,
                      bin_labels: Optional[List[str]] = None,
                      return_data: bool = False,
                      **kwargs) -> object:
    """
    Plot summary statistics from aggregated polars DataFrames using an extensible framework.
    
    This function provides a modular interface for creating different types of statistical plots
    with confidence intervals. New plot types can be easily added by implementing additional
    plot functions and registering them in the PLOT_TYPES dictionary.
    
    Parameters:
    -----------
    list_of_dfs : List[pl.DataFrame]
        List of DataFrames to analyze. Each DataFrame should contain the grouper column
        and all columns specified in int_columns.
    labels : List[str]
        Labels for each DataFrame in the legend. Must have the same length as list_of_dfs.
    grouper : str
        Column name to group by. This column must exist in all DataFrames.
    int_columns : List[str]
        Column names to calculate statistics for. These columns must exist in all DataFrames
        and should contain numeric data suitable for mean/CI calculations.
    ax : matplotlib.axes.Axes or None
        Matplotlib axes object to plot on. Can be None when return_data=True.
    plot_type : str, default="scatter"
        Type of plot to generate. Currently supported: "scatter", "bar".
        Additional plot types can be added by implementing new plot functions.
    show_error : bool, default=True
        Whether to show confidence interval error bars.
    offset : float, default=0.2
        Offset between points for different DataFrames (used in scatter plots).
    bin_edges : Optional[List[float]], default=None
        List of bin edges for binning the grouper column. If provided, data will be binned
        before aggregation.
    bin_labels : Optional[List[str]], default=None
        Custom labels for the bins. If None, labels will be generated automatically.
    return_data : bool, default=False
        If True, returns the sorted aggregated data table instead of creating a plot.
        When True, the 'ax' parameter can be None.
    **kwargs : dict
        Additional keyword arguments passed to the specific plot function.
    
    Returns:
    --------
    matplotlib.axes.Axes or pl.DataFrame
        If return_data is False: The matplotlib axes object with the completed plot.
        If return_data is True: A polars DataFrame containing the sorted aggregated data.
        
    Raises:
    -------
    ValueError
        If plot_type is not supported, or if input validation fails.
    """
    # Define supported plot types and their corresponding functions
    PLOT_TYPES: Dict[str, Callable] = {
        "scatter": scatter_plot_with_cis,
        "bar": bar_plot_with_cis,
    }
    
    # Validate plot type
    if plot_type not in PLOT_TYPES:
        available_types = ", ".join(f"'{t}'" for t in PLOT_TYPES.keys())
        raise ValueError(f"Unsupported plot_type '{plot_type}'. "
                        f"Available plot types: {available_types}")
    
    # Validate inputs
    _validate_plot_inputs(list_of_dfs, labels, grouper, int_columns, return_data)
    
    # Prepare data for plotting
    try:
        combined_df, actual_grouper, unique_groups = _prepare_plot_data(
            list_of_dfs, grouper, int_columns, bin_edges
        )
    except Exception as e:
        raise ValueError(f"Error preparing plot data: {str(e)}. "
                        f"Please check that all DataFrames contain the required columns "
                        f"and that the data is properly formatted.") from e
    
    # If return_data is True, return the sorted data table instead of plotting
    if return_data:
        # Add labels column to identify each DataFrame
        labeled_df = combined_df.with_columns(
            pl.col('df_index').map_elements(lambda x: labels[x], return_dtype=pl.String).alias('dataset_label')
        )
        # Sort by the actual grouper to maintain consistency with plot ordering
        return labeled_df.sort([actual_grouper, 'df_index'])
    
    # Get the appropriate plot function and execute it
    plot_function = PLOT_TYPES[plot_type]
    try:
        plot_function(
            combined_df=combined_df,
            actual_grouper=actual_grouper,
            unique_groups=unique_groups,
            list_of_dfs=list_of_dfs,
            labels=labels,
            int_columns=int_columns,
            ax=ax,
            show_error=show_error,
            offset=offset,
            **kwargs
        )
    except Exception as e:
        raise ValueError(f"Error creating {plot_type} plot: {str(e)}. "
                        f"This may be due to data format issues or missing values "
                        f"in your DataFrames.") from e
    
    # Configure plot aesthetics
    _configure_plot_aesthetics(ax, unique_groups, bin_edges, bin_labels, 
                              list_of_dfs, int_columns)
    
    return ax


def _configure_plot_aesthetics(ax, 
                              unique_groups: List[str],
                              bin_edges: Optional[List[float]],
                              bin_labels: Optional[List[str]],
                              list_of_dfs: List[pl.DataFrame],
                              int_columns: List[str]) -> None:
    """
    Configure common plot aesthetics like axis labels, formatting, and legend.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object
    unique_groups : List[str]
        Sorted unique group values for x-axis labels
    bin_edges : Optional[List[float]]
        Bin edges if binning was used
    bin_labels : Optional[List[str]]
        Custom bin labels
    list_of_dfs : List[pl.DataFrame]
        Original DataFrames (for determining if legend is needed)
    int_columns : List[str]
        Columns being plotted (for determining if legend is needed)
    """
    # Set the x-ticks to be in the middle of each group and use the actual group labels
    ax.set_xticks(range(len(unique_groups)))
    
    # Use bin_labels for x-axis labels if provided and binning is used, otherwise use unique_groups
    if bin_edges is not None and bin_labels is not None:
        # Create a mapping from unique_groups to bin_labels to ensure proper alignment
        # This assumes bin_labels are provided in the same order as the original bins
        if len(bin_labels) == len(unique_groups):
            ax.set_xticklabels(bin_labels)
        else:
            # Fallback to unique_groups if lengths don't match
            ax.set_xticklabels(unique_groups)
    else:
        ax.set_xticklabels(unique_groups)
    
    # Format the y-axis to show percentages
    # Use 100.0 because relative_lift_pct is already multiplied by 100
    ax.yaxis.set_major_formatter(PercentFormatter(100.0))
    
    # Add grid, legend, and other styling elements
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if len(list_of_dfs) > 1 or len(int_columns) > 1:
        ax.legend(loc='upper right')


# Example of how to extend the framework with a new plot type:
#
# def line_plot_with_cis(combined_df: pl.DataFrame,
#                        actual_grouper: str,
#                        unique_groups: List[str],
#                        list_of_dfs: List[pl.DataFrame],
#                        labels: List[str],
#                        int_columns: List[str],
#                        ax,
#                        show_error: bool = True,
#                        **kwargs) -> None:
#     """
#     Create line plots with confidence intervals.
#     """
#     position_map = {group: i for i, group in enumerate(unique_groups)}
#     
#     for col in int_columns:
#         for i in range(len(list_of_dfs)):
#             group_df = combined_df.filter(pl.col('df_index') == i)
#             
#             base_positions = [position_map[g] for g in group_df[actual_grouper]]
#             y_values = group_df[f'{col}_mean'].to_numpy()
#             
#             if show_error:
#                 lower_yerr = group_df[f'{col}_mean'] - group_df[f'{col}_ci_lower']
#                 lower_yerr = np.maximum(lower_yerr.to_numpy(), 0)
#                 yerr = [lower_yerr, (group_df[f'{col}_ci_upper'] - group_df[f'{col}_mean']).to_numpy()]
#             else:
#                 yerr = None
#
#             # Line plot implementation
#             ax.plot(base_positions, y_values, 
#                     marker='o', linestyle='-',
#                     label=f'{labels[i]} - {col}' if len(int_columns) > 1 else f'{labels[i]}')
#             
#             if show_error:
#                 ax.errorbar(base_positions, y_values, yerr=yerr, 
#                            fmt='none', capsize=3, alpha=0.7)
#
# To use the new plot type, simply add it to the PLOT_TYPES dictionary:
# PLOT_TYPES["line"] = line_plot_with_cis
#
# Then call: plot_summary_stats(..., plot_type="line")


def _prepare_lift_data(df: pl.LazyFrame,
                      grouper: str,
                      treatment_value: str,
                      control_value: str,
                      outcome_col: str,
                      segment_col: Optional[str] = None,
                      segment_values: Optional[List[str]] = None,
                      alpha: float = 0.05) -> tuple[pl.DataFrame, str, List[str]]:
    """
    Prepare lift data in the format expected by existing plot functions.
    
    This function transforms lift results into a format compatible with 
    scatter_plot_with_cis and bar_plot_with_cis by creating mock confidence intervals
    in the expected column naming convention.
    """
    # Prepare segments
    if segment_col is None:
        segments = ["Overall"]
    else:
        if segment_values is None:
            segment_values = df.select(segment_col).unique().collect()[segment_col].to_list()
        segments = segment_values
    
    # Calculate lift statistics for each segment
    lift_results = []
    for segment in segments:
        if segment_col is None or segment == "Overall":
            segment_df = df
            segment_label = "Overall"
        else:
            segment_df = df.filter(pl.col(segment_col) == segment)
            segment_label = str(segment)
        
        try:
            result = compare_groups_lazy(
                df=segment_df,
                grouper=grouper,
                treatment_value=treatment_value,
                control_value=control_value,
                outcome_col=outcome_col,
                alpha=alpha
            )
            result = result.with_columns(pl.lit(segment_label).alias('segment'))
            lift_results.append(result)
        except Exception as e:
            print(f"Warning: Could not calculate lift for segment '{segment_label}': {e}")
            continue
    
    if not lift_results:
        raise ValueError("No valid lift calculations could be performed")
    
    # Combine results
    combined_results = pl.concat(lift_results)
    
    # Transform to format expected by existing plot functions
    # We need columns like: {metric}_mean, {metric}_ci_lower, {metric}_ci_upper, {metric}_n, etc.
    formatted_df = combined_results.select([
        pl.col('segment'),
        # Transform relative_lift_pct to look like a "_mean" column
        pl.col('relative_lift_pct').alias('relative_lift_pct_mean'),
        pl.col('rel_lift_ci_lower').alias('relative_lift_pct_ci_lower'),
        pl.col('rel_lift_ci_upper').alias('relative_lift_pct_ci_upper'),
        # Add dummy standard deviation and count for compatibility
        pl.lit(1.0).alias('relative_lift_pct_std'),
        pl.col('treatment_n').alias('relative_lift_pct_n'),
        pl.lit(0.0).alias('relative_lift_pct_se'),  # Not used but expected
        # Keep significance info for coloring
        pl.col('is_significant'),
        # Add df_index for compatibility with existing functions (single dataset)
        pl.lit(0).alias('df_index'),
        # Keep original columns for table output
        pl.col('treatment_mean'),
        pl.col('control_mean'),
        pl.col('treatment_n').alias('treatment_count'),
        pl.col('control_n'),
        pl.col('absolute_lift'),
        pl.col('abs_lift_ci_lower'),
        pl.col('abs_lift_ci_upper')
    ])
    
    # Get sorted unique segments
    unique_segments = [str(s) for s in sorted(segments, key=str)]
    
    return formatted_df, 'segment', unique_segments


def plot_relative_lift(df: pl.LazyFrame,
                      grouper: str,
                      treatment_value: str,
                      control_value: str,
                      outcome_col: str,
                      segment_col: Optional[str] = None,
                      segment_values: Optional[List[str]] = None,
                      alpha: float = 0.05,
                      plot_type: str = "bar",
                      ax=None,
                      show_error: bool = True,
                      title: Optional[str] = None,
                      **kwargs) -> object:
    """
    Plot relative lift analysis results with statistical significance testing.
    
    This function leverages the existing plot infrastructure (scatter_plot_with_cis and 
    bar_plot_with_cis) by transforming lift data into the expected format.
    
    Parameters:
    -----------
    df : pl.LazyFrame
        Input LazyFrame containing the experimental data
    grouper : str
        Column name containing group assignments (treatment/control)
    treatment_value : str
        Value identifying the treatment group
    control_value : str
        Value identifying the control group
    outcome_col : str
        Column name for the outcome variable to analyze
    segment_col : Optional[str], default=None
        Column name for segmentation. If provided, analysis will be done for each segment.
    segment_values : Optional[List[str]], default=None
        Specific segment values to analyze. If None, all unique values in segment_col will be used.
    alpha : float, default=0.05
        Significance level for confidence intervals
    plot_type : str, default="bar"
        Type of plot: "bar" for bar charts, "scatter" for scatter plots, "table" for table output
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates new axes.
    show_error : bool, default=True
        Whether to show confidence interval error bars
    title : Optional[str], default=None
        Plot title. If None, generates automatic title.
    **kwargs : dict
        Additional arguments passed to plotting functions
        
    Returns:
    --------
    matplotlib.axes.Axes or pl.DataFrame
        If plot_type is "table": returns DataFrame with lift analysis results
        Otherwise: returns matplotlib axes with the plot
        
    Raises:
    -------
    ValueError
        If required columns are missing or plot_type is invalid
    """
    # Validate inputs
    df_schema = df.collect_schema()
    required_cols = [grouper, outcome_col]
    if segment_col:
        required_cols.append(segment_col)
        
    missing_cols = [col for col in required_cols if col not in df_schema.names()]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Define supported plot types - reuse existing functions from plot_summary_stats
    PLOT_TYPES: Dict[str, Callable] = {
        "scatter": scatter_plot_with_cis,
        "bar": bar_plot_with_cis,
    }
    
    # Validate plot type
    if plot_type == "table":
        # Handle table output separately
        combined_results, _, _ = _prepare_lift_data(
            df, grouper, treatment_value, control_value, outcome_col,
            segment_col, segment_values, alpha
        )
        return _format_lift_table(combined_results)
    elif plot_type not in PLOT_TYPES:
        available_types = list(PLOT_TYPES.keys()) + ["table"]
        raise ValueError(f"plot_type must be one of {available_types}, got '{plot_type}'")
    
    # Prepare lift data using the helper function
    combined_df, actual_grouper, unique_groups = _prepare_lift_data(
        df, grouper, treatment_value, control_value, outcome_col,
        segment_col, segment_values, alpha
    )
    
    # Create plot if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use existing plot infrastructure directly - same as plot_summary_stats
    plot_function = PLOT_TYPES[plot_type]
    plot_function(
        combined_df=combined_df,
        actual_grouper=actual_grouper,
        unique_groups=unique_groups,
        list_of_dfs=[combined_df],  # Single "dataset" 
        labels=["Lift Analysis"],
        int_columns=["relative_lift_pct"],
        ax=ax,
        show_error=show_error,
        **kwargs
    )
    
    # Configure standard plot aesthetics
    _configure_plot_aesthetics(ax, unique_groups, None, None, [combined_df], ["relative_lift_pct"])
    
    # Add lift-specific customizations
    _configure_lift_plot_aesthetics(ax, unique_groups, title, outcome_col)
    
    return ax


def _format_lift_table(combined_df: pl.DataFrame) -> pl.DataFrame:
    """Format lift results for table output with comprehensive metrics."""
    return combined_df.select([
        pl.col('segment'),
        pl.col('treatment_mean'),
        pl.col('control_mean'),
        pl.col('treatment_count').alias('treatment_n'),
        pl.col('control_n'),
        pl.col('absolute_lift'),
        pl.col('relative_lift_pct_mean').alias('relative_lift_pct'),
        pl.col('abs_lift_ci_lower'),
        pl.col('abs_lift_ci_upper'),
        pl.col('relative_lift_pct_ci_lower').alias('rel_lift_ci_lower'),
        pl.col('relative_lift_pct_ci_upper').alias('rel_lift_ci_upper'),
        pl.col('is_significant')
    ])




def _configure_lift_plot_aesthetics(ax, 
                                   unique_groups: List[str],
                                   title: Optional[str], 
                                   outcome_col: str) -> None:
    """Configure aesthetics for lift plots - simplified version."""
    # Set y-axis label to percentage
    ax.set_ylabel('Relative Lift (%)')
    
    # Add horizontal line at zero for easy interpretation
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=0.8)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Relative Lift Analysis - {outcome_col}')
    
    # The rest of the styling (x-axis, grid, etc.) is handled by _configure_plot_aesthetics