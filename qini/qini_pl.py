import logging
from typing import List, Optional, Tuple, Union
import polars as pl
import matplotlib.pyplot as plt

from ml_models.martech_intelligence.uplift.src.shared.utils.pl_utils import (
    load_array,
)


def qini_val(
    df: pl.LazyFrame,
    assignment_col: str = "assignment",
    outcome_col: str = "y",
    control_val: str = "control",
    treatment_val: str = "treatment",
) -> float:
    """
    Calculate raw Qini statistic for input data.

    Args:
        df: LazyFrame containing the experimental data
        assignment_col: Column name containing treatment assignment values. Defaults to "assignment".
        outcome_col: Column name containing outcome values (0/1). Defaults to "y".
        control_val: Value in assignment_col indicating control group. Defaults to "control".
        treatment_val: Value in assignment_col indicating treatment group. Defaults to "treatment".

    Returns:
        float: Raw qini value representing net incremental outcomes

    Raises:
        ValueError: If no control or treatment observations found
    """
    df = load_array(df)

    # Get subset statistics in single query
    df_stats = (
        df.group_by(assignment_col)
        .agg([pl.col(outcome_col).sum().alias("Y"), pl.len().alias("N")])
        .collect()
    )

    T_subset = df_stats.filter(pl.col(assignment_col) == treatment_val)
    C_subset = df_stats.filter(pl.col(assignment_col) == control_val)

    # Handle edge cases
    if len(C_subset) == 0 or C_subset["N"].item() == 0:
        raise ValueError("No control observations found in data")
    if len(T_subset) == 0 or T_subset["N"].item() == 0:
        raise ValueError("No treatment observations found in data")

    # Raw uplift calculation
    yt = T_subset["Y"].item()  # Treated positive outcomes in subset
    yc = C_subset["Y"].item()  # Control positive outcomes in subset
    nt = T_subset["N"].item()  # Treated count in subset
    nc = C_subset["N"].item()  # Control count in subset

    # Calculate raw qini
    qini = yt - (nt * yc) / nc

    return qini


def calc_qini_at_threshold(
    df: pl.LazyFrame,
    qini_col: str,
    threshold: Optional[float] = None,
    percentile: Optional[float] = None,
    assignment_col: str = "assignment",
    outcome_col: str = "y",
    control_val: str = "control",
    treatment_val: str = "treatment",
) -> float:
    """
    Apply threshold or percentile filtering and calculate Qini statistic.

    Args:
        df: LazyFrame containing the experimental data
        qini_col: Column name to apply threshold/percentile filtering on
        threshold: Optional threshold value to filter data (qini_col >= threshold)
        percentile: Optional percentile value (0-1) to calculate threshold from qini_col
        assignment_col: Column name containing treatment assignment values. Defaults to "assignment".
        outcome_col: Column name containing outcome values (0/1). Defaults to "y".
        control_val: Value in assignment_col indicating control group. Defaults to "control".
        treatment_val: Value in assignment_col indicating treatment group. Defaults to "treatment".

    Returns:
        float: Raw qini value for filtered data

    Raises:
        ValueError: If both threshold and percentile are provided, or if neither is provided
    """
    if threshold is not None and percentile is not None:
        raise ValueError("Cannot specify both threshold and percentile")
    if threshold is None and percentile is None:
        raise ValueError("Must specify either threshold or percentile")

    df = load_array(df)

    if percentile is not None:
        if not 0 <= percentile <= 1:
            raise ValueError("Percentile must be between 0 and 1")
        # Calculate threshold from percentile
        threshold = df.select(pl.col(qini_col).quantile(percentile)).collect().item()

    # Apply threshold filtering
    filtered_df = df.filter(pl.col(qini_col) >= threshold)

    # Calculate qini on filtered data
    return qini_val(
        filtered_df,
        assignment_col=assignment_col,
        outcome_col=outcome_col,
        control_val=control_val,
        treatment_val=treatment_val,
    )


class QiniEval:
    """
    Qini evaluation class for uplift model evaluation.
    """

    def __init__(
        self,
        model_vars: List[str],
        df: pl.DataFrame,
        assignment_col: str = "assignment",
        outcome_col: str = "y",
        control_val="control",
        treatment_val="treatment",
    ):
        """
        Initialize QiniEval with experimental data and model predictions.

        Args:
            df: DataFrame containing experimental data
            model_vars: List of model column names to evaluate
            assignment_col: Column indicating treatment assignment
            outcome_col: Column indicating binary outcome (0/1)
            control_val: Value indicating control group
            treatment_val: Value indicating treatment group
        """
        # Validate inputs
        if not model_vars:
            raise ValueError("model_vars cannot be empty")

        # Store parameters - keep dataframe lazy
        self.df = load_array(df)
        # create treatment_binary for cumulation
        self.df = df.with_columns(
            pl.when(pl.col("assignment") == control_val).then(0).otherwise(1).alias("treatment")
        )

        self.model_vars = model_vars
        self.assignment_col = assignment_col
        self.outcome_col = outcome_col
        self.control_val = control_val
        self.treatment_val = treatment_val

        # Validate that required columns exist
        df_cols = self.df.collect_schema().names()
        if self.assignment_col not in df_cols:
            raise ValueError(f"Assignment column '{self.assignment_col}' not found in DataFrame")
        if self.outcome_col not in df_cols:
            raise ValueError(f"Outcome column '{self.outcome_col}' not found in DataFrame")
        missing_models = [m for m in model_vars if m not in df_cols]
        if missing_models:
            raise ValueError(f"Model columns not found in DataFrame: {missing_models}")

        # Lazy-computed properties
        self._baseline_stats = None
        self.qini_df = None
        self.max_performance = None
        self._calculated = False

        # Initialize lift constants (computed once)
        self._control_lift = None
        self._treatment_lift = None
        self._total_population = None

        self.calc()

    @property
    def baseline_stats(self):
        """Lazy-computed baseline statistics and expected outcomes."""
        if self._baseline_stats is None:
            stats = (
                self.df.group_by(self.assignment_col)
                .agg(
                    [
                        pl.len().alias("N"),
                        pl.col(self.outcome_col).sum().alias("Y"),
                        pl.col(self.outcome_col).mean().alias("rate"),
                    ]
                )
                .collect()
            )

            control = stats.filter(pl.col(self.assignment_col) == self.control_val)
            treatment = stats.filter(pl.col(self.assignment_col) == self.treatment_val)

            control_dict = control.to_dicts()[0]
            treatment_dict = treatment.to_dicts()[0]

            self._baseline_stats = {"control": control_dict, "treatment": treatment_dict}

            # Compute expected outcomes
            self._total_population = control_dict["N"] + treatment_dict["N"]
            self._control_lift = control_dict["rate"] * self._total_population
            self._treatment_lift = treatment_dict["rate"] * self._total_population

        return self._baseline_stats

    def _validate(self, models: Union[str, List[str]]):
        """
        Flexible validation method.

        Args:
            models: List of models to validate or single model string
            calc: If True, ensure calc() has been called
        """
        if models is not None:
            # Handle both single model and list of models
            model_list = [models] if isinstance(models, str) else models
            for model in model_list:
                if model not in self.model_vars:
                    raise ValueError(f"Model '{model}' not found in model_vars: {self.model_vars}")

    def _cum_qini(
        self,
        df: pl.LazyFrame,
        model: str,
    ):
        """
        Calculate cumulative Qini statistic for a model.

        Args:
            df_subset: LazyFrame containing the data (will be sorted by model)
            model: Model column name to sort by and calculate Qini for
            outcome_col: Column containing binary outcomes (0/1)
            treatment_col: Column containing treatment assignment (0/1)

        Returns:
            pl.LazyFrame: DataFrame with model scores and cumulative Qini values
        """
        sorted_df = df.sort(model, descending=True)

        # Add treatment/control cumsums
        sorted_df = sorted_df.with_columns((1 - pl.col("treatment")).alias("control"))

        # Sums required to compute qini
        sorted_df = sorted_df.with_columns(
            pl.cum_sum("treatment").alias("nt"),
            pl.cum_sum("control").alias("nc"),
            (pl.col("treatment") * pl.col(self.outcome_col)).cum_sum().alias("yt"),
            (pl.col("control") * pl.col(self.outcome_col)).cum_sum().alias("yc"),
        )

        sorted_df = sorted_df.with_columns(
            pl.col("yt")
            .sub(pl.col("yc").mul(pl.col("nt").truediv(pl.col("nc"))))
            .fill_nan(0)
            .alias("qini"),
            (pl.col("nt") + pl.col("nc")).alias("N"),
        )

        return sorted_df.select(
            [
                pl.col(model),  # Include the sorted model column
                pl.col("qini").alias(f"{model}_qini"),
            ]
        )

    def qini_models(
        self,
        df=None,
        models: Optional[List] = None,
    ):
        """
        Calculate Qini curves for all models using parallelized collection.

        Args:
            df: DataFrame to use (defaults to self.df)
            models: List of models to calculate (defaults to self.model_vars)

        Returns:
            pl.DataFrame: Combined DataFrame with qini curves for all models
        """
        df = df or self.df
        df = load_array(df)
        models = models or self.model_vars

        # Creates a dictionary of lazy expressions that can be collected simultaneously
        lazy_qini_queries = {f"{model}_qini": self._cum_qini(df, model=model) for model in models}

        # Add row_number as first query in the dict
        lazy_qini_queries = {
            "row": df.with_row_count("row", offset=1).select("row"),
            **lazy_qini_queries,
        }

        qini_results = pl.collect_all(list(lazy_qini_queries.values()))

        # Create DataFrame from collected results - extract columns properly
        qini_df = pl.concat(qini_results, how="horizontal")

        qini_df = qini_df.with_columns(
            pl.col("row").truediv(self._total_population).alias("percentile"),
            pl.col("row")
            .mul(self.baseline_stats["treatment"]["rate"] - self.baseline_stats["control"]["rate"])
            .alias("random_baseline"),
        )

        return qini_df

    def _get_max_performance(self):
        """Calculate max percentile information for all models from curves DataFrame.

        Returns:
            pl.DataFrame: Max percentile info for each model
        """
        max_performance = []

        for model in self.model_vars:
            qini_col = f"{model}_qini"
            df_model = (
                self.qini_df[model, qini_col, "percentile", "random_baseline"]
                .clone()
                .sort("percentile")
            )
            df_model = df_model.unique(subset=model, keep="last")

            max_qini = df_model[qini_col].max()
            r = df_model.filter(pl.col(qini_col) == max_qini)
            max_performance.append(
                {
                    "model": model,
                    "max_qini": max_qini,
                    "max_percentile": r["percentile"].item(),
                    "untargeted_lift": r["random_baseline"].item(),
                    "lift_relative_to_random": abs(max_qini / r["random_baseline"].item()),
                    "threshold": r[model].item(),
                }
            )
        return pl.DataFrame(max_performance)

    def get_optimal_threshold(self) -> Tuple[str, float]:
        """
        Get the optimal model and its cutoff threshold.

        Returns:
            Tuple[str, float]: (optimal_model_name, threshold_value)
        """
        self.calc()

        # Best Model
        optimal_model = self.max_performance.filter(
            pl.col("max_qini") == self.max_performance["max_qini"].max()
        )
        return (optimal_model["model"].item(), optimal_model["threshold"].item())

    def calc_qini_at_optimal_threshold(self, df: pl.LazyFrame) -> float:
        """
        Calculate Qini statistic on external dataframe using optimal model and threshold.

        Args:
            df: LazyFrame containing the experimental data to evaluate

        Returns:
            float: Raw qini value for filtered data using optimal threshold
        """
        df = load_array(df)
        model_col, threshold = self.get_optimal_threshold()

        return calc_qini_at_threshold(
            df=df,
            qini_col=model_col,
            threshold=threshold,
            assignment_col=self.assignment_col,
            outcome_col=self.outcome_col,
            control_val=self.control_val,
            treatment_val=self.treatment_val,
        )

    def calc(self):
        """
        Calculate qini curves and performance metrics for all models.

        Returns:
            self: Returns self for method chaining
        """
        # Calculate qini curves using class method
        if self._calculated:
            return self
        logging.info("Calculating Qini Statistics")

        # First compute baselinestats
        _ = self.baseline_stats

        self.qini_df = self.qini_models()
        self.max_performance = self._get_max_performance()

        self._calculated = True

        return self

    def plot_qini_curve(
        self,
        models: List[str] = None,
        normalize: str = None,
        show_random_baseline: bool = True,
        ax=None,
        figsize: tuple = (10, 6),
        title: str = None,
        xlabel: str = "Percent of Reachable Audience Targeted",
        ylabel: str = None,
    ):
        """
        Plot raw Qini curves for specified models with optional random baseline.

        Args:
            models: List of model names to plot. If None, plots all models.
            normalize: Normalization type ("control", "treatment", or None)
            show_random_baseline: Whether to show random baseline
            ax: Matplotlib axes to plot on. If None, creates new figure.
            figsize: Figure size tuple (width, height) - only used if ax is None
            title: Plot title (defaults based on normalization if None)
            xlabel: X-axis label
            ylabel: Y-axis label (defaults based on normalization if None)

        Returns:
            matplotlib figure and axes objects
        """
        # Default to all models if none specified
        if models is None:
            models = self.model_vars

        self._validate(models)
        self.calc()

        # Set default title and ylabel based on normalization
        if title is None:
            if normalize is None:
                title = "Raw Qini Curves"
            else:
                title = f"Normalized Qini Curves (Percent lift Relative to {normalize.title()}-Baseline)"

        if ylabel is None:
            if normalize is None:
                ylabel = "Raw Uplift"
            else:
                ylabel = f"Normalized Uplift (% change relative to {normalize.title()} Baseline)"

        # Create plot if ax not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # X values from percentile column (convert to 0-100 scale)
        x_values = (self.qini_df["percentile"] * 100).to_list()

        # Plot each model's qini curve
        for model in models:
            qini_col = f"{model}_qini"  # Correct column name (without "_curve")
            if qini_col in self.qini_df.columns:
                y_values = self.qini_df[qini_col].to_list()
                if normalize is not None:
                    # Apply normalization
                    if normalize == "control":
                        y_values = [y / self._control_lift for y in y_values]
                    elif normalize == "treatment":
                        y_values = [y / self._treatment_lift for y in y_values]
                ax.plot(x_values, y_values, marker="o", label=model, linewidth=2, markersize=1)

        # Plot random baseline if requested and available
        if show_random_baseline and "random_baseline" in self.qini_df.columns:
            baseline_values = self.qini_df["random_baseline"].to_list()
            if normalize is not None:
                # Apply normalization to baseline
                if normalize == "control":
                    baseline_values = [b / self._control_lift for b in baseline_values]
                elif normalize == "treatment":
                    baseline_values = [b / self._treatment_lift for b in baseline_values]

            ax.plot(
                x_values,
                baseline_values,
                linestyle="--",
                color="gray",
                alpha=0.7,
                label="Random Baseline",
                linewidth=0.5,
            )

        # Add zero line for reference
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Set x-axis to show percentages from 0-100% with 10% steps
        ax.set_xlim(0, 100)
        x_ticks = list(range(0, 101, 10))  # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x}%" for x in x_ticks])

        plt.tight_layout()
        return fig, ax