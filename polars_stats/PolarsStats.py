import polars as pl
from scipy import stats
from typing import Optional, List, Tuple, Union

def lazy_ci(col: str, alpha = 0.05):
    z_value = stats.norm.ppf(1 - alpha/2)

    return [
        pl.mean(col).alias(f'{col}_mean'),
        pl.std(col).alias(f'{col}_std'),
        pl.col(col).count().alias(f'{col}_n'),
        (pl.std(col) / pl.col(col).count().sqrt()).alias(f'{col}_se'),
        (pl.mean(col) - z_value * (pl.std(col) / pl.col(col).count().sqrt())).alias(f'{col}_ci_lower'),
        (pl.mean(col) + z_value * (pl.std(col) / pl.col(col).count().sqrt())).alias(f'{col}_ci_upper')
    ]

def aggregate_with_ci_lazy(df: pl.LazyFrame, 
                     grouper: str, 
                     int_columns: List[str], 
                     max_value: Optional[int] = 5,
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
    # Create aggregation expressions
    agg_exprs = []  # Updated from pl.count() to pl.len()
    
    # For each column, calculate mean and standard error
    for col in int_columns:
        agg_exprs.extend(lazy_ci(col, alpha))
    # Perform the aggregation
    result_df = df.group_by(grouper).agg(agg_exprs).collect()
        
    return result_df


def relative_lift(treatment_col: str, control_col: str, alpha: float = 0.05):
    """
    Calculate relative lift with statistical significance testing using lazy evaluation.
    
    Returns expressions for absolute difference, percentage difference, and confidence intervals
    for the difference between two group means.
    
    Parameters:
    -----------
    treatment_col : str
        Column name for treatment group values
    control_col : str  
        Column name for control group values
    alpha : float
        Significance level for confidence intervals (default: 0.05 for 95% CI)
        
    Returns:
    --------
    list of polars expressions for:
        - absolute_lift: treatment_mean - control_mean
        - relative_lift_pct: ((treatment_mean - control_mean) / control_mean) * 100
        - pooled_se: pooled standard error for the difference
        - t_statistic: t-statistic for significance test
        - p_value: p-value for two-tailed t-test
        - abs_lift_ci_lower: lower bound of absolute lift CI
        - abs_lift_ci_upper: upper bound of absolute lift CI
        - rel_lift_ci_lower: lower bound of relative lift CI (%)
        - rel_lift_ci_upper: upper bound of relative lift CI (%)
    """
    # Calculate basic statistics for each group
    treatment_mean = pl.mean(treatment_col)
    control_mean = pl.mean(control_col)
    treatment_std = pl.std(treatment_col)
    control_std = pl.std(control_col)
    treatment_n = pl.col(treatment_col).count()
    control_n = pl.col(control_col).count()
    
    # Calculate absolute and relative lift
    abs_lift = treatment_mean - control_mean
    rel_lift_pct = ((treatment_mean - control_mean) / control_mean) * 100
    
    # Calculate pooled standard error for two-sample t-test
    pooled_var = (((treatment_n - 1) * treatment_std.pow(2)) + 
                  ((control_n - 1) * control_std.pow(2))) / (treatment_n + control_n - 2)
    pooled_se = pooled_var.sqrt() * ((1 / treatment_n) + (1 / control_n)).sqrt()
    
    # Calculate t-statistic and degrees of freedom
    t_stat = abs_lift / pooled_se
    df = treatment_n + control_n - 2
    
    # For p-value calculation, we'll use a conservative approach since polars doesn't have t-distribution
    # Use normal approximation (valid for large samples)
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # Confidence intervals for absolute lift
    abs_lift_margin = z_critical * pooled_se
    abs_lift_ci_lower = abs_lift - abs_lift_margin
    abs_lift_ci_upper = abs_lift + abs_lift_margin
    
    # Confidence intervals for relative lift (using delta method approximation)
    rel_lift_se_pct = (pooled_se / control_mean) * 100
    rel_lift_margin = z_critical * rel_lift_se_pct
    rel_lift_ci_lower = rel_lift_pct - rel_lift_margin
    rel_lift_ci_upper = rel_lift_pct + rel_lift_margin
    
    return [
        abs_lift.alias('absolute_lift'),
        rel_lift_pct.alias('relative_lift_pct'), 
        pooled_se.alias('pooled_se'),
        t_stat.alias('t_statistic'),
        # Store critical value for significance testing
        pl.lit(z_critical).alias('critical_value'),
        abs_lift_ci_lower.alias('abs_lift_ci_lower'),
        abs_lift_ci_upper.alias('abs_lift_ci_upper'),
        rel_lift_ci_lower.alias('rel_lift_ci_lower'), 
        rel_lift_ci_upper.alias('rel_lift_ci_upper'),
        # Add significance flag
        (t_stat.abs() > z_critical).alias('is_significant')
    ]


def compare_groups_lazy(df: pl.LazyFrame, 
                       grouper: str,
                       treatment_value: str,
                       control_value: str, 
                       outcome_col: str,
                       alpha: float = 0.05) -> pl.DataFrame:
    """
    Compare two groups using relative lift with statistical significance testing.
    
    Parameters:
    -----------
    df : polars.LazyFrame
        Input dataframe
    grouper : str
        Column name containing group assignments
    treatment_value : str
        Value identifying treatment group
    control_value : str
        Value identifying control group  
    outcome_col : str
        Column name for outcome variable
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    polars.DataFrame
        Results with lift calculations and significance tests
    """
    # Filter and aggregate by group
    treatment_data = df.filter(pl.col(grouper) == treatment_value).select(outcome_col)
    control_data = df.filter(pl.col(grouper) == control_value).select(outcome_col)
    
    # Combine data with group indicators for relative_lift function
    combined = df.filter(pl.col(grouper).is_in([treatment_value, control_value])).with_columns([
        pl.when(pl.col(grouper) == treatment_value)
        .then(pl.col(outcome_col))
        .otherwise(None)
        .alias('treatment_outcome'),
        pl.when(pl.col(grouper) == control_value)  
        .then(pl.col(outcome_col))
        .otherwise(None)
        .alias('control_outcome')
    ])
    
    # Calculate lift statistics
    result = combined.select([
        # Basic group statistics
        pl.col('treatment_outcome').drop_nulls().mean().alias('treatment_mean'),
        pl.col('control_outcome').drop_nulls().mean().alias('control_mean'),
        pl.col('treatment_outcome').drop_nulls().count().alias('treatment_n'),
        pl.col('control_outcome').drop_nulls().count().alias('control_n'),
        # Lift calculations using the outcomes
        *relative_lift('treatment_outcome', 'control_outcome', alpha)
    ]).collect()
    
    return result