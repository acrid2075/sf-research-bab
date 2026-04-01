import polars as pl
import numpy as np
import sf_quant.data as sfd
import sf_quant.backtester as sfb
import sf_quant.performance as sfp
import sf_quant.optimizer as sfo

import polars as pl
import os
import ray
import datetime as dt
from ray.experimental import tqdm_ray
from sf_quant.data.benchmark import load_benchmark
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.optimizers import dynamic_mve_optimizer
from sf_quant.optimizer.constraints import Constraint
from sf_quant.schema.portfolio_schema import PortfolioSchema
import matplotlib.pyplot as plt


@ray.remote
def _construct_portfolio(
    date_: dt.date,
    data: pl.DataFrame,
    constraints: list[Constraint],
    gamma: float,
    target_active_risk: float | None = None,
    benchmark_df: pl.DataFrame | None = None,
    progress_bar: tqdm_ray.tqdm | None = None,
):
    """
    Construct a single optimized portfolio for a given date.

    This is a Ray-remote helper function for parallel portfolio construction.
    It filters data to a specific date, constructs the factor model components,
    and solves a mean-variance optimization problem.

    Parameters
    ----------
    date_ : dt.date
        The date for which to construct the portfolio.
    data : pl.DataFrame
        Full dataset with columns 'date', 'barrid', 'alpha', optionally 'predicted_beta'.
    constraints : list[Constraint]
        Portfolio constraints to enforce.
    gamma : float
        Risk aversion parameter.
    target_active_risk : float, optional
        Target active risk for gamma calibration.
    benchmark_df : pl.DataFrame, optional
        Benchmark weights data with columns 'date', 'barrid', 'benchmark_weight'.
    progress_bar : tqdm_ray.tqdm, optional
        Ray-based progress bar for tracking completion.

    Returns
    -------
    pl.DataFrame
        Portfolio weights for the given date, with columns 'date', 'barrid', 'weight'.
    """
    subset = data[date_].sort("barrid")
    barrids = subset["barrid"].to_list()
    alphas = subset["alpha"].to_numpy()

    betas = (
        subset["predicted_beta"].to_numpy()
        if "predicted_beta" in subset.columns
        else None
    )

    benchmark_weights = None
    if benchmark_df is not None:
        # Get benchmark weights for portfolio barrids on this date
        bmk_subset = benchmark_df.filter(pl.col("date").eq(date_))
        # Join with portfolio to ensure alignment
        aligned = (
            pl.DataFrame({"barrid": barrids})
            .join(bmk_subset[["barrid", "benchmark_weight"]], on="barrid", how="left")
            .fill_null(0.0)
        )
        benchmark_weights = aligned["benchmark_weight"].to_numpy()

    (
        factor_exposures,
        factor_covariance,
        specific_risk,
    ) = construct_factor_model_components(date_, barrids)

    portfolio = dynamic_mve_optimizer(
        ids=barrids,
        alphas=alphas,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        constraints=constraints,
        initial_gamma=gamma,
        betas=betas,
        target_active_risk=target_active_risk,
        benchmark_weights=benchmark_weights,
    )

    portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
        "date", "barrid", "weight", 'gamma', 'active_risk'
    )

    if progress_bar is not None:
        progress_bar.update.remote(1)

    return portfolio

def dynamic_backtest_parallel(
    data: pl.DataFrame,
    constraints: list[Constraint],
    initial_gamma: float = 100,
    target_active_risk: float = 0.05,
    n_cpus: int | None = None,
) -> pl.DataFrame:
    """
    Run a parallelized backtest of portfolio optimization using Ray.

    This function distributes portfolio construction tasks across multiple CPUs
    using Ray, solving mean–variance optimization problems for each date in parallel.
    A Ray-based progress bar tracks computation progress.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataset containing at least the following columns:

        - ``date`` : datetime-like, the date of each observation.
        - ``barrid`` : str, unique identifier for each asset.
        - ``alpha`` : float, expected return (alpha) for each asset.
        - ``predicted_beta`` : float, optional, factor exposures for constraints.

    constraints : list[Constraint]
        List of portfolio constraints to enforce during optimization.
    initial_gamma : float, optional
        Risk aversion parameter. Higher values penalize portfolio variance
        more strongly. Used as the starting gamma for calibration if
        ``target_active_risk`` is specified. Default is 100.
    target_active_risk : float, optional
        If specified, automatically calibrate gamma for each date to achieve this
        target annualized active risk (e.g., 0.05 for 5%). Requires benchmark data
        to be available via :func:`~sf_quant.data.benchmark.load_benchmark`.
    n_cpus : int, optional
        Number of CPUs to allocate to Ray. If ``None``, defaults to
        ``os.cpu_count()`` but is capped at the number of unique dates.

    Returns
    -------
    pl.DataFrame
        A PortfolioSchema-validated Polars DataFrame containing optimized
        portfolio weights across all backtest dates, with columns:

        - ``date`` : datetime, portfolio date.
        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.
        - ``gamma`` : float, calibrated risk aversion parameter for each date.
        - ``active_risk`` : float, achieved annualized active risk for each date.

    Notes
    -----
    - Benchmark data is automatically loaded when ``target_active_risk`` is specified,
      via :func:`~sf_quant.data.benchmark.load_benchmark`.
    - Ray is initialized with ``ignore_reinit_error=True``, allowing safe
      re-invocation within the same process.

    See Also
    --------
    backtest_parallel : Sequential version without active risk calibration.
    dynamic_mve_optimizer : Underlying optimizer with active risk calibration.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import sf_quant.backtester as sfb
    >>> import sf_quant.optimizer as sfo
    >>> import polars as pl
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 1, 10)
    >>> columns = ['date', 'barrid']
    >>> data = (
    ...     sfd.load_assets(
    ...         start=start,
    ...         end=end,
    ...         in_universe=True,
    ...         columns=columns
    ...     )
    ...     .with_columns(
    ...         pl.lit(0).alias('alpha')
    ...     )
    ... )
    >>> constraints = [sfo.FullInvestment()]
    >>> weights = sfb.dynamic_backtest_parallel(
    ...     data=data,
    ...     constraints=constraints,
    ...     initial_gamma=100,
    ... )
    shape: (5, 5)
    ┌────────────┬─────────┬───────────┬───────┬──────────────┐
    │ date       ┆ barrid  ┆ weight    ┆ gamma ┆ active_risk  │
    │ ---        ┆ ---     ┆ ---       ┆ ---   ┆ ---          │
    │ date       ┆ str     ┆ f64       ┆ f64   ┆ f64          │
    ╞════════════╪═════════╪═══════════╪═══════╪══════════════╡
    │ 2024-01-02 ┆ USA06Z1 ┆ -0.000639 ┆ 100.0 ┆ 0.047        │
    │ 2024-01-02 ┆ USA0771 ┆ -0.000083 ┆ 100.0 ┆ 0.047        │
    │ 2024-01-02 ┆ USA0C11 ┆ -0.003044 ┆ 100.0 ┆ 0.047        │
    │ 2024-01-02 ┆ USA0SY1 ┆ -0.002177 ┆ 100.0 ┆ 0.047        │
    │ 2024-01-02 ┆ USA11I1 ┆ 0.001475  ┆ 100.0 ┆ 0.047        │
    └────────────┴─────────┴───────────┴───────┴──────────────┘
    """
    # Get dates
    # 1. Split the data by date into a dictionary
    dates = data["date"].unique().sort().to_list()
    raw_dict = data.partition_by("date", as_dict=True)

    # 2. Extract the date from the tuple key and build the lookup
    data_by_date = {
        date_tuple[0]: df for date_tuple, df in raw_dict.items()
    }

    start_date = dates[0]
    end_date = dates[-1]
    benchmark_df = load_benchmark(start_date, end_date).rename({"weight": "benchmark_weight"})

    # Set up ray
    # Check if we are in a Slurm allocation
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    n_cpus = n_cpus or (int(slurm_cpus) if slurm_cpus else os.cpu_count())
    n_cpus = min(len(dates), n_cpus)
    # Empty runtime_env = no environment management. Workers use parent's Python directly.
    # This prevents Ray from trying to serialize code or rebuild packages on air-gapped clusters.
    ray.init(ignore_reinit_error=True, num_cpus=n_cpus, runtime_env={})

    # Set up ray progress bar
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    progress_bar = remote_tqdm.remote(
        total=len(dates), desc=f"Computing portfolios with {n_cpus} cpus"
    )

    # 1. Put the data into shared memory ONCE
    # data_ref = ray.put(data)
    data_dict_ref = ray.put(data_by_date)
    benchmark_ref = ray.put(benchmark_df)

    # 2. Pass the reference ID, not the raw object
    portfolio_futures = [
        _construct_portfolio.remote(
            date_=date_,
            data=data_dict_ref,  # Pass the ref here!
            constraints=constraints,
            gamma=initial_gamma,
            target_active_risk=target_active_risk,
            benchmark_df=benchmark_ref,  # Pass the ref here too!
            progress_bar=progress_bar,
        )
        for date_ in dates
    ]

    portfolio_list = ray.get(portfolio_futures)

    progress_bar.close.remote()
    ray.shutdown()

    return pl.concat(portfolio_list)


if __name__ == "__main__":
    signal_df = pl.read_parquet('zero_data/bab_signal.parquet')

    # print("Starting zero_constraints")
    # zero_constraints = [sfo.ZeroBeta()]
    # zero_weights = dynamic_backtest_parallel(signal_df, zero_constraints)
    # zero_weights.write_parquet("weights/zero_beta.parquet")

    # print("Starting unit_constraints")
    # unit_constraints = [sfo.UnitBeta()]
    # unit_weights = dynamic_backtest_parallel(signal_df, unit_constraints)
    # unit_weights.write_parquet("weights/unit_beta.parquet")

    # print("Starting full_constraints")
    # full_constraints = [sfo.UnitBeta(), sfo.LongOnly(), sfo.FullInvestment()]
    # full_weights = dynamic_backtest_parallel(signal_df, full_constraints)
    # full_weights.write_parquet("weights/full_beta.parquet")

    print("Getting portfolio weights and returns")
    zero_weights = pl.read_parquet("weights/zero_beta.parquet")
    zero_returns = sfp.generate_returns_from_weights(zero_weights)
    zero_df = (
    zero_weights.rename({"weight": "zero_weight", "gamma": "zero_gamma"})
        .with_columns(
            pl.col("zero_weight")
            #.truediv(pl.col("active_risk").mean().over("date")).mul(0.05)
            )
        )
    
    unit_weights = pl.read_parquet("weights/unit_beta.parquet")
    unit_returns = sfp.generate_returns_from_weights(unit_weights)
    unit_df = (unit_weights.rename({"weight": "unit_weight", "gamma": "unit_gamma"})
                    # .with_columns(pl.col("unit_weight").truediv(pl.col("active_risk").mean().over("date")).mul(.05))
                    )


    full_weights = pl.read_parquet("weights/full_beta.parquet")
    full_returns = sfp.generate_returns_from_weights(full_weights)
    full_df = (full_weights.rename({"weight": "full_weight", "gamma": "full_gamma"})
                    #.with_columns(pl.col("full_weight").truediv(pl.col("active_risk").mean()).mul(.05))
                    )
    
    _min = zero_df.select("date").min().item()
    _max = zero_df.select("date").max().item()
    bmk_weights = sfd.load_benchmark(_min, _max).rename({"weight": "bmk_weight"})

    print("Joining weights")
    weights_df = (
        signal_df.join(bmk_weights, on=["date", "barrid"])
        .join(zero_df.select([pl.col("date"), pl.col("barrid"), pl.col("zero_weight")]), on=["date", "barrid"])
        .join(unit_df.select([pl.col("date"), pl.col("barrid"), pl.col("unit_weight")]), on=["date", "barrid"])
        .join(full_df.select([pl.col("date"), pl.col("barrid"), pl.col("full_weight")]), on=["date", "barrid"])
        )
    
    print(weights_df.head())
    
    
    bmk_returns = sfd.load_benchmark_returns(_min, _max).with_columns(pl.col("bmk_return").truediv(100))
    print("Joining returns")
    returns = (
        bmk_returns.join(zero_returns.rename({"return": "zero_return"}), on="date")
        .join(unit_returns.rename({"return": "unit_return"}), on="date")
        .join(full_returns.rename({"return": "full_return"}), on="date")
    )

    print(returns.head())

    print("Performing regressions")
    cov_xy = pl.cov("unit_return", "bmk_return")
    var_x = pl.col("bmk_return").var()

    beta_expr = cov_xy / var_x

    errors = (
        returns.with_columns(beta=beta_expr)
        .with_columns(e=pl.col("unit_return") - (pl.col("bmk_return") * pl.col("beta")))
    )

    print(errors.head(5))

    # Correlation of residuals with zero-beta returns
    print(errors.select(pl.corr("e", "zero_return")))



    cov_xy = pl.cov("full_return", "bmk_return")
    var_x = pl.col("bmk_return").var()

    beta_expr = cov_xy / var_x

    errors = (
        returns.with_columns(beta=beta_expr)
        .with_columns(e=pl.col("full_return") - (pl.col("bmk_return") * pl.col("beta")))
    )

    print(errors.head(5))

    # Correlation of residuals with zero-beta returns
    print(errors.select(pl.corr("e", "zero_return")))


    cum_returns = errors.with_columns([
        (pl.col("bmk_return") + 1).cum_prod().alias("cum_bmk"),
        (pl.col("zero_return") + 1).cum_prod().alias("cum_zero"),
        (pl.col("unit_return") + 1).cum_prod().alias("cum_unit"),
        (pl.col("full_return") + 1).cum_prod().alias("cum_full")
    ])

    
    dates = cum_returns["date"].to_list()

    # plt.clf()
    # plt.plot(dates, cum_returns["cum_bmk"].to_list(), label='Benchmark')
    # plt.plot(dates, cum_returns["cum_zero"].to_list(), label='Zero Beta')
    # plt.plot(dates, cum_returns["cum_unit"].to_list(), label='Unit Beta')
    # plt.plot(dates, cum_returns["cum_full"].to_list(), label='Full Constraints')
    # plt.yscale('log')
    # plt.xlabel('Date')
    # plt.ylabel('Cumulative Return (Log Scale)')
    # plt.legend()
    # plt.title("Constrained BAB")
    # plt.tight_layout()
    # plt.savefig('constrained_beta.png')
    # plt.clf()

    # plt.clf()
    # plt.plot(dates, cum_returns["cum_zero"].to_list(), label='Zero Beta')
    # plt.yscale('log')
    # plt.xlabel('Date')
    # plt.ylabel('Cumulative Return (Log Scale)')
    # plt.legend()
    # plt.title("Zero Beta BAB")
    # plt.tight_layout()
    # plt.savefig('zero_beta.png')
    # plt.clf()





    last_df = (weights_df
               .with_columns(pl.col("full_weight")
                             .sub(pl.col("bmk_weight"))
                             .alias("effective_active_weight")
                             )
                .with_columns(
                    pl.corr("zero_weight", "effective_active_weight")
                    .over("date")
                    .alias("transfer_coeff")
                )
    )
    
    plt.clf()
    plt.plot(last_df["date"].to_list(), last_df["transfer_coeff"].to_list())
    plt.title("Transfer Coefficient, BAB")
    plt.xlabel("Date")
    plt.ylabel("TC")
    plt.show()
    plt.clf()

                             


                
