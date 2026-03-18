import os
import polars as pl
import datetime as dt
from dotenv import load_dotenv
import sf_quant.data as sfd
import numpy as np
from scipy.stats import norm

def load_data() -> pl.DataFrame:
    """
    Load and prepare market data for signal creation.

    Returns:
        pl.DataFrame: Market data with required columns
    """
    # TODO: Load data from source (API, file, database)
    start = dt.date(2000, 1, 1)
    end = dt.date(2024, 12, 31)

    data = sfd.load_assets(
        start=start,
        end=end,
        columns=[
            "date",
            "barrid",
            "price",
            "return",
            "specific_risk",
            "predicted_beta",
            "daily_volume",
            "market_cap",
        ],
        in_universe=True,
        ).with_columns(pl.col("return", "specific_risk").truediv(100))

    # TODO: Filter data as needed (date range, symbols, quality checks)

    return data

def create_signal():
    """
    Loads data, creates a simple signal, and saves it to parquet.
    """
    # Load environment variables from .env file
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/bab_signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # TODO: Load Data
    df = load_data()



    # TODO: Add your signal logic here (remember alpha logic)
    print("ranking")

    df = df.filter(
        (pl.col("predicted_beta").is_not_null()) &
        (pl.col("price") >= 5)
    )

    ranked_df = (
        df.sort(["date", "predicted_beta"])
            .with_columns(
                (
                    (pl.col("predicted_beta").rank("average").over("date") - 1)
                    / (pl.col("predicted_beta").count().over("date") - 1)
                ).alias("rank_scaled")
            )
        )
    print("rank scaling")
    # We then apply the inverse CDF of the standard normal distribution to remove skewness
    transformed_df = ranked_df.with_columns(
        pl.col("rank_scaled").map_elements(lambda x: norm.ppf(np.clip(x, 1e-6, 1 - 1e-6)), return_dtype=pl.Float64).alias("bab")
    )

    print("building signal")

    # We then multiply by -1 to get the desired signal (Long low beta and short high beta)
    df = (
        transformed_df
        .sort(["barrid", "date"])
        .with_columns(
            (pl.col("bab") * -1).shift(1).over("barrid").alias("bab")
        )
    )

    df = df.filter(
        (pl.col("bab").is_not_null() &
        pl.col("specific_risk").is_not_null() &
        pl.col("return").is_not_null())
    )

    #ic
    IC = 0.05

    #compute z scores for alpha
    scores = df.select(
        "date",
        "barrid",
        "predicted_beta",
        "specific_risk",
        "bab",
        "return",
        pl.col('bab')
        .sub(pl.col('bab').mean().over("date"))
        .truediv(pl.col('bab').std().over("date"))
        .alias("score"),
    )

    scores = scores

    #compute alphas
    alphas = (
        scores.with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
        # .select("date", "barrid", "alpha", "signal", "return", "predicted_beta")
        .sort("date", "barrid")
    )

    # TODO: Save to data/bab_signal.parquet

    alphas.write_parquet(output_path)

if __name__ == "__main__":
    create_signal()
