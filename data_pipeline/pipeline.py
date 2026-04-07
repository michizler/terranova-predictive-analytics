"""
Production-ready pandas pipeline for FEMA data cleaning and feature engineering.

Outputs:
    model_data.csv

Example usage:
    python pipeline.py \
        --declarations data/declarations.csv \
        --dis_summ data/disaster_summaries.csv \
        --pub_assistance data/public_assistance.csv \
        --output model_data.csv

Or with a single directory:
    python pipeline.py --input-dir ../fema_data --output model_data.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DROP_COLS = [
    # ids / raw dates - little predictive value and potential for leakage
    "disasterNumber",
    "declarationDate",
    "incidentBeginDate",
    "incidentEndDate",
    
    # targets - possible prediction targets. They'll be handled seperately in model training pipeline.
    # Only here for reference.
    
    # "target_total_recovery_cost",
    # "target_pa_cost",
    # "log_target_pa_cost",
    # "log_target_total_cost",
    
    # leakage: cost components
    "totalAmountIhpApproved",
    "totalAmountHaApproved",
    "totalAmountOnaApproved",
    "totalObligatedAmountPa",
    "totalObligatedAmountHmgp",
    # leakage: missing flags for target components
    "totalAmountIhpApproved_missing",
    "totalAmountHaApproved_missing",
    "totalAmountOnaApproved_missing",
    "totalObligatedAmountPa_missing",
    "totalObligatedAmountHmgp_missing",
    # leakage: post-event PA aggregates
    "pa_total_obligated_sum",
    "pa_total_obligated_mean",
    "pa_total_obligated_median",
    "pa_total_obligated_max",
    # project scope is prediction of recovery cost before declaration
    "pa_project_count",
    "large",
    "small",
    "large_project_ratio",
    # possible time leakage
    "incident_duration_days",
]

DECLARATION_DATE_COLS = [
    "declarationDate",
    "incidentBeginDate",
    "incidentEndDate",
]

SUMMARY_NUMERIC_COLS = [
    "totalAmountIhpApproved",
    "totalAmountHaApproved",
    "totalAmountOnaApproved",
    "totalObligatedAmountPa",
    "totalObligatedAmountHmgp",
]

PA_NUMERIC_COLS = [
    "totalObligated",
]

SUMMARY_COLS_FOR_MISSING_FLAGS = [
    "totalAmountIhpApproved",
    "totalAmountHaApproved",
    "totalAmountOnaApproved",
    "totalObligatedAmountPa",
    "totalObligatedAmountHmgp",
]

PA_FEATURE_COLS = [
    "pa_project_count",
    "pa_total_obligated_sum",
    "pa_total_obligated_mean",
    "pa_total_obligated_median",
    "pa_total_obligated_max",
    "large",
    "small",
    "large_project_ratio",
]


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build production-ready model_data.csv from FEMA source files."
    )
    parser.add_argument("--declarations", type=str, help="Path to declarations.csv")
    parser.add_argument("--dis_summ", type=str, help="Path to disaster_summaries.csv")
    parser.add_argument(
        "--pub_assistance", type=str, help="Path to public_assistance.csv"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing declarations.csv, disaster_summaries.csv, and public_assistance.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_data.csv",
        help="Output CSV path. Default: model_data.csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def resolve_input_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    if args.input_dir:
        input_dir = Path(args.input_dir)
        declarations_path = input_dir / "declarations.csv"
        dis_summ_path = input_dir / "disaster_summaries.csv"
        pub_assistance_path = input_dir / "public_assistance.csv"
    else:
        missing = [
            name
            for name, value in {
                "--declarations": args.declarations,
                "--dis_summ": args.dis_summ,
                "--pub_assistance": args.pub_assistance,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Provide either --input-dir or all three explicit file paths: "
                + ", ".join(missing)
            )
        declarations_path = Path(args.declarations)
        dis_summ_path = Path(args.dis_summ)
        pub_assistance_path = Path(args.pub_assistance)

    for path in [declarations_path, dis_summ_path, pub_assistance_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    return declarations_path, dis_summ_path, pub_assistance_path


def load_data(
    declarations_path: Path,
    dis_summ_path: Path,
    pub_assistance_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Loading source datasets...")
    declarations = pd.read_csv(declarations_path)
    dis_summ = pd.read_csv(dis_summ_path)
    pub_assistance = pd.read_csv(pub_assistance_path)

    LOGGER.info("Declarations shape: %s", declarations.shape)
    LOGGER.info("Disaster summaries shape: %s", dis_summ.shape)
    LOGGER.info("Public assistance shape: %s", pub_assistance.shape)

    return declarations, dis_summ, pub_assistance


def preprocess_declarations(declarations: pd.DataFrame) -> pd.DataFrame:
    df = declarations.copy()

    for col in DECLARATION_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["declaration_year"] = df["declarationDate"].dt.year
    df["declaration_month"] = df["declarationDate"].dt.month
    df["declaration_quarter"] = df["declarationDate"].dt.quarter

    df["incident_duration_days"] = (
        df["incidentEndDate"] - df["incidentBeginDate"]
    ).dt.days

    df["declaration_lag_days"] = (
        df["declarationDate"] - df["incidentBeginDate"]
    ).dt.days

    df["incident_open_flag"] = df["incidentEndDate"].isna().astype(int)

    state_freq = (
        df.groupby("state")["disasterNumber"]
        .count()
        .rename("state_disaster_frequency")
        .reset_index()
    )

    df = df.merge(state_freq, on="state", how="left")
    return df


def preprocess_disaster_summaries(dis_summ: pd.DataFrame) -> pd.DataFrame:
    df = dis_summ.copy()

    for col in SUMMARY_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target_pa_cost"] = df["totalObligatedAmountPa"].fillna(0)

    df["target_total_recovery_cost"] = (
        df["totalAmountIhpApproved"].fillna(0)
        + df["totalObligatedAmountPa"].fillna(0)
        + df["totalObligatedAmountHmgp"].fillna(0)
    )

    df["log_target_pa_cost"] = np.log1p(df["target_pa_cost"])
    df["log_target_total_cost"] = np.log1p(df["target_total_recovery_cost"])

    return df


def preprocess_public_assistance(
    pub_assistance: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pub_assistance.copy()

    for col in PA_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pa_agg = (
        df.groupby("disasterNumber")
        .agg(
            pa_project_count=("pwNumber", "count"),
            pa_total_obligated_sum=("totalObligated", "sum"),
            pa_total_obligated_mean=("totalObligated", "mean"),
            pa_total_obligated_median=("totalObligated", "median"),
            pa_total_obligated_max=("totalObligated", "max"),
        )
        .reset_index()
    )

    size_counts = df.pivot_table(
        index="disasterNumber",
        columns="projectSize",
        values="pwNumber",
        aggfunc="count",
        fill_value=0,
    ).reset_index()

    size_counts.columns = [str(col).lower() for col in size_counts.columns]
    size_counts = size_counts.rename(columns={"disasternumber": "disasterNumber"})

    if "large" not in size_counts.columns:
        size_counts["large"] = 0
    if "small" not in size_counts.columns:
        size_counts["small"] = 0

    denominator = size_counts["small"] + size_counts["large"]
    size_counts["large_project_ratio"] = np.where(
        denominator > 0,
        size_counts["large"] / denominator,
        0,
    )

    return pa_agg, size_counts


def build_feature_table(
    declarations: pd.DataFrame,
    dis_summ: pd.DataFrame,
    pub_assistance: pd.DataFrame,
) -> pd.DataFrame:
    decl = preprocess_declarations(declarations)
    summ = preprocess_disaster_summaries(dis_summ)
    pa_agg, size_counts = preprocess_public_assistance(pub_assistance)

    model_df = (
        decl.merge(summ, on="disasterNumber", how="left")
        .merge(pa_agg, on="disasterNumber", how="left")
        .merge(size_counts, on="disasterNumber", how="left")
    )

    for col in SUMMARY_COLS_FOR_MISSING_FLAGS:
        model_df[f"{col}_missing"] = model_df[col].isna().astype(int)

    existing_summary_fill_cols = [
        c for c in SUMMARY_COLS_FOR_MISSING_FLAGS if c in model_df.columns
    ]
    existing_pa_fill_cols = [c for c in PA_FEATURE_COLS if c in model_df.columns]

    model_df[existing_summary_fill_cols] = model_df[existing_summary_fill_cols].fillna(
        0
    )
    model_df[existing_pa_fill_cols] = model_df[existing_pa_fill_cols].fillna(0)

    model_df = model_df[
        model_df["target_total_recovery_cost"].notna()
        & model_df["target_pa_cost"].notna()
    ].copy()

    return model_df


def drop_pretraining_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing_drop_cols = [col for col in DROP_COLS if col in df.columns]
    LOGGER.info(
        "Dropping %d specified columns before model training.", len(existing_drop_cols)
    )
    return df.drop(columns=existing_drop_cols, errors="ignore")


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    final_df = df.copy().reset_index(drop=True)

    LOGGER.info("Final dataset shape: %s", final_df.shape)
    missing_ratio = final_df.isna().mean().sort_values(ascending=False).head(10)
    LOGGER.info("Top remaining missingness:\n%s", missing_ratio.to_string())

    return final_df


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved output to %s", output_path)


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    declarations_path, dis_summ_path, pub_assistance_path = resolve_input_paths(args)

    declarations, dis_summ, pub_assistance = load_data(
        declarations_path=declarations_path,
        dis_summ_path=dis_summ_path,
        pub_assistance_path=pub_assistance_path,
    )

    model_df = build_feature_table(
        declarations=declarations,
        dis_summ=dis_summ,
        pub_assistance=pub_assistance,
    )
    model_df = drop_pretraining_columns(model_df)
    model_df = finalize_dataset(model_df)

    save_output(model_df, Path(args.output))


if __name__ == "__main__":
    main()
