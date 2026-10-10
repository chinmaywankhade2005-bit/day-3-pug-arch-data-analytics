"""Clean and analyze facility inspection data, then save tables and charts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_PATH = ROOT / "data" / "raw" / "facility_inspections.csv"
OUTPUT_DIR = ROOT / "outputs"
NUMERIC_COLUMNS = ["cleanliness_score", "odor_score", "waste_level", "footfall", "complaints"]
SCORE_COLUMNS = ["cleanliness_score", "odor_score", "waste_level"]
EXPECTED_COLUMNS = [
    "facility_id", "location", "cleanliness_score", "odor_score", "waste_level",
    "water_availability", "footfall", "complaints", "inspection_date",
]


def numpy_demonstration() -> dict[str, Any]:
    """Exercise array creation, dimensions, shape, indexing, slicing and reductions."""
    readings = np.array([[82, 17, 13], [74, 28, 21], [91, 12, 9]], dtype=float)
    return {
        "array": readings.tolist(),
        "dimensions": int(readings.ndim),
        "shape": list(readings.shape),
        "first_facility": readings[0].tolist(),
        "cleanliness_column_slice": readings[:, 0].tolist(),
        "plus_five_math_operation": (readings + 5).tolist(),
        "column_aggregations": {
            "mean": np.mean(readings, axis=0).round(2).tolist(),
            "sum": np.sum(readings, axis=0).tolist(),
            "minimum": np.min(readings, axis=0).tolist(),
        },
    }


def pandas_demonstration(cleaned: pd.DataFrame) -> dict[str, Any]:
    """Exercise Series/DataFrame, filtering, sorting, grouping and aggregation."""
    series = cleaned["cleanliness_score"]
    filtered = cleaned.loc[cleaned["cleanliness_score"] < 70, ["facility_id", "location", "cleanliness_score"]]
    sorted_rows = cleaned.sort_values("complaints", ascending=False).head(3)
    grouped = cleaned.groupby("location", dropna=False).agg(
        facility_records=("facility_id", "size"),
        mean_cleanliness=("cleanliness_score", "mean"),
        total_complaints=("complaints", "sum"),
    )
    return {
        "series_name": str(series.name),
        "series_length": int(series.size),
        "dataframe_shape": list(cleaned.shape),
        "filtered_low_cleanliness_count": int(len(filtered)),
        "top_complaint_rows": sorted_rows[["facility_id", "complaints"]].to_dict(orient="records"),
        "grouped_by_location": grouped.round(2).reset_index().to_dict(orient="records"),
        "csv_read_method": "pandas.read_csv",
        "data_transformation": "scores/counts parsed to numeric; dates parsed; location and water labels normalized; numeric outliers capped to IQR fences",
    }


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return cleaned records, an issue log, and measured cleaning counts."""
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if raw.empty:
        raise ValueError("Input dataset is empty; no facility records can be analyzed.")

    frame = raw[EXPECTED_COLUMNS].copy()
    initial_count = int(len(frame))
    issue_records: list[dict[str, Any]] = []
    duplicate_mask = frame.duplicated(keep="first")
    exact_duplicates = int(duplicate_mask.sum())
    for index in frame.index[duplicate_mask]:
        issue_records.append({"source_row": int(index) + 2, "facility_id": str(frame.at[index, "facility_id"]), "issue": "exact_duplicate", "field": "*", "original_value": "identical row", "decision": "Removed duplicate; retained first occurrence."})
    frame = frame.drop_duplicates(keep="first").copy()
    frame["facility_id"] = frame["facility_id"].astype("string").str.strip()
    frame["location"] = frame["location"].astype("string").str.strip().replace("", pd.NA)
    frame["water_availability"] = (
        frame["water_availability"].astype("string").str.strip().str.title()
    )
    frame.loc[~frame["water_availability"].isin(["Available", "Unavailable"]), "water_availability"] = pd.NA

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_scores = 0
    for column in SCORE_COLUMNS:
        invalid = frame[column].notna() & ~frame[column].between(0, 100)
        invalid_scores += int(invalid.sum())
        for index in frame.index[invalid]:
            issue_records.append({"source_row": int(index) + 2, "facility_id": str(frame.at[index, "facility_id"]), "issue": "invalid_score", "field": column, "original_value": str(frame.at[index, column]), "decision": "Set to missing and imputed by location median, then overall median."})
        frame.loc[invalid, column] = np.nan
    invalid_counts = 0
    for column in ["footfall", "complaints"]:
        invalid = frame[column].notna() & (frame[column] < 0)
        invalid_counts += int(invalid.sum())
        for index in frame.index[invalid]:
            issue_records.append({"source_row": int(index) + 2, "facility_id": str(frame.at[index, "facility_id"]), "issue": "invalid_negative_count", "field": column, "original_value": str(frame.at[index, column]), "decision": "Set to missing and imputed by location median, then overall median."})
        frame.loc[invalid, column] = np.nan

    for column in NUMERIC_COLUMNS:
        for index in frame.index[frame[column].isna()]:
            issue_records.append({"source_row": int(index) + 2, "facility_id": str(frame.at[index, "facility_id"]), "issue": "missing_numeric", "field": column, "original_value": "missing or invalid", "decision": "Imputed by location median, then overall median."})

    invalid_water = ~frame["water_availability"].isin(["Available", "Unavailable"])
    for index in frame.index[invalid_water]:
        issue_records.append({"source_row": int(index) + 2, "facility_id": str(frame.at[index, "facility_id"]), "issue": "missing_or_unrecognized_water", "field": "water_availability", "original_value": str(frame.at[index, "water_availability"]), "decision": "Replaced with Unknown."})

    frame["inspection_date"] = pd.to_datetime(frame["inspection_date"], errors="coerce")
    missing_id = frame["facility_id"].isna() | frame["facility_id"].eq("")
    invalid_date = frame["inspection_date"].isna()
    unrepairable = missing_id | invalid_date
    dropped_unrepairable = int(unrepairable.sum())
    for index in frame.index[unrepairable]:
        issue = "missing_facility_id" if missing_id.at[index] else "invalid_inspection_date"
        field = "facility_id" if missing_id.at[index] else "inspection_date"
        issue_records.append({"source_row": int(index) + 2, "facility_id": str(frame.at[index, "facility_id"]), "issue": issue, "field": field, "original_value": str(raw.at[index, field]), "decision": "Dropped row because it cannot be reliably identified or placed in time."})
    frame = frame.loc[~unrepairable].copy()
    if frame.empty:
        raise ValueError("No valid records remain after removing rows without a facility ID or valid inspection date.")

    missing_numeric_before = {column: int(frame[column].isna().sum()) for column in NUMERIC_COLUMNS}
    for column in NUMERIC_COLUMNS:
        location_medians = frame.groupby("location", dropna=False)[column].transform("median")
        frame[column] = frame[column].fillna(location_medians).fillna(frame[column].median())
        if frame[column].isna().any():
            frame[column] = frame[column].fillna(0)
    frame["location"] = frame["location"].fillna("Unknown")
    frame["water_availability"] = frame["water_availability"].fillna("Unknown")

    outlier_flags: dict[str, int] = {}
    outlier_union = pd.Series(False, index=frame.index)
    iqr_bounds: dict[str, list[float]] = {}
    for column in NUMERIC_COLUMNS:
        q1 = float(frame[column].quantile(0.25))
        q3 = float(frame[column].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = (frame[column] < lower) | (frame[column] > upper)
        outlier_flags[column] = int(flagged.sum())
        outlier_union |= flagged
        for index in frame.index[flagged]:
            issue_records.append({"source_row": int(index) + 2, "facility_id": str(frame.at[index, "facility_id"]), "issue": "iqr_outlier", "field": column, "original_value": str(frame.at[index, column]), "decision": f"Capped to IQR fence [{lower:.3f}, {upper:.3f}] in cleaned copy."})
        iqr_bounds[column] = [round(lower, 3), round(upper, 3)]
        frame[column] = frame[column].clip(lower=lower, upper=upper)

    # Use integer types for count columns after imputation and outlier capping.
    for column in ["footfall", "complaints"]:
        frame[column] = frame[column].round().astype("int64")
    frame["inspection_date"] = frame["inspection_date"].dt.strftime("%Y-%m-%d")
    frame = frame[EXPECTED_COLUMNS].sort_values(["inspection_date", "facility_id"]).reset_index(drop=True)

    log_rows = [
        {"issue": "exact_duplicate_rows", "count": exact_duplicates, "decision": "Keep the first exact row and remove later identical copies."},
        {"issue": "invalid_score_values", "count": invalid_scores, "decision": "Treat scores outside 0-100 as missing; impute by location median, then overall median."},
        {"issue": "invalid_negative_counts", "count": invalid_counts, "decision": "Treat negative footfall/complaints as missing; impute by location median, then overall median."},
        {"issue": "missing_or_unrecognized_water_labels", "count": int(raw["water_availability"].astype("string").str.strip().str.title().isin(["Available", "Unavailable"]).eq(False).sum()), "decision": "Normalize valid labels; replace missing or unknown labels with Unknown."},
        {"issue": "invalid_or_missing_inspection_date_rows", "count": dropped_unrepairable, "decision": "Drop rows without a parseable inspection date; the observation cannot be placed in time."},
        {"issue": "numeric_outlier_rows", "count": int(outlier_union.sum()), "decision": "Flag values beyond 1.5x IQR fences and cap each numeric value to its column fence; preserve raw source unchanged."},
    ]
    details = {
        "input_rows": initial_count,
        "exact_duplicate_rows_removed": exact_duplicates,
        "rows_dropped_for_missing_id_or_invalid_date": dropped_unrepairable,
        "output_rows": int(len(frame)),
        "invalid_score_values": invalid_scores,
        "invalid_negative_count_values": invalid_counts,
        "missing_numeric_values_imputed": missing_numeric_before,
        "outlier_values_by_column": outlier_flags,
        "rows_with_at_least_one_outlier": int(outlier_union.sum()),
        "iqr_bounds_by_column": iqr_bounds,
        "quality_issue_count": len(issue_records),
        "cleaning_decisions": [row["decision"] for row in log_rows],
    }
    return frame, pd.DataFrame(log_rows), details, pd.DataFrame(issue_records, columns=["source_row", "facility_id", "issue", "field", "original_value", "decision"])


def calculate_analysis(cleaned: pd.DataFrame, cleaning: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate descriptive statistics, location summaries, and data-backed insights."""
    numeric_summary = cleaned[NUMERIC_COLUMNS].describe().round(2).to_dict()
    by_location = cleaned.groupby("location").agg(
        records=("facility_id", "size"),
        mean_cleanliness=("cleanliness_score", "mean"),
        mean_odor=("odor_score", "mean"),
        mean_waste=("waste_level", "mean"),
        mean_footfall=("footfall", "mean"),
        total_complaints=("complaints", "sum"),
        mean_complaints=("complaints", "mean"),
        unavailable_water_records=("water_availability", lambda s: int(s.eq("Unavailable").sum())),
    ).sort_values("mean_cleanliness", ascending=False)
    water_summary = cleaned.groupby("water_availability").agg(
        records=("facility_id", "size"),
        mean_cleanliness=("cleanliness_score", "mean"),
        mean_complaints=("complaints", "mean"),
    ).round(2)
    correlation = cleaned[["cleanliness_score", "odor_score", "waste_level", "footfall", "complaints"]].corr().round(3)
    best = by_location["mean_cleanliness"].idxmax()
    worst = by_location["mean_cleanliness"].idxmin()
    insights = [
        f"{best} has the highest mean cleanliness score ({by_location.loc[best, 'mean_cleanliness']:.2f}); {worst} has the lowest ({by_location.loc[worst, 'mean_cleanliness']:.2f}).",
        f"Across cleaned records, cleanliness and complaints have a Pearson correlation of {correlation.loc['cleanliness_score', 'complaints']:.3f}.",
    ]
    if "Unavailable" in water_summary.index and "Available" in water_summary.index:
        diff = float(water_summary.loc["Unavailable", "mean_complaints"] - water_summary.loc["Available", "mean_complaints"])
        direction = "higher" if diff > 0 else "lower"
        insights.append(f"Records marked Unavailable average {abs(diff):.2f} {direction} complaints than records marked Available (mean difference {diff:+.2f}).")
    else:
        insights.append("Water availability categories do not both occur in the cleaned data, so a two-group comparison is unavailable.")
    summary = {
        "dataset_note": "Synthetic sample data; not operational facility observations.",
        "cleaned_record_count": int(len(cleaned)),
        "location_count": int(cleaned["location"].nunique()),
        "cleaning": cleaning,
        "numeric_descriptive_statistics": numeric_summary,
        "pearson_correlation": correlation.to_dict(),
        "water_availability_summary": water_summary.reset_index().to_dict(orient="records"),
        "insights": insights,
        "learning_demonstrations": {
            "numpy": numpy_demonstration(),
            "pandas": pandas_demonstration(cleaned),
        },
    }
    return summary, by_location.round(2).reset_index()


def create_visualizations(cleaned: pd.DataFrame, by_location: pd.DataFrame, visual_dir: Path) -> list[Path]:
    visual_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    charts = [
        ("bar_mean_cleanliness_by_location.png", "Mean cleanliness score by location", "Mean cleanliness score", "mean_cleanliness"),
        ("bar_total_complaints_by_location.png", "Total complaints by location", "Total complaints", "total_complaints"),
    ]
    for filename, title, ylabel, column in charts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(by_location["location"], by_location[column], color="#277DA1")
        ax.set(title=title, xlabel="Location", ylabel=ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = visual_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cleaned["cleanliness_score"], bins=10, color="#43AA8B", edgecolor="white")
    ax.set(title="Distribution of cleanliness scores", xlabel="Cleanliness score", ylabel="Record count")
    fig.tight_layout()
    path = visual_dir / "histogram_cleanliness_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(cleaned["odor_score"], cleaned["complaints"], c="#F9844A", alpha=0.8)
    ax.set(title="Odor score and complaints", xlabel="Odor score (higher means more severe)", ylabel="Complaints")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = visual_dir / "scatter_odor_vs_complaints.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    availability = pd.crosstab(cleaned["location"], cleaned["water_availability"])
    fig, ax = plt.subplots(figsize=(8, 5))
    availability.plot(kind="bar", stacked=True, ax=ax, color=["#F9C74F", "#577590", "#BDBDBD"][: len(availability.columns)])
    ax.set(title="Water availability records by location", xlabel="Location", ylabel="Record count")
    ax.legend(title="Water availability")
    fig.tight_layout()
    path = visual_dir / "additional_water_availability_by_location.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def run_analysis(raw_path: Path = RAW_PATH, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    raw = pd.read_csv(raw_path)
    cleaned, cleaning_log, cleaning_details, quality_issues = clean_data(raw)
    summary, location_summary = calculate_analysis(cleaned, cleaning_details)
    visual_dir = output_dir / "visualizations"
    outputs = [
        output_dir / "facility_inspections_cleaned.csv",
        output_dir / "cleaning_log.csv",
        output_dir / "quality_issues.csv",
        output_dir / "analysis_summary.json",
        output_dir / "statistics_by_location.csv",
        *(visual_dir / name for name in [
            "bar_mean_cleanliness_by_location.png", "bar_total_complaints_by_location.png",
            "histogram_cleanliness_scores.png", "scatter_odor_vs_complaints.png",
            "additional_water_availability_by_location.png",
        ]),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing output(s): " + ", ".join(str(p) for p in existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(outputs[0], index=False)
    cleaning_log.to_csv(outputs[1], index=False)
    quality_issues.to_csv(outputs[2], index=False)
    outputs[3].write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    location_summary.to_csv(outputs[4], index=False)
    chart_paths = create_visualizations(cleaned, location_summary, visual_dir)
    return {"summary": summary, "outputs": [str(p) for p in outputs], "charts": [str(p) for p in chart_paths]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_PATH, help="Input facility CSV (default: bundled sample)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory (must not contain generated outputs)")
    args = parser.parse_args()
    result = run_analysis(args.input, args.output)
    print(f"Analyzed {result['summary']['cleaned_record_count']} cleaned facility records.")
    for insight in result["summary"]["insights"]:
        print(f"- {insight}")
    print(f"Saved {len(result['charts'])} visualizations and analysis outputs under {args.output}")


if __name__ == "__main__":
    main()
