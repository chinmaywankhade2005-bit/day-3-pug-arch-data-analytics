# Day 3: Data Analysis & Python for AI/ML

## Project overview and problem statement

This project practices NumPy, Pandas, and Matplotlib by inspecting, cleaning, analyzing, and visualizing facility inspection records. It is an educational implementation of the Day 3 assignment in the Pug Arch 10-Day Intern Technical Training and Domain Assessment Program.

The bundled facility records are **synthetic sample data** created for this assignment. They are not measurements from real facilities and should not be used to make operational decisions. The raw input is kept separately from every cleaned or derived output.

## Dataset and fields

The source is `data/raw/facility_inspections.csv`, generated reproducibly by `generate_sample_data.py`. It contains 50 rows: 48 patterned facility observations, one exact duplicate, and one additional extreme but range-valid observation. Deliberate data-quality examples include a missing score, an unrecognized water label, an out-of-range score, an invalid negative count, an invalid inspection date, and high numeric values that exercise outlier detection.

| Field | Meaning and accepted values |
| --- | --- |
| `facility_id` | Facility identifier; non-empty text required. |
| `location` | Synthetic site group. |
| `cleanliness_score` | Score from 0 to 100; higher means cleaner. |
| `odor_score` | Score from 0 to 100; higher means more severe odor. |
| `waste_level` | Score from 0 to 100; higher means more observed waste. |
| `water_availability` | `Available` or `Unavailable`; unknown source labels are represented as `Unknown` after cleaning. |
| `footfall` | Nonnegative visitor count. |
| `complaints` | Nonnegative complaint count. |
| `inspection_date` | Parseable inspection date in ISO `YYYY-MM-DD` form in cleaned output. |

## Technology stack

- Python 3.14 (the project also uses commonly supported Python 3 releases)
- NumPy for arrays, indexing, vector operations, and aggregation
- Pandas for CSV loading, Series/DataFrames, filtering, sorting, grouping, aggregation, and data cleaning
- Matplotlib for five saved chart images
- `unittest` for workflow and edge-case tests

## Project structure

```text
day-03/
├── analysis.py
├── generate_sample_data.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/facility_inspections.csv
├── outputs/
│   ├── facility_inspections_cleaned.csv
│   ├── cleaning_log.csv
│   ├── quality_issues.csv
│   ├── analysis_summary.json
│   ├── statistics_by_location.csv
│   └── visualizations/
└── tests/test_analysis.py
```

The cleaned dataset is written to `outputs/facility_inspections_cleaned.csv`, separate from the preserved raw CSV.

## Data-cleaning decisions

`analysis.py` reads the bundled CSV with `pandas.read_csv` and validates the required fields. The raw CSV is never edited by the analysis. Cleaning totals and decisions are recorded in `outputs/cleaning_log.csv`; `outputs/quality_issues.csv` identifies affected source row numbers, facility IDs, fields, original values, and actions.

1. Remove later copies of exact duplicate rows while keeping the first. A repeated `facility_id` alone is not treated as a duplicate because facilities can be inspected more than once.
2. Parse score and count fields as numeric. Scores outside 0–100 and negative footfall/complaint counts are invalid; treat those cells as missing.
3. Parse inspection dates. Drop records without a facility ID or parseable inspection date because they cannot be reliably identified or placed in time.
4. Impute missing numeric cells with the location median, then the overall median if that location has no usable value. Use zero only as a final fallback if the whole column has no usable values.
5. Normalize water labels. Missing or unrecognized labels become `Unknown`; missing locations become `Unknown`.
6. Detect numeric outliers using the 1.5×IQR fences for each numeric field. Cap outlier values at the relevant fence in the cleaned copy and log field counts and bounds. The raw values remain intact in the separate raw file. This transparent educational rule can cap legitimate extremes, so inspect the log and domain context before using the method with real data.

An empty dataset, missing required columns, or a dataset with no remaining valid dated records raises a clear `ValueError`. The program refuses to overwrite generated output files that already exist. To rerun, select a new output path with `--output`.

## Analysis findings and insights

The script computes descriptive statistics for all five numeric measures, a location-level summary, water availability group summaries, and Pearson correlations. It also reports three findings based on the generated cleaned data. **Because the supplied data is synthetic, these findings demonstrate analysis only and do not describe real facilities.**

On the bundled synthetic sample, cleaning starts with 50 rows, removes 1 exact duplicate, drops 1 row with an invalid inspection date, and produces 48 cleaned rows. It treats 1 out-of-range score and 1 negative count as missing; missing numeric values are imputed as recorded in the summary JSON. Two records contain at least one IQR-flagged numeric outlier; the per-field flags and fences are in the report.

The current run produced these findings from the cleaned sample:

1. West has the highest mean cleanliness score (84.50), while South has the lowest (63.50).
2. Cleanliness and complaint counts have a Pearson correlation of -0.801 across the 48 cleaned records.
3. Records marked `Unavailable` average 0.42 more complaints than records marked `Available` (5.80 versus 5.38 per record).

These are descriptive patterns in constructed sample data, not claims about real facilities. Other measured summary values are saved in `outputs/analysis_summary.json` and `outputs/statistics_by_location.csv`.

Run the analysis to reproduce and print the findings. Since this workspace already contains generated outputs, choose a fresh output directory so existing files are preserved:

```powershell
python analysis.py --output outputs\run-2
```

The exact results are saved in `outputs/analysis_summary.json` and `outputs/statistics_by_location.csv`. Do not interpret correlation as causation.

## Visualizations

Five PNG files are generated under `outputs/visualizations/`:

1. Mean cleanliness score by location (bar chart)
2. Total complaints by location (bar chart)
3. Cleanliness score distribution (histogram)
4. Odor score versus complaints (scatter plot)
5. Water availability records by location (stacked bar chart)

## Installation

Use an isolated environment inside this Day 3 folder if you want to install dependencies without changing another project's environment:

```powershell
cd D:\Pugarch\Pugarch-Internship\day-03
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No virtual environment is required if compatible NumPy, Pandas, and Matplotlib are already available in the Python environment you choose.

## How to run

From `day-03/`, the bundled sample dataset can be regenerated only if the raw CSV does not already exist:

```powershell
python generate_sample_data.py
```

Run the analysis and create the cleaned dataset, cleaning log, row-level quality report, summary, location statistics, and five charts. Pick a new output path each time; the script stops if any target output already exists:

```powershell
python analysis.py --output outputs\run-2
```

To analyze another CSV with the same required columns, provide a separate input and a fresh output directory:

```powershell
python analysis.py --input path\to\facility_inspections.csv --output path\to\new_outputs
```

## Testing

Run the tests from this folder:

```powershell
python -m unittest discover -s tests -v
```

The test suite covers the known sample-data quality issues, source preservation, empty data, missing columns, unrepairable dates, all five chart outputs, analysis findings, and output overwrite protection.

Latest test run: **5 tests passed** with `python -m unittest discover -s tests -v`.

## Challenges and solutions

- **Synthetic data has no external validity:** mark the dataset and every generated report as synthetic, and keep the sample separate from raw data supplied by an organization.
- **Some records cannot be repaired reliably:** discard rows lacking a facility ID or valid inspection date, while imputing recoverable numeric cells and preserving an audit log.
- **An extreme value may be real:** cap outliers only in the cleaned copy, retain the original CSV unchanged, and record the IQR bounds so the choice can be reviewed.
- **Repeat runs could replace evidence:** check expected output paths first and refuse to overwrite them; use a new output directory for another run.

## Future improvements

- Replace the sample with an approved, documented real facility dataset and revise score validation rules with domain owners.
- Compare alternative outlier methods and imputation strategies, documenting their effects on findings.
- Add time-series analysis across repeated inspections and more detailed validation reporting.
- Use a small user interface or notebook for interactive exploration and stakeholder review.
