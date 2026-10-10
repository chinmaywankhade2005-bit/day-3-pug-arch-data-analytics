"""Create the fixed, reproducible synthetic facility dataset used in this project."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "raw" / "facility_inspections.csv"


def build_rows() -> list[dict[str, object]]:
    """Build 48 patterned observations and five deliberate data-quality cases."""
    rows: list[dict[str, object]] = []
    sites = ["North", "South", "East", "West"]
    cleanliness_offsets = {"North": 0, "South": -17, "East": -10, "West": 4}
    odor_offsets = {"North": 0, "South": 14, "East": 8, "West": -3}
    waste_offsets = {"North": 0, "South": 13, "East": 6, "West": -2}

    for site_index, location in enumerate(sites):
        for facility_number in range(1, 13):
            step = facility_number - 1
            cleanliness = 84 + cleanliness_offsets[location] - (step % 5) * 2
            odor = 13 + odor_offsets[location] + (step % 4) * 2
            waste = 17 + waste_offsets[location] + (step % 3) * 3
            footfall = 110 + site_index * 18 + (step % 6) * 13
            complaints = max(0, round((100 - cleanliness) / 12 + odor / 18 + waste / 22 + (step % 3)))
            water = "Unavailable" if (step + site_index) % 5 == 0 else "Available"
            rows.append(
                {
                    "facility_id": f"FAC-{site_index + 1:02d}-{facility_number:03d}",
                    "location": location,
                    "cleanliness_score": cleanliness,
                    "odor_score": odor,
                    "waste_level": waste,
                    "water_availability": water,
                    "footfall": footfall,
                    "complaints": complaints,
                    "inspection_date": f"2026-{(step % 6) + 1:02d}-{((step * 2) % 27) + 1:02d}",
                }
            )

    # Intentional, documented quality issues for exercising the cleaning workflow.
    rows[2]["cleanliness_score"] = ""
    rows[8]["water_availability"] = "Unknown status"
    rows[15]["odor_score"] = 145  # outside the documented 0-100 score range
    rows[22]["footfall"] = 25000  # IQR outlier, but a valid nonnegative count
    rows[31]["inspection_date"] = "not-a-date"
    rows[40]["complaints"] = -4  # invalid negative count
    rows.append(dict(rows[5]))  # exact duplicate record
    rows.append(
        {
            "facility_id": "FAC-99-001",
            "location": "East",
            "cleanliness_score": 1,
            "odor_score": 99,
            "waste_level": 98,
            "water_availability": "Unavailable",
            "footfall": 120,
            "complaints": 34,
            "inspection_date": "2026-06-18",
        }
    )  # valid values but deliberate multivariate extreme observations
    return rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing sample data: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} sample rows at {OUTPUT}")


if __name__ == "__main__":
    main()
