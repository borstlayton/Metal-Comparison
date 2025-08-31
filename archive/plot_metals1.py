from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_2000 = Path("assets/metals_2000.csv")
CSV_FILE_2025 = Path("assets/metals_2025.csv")
OUT_PATH = Path("assets/Figure.png")

ID_LIKE_COLUMNS = {"sample id", "sample_id", "id", "sample"}

def read_header_and_units(path: Path) -> tuple[list[str], dict[str, str]]:
    # Read header + two rows (units row + potentially blank/data)
    head = pd.read_csv(
        path,
        nrows=3,
        encoding="utf-8-sig",
        keep_default_na=True,
    )
    headers = [str(c).strip() for c in head.columns]
    units_row = head.iloc[0]  # first data row should be units

    units_map: dict[str, str] = {}
    for col, raw in zip(headers, units_row):
        if pd.isna(raw):
            continue
        val = str(raw).strip()
        if val.startswith("(") and val.endswith(")"):
            val = val[1:-1].strip()
        val = " ".join(val.split())
        if val:
            units_map[col] = val
    return headers, units_map

def load_numeric_data(path: Path) -> pd.DataFrame:
    """
    Load data skipping:
      - Row 1: units
      - Row 2: blank/NaN (if present)
    Coerces to numeric where possible. Drops ID-like columns.
    """
    # Peek first 3 rows to decide if row 2 is blank
    head3 = pd.read_csv(path, nrows=3, encoding="utf-8-sig")
    skip = [1]
    if len(head3) >= 2:
        # Consider row index 1 (second data row, zero-based within file is 2) possibly blank
        row1 = head3.iloc[1]
        if row1.isna().all() or (row1.astype(str).str.strip() == "").all():
            skip.append(2)

    df = pd.read_csv(
        path,
        skiprows=skip,
        encoding="utf-8-sig",
        keep_default_na=True,
    )

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop ID-like columns (case-insensitive, with whitespace stripped)
    drop_cols = [c for c in df.columns if c.lower().strip() in ID_LIKE_COLUMNS]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Coerce numerics; non-numeric -> NaN
    df = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # Drop columns that became entirely NaN (e.g., textual columns that slipped through)
    df = df.dropna(axis=1, how="all")

    return df

def harmonize_units(units_a: Dict[str, str], units_b: Dict[str, str], cols: List[str]) -> Dict[str, str]:
    """
    Ensure the same unit per column across both datasets. If a mismatch is found,
    prefer the non-empty unit from 'a', but note the mismatch on stderr and tag the label.
    """
    final: Dict[str, str] = {}
    for c in cols:
        ua = (units_a.get(c) or "").strip()
        ub = (units_b.get(c) or "").strip()
        if ua and ub and ua != ub:
            print(f"Warning: Unit mismatch for column '{c}': 2000='{ua}' vs 2025='{ub}'", file=sys.stderr)
            final[c] = f"{ua}*"
        else:
            final[c] = ua or ub or ""
    return final

def compute_common_means(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
    means_a = df_a.mean(numeric_only=True)
    means_b = df_b.mean(numeric_only=True)
    common = sorted(set(means_a.index).intersection(means_b.index))
    if not common:
        sys.exit("No overlapping numeric columns found. Ensure column names align (e.g., 'As', 'Cd', ...).")
    return common, means_a[common].to_numpy(dtype=float), means_b[common].to_numpy(dtype=float)


def main() -> None:
    if not CSV_FILE_2000.exists() or not CSV_FILE_2025.exists():
        sys.exit("Input CSV file(s) not found in assets/.")

    # Read headers and units from both files
    headers_2000, units_2000 = read_header_and_units(CSV_FILE_2000)
    headers_2025, units_2025 = read_header_and_units(CSV_FILE_2025)

    # Load numeric data
    df_2000 = load_numeric_data(CSV_FILE_2000)
    df_2025 = load_numeric_data(CSV_FILE_2025)

    # Compute common columns and their means
    metals, y2000, y2025 = compute_common_means(df_2000, df_2025)

    # Build a unified units map for those columns
    unified_units = harmonize_units(units_2000, units_2025, metals)
    
    # Determine if all units are identical (ignoring mismatch flags)
    unique_units = {u.rstrip("*") for u in unified_units.values() if u}

    # Plot
    x = np.arange(len(metals))
    width = 0.38
    figsize = (max(10, int(len(metals) * 0.7)), 6)

    # Build labels first (no ax usage here)
    xticklabels: list[str] = []
    only_unit: str | None = None

    if len(unique_units) > 1:
        for c in metals:
            u = unified_units.get(c, "")
            if not u:
                xticklabels.append(c)
            else:
                xticklabels.append(f"{c} ({u[:-1]})*" if u.endswith("*") else f"{c} ({u})")
    else:
        xticklabels = metals[:]
        only_unit = next(iter(unique_units), None)

    # ... now create the plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    bars1 = ax.bar(x - width/2, y2000, width=width, label="2000", color="#4C78A8")
    bars2 = ax.bar(x + width/2, y2025, width=width, label="2025", color="#F58518")

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    # Set ylabel once, now that ax exists
    if only_unit:
        ax.set_ylabel(f"Average concentration ({only_unit})")
    else:
        # Fall back if mixed units or none detected
        ax.set_ylabel("Average concentration (per metal units)")

    ax.set_title("Average Metal Concentrations: 2000 vs 2025")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()