from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_CSV_2000 = Path("assets/metals_2000.csv")
DEFAULT_CSV_2025 = Path("assets/metals_2025.csv")
DEFAULT_OUT = Path("assets/metal_concentrations.png")

# Canonicalize: lowercase, strip, remove non-alphanumerics, collapse spaces/underscores.
_RAW_ID_KEYS = {"sample id", "sample_id", "id", "sample"}
ID_LIKE_KEYS = {"".join(ch for ch in k.lower().strip() if ch.isalnum()) for k in _RAW_ID_KEYS}

@dataclass(slots=True)
class Inputs:
    csv_2000: Path
    csv_2025: Path
    out_path: Path
    show: bool

def _canon(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch == "_")
    s = s.replace("_", " ")
    return " ".join(s.split())

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
        val = str(raw).strip().strip('"').strip("'")
        # Unwrap common wrappers
        if (val.startswith("(") and val.endswith(")")) or (val.startswith("[") and val.endswith("]")):
            val = val[1:-1].strip()
        val = " ".join(val.split())
        if val:
            units_map[col] = val
    return headers, units_map

def load_numeric_data(path: Path) -> pd.DataFrame:
    """
    Load data skipping:
      - Row 1: units
      - Additional early rows that are blank/NaN
    Coerces to numeric where possible. Drops ID-like columns.
    """
    head3 = pd.read_csv(path, nrows=3, encoding="utf-8-sig")
    # rows are 0-based in the DataFrame but 1-based skiprows for the file after header line
    skip = [1]
    for i in range(1, min(3, len(head3))):
        row = head3.iloc[i]
        if row.isna().all() or (row.astype(str).str.strip() == "").all():
            skip.append(i + 1)

    df = pd.read_csv(
        path,
        skiprows=sorted(set(skip)),
        encoding="utf-8-sig",
        keep_default_na=True,
    )

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop ID-like columns using canonical comparison
    drop_cols = [c for c in df.columns if _canon(c).replace(" ", "") in ID_LIKE_KEYS]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Coerce numerics; non-numeric -> NaN
    df = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # Drop columns that became entirely NaN (e.g., textual columns that slipped through)
    df = df.dropna(axis=1, how="all")

    return df

def harmonize_units(units_a: dict[str, str], units_b: dict[str, str], cols: list[str]) -> dict[str, str]:
    """
    Ensure the same unit per column across both datasets. If a mismatch is found,
    prefer the non-empty unit from 'a', but note the mismatch on stderr and tag the label.
    """
    final: dict[str, str] = {}
    for c in cols:
        ua = (units_a.get(c) or "").strip()
        ub = (units_b.get(c) or "").strip()
        if ua and ub and ua != ub:
            print(f"Warning: Unit mismatch for column '{c}': 2000='{ua}' vs 2025='{ub}'", file=sys.stderr)
            final[c] = f"{ua}*"
        else:
            final[c] = ua or ub or ""
    return final

def compute_common_means(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    means_a = df_a.mean(numeric_only=True)
    means_b = df_b.mean(numeric_only=True)
    common = sorted(set(means_a.index).intersection(means_b.index))
    if not common:
        raise SystemExit(
            "No overlapping numeric columns found between files. "
            "Ensure column names align (e.g., 'As', 'Cd', ...)."
        )
    return common, means_a[common].to_numpy(dtype=float), means_b[common].to_numpy(dtype=float)

def _default_should_show() -> bool:
    # Heuristic: interactive if both stdin and stdout are ttys
    return sys.stdin.isatty() and sys.stdout.isatty()

def parse_args() -> Inputs:
    p = argparse.ArgumentParser(description="Compare average metal concentrations between two CSVs.")
    p.add_argument("--csv-2000", type=Path, default=DEFAULT_CSV_2000, help="Path to 2000 CSV")
    p.add_argument("--csv-2025", type=Path, default=DEFAULT_CSV_2025, help="Path to 2025 CSV")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output figure path (PNG)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--show", dest="show", action="store_true", help="Display the plot window")
    g.add_argument("--no-show", dest="show", action="store_false", help="Do not display the plot window")
    p.set_defaults(show=_default_should_show())
    args = p.parse_args()
    return Inputs(csv_2000=args.csv_2000, csv_2025=args.csv_2025, out_path=args.out, show=args.show)

def main() -> None:
    inp = parse_args()

    if not inp.csv_2000.exists():
        raise SystemExit(f"Input CSV not found: {inp.csv_2000}")
    if not inp.csv_2025.exists():
        raise SystemExit(f"Input CSV not found: {inp.csv_2025}")

    # Read headers and units from both files
    headers_2000, units_2000 = read_header_and_units(inp.csv_2000)
    headers_2025, units_2025 = read_header_and_units(inp.csv_2025)
    _ = (headers_2000, headers_2025)  # currently unused, but kept for future validation

    # Load numeric data
    df_2000 = load_numeric_data(inp.csv_2000)
    df_2025 = load_numeric_data(inp.csv_2025)

    # Compute common columns and their means
    metals, y2000, y2025 = compute_common_means(df_2000, df_2025)

    # Build a unified units map for those columns
    unified_units = harmonize_units(units_2000, units_2025, metals)

    # Determine if all units are identical (ignoring mismatch flags)
    unique_units = {u.rstrip("*") for u in unified_units.values() if u}

    # Plot
    x = np.arange(len(metals))
    width = 0.38
    # Clamp figure width for very wide sets
    fig_width = max(10, min(18, int(len(metals) * 0.7)))
    figsize = (fig_width, 6)

    # Build labels
    xticklabels: list[str] = []
    only_unit: str | None = None

    if len(unique_units) > 1:
        for c in metals:
            u = unified_units.get(c, "")
            if not u:
                xticklabels.append(c)
            else:
                label = f"{c} ({u[:-1]})*" if u.endswith("*") else f"{c} ({u})"
                xticklabels.append(label)
    else:
        xticklabels = metals[:]
        only_unit = next(iter(unique_units), None)

    # Style baseline for consistency
    plt.rcParams.update({
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    bars1 = ax.bar(x - width / 2, y2000, width=width, label="2000", color="#4C78A8")
    bars2 = ax.bar(x + width / 2, y2025, width=width, label="2025", color="#F58518")

    ax.set_xticks(x, xticklabels)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    ylabel = f"Average concentration ({only_unit})" if only_unit else "Average concentration (per metal units)"
    ax.set_ylabel(ylabel)
    ax.set_title("Average Metal Concentrations: 2000 vs 2025")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=8)

    # Add headroom so labels don't crowd the top
    ymax_2000 = float(np.max(y2000)) if y2000.size else 0.0
    ymax_2025 = float(np.max(y2025)) if y2025.size else 0.0
    ymax = max(ymax_2000, ymax_2025)

    if ymax > 0:
        ax.set_ylim(top=ymax * 1.15)

    inp.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(inp.out_path, dpi=200)
    if inp.show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()