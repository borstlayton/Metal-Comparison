# Metal concentration comparison (2000 vs 2025)

This project creates a simple bar chart comparing the average concentrations of metals measured in the years 2000 and 2025 from two csv files. 

---

## What’s included

- `metals_2000.csv` — Excel data for year 2000
- `metals_2025.csv` — Excel data for year 2025
- `plot_metals.py` — Python script that reads the Excel files and makes the chart
- `requirements.txt` — List of Python packages the script needs
- `Figure.png` — Pre-generated graph

![This is `Figure.png`](.\Figure.png)

> The script computes the average value for each metal, finds metals present in both files, and plots them side-by-side.

---

## Overview

- You’ll use a terminal called PowerShell to run a few commands.
- You’ll install Python.
- You’ll run one script to see the chart.

---

## Step 1 — Install Python on Windows

1. Go to: https://www.python.org/downloads
2. Click “Download Python” (latest version).
3. Run the installer.
4. Important: On the first screen, check the box:
   - “Add Python to PATH”
5. Click “Install Now” and complete the setup.

Verify the installation:
- Press Windows key, type “PowerShell”, and open “Windows PowerShell”.
- In PowerShell, type:

```
python --version
```

You should see something like Python 3.x.x.
Check pip (Python’s package installer):

```
pip --version
```

If both show a version number, you’re good to go.
If pip doesn’t work, try:

```
python -m pip --version
```

---

## Step 2 — Put the files in a folder

Create a folder (for example, C:\MetalChart), and place these files inside:
- metals_2000.csv
- metals_2025.csv
- plot_metals.py
- requirements.txt

Open PowerShell in that folder:

- In File Explorer, open the folder.
- Click in the address bar, type powershell, and press Enter.

---

## Step 3 — Install the project’s dependencies

In PowerShell, run:

```
python -m pip install -r requirements.txt
```

This installs:
- pandas (reads csv files and handles data)
- matplotlib (makes the chart)

---

## Step 4 — Run the script

Still in PowerShell, run:

```
python plot_metals.py
```

A chart window will appear showing average concentrations for metals found in both years (2000 vs 2025). Close the chart window when you’re done.
