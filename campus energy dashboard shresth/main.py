import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================
# OOP CLASSES (Fixed)
# =========================================

class MeterReading:
    def __init__(self, timestamp, kwh):
        self.timestamp = timestamp
        self.kwh = kwh


class Building:
    def __init__(self, name):
        self.name = name
        self.meter_readings = []

    def add_reading(self, meter_reading):
        self.meter_readings.append(meter_reading)

    def total_consumption(self):
        return sum(r.kwh for r in self.meter_readings)

    def generate_report(self):
        total = self.total_consumption()
        return f"Building {self.name}: total consumption = {total:.2f} kWh"


class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def get_or_create_building(self, name):
        if name not in self.buildings:
            self.buildings[name] = Building(name)
        return self.buildings[name]

    def add_reading(self, building_name, timestamp, kwh):
        building = self.get_or_create_building(building_name)
        building.add_reading(MeterReading(timestamp, kwh))

    def generate_all_reports(self):
        return [b.generate_report() for b in self.buildings.values()]


# =========================================
# DATA INGESTION
# =========================================

def load_and_merge_data(data_dir, log_path):
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    frames = []
    logs = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, on_bad_lines="skip")

            if not {"timestamp", "kwh"}.issubset(df.columns):
                logs.append(f"{file.name}: Missing required columns.\n")
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "kwh"])
            df["building"] = file.stem
            frames.append(df)

        except Exception as e:
            logs.append(f"{file.name}: Error - {e}\n")

    with open(log_path, "w") as f:
        f.writelines(logs)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================================
# AGGREGATION FUNCTIONS (INDEX FIXED)
# =========================================

def calculate_daily_totals(df):
    df = df.set_index("timestamp")
    return df.resample("D")["kwh"].sum().reset_index()


def calculate_weekly_aggregates(df):
    df = df.set_index("timestamp")
    return df.resample("W")["kwh"].sum().reset_index()


def building_wise_summary(df):
    summary = df.groupby("building")["kwh"].agg(
        mean="mean",
        min="min",
        max="max",
        total="sum"
    )
    return summary.reset_index()


# =========================================
# DASHBOARD
# =========================================

def create_dashboard(df, output_path):
    df = df.set_index("timestamp")

    daily = df.resample("D")["kwh"].sum()
    weekly_avg = df.groupby("building").resample("W")["kwh"].sum().groupby("building").mean()

    hourly = df.resample("h")["kwh"].sum().reset_index()
    peak = hourly.nlargest(50, "kwh")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].plot(daily.index, daily.values)
    ax[0].set_title("Daily Campus Consumption")

    ax[1].bar(weekly_avg.index, weekly_avg.values)
    ax[1].set_title("Avg Weekly Usage Per Building")

    ax[2].scatter(peak["timestamp"], peak["kwh"], s=10)
    ax[2].set_title("Peak Hour Consumption")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================================
# SUMMARY GENERATOR
# =========================================

def generate_summary_text(df, summary_df, output_txt):
    total = df["kwh"].sum()
    highest = summary_df.sort_values("total", ascending=False).iloc[0]

    peak = df.loc[df["kwh"].idxmax()]

    daily = calculate_daily_totals(df)
    weekly = calculate_weekly_aggregates(df)

    lines = [
        "Campus Energy Use Summary\n",
        "=========================\n\n",
        f"Total campus consumption: {total:.2f} kWh\n",
        f"Highest consuming building: {highest['building']} ({highest['total']:.2f} kWh)\n",
        f"Peak load time: {peak['timestamp']} with {peak['kwh']:.2f} kWh\n\n",
        f"Daily points: {len(daily)}\n",
        f"Weekly points: {len(weekly)}\n",
    ]

    with open(output_txt, "w") as f:
        f.writelines(lines)

    print("".join(lines))


# =========================================
# MAIN SCRIPT
# =========================================

if __name__ == "__main__":
    base = Path(__file__).parent
    data_dir = base / "data"
    out = base / "output"
    out.mkdir(exist_ok=True)

    log_file = out / "log.txt"
    df = load_and_merge_data(data_dir, log_file)

    if df.empty:
        print("No valid CSV data found.")
        exit()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    summary_df = building_wise_summary(df)

    # ---- OOP Manager ----
    manager = BuildingManager()
    for row in df.itertuples(index=False):
        manager.add_reading(row.building, row.timestamp, row.kwh)

    for r in manager.generate_all_reports():
        print(r)

    # ---- Dashboard and Text Summary ----
    create_dashboard(df, out / "dashboard.png")
    generate_summary_text(df, summary_df, out / "summary.txt")

    print("All tasks completed. Check output folder.")
