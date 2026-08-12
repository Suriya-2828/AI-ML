# log_file_analyzer.py

import pandas as pd

# ----------------------------------------
# Read Log File
# ----------------------------------------

log_file = "application.log"

records = []

with open(log_file, "r") as file:
    for line in file:
        parts = line.strip().split()

        date = parts[0]
        time = parts[1]
        level = parts[2]
        module = parts[3]
        message = " ".join(parts[4:])

        records.append([date, time, level, module, message])

# ----------------------------------------
# Create DataFrame
# ----------------------------------------

df = pd.DataFrame(
    records,
    columns=[
        "Date",
        "Time",
        "Level",
        "Module",
        "Message"
    ]
)

print("\nOriginal Log Entries\n")
print(df.head())

# ----------------------------------------
# Filter ERROR and WARNING
# ----------------------------------------

filtered = df[df["Level"].isin(["ERROR", "WARNING"])]

print("\nFiltered Logs\n")
print(filtered)

# ----------------------------------------
# Error/Warning Frequency by Module
# ----------------------------------------

report = (
    filtered.groupby(["Module", "Level"])
    .size()
    .reset_index(name="Frequency")
)

print("\nFrequency Report\n")
print(report)

# ----------------------------------------
# Save CSV Report
# ----------------------------------------

report.to_csv("log_analysis_report.csv", index=False)

print("\nCSV Report Generated Successfully.")
print("File Saved: log_analysis_report.csv")
