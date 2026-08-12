# Log File Analyzer Report

## 1. Objective

The objective of this project is to analyze an application log file using Python. The program identifies all ERROR and WARNING log entries, groups them according to the application module, and generates a CSV report showing the frequency of each log level for every module.

---

## 2. Dataset

A sample application log file named `application.log` was used.

Each log entry contains:

* Date
* Time
* Log Level
* Module Name
* Message

---

## 3. Libraries Used

* pandas

---

## 4. Methodology

The following steps were performed:

1. Read the application log file line by line.
2. Split each log entry into its components.
3. Store the entries in a Pandas DataFrame.
4. Filter only the ERROR and WARNING records.
5. Group the filtered records by Module and Log Level.
6. Count the frequency of each group.
7. Export the final report as a CSV file.

---

## 5. Output

The program generates:

* log_analysis_report.csv

The report contains:

* Module
* Log Level
* Frequency

---

## 6. Conclusion

The log analyzer successfully identifies all ERROR and WARNING events and summarizes their frequency by application module. This type of analysis helps developers quickly identify modules experiencing the highest number of issues.

