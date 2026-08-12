# Log File Analyzer

## Overview

This project analyzes an application log file and extracts all ERROR and WARNING entries. The filtered logs are grouped by application module, and a CSV report containing the frequency of each log level is generated.

---

## Requirements

* Python 3.11+
* pandas

Install:

```bash
pip install pandas
```

---

## Files

```
Question2/
│
├── application.log
├── log_file_analyzer.py
├── log_analysis_report.csv
├── README.md
└── report.pdf
```

---

## How to Run

```bash
python log_file_analyzer.py
```

---

## Output

Generated file:

* log_analysis_report.csv

The report contains:

* Module
* Log Level
* Frequency

---

## Author

Suriya

