# Manufacturing Defect Detection

## Requirements

- Python 3.11+
- pandas
- numpy
- pillow
- scikit-learn
- joblib

## Install

```bash
pip install numpy pillow scikit-learn joblib
```

## Dataset

Download the NEU Surface Defect Dataset.

Place it like:

```
dataset/
    Crazing/
    Inclusion/
    Patches/
    Pitted/
    Rolled/
    Scratches/
```

## Run

```bash
python defect_detection.py
```

## Output

- Accuracy
- Classification Report
- defect_model.pkl
